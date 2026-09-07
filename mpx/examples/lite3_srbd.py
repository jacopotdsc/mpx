"""Standalone SRBD MPC + whole-body controller on the Lite3. No RL.

Mirror of srbd_quad.py (Aliengo) with the Lite3 model and configuration.

    python lite3_srbd.py                 # viewer; arrows drive vx/wz,
                                         # Home/End (or PgUp/PgDn) drive vy
    python lite3_srbd.py --headless      # no viewer
    python lite3_srbd.py --headless --metrics out.csv --cmd 0.3 0 0
"""

import argparse
import os
import sys
import time
from timeit import default_timer as timer

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, "..")))
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np

import mpx.config.config_lite3 as config
import mpx.utils.mpc_wrapper_srbd as mpc_wrapper_srbd
import mpx.utils.sim as sim_utils

from mujoco_playground._src.locomotion.lite3 import lite3_constants as consts

# Lateral (vy) keys. Not letters: MuJoCo's viewer reserves every key A-Z for its
# own visualization flags (mjVISSTRING/mjRNDSTRING), so a letter binding never
# reaches the callback. Home/End and PageUp/PageDown are free.
KEY_PAGE_UP, KEY_PAGE_DOWN = 266, 267
KEY_HOME, KEY_END = 268, 269
FORWARD_STEP = 0.1
FORWARD_LIMIT = 3.0
LATERAL_STEP = 0.1
LATERAL_LIMIT = 2.0
YAW_STEP = 0.1
YAW_LIMIT = 2.5

jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


def _reset_to_initial_state(model, data):
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
        data.qvel[:] = 0.0
    else:
        data.qpos = np.asarray(jnp.concatenate([config.p0, config.quat0, config.q0]))
        data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def _srbd_state(qpos, qvel):
    return jnp.concatenate([
        jnp.asarray(qpos[:3]),
        jnp.asarray(qpos[3:7]),
        jnp.asarray(qvel[:3]),
        jnp.asarray(qvel[3:6]),
    ])


def main(headless=False, steps=500, metrics=None, cmd=None):
    model = mujoco.MjModel.from_xml_path(consts.task_to_xml("flat_terrain").as_posix())
    data = mujoco.MjData(model)

    sim_frequency = float(config.whole_body_frequency)
    model.opt.timestep = 1.0 / sim_frequency

    contact_ids = sim_utils.geom_ids(model, config.contact_frame)
    touch_adr = [model.sensor_adr[model.sensor(f"{f}_floor_found").id]
                 for f in config.contact_frame]

    command_handle = sim_utils.KeyboardVelocityCommand(
        vx=0.0,
        vy=0.0,
        wz=0.0,
        forward_step=FORWARD_STEP,
        yaw_step=YAW_STEP,
        forward_limits=(-FORWARD_LIMIT, FORWARD_LIMIT),
        yaw_limits=(-YAW_LIMIT, YAW_LIMIT),
    )
    # `cmd` is either a fixed (vx, vy, wz) or a callable step -> (vx, vy, wz),
    # the latter to script command transitions in headless validation runs.
    cmd_fn = cmd if callable(cmd) else (lambda k: cmd)
    if cmd is not None:
        command_handle.vx, command_handle.vy, command_handle.wz = cmd_fn(0)

    mpc = mpc_wrapper_srbd.BatchedMPCControllerWrapper(config, n_env=1)
    _reset_to_initial_state(model, data)

    print(f"  model      {consts.task_to_xml('flat_terrain').name}  "
          f"nq/nv/nu {model.nq}/{model.nv}/{model.nu}  mass {model.body_subtreemass[1]:.3f} kg")
    print(f"  feet       {config.contact_frame} -> geoms {list(contact_ids)}")
    print(f"  sim {sim_frequency:.0f} Hz | mpc {config.mpc_frequency} Hz | "
          f"horizon {config.N}x{config.dt}s = {config.N*config.dt:.2f}s")

    def read_contact():
        return jnp.asarray([data.sensordata[a] > 0 for a in touch_adr], dtype=jnp.float32)

    # Warm-up: compile everything before the loop starts.
    foot = jnp.asarray(sim_utils.geom_positions(data, contact_ids))
    x0 = _srbd_state(data.qpos, data.qvel)
    command = jnp.asarray(command_handle.mpc_input(config.robot_height))
    mpc_state = mpc.init_state()
    mpc_state = mpc.run(mpc_state, x0[None, :], command[None, :],
                        foot[None, :], read_contact()[None, :])
    tau_warm, _ = mpc.whole_body_run(mpc_state,
                                     jnp.asarray(data.qpos)[None, :],
                                     jnp.asarray(data.qvel)[None, :])
    tau_warm.block_until_ready()
    mpc.reset()
    _reset_to_initial_state(model, data)

    period = int(sim_frequency / config.mpc_frequency)
    counter = 0
    log = []

    def step_controller(mpc_state):
        nonlocal counter
        if cmd is not None:
            command_handle.vx, command_handle.vy, command_handle.wz = cmd_fn(counter)
        qpos, qvel = data.qpos.copy(), data.qvel.copy()

        if counter % period == 0:
            foot = jnp.asarray(sim_utils.geom_positions(data, contact_ids))
            command = jnp.asarray(command_handle.mpc_input(config.robot_height))
            mpc_state = mpc.run(mpc_state, _srbd_state(qpos, qvel)[None, :],
                                command[None, :], foot[None, :], read_contact()[None, :])

        tau_cmd, _ = mpc.whole_body_run(mpc_state,
                                        jnp.asarray(qpos)[None, :],
                                        jnp.asarray(qvel)[None, :])
        tau = np.asarray(tau_cmd[0])
        data.ctrl = tau
        mujoco.mj_step(model, data)
        counter += 1

        if metrics is not None:
            log.append(_sample(model, data, mpc_state, tau, touch_adr, command_handle))
        return mpc_state

    def key_callback(keycode):
        """Arrow keys drive vx/wz through the shared handle; Home/End and
        PageUp/PageDown drive vy, which the handle neither binds nor clips."""
        if keycode in (KEY_HOME, KEY_PAGE_UP):
            command_handle.vy = float(
                np.clip(command_handle.vy + LATERAL_STEP, -LATERAL_LIMIT, LATERAL_LIMIT))
        elif keycode in (KEY_END, KEY_PAGE_DOWN):
            command_handle.vy = float(
                np.clip(command_handle.vy - LATERAL_STEP, -LATERAL_LIMIT, LATERAL_LIMIT))
        else:
            command_handle.key_callback(keycode)

    def overlay_text():
        """Two aligned columns: the key that drives each command, and the
        commanded value next to the one the robot is actually achieving.
        Linear velocities are in the base frame, wz is the yaw rate."""
        R = data.xmat[1].reshape(3, 3)
        v = R.T @ data.qvel[:3]
        return (
            "Key\n"
            "Up/Down\n"
            "Home/End or PgUp/PgDn\n"
            "Left/Right\n"
            "Space",
            f"        cmd   actual\n"
            f"vx    {command_handle.vx:+6.2f}   {v[0]:+6.2f}\n"
            f"vy    {command_handle.vy:+6.2f}   {v[1]:+6.2f}\n"
            f"wz    {command_handle.wz:+6.2f}   {data.qvel[5]:+6.2f}\n"
            f"stop  (vy limit +-{LATERAL_LIMIT:.1f})",
        )

    if headless:
        for _ in range(steps):
            mpc_state = step_controller(mpc_state)
    else:
        with mujoco.viewer.launch_passive(
            model, data, key_callback=key_callback
        ) as viewer:
            viewer.sync()
            while viewer.is_running():
                viewer.set_texts((None, None, *overlay_text()))
                tic = timer()
                mpc_state = step_controller(mpc_state)
                toc = timer()
                if toc - tic < model.opt.timestep:
                    time.sleep(model.opt.timestep - (toc - tic))
                viewer.sync()

    if metrics is not None and log:
        keys = list(log[0].keys())
        arr = np.array([[r[k] for k in keys] for r in log])
        np.savetxt(metrics, arr, delimiter=",", header=",".join(keys), comments="")
        print(f"  metrics -> {metrics}  ({len(log)} rows)")
    return log


def _sample(model, data, mpc_state, tau, touch_adr, command_handle):
    """One row of diagnostics. Only built when --metrics is given."""
    w, x, y, z = data.qpos[3:7]
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    R = data.xmat[1].reshape(3, 3)
    v_local = R.T @ data.qvel[:3]
    grf = np.asarray(mpc_state.grf[0])
    planned = np.asarray(mpc_state.contact[0])
    row = dict(
        x=data.qpos[0], y=data.qpos[1], base_z=data.qpos[2],
        roll=roll, pitch=pitch, yaw=yaw,
        vx=v_local[0], vy=v_local[1], vz=v_local[2],
        wx=data.qvel[3], wy=data.qvel[4], wz=data.qvel[5],
        cmd_vx=command_handle.vx, cmd_vy=command_handle.vy, cmd_wz=command_handle.wz,
        grf_z_sum=float(grf[2::3].sum()),
    )
    for i, f in enumerate(["FL", "FR", "HL", "HR"]):
        row[f"{f}_contact"] = float(data.sensordata[touch_adr[i]] > 0)
        row[f"{f}_planned"] = float(planned[i])
        row[f"{f}_grf_z"] = float(grf[3 * i + 2])
    for j in range(12):
        row[f"tau{j}"] = float(tau[j])
        row[f"q{j}"] = float(data.qpos[7 + j])
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--metrics", type=str, default=None,
                        help="save a diagnostics CSV (off by default)")
    parser.add_argument("--cmd", nargs=3, type=float, default=None,
                        metavar=("VX", "VY", "WZ"))
    args = parser.parse_args()
    main(headless=args.headless, steps=args.steps, metrics=args.metrics, cmd=args.cmd)
