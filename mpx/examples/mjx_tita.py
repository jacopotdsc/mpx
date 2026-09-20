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

import mpx.config.config_dfcip as config
import mpx.utils.mpc_wrapper_dfcip as mpc_wrapper_dfcip
import mpx.utils.mpc_utils as mpc_utils
import mpx.utils.sim as sim_utils

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


def _reset_to_initial_state(model, data):
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def _gather_raw_state(model, data, base_body_id, contact_ids):
    mujoco.mj_subtreeVel(model, data)
    pcom = data.subtree_com[base_body_id].copy()
    vcom = data.subtree_linvel[base_body_id].copy()

    centers = data.geom_xpos[contact_ids].copy()               # (2, 3)
    Rs = data.geom_xmat[contact_ids].reshape(2, 3, 3).copy()   # (2, 3, 3)
    radii = model.geom_size[contact_ids, 0].copy()             # (2,)

    feet_vel = np.zeros((2, 3))
    vel = np.zeros(6)
    for i, g in enumerate(contact_ids):
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_GEOM, int(g), vel, 0)
        feet_vel[i] = vel[3:6]
    return pcom, vcom, centers, Rs, radii, feet_vel


@jax.jit
def _process_state(pcom, vcom, centers, Rs, radii, feet_vel, theta_prev):
    l_rcp = mpc_utils.get_rCP(Rs[0], radii[0])
    r_rcp = mpc_utils.get_rCP(Rs[1], radii[1])

    pl_world = centers[0] + l_rcp
    pr_world = centers[1] + r_rcp
    dpl_world, dpr_world = feet_vel[0], feet_vel[1]

    tita_state = jnp.concatenate([pcom, vcom, pl_world, pr_world, dpl_world, dpr_world])

    c_world = (pl_world + pr_world) / 2.0
    vc_world = (dpl_world + dpr_world) / 2.0

    diff = pl_world - pr_world
    theta_wrapped = jnp.arctan2(-diff[0], diff[1])
    a = (theta_wrapped - theta_prev + jnp.pi) % (2 * jnp.pi)
    a = jnp.where(a < 0, a + 2 * jnp.pi, a) - jnp.pi
    theta = theta_prev + a

    ct, st = jnp.cos(theta), jnp.sin(theta)
    R = jnp.array([[ct, -st, 0.0], [st, ct, 0.0], [0.0, 0.0, 1.0]])
    dpl_b = R.T @ dpl_world
    dpr_b = R.T @ dpr_world
    w = (dpr_b[0] - dpl_b[0]) / config.d
    v = (dpr_b[0] + dpl_b[0]) / 2.0

    x0 = jnp.concatenate([
        pcom,
        vcom,
        c_world,
        jnp.array([vc_world[2]]),
        jnp.array([theta]),
        jnp.array([v]),
        jnp.array([w]),
    ])
    return tita_state, x0, theta


def main(headless=False, steps=500, scene="flat"):
    model = mujoco.MjModel.from_xml_path(dir_path + f"/../data/tita/scene_{scene}.xml")
    data = mujoco.MjData(model)
    sim_frequency = float(config.whole_body_frequency)
    model.opt.timestep = 1.0 / sim_frequency

    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, config.base_body_name)
    contact_ids = sim_utils.geom_ids(model, config.contact_frame)
    command_handle = sim_utils.KeyboardVelocityCommand(
        vx=0.0,
        vy=0.0,
        wz=0.0,
        forward_step=0.1,
        yaw_step=0.2,
        forward_limits=(-10.0, 10.0),
        yaw_limits=(-1.5, 1.5),
    )
    mpc = mpc_wrapper_dfcip.BatchedMPCControllerWrapper(config, n_env=1)

    _reset_to_initial_state(model, data)

    theta_prev = 0.0
    raw = _gather_raw_state(model, data, base_body_id, contact_ids)
    tita_state, x0, theta_prev = _process_state(*raw, theta_prev)
    command = jnp.asarray(command_handle.mpc_wheeled_input(config.com_z_to_track))
    mpc_state = mpc.init_state()
    mpc_state, _ = mpc.run(mpc_state, x0[None, :], command[None, :])
    mpc_state, tau, qddot, _, _, _ = mpc.whole_body_run(
        mpc_state,
        x0,
        jnp.asarray(data.qpos)[None, :],
        jnp.asarray(data.qvel)[None, :],
        tita_state[6:9][None, :],
        tita_state[9:12][None, :],
        tita_state[12:15][None, :],
        tita_state[15:18][None, :],
    )
    tau.block_until_ready()
    mpc.reset()

    period = int(sim_frequency / config.mpc_frequency)
    print(f"sim_frequency: {sim_frequency} Hz, mpc_frequency: {config.mpc_frequency} Hz")
    counter = 0

    def step_controller(mpc_state, tau, qddot, theta_prev):
        nonlocal counter

        qpos = data.qpos.copy()
        qvel = data.qvel.copy()

        raw = _gather_raw_state(model, data, base_body_id, contact_ids)
        tita_state, x0, theta_prev = _process_state(*raw, theta_prev)

        if counter % period == 0:
            command = jnp.asarray(command_handle.mpc_wheeled_input(config.com_z_to_track))
            mpc_state, _ = mpc.run(mpc_state, x0[None, :], command[None, :])

        mpc_state, tau, qddot, _, _, _ = mpc.whole_body_run(
            mpc_state,
            x0,
            jnp.asarray(qpos)[None, :],
            jnp.asarray(qvel)[None, :],
            tita_state[6:9][None, :],
            tita_state[9:12][None, :],
            tita_state[12:15][None, :],
            tita_state[15:18][None, :],
        )

        dt = model.opt.timestep
        q_joint = qpos[7:]
        dq_joint = qvel[6:]
        qddot_joint = np.asarray(qddot[0, 6:])

        dq_des = dq_joint + qddot_joint * dt
        q_des = q_joint + dq_joint * dt + 0.5 * qddot_joint * dt**2

        pd_ctrl = 35.0 * (q_des - q_joint) + 10.0 * (dq_des - dq_joint)
        pd_ctrl[[3, 7]] = 0.0  # wheels: no position term

        data.ctrl = pd_ctrl + np.asarray(tau[0])
        mujoco.mj_step(model, data)
        counter += 1

        return mpc_state, tau, qddot, theta_prev

    if headless:
        for _ in range(steps):
            mpc_state, tau, qddot, theta_prev = step_controller(mpc_state, tau, qddot, theta_prev)
        return mpc_state

    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=command_handle.key_callback,
    ) as viewer:
        viewer.cam.distance *= 5.5
        viewer.sync()
        while viewer.is_running():
            overlay_text = command_handle.consume_overlay_text()
            tic = timer()
            if overlay_text is not None:
                viewer.set_texts((None, None, *overlay_text))
            mpc_state, tau, qddot, theta_prev = step_controller(mpc_state, tau, qddot, theta_prev)
            toc = timer()
            if toc - tic < model.opt.timestep:
                time.sleep(model.opt.timestep - (toc - tic))
            viewer.sync()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--scene", type=str, default="flat")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    main(
        headless=args.headless,
        steps=args.steps,
        scene=args.scene,
    )