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

import mpx.config.config_srbd as config
import mpx.utils.mpc_wrapper_srbd as mpc_wrapper_srbd
import mpx.utils.sim as sim_utils

jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


def _reset_to_initial_state(model, data):
    try:
        mujoco.mj_resetDataKeyframe(model, data, 0)
        data.qvel[:] = 0.0
    except Exception as e:
        print(f"Failed to reset to initial state: {e}")
    mujoco.mj_forward(model, data)


def _srbd_state(qpos, qvel):
    return jnp.concatenate(
        [
            jnp.asarray(qpos[:3]),
            jnp.asarray(qpos[3:7]),
            jnp.asarray(qvel[:3]),
            jnp.asarray(qvel[3:6]),
        ]
    )

def get_dfip_current_state(self, tita_state: jax.Array, theta_prev: float) -> jax.Array:
        
    def unwrapNear(self, theta_wrapped: float, theta_prev: float) -> float:
        # wrapping to pi
        a = theta_wrapped - theta_prev # input

        a = (a + jnp.pi) % (2 * jnp.pi)
        a = jnp.where(a < 0, a + 2*jnp.pi, a)
        a = a - jnp.pi

        return theta_prev + a

    pcom        = tita_state[0:3]
    vcom        = tita_state[3:6]
    pl_world    = tita_state[6:9]
    pr_world    = tita_state[9:12]
    dpl_world   = tita_state[12:15]
    dpr_world   = tita_state[15:18]

    c_world     = (pl_world + pr_world) / 2.0
    vc_world    = (dpl_world + dpr_world) / 2.0

    # extract theta
    diff = pl_world - pr_world
    theta_wrapped = jnp.atan2( -diff[0], diff[1])
    theta = unwrapNear(theta_wrapped, theta_prev)

    R = jnp.array([
        [ jnp.cos(theta), -jnp.sin(theta), 0.],
        [ jnp.sin(theta),  jnp.cos(theta), 0.],
        [ 0.,              0.,             1.]
    ])

    dpl_body = R.T @ dpl_world
    dpr_body = R.T @ dpr_world

    w = (dpr_body[0] - dpl_body[0]) / self.d # 0.1
    v = (dpr_body[0] + dpl_body[0]) / 2.0

    x0 = jnp.concatenate([
        pcom,
        vcom,
        c_world,
        jnp.array([vc_world[2]]),
        jnp.array([theta]),
        jnp.array([v]),
        jnp.array([w]),
    ])

    return x0, theta

def main(headless=False, steps=500, scene="flat"):
    model = mujoco.MjModel.from_xml_path(
        dir_path + f"/../data/tita/tita_world.xml"
    )
    data = mujoco.MjData(model)
    sim_frequency = float(config.whole_body_frequency)
    model.opt.timestep = 1.0 / sim_frequency
    
    contact_ids = sim_utils.geom_ids(model, config.contact_frame)
    command_handle = sim_utils.KeyboardVelocityCommand(vx=0.0, vy=0.0, wz=0.0)
    # TODO: use mpc_wrapper_dfcip
    mpc = mpc_wrapper_srbd.BatchedMPCControllerWrapper(config, n_env=1)

    _reset_to_initial_state(model, data)
    
    foot = jnp.asarray(sim_utils.geom_positions(data, contact_ids))
    x0 = _srbd_state(data.qpos, data.qvel)
    command = jnp.asarray(command_handle.mpc_input(config.robot_height))
    contact = jnp.asarray(sim_utils.estimate_contacts(data, contact_ids))

    mpc.run(x0[None, :], command[None, :], foot[None, :], contact[None, :])

    period = int(sim_frequency / config.mpc_frequency)
    print(f"sim_frequency: {sim_frequency} Hz, mpc_frequency: {config.mpc_frequency} Hz")
    #print(f"Controller period: {period} steps at {sim_frequency} Hz simulation frequency.")
    counter = 0

    def step_controller():
        nonlocal counter

        qpos = data.qpos.copy()
        qvel = data.qvel.copy()

        if counter % period == 0:
            foot = jnp.asarray(sim_utils.geom_positions(data, contact_ids))
            command = jnp.asarray(command_handle.mpc_input(config.robot_height))
            contact = jnp.asarray(sim_utils.estimate_contacts(data, contact_ids))
            x0 = _srbd_state(qpos, qvel)

            #print(f"Contact: {contact}")
            #print(foot)
            #print(f"Command: {command}")

            start = timer()
            mpc.run(x0[None, :], command[None, :], foot[None, :], contact[None, :])
            stop = timer()
            #print(f"MPC time: {1e3 * (stop - start):.2f} ms")

        tau_cmd, _ = mpc.whole_body_run(
            jnp.asarray(qpos)[None, :],
            jnp.asarray(qvel)[None, :],
        )
        data.ctrl = np.asarray(tau_cmd[0])
        mujoco.mj_step(model, data)
        counter += 1

    if headless:
        for _ in range(steps):
            step_controller()
        return

    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=command_handle.key_callback,
    ) as viewer:
        viewer.sync()
        while viewer.is_running():
            overlay_text = command_handle.consume_overlay_text()
            tic = timer()
            if overlay_text is not None:
                viewer.set_texts((None, None, *overlay_text))
            #step_controller()
            data.ctrl = np.zeros(model.nu) 
            mujoco.mj_step(model, data)
            counter += 1

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
