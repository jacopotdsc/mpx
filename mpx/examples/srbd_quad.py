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
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
        data.qvel[:] = 0.0
    else:
        data.qpos = np.asarray(jnp.concatenate([config.p0, config.quat0, config.q0]))
        data.qvel[:] = 0.0
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


def main(headless=False, steps=500, scene="flat"):
    model = mujoco.MjModel.from_xml_path(
        dir_path + f"/../data/aliengo/scene_{scene}.xml"
    )
    data = mujoco.MjData(model)
    sim_frequency = float(config.whole_body_frequency)
    model.opt.timestep = 1.0 / sim_frequency

    contact_ids = sim_utils.geom_ids(model, config.contact_frame)
    command_handle = sim_utils.KeyboardVelocityCommand(vx=0.0, vy=0.0, wz=0.0)
    mpc = mpc_wrapper_srbd.BatchedMPCControllerWrapper(config, n_env=1)

    _reset_to_initial_state(model, data)

    foot = jnp.asarray(sim_utils.geom_positions(data, contact_ids))
    x0 = _srbd_state(data.qpos, data.qvel)
    command = jnp.asarray(command_handle.mpc_input(config.robot_height))
    contact = jnp.asarray(sim_utils.estimate_contacts(data, contact_ids))
    mpc_state = mpc.init_state()
    mpc_state = mpc.run(mpc_state, x0[None, :], command[None, :], foot[None, :], contact[None, :])
    tau_warm, _ = mpc.whole_body_run(
        mpc_state,
        jnp.asarray(data.qpos)[None, :],
        jnp.asarray(data.qvel)[None, :],
    )
    tau_warm.block_until_ready()
    mpc.reset()

    period = int(sim_frequency / config.mpc_frequency)
    print(f"sim_frequency: {sim_frequency} Hz, mpc_frequency: {config.mpc_frequency} Hz")
    #print(f"Controller period: {period} steps at {sim_frequency} Hz simulation frequency.")
    counter = 0
    grf = jnp.zeros(config.n_contact * 3)
    J = jnp.zeros((model.nv, config.n_contact * 3))

    def step_controller(mpc_state, grf, J):
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
            mpc_state = mpc.run(mpc_state, x0[None, :], command[None, :], foot[None, :], contact[None, :])
            stop = timer()
            
            #print(f"GRF: {np.array(grf)}")
            #print(f"MPC time: {1e3 * (stop - start):.2f} ms")

        tau_cmd, J = mpc.whole_body_run(
            mpc_state,
            jnp.asarray(qpos)[None, :],
            jnp.asarray(qvel)[None, :],
        )
        grf = mpc_state.grf[0]

        def tau_to_qdes(model, data, tau, grf, J, dt):
            qpos = data.qpos.copy()
            qvel = data.qvel.copy()

            mujoco.mj_forward(model, data)
            
            bias = data.qfrc_bias.copy()
            vec = np.zeros(model.nv)
            J_j = J[0, 6:, :] ¯
            vec[6:] = tau - bias[6:] + J_j@grf
           
            M = np.zeros((model.nv, model.nv), dtype=np.float64)
            mujoco.mj_fullM(model, M, data.qM)
            qacc = np.linalg.solve(M, vec)
            qacc_joints = qacc[6:]

            dq_des = qvel[6:] + qacc_joints * dt
            q_des = qpos[7:]+ qvel[6:] * dt + qacc_joints * (dt**2) / 2
            #print(f"tau_cmd      : {tau_ctrl}")
        
            return q_des, dq_des, qacc_joints

        dt = model.opt.timestep
        tau_ctrl = np.asarray(tau_cmd[0])
        q_des, dq_des, qacc_joints = tau_to_qdes(model, data, tau_ctrl, grf, J, model.opt.timestep)
        pd_ctrl = tau_ctrl + 50 * (q_des - qpos[7:]) + 5 * (dq_des - qvel[6:])
        #print(config.q0)
        #print(config.q0.shape)
        #pd_ctrl = 50*(config.q0 - qpos[7:]) - 0 * ( - qvel[6:])
        
        #print(f"bias[6:]     : {data.qfrc_bias[6:]}")
        #print(f"vec (netto)  : {tau_ctrl - data.qfrc_bias[6:]}")
        #print(f"qacc_joints  : {qacc_joints}")
        #print(f"q_des - q    : {q_des - data.qpos[7:]}")
        #print(f"pd_ctrl      : {pd_ctrl}")
        #print(f"tau_cmd[0]   : {tau_ctrl}\n-----------------------")

        data.ctrl = np.asarray(pd_ctrl)
        #data.ctrl = np.asarray(tau_ctrl)
        mujoco.mj_step(model, data)
        counter += 1


        return mpc_state, tau_cmd, grf, J

    if headless:
        for _ in range(steps):
            mpc_state, tau_cmd, grf, J = step_controller(mpc_state, grf, J)
        return mpc_state

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
            mpc_state, tau_cmd, grf, J = step_controller(mpc_state, grf, J)
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
