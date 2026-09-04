"""Headless validation harness for the DFCIP MPC+WBC controller (mjx_tita.py's
control law), used to check velocity-tracking / stability at the vx/omega
operating points requested for Objective A. No viewer, no video, no
matplotlib dependency -- reuses only the core wrapper/config modules so it
can run on CPU quickly for iteration.

Usage:
    python validate_dfcip_controller.py
"""
import os
import sys
import time
import json

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import mujoco

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, "..")))
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")

import mpx.config.config_dfcip as config
import mpx.utils.mpc_wrapper_dfcip as mpc_wrapper_dfcip
import mpx.utils.mpc_utils as mpc_utils
import mpx.utils.sim as sim_utils


def build_tita_state_np(model, data, base_body_id, contact_ids):
    mujoco.mj_subtreeVel(model, data)
    pcom = data.subtree_com[base_body_id].copy()
    vcom = data.subtree_linvel[base_body_id].copy()
    centers = data.geom_xpos[contact_ids].copy()
    Rs = data.geom_xmat[contact_ids].reshape(2, 3, 3).copy()
    radii = model.geom_size[contact_ids, 0].copy()
    feet_vel = np.zeros((2, 3))
    vel = np.zeros(6)
    for i, g in enumerate(contact_ids):
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_GEOM, int(g), vel, 0)
        feet_vel[i] = vel[3:6]
    return pcom, vcom, centers, Rs, radii, feet_vel


@jax.jit
def process_tita_state(pcom, vcom, centers, Rs, radii, feet_vel, theta_prev):
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
    R = jnp.array([[ct, -st, 0.], [st, ct, 0.], [0., 0., 1.]])
    dpl_b = R.T @ dpl_world
    dpr_b = R.T @ dpr_world
    w = (dpr_b[0] - dpl_b[0]) / config.d
    v = (dpr_b[0] + dpl_b[0]) / 2.0
    x0 = jnp.concatenate([pcom, vcom, c_world, jnp.array([vc_world[2]]),
                          jnp.array([theta]), jnp.array([v]), jnp.array([w])])
    return tita_state, x0, theta


def reset_to_initial_state(model, data):
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def run_case(mpc, solve_mpc, solve_wbc, model, base_body_id, contact_ids,
             vx_cmd, wz_cmd, duration_s=3.0, ramp_s=1.0, label=""):
    data = mujoco.MjData(model)
    reset_to_initial_state(model, data)

    sim_frequency = float(config.whole_body_frequency)
    period = int(sim_frequency / config.mpc_frequency)
    n_steps = int(duration_s * sim_frequency)
    n_ramp_steps = max(1, int(ramp_s * sim_frequency))

    counter = 0
    theta_prev = 0.0
    mpc_state = mpc.init_state()
    command = np.array([0.0, 0.0, 0.0, config.com_z_to_track], dtype=np.float64)
    COMMAND_SMOOTHING = 0.02

    tau = jnp.zeros((1, model.nu))
    qddot = jnp.zeros((1, model.nv))

    log = {"t": [], "vx": [], "wz": [], "height": [], "roll": [], "pitch": [],
           "tau_max": [], "fell": False}

    fell = False
    for counter in range(n_steps):
        ramp = min(1.0, counter / n_ramp_steps)
        target_command = np.array([vx_cmd * ramp, 0.0, wz_cmd * ramp, config.com_z_to_track])

        raw = build_tita_state_np(model, data, base_body_id, contact_ids)
        tita_state, x0, theta_prev_j = process_tita_state(*raw, theta_prev)
        theta_prev = float(theta_prev_j)

        if counter % period == 0:
            command += COMMAND_SMOOTHING * (target_command - command)
            command[3] = target_command[3]
            mpc_state, reference = solve_mpc(mpc_state, x0[None, :], jnp.asarray(command)[None, :])

            pl_world_wbc = tita_state[6:9][None, :]
            pr_world_wbc = tita_state[9:12][None, :]
            dpl_world_wbc = tita_state[12:15][None, :]
            dpr_world_wbc = tita_state[15:18][None, :]
            mpc_state, tau, qddot, fl, fr, desired = solve_wbc(
                mpc_state, x0, jnp.asarray(data.qpos)[None, :], jnp.asarray(data.qvel)[None, :],
                pl_world_wbc, pr_world_wbc, dpl_world_wbc, dpr_world_wbc,
            )

        qpos_joint = data.qpos[7:]
        qvel_joint = data.qvel[6:]
        qddot_joint = np.asarray(qddot[0, 6:])
        dt = model.opt.timestep
        dq_desired = qvel_joint + qddot_joint * dt
        q_desired = qpos_joint + qvel_joint * dt + 0.5 * qddot_joint * dt ** 2
        p_ctrl = 35.0 * (q_desired - qpos_joint)
        d_ctrl = 10.0 * (dq_desired - qvel_joint)
        p_ctrl[[3, 7]] = 0.0
        total_ctrl = p_ctrl + d_ctrl + np.asarray(tau[0])
        data.ctrl = total_ctrl

        mujoco.mj_step(model, data)

        quat = data.qpos[3:7]
        # roll/pitch from quaternion (wxyz)
        w_, x_, y_, z_ = quat
        roll = np.arctan2(2 * (w_ * x_ + y_ * z_), 1 - 2 * (x_ * x_ + y_ * y_))
        pitch = np.arcsin(np.clip(2 * (w_ * y_ - z_ * x_), -1.0, 1.0))
        upvector_z = 1 - 2 * (x_ * x_ + y_ * y_)

        actual_vx = float(x0[11])  # unicycle forward speed v (body-x)
        actual_wz = float(x0[12])  # unicycle yaw rate w

        if counter % 25 == 0:
            log["t"].append(counter * dt)
            log["vx"].append(actual_vx)
            log["wz"].append(actual_wz)
            log["height"].append(float(data.qpos[2]))
            log["roll"].append(float(roll))
            log["pitch"].append(float(pitch))
            log["tau_max"].append(float(np.max(np.abs(total_ctrl))))

        if upvector_z < 0.3 or data.qpos[2] < 0.15:
            fell = True
            break

    log["fell"] = fell
    log["final_step"] = counter
    # summary stats over the last 1s (steady state)
    steady_n = max(1, int(1.0 * sim_frequency / 25))
    vx_ss = np.array(log["vx"][-steady_n:]) if log["vx"] else np.array([0.0])
    wz_ss = np.array(log["wz"][-steady_n:]) if log["wz"] else np.array([0.0])
    summary = {
        "label": label, "vx_cmd": vx_cmd, "wz_cmd": wz_cmd,
        "fell": fell, "fell_at_step": counter if fell else None,
        "steady_vx_mean": float(np.mean(vx_ss)), "steady_vx_std": float(np.std(vx_ss)),
        "steady_wz_mean": float(np.mean(wz_ss)), "steady_wz_std": float(np.std(wz_ss)),
        "height_min": float(np.min(log["height"])) if log["height"] else None,
        "height_max": float(np.max(log["height"])) if log["height"] else None,
        "roll_max_abs": float(np.max(np.abs(log["roll"]))) if log["roll"] else None,
        "pitch_max_abs": float(np.max(np.abs(log["pitch"]))) if log["pitch"] else None,
        "tau_max": float(np.max(log["tau_max"])) if log["tau_max"] else None,
    }
    return summary, log


def main():
    model = mujoco.MjModel.from_xml_path(dir_path + "/../data/tita/scene_flat.xml")
    model.opt.timestep = 1.0 / float(config.whole_body_frequency)
    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, config.base_body_name)
    contact_ids = sim_utils.geom_ids(model, config.contact_frame)

    mpc = mpc_wrapper_dfcip.BatchedMPCControllerWrapper(config, n_env=1)

    @jax.jit
    def solve_mpc(mpc_state, x0, command):
        return mpc.run(mpc_state, x0, command)

    @jax.jit
    def solve_wbc(mpc_state, x0, qpos, qvel, pl_world, pr_world, dpl_world, dpr_world):
        return mpc.whole_body_run(mpc_state, x0, qpos, qvel, pl_world, pr_world, dpl_world, dpr_world)

    cases = [
        ("vx=0.0", 0.0, 0.0),
        ("vx=0.2", 0.2, 0.0),
        ("vx=0.6", 0.6, 0.0),
        ("vx=1.0", 1.0, 0.0),
        ("omega=0.2", 0.0, 0.2),
        ("omega=0.4", 0.0, 0.4),
        ("omega=0.8", 0.0, 0.8),
        ("vx=0.3,omega=0.2", 0.3, 0.2),
        ("vx=0.6,omega=0.4", 0.6, 0.4),
    ]

    results = []
    for label, vx_cmd, wz_cmd in cases:
        t0 = time.time()
        summary, log = run_case(mpc, solve_mpc, solve_wbc, model, base_body_id, contact_ids,
                                 vx_cmd, wz_cmd, duration_s=3.0, ramp_s=1.0, label=label)
        dt_wall = time.time() - t0
        summary["wall_s"] = dt_wall
        results.append(summary)
        print(f"[{label}] fell={summary['fell']} steady_vx={summary['steady_vx_mean']:.3f}"
              f"+-{summary['steady_vx_std']:.3f} steady_wz={summary['steady_wz_mean']:.3f}"
              f"+-{summary['steady_wz_std']:.3f} height=[{summary['height_min']:.3f},{summary['height_max']:.3f}]"
              f" roll_max={summary['roll_max_abs']:.3f} pitch_max={summary['pitch_max_abs']:.3f}"
              f" tau_max={summary['tau_max']:.1f} wall={dt_wall:.1f}s")

    out_path = os.path.join(dir_path, "validation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
