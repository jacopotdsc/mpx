"""Run four Lite3 control steps and compare policy observations with controller inputs.

Run from the Lite3 workspace with its MPX and mujoco_playground on PYTHONPATH:
  python check_lite3_shared_noise.py
"""

import jax
import jax.numpy as jnp
import numpy as np
from mujoco_playground._src.locomotion.lite3.joystick import Joystick, default_config


def snapshot(state, env):
    # Actor layout: linvel(3), gyro(3), gravity(3), joint position(12),
    # joint velocity(12), previous action(12), command(3).
    actor = np.asarray(state.obs["state"])
    qpos = np.asarray(state.info["qpos_measured"])
    qvel = np.asarray(state.info["qvel_measured"])
    return {
        "actor_q": actor[9:21].copy(),
        "actor_dq": actor[21:33].copy(),
        "qpos": qpos.copy(),
        "qvel": qvel.copy(),
        "true_q": np.asarray(state.data.qpos[7:]).copy(),
        "true_dq": np.asarray(state.data.qvel[6:]).copy(),
        "default_q": np.asarray(env._default_pose).copy(),
    }


def main():
    cfg = default_config()
    cfg.pert_config.enable = False
    env = Joystick(task="flat_terrain", config=cfg)
    state = env.reset(jax.random.PRNGKey(0))
    snapshots = [snapshot(state, env)]
    controller_inputs = []

    for _ in range(4):
        # At this point state.obs["state"] is the would-be policy input.
        # step() reads these stored measurements for MPC and first WBC substep.
        controller_inputs.append((
            np.asarray(state.info["qpos_measured"]).copy(),
            np.asarray(state.info["qvel_measured"]).copy(),
        ))
        # No policy/checkpoint is loaded: use a zero action only to advance.
        state = env.step(state, jnp.zeros(env.mjx_model.nu))
        snapshots.append(snapshot(state, env))

    # Inspect only after all four steps have completed. Show three successive
    # post-reset states (steps 1, 2, 3), each consumed by the next env.step().
    for i, (snap, (wbc_q, wbc_dq)) in enumerate(zip(snapshots, controller_inputs)):
        assert snap["qpos"].shape == (env.mjx_model.nq,)
        assert snap["qvel"].shape == (env.mjx_model.nv,)
        pos_equal = np.allclose(snap["actor_q"] + snap["default_q"], wbc_q[7:], atol=1e-6, rtol=0)
        vel_equal = np.allclose(snap["actor_dq"], wbc_dq[6:], atol=1e-6, rtol=0)
        mpc_base_equal = np.array_equal(snap["qpos"][:7], wbc_q[:7]) and np.array_equal(snap["qvel"][:6], wbc_dq[:6])
        if 1 <= i <= 3:
            obs_q = snap["actor_q"] + snap["default_q"]
            obs_dq = snap["actor_dq"]
            print(f"\nStep {i}")
            print("joint | obs pos joint | obs vel joint | mpc pos joint | mpc vel joint | same")
            for j in range(env.mjx_model.nu):
                same = (np.isclose(obs_q[j], wbc_q[7 + j], atol=1e-6, rtol=0)
                        and np.isclose(obs_dq[j], wbc_dq[6 + j], atol=1e-6, rtol=0))
                print(f"{j:5d} | {obs_q[j]:13.7f} | {obs_dq[j]:13.7f} | "
                      f"{wbc_q[7 + j]:13.7f} | {wbc_dq[6 + j]:13.7f} | {same}")
        assert pos_equal and vel_equal and mpc_base_equal, f"Measurement mismatch at step {i}"

    print("\nPASS: policy observation and first WBC substep share joint measurements in all four steps.")
    print("No neural network was run; actions were zero. This checks its input, not its output.")
    print("The columns named 'mpc' are the measured joint inputs to WBC; the SRBD MPC uses only the base.")


if __name__ == "__main__":
    main()
