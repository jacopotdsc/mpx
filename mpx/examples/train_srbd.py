"""
train_srbd.py
-------------
PPO training for QuadrupedMPCEnv.
Follows the pattern from locomotion.ipynb (MuJoCo Playground).
"""

from __future__ import annotations

import argparse
import functools
import os
import pickle
import sys
import time
from datetime import datetime

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from brax.training.acme import running_statistics
from brax.envs.wrappers.training import EpisodeWrapper, VmapWrapper, AutoResetWrapper
import mujoco
import mujoco.viewer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl_env_srbd import QuadrupedMPCEnv


def wrap_for_brax_training(env, episode_length, action_repeat=1, randomization_fn=None, **kwargs):
    """Standard Brax wrapper: does not use mujoco_playground (which requires state.data)."""
    env = EpisodeWrapper(env, episode_length, action_repeat)
    env = VmapWrapper(env)
    env = AutoResetWrapper(env)
    return env

# ─────────────────────────────────────────────────────────────────────────────
#  PPO parameters  (edit here)
# ─────────────────────────────────────────────────────────────────────────────

PPO_PARAMS = dict(
    num_timesteps          = 1_000_000,
    num_evals              = 10,
    reward_scaling         = 1.0,
    episode_length         = 500,
    normalize_observations = True,
    action_repeat          = 1,
    unroll_length          = 20,
    num_minibatches        = 32,
    num_updates_per_batch  = 4,
    discounting            = 0.97,
    learning_rate          = 3e-4,
    entropy_cost           = 1e-2,
    num_envs               = 128,
    batch_size             = 256,
    seed                   = 0,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Create environments
# ─────────────────────────────────────────────────────────────────────────────

env      = QuadrupedMPCEnv(episode_steps=PPO_PARAMS["episode_length"])
eval_env = QuadrupedMPCEnv(episode_steps=PPO_PARAMS["episode_length"])

# ─────────────────────────────────────────────────────────────────────────────
#  Progress callback
# ─────────────────────────────────────────────────────────────────────────────

x_data, y_data, y_dataerr = [], [], []
times = [datetime.now()]


def progress(num_steps, metrics):
    times.append(datetime.now())
    x_data.append(num_steps)
    y_data.append(metrics["eval/episode_reward"])
    y_dataerr.append(metrics["eval/episode_reward_std"])

    plt.clf()
    plt.xlim([0, PPO_PARAMS["num_timesteps"] * 1.25])
    plt.xlabel("# environment steps")
    plt.ylabel("reward per episode")
    plt.title(f"y={y_data[-1]:.3f}")
    plt.errorbar(x_data, y_data, yerr=y_dataerr, color="blue")
    plt.savefig("training_curve.png", dpi=120)
    plt.close()

    elapsed = int((times[-1] - times[0]).total_seconds())
    elapsed_m, elapsed_s = divmod(elapsed, 60)
    print(
        f"  eval {num_steps:>12,} | "
        f"reward = {y_data[-1]:+.3f} ± {y_dataerr[-1]:.3f} | "
        f"elapsed {elapsed_m:02d}:{elapsed_s:02d}"
    )

# ─────────────────────────────────────────────────────────────────────────────
#  Network factory
# ─────────────────────────────────────────────────────────────────────────────

network_factory = ppo_networks.make_ppo_networks

# ─────────────────────────────────────────────────────────────────────────────
#  train_fn
# ─────────────────────────────────────────────────────────────────────────────

train_fn = functools.partial(
    ppo.train,
    **PPO_PARAMS,
    network_factory=network_factory,
    progress_fn=progress,
)

def save_params(params, ckpt_dir: str):
    os.makedirs(ckpt_dir, exist_ok=True)

    leaves = jax.tree_util.tree_leaves(params)
    np.savez(
        os.path.join(ckpt_dir, "params_final.npz"),
        **{f"leaf_{i}": np.array(v) for i, v in enumerate(leaves)},
    )

    with open(os.path.join(ckpt_dir, "params_final.pkl"), "wb") as f:
        pickle.dump(params, f)

    print(f"Params saved to {ckpt_dir}/params_final.npz")
    print(f"Params saved to {ckpt_dir}/params_final.pkl")


def load_params(ckpt_dir: str):
    pkl_path = os.path.join(ckpt_dir, "params_final.pkl")
    if not os.path.exists(pkl_path):
        return None
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def run_viewer_rollout(inference_fn=None):
    print("\n" + "=" * 60)
    print("  Rollout viewer" + (" with loaded policy" if inference_fn else " with zero action"))
    print("=" * 60)

    jit_reset = jax.jit(eval_env.reset)
    jit_step = jax.jit(eval_env.step)
    if inference_fn is not None:
        jit_infer = jax.jit(inference_fn)

    episode_length = PPO_PARAMS["episode_length"]

    rng = jax.random.PRNGKey(42)
    rng, reset_rng = jax.random.split(rng)
    state = jit_reset(reset_rng)

    viewer_model_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "aliengo", "scene_flat.xml")
    )
    viewer_model = mujoco.MjModel.from_xml_path(viewer_model_path)
    viewer_data = mujoco.MjData(viewer_model)

    rewards = []
    zero_action = jnp.zeros((eval_env.action_size,), dtype=jnp.float32)
    steps_done = 0

    with mujoco.viewer.launch_passive(viewer_model, viewer_data) as viewer:
        # Sync viewer immediately to the env reset pose before first control step.
        viewer_data.qpos[:] = np.array(state.pipeline_state.q)
        viewer_data.qvel[:] = np.array(state.pipeline_state.qd)
        mujoco.mj_forward(viewer_model, viewer_data)
        cmd0 = state.info["command"]
        viewer.set_texts([(
            mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
            "Command",
            f"vx={float(cmd0[0]):+.2f}  vy={float(cmd0[1]):+.2f}  wz={float(cmd0[2]):+.2f}",
        )])
        viewer.sync()

        for i in range(episode_length):
            if not viewer.is_running():
                break

            if inference_fn is not None:
                rng, act_rng = jax.random.split(rng)
                action, _ = jit_infer(state.obs, act_rng)
                #action = zero_action
            else:
                action = zero_action

            state = jit_step(state, action)
            steps_done = i + 1
            rewards.append(float(state.reward))

            viewer_data.qpos[:] = np.array(state.pipeline_state.q)
            viewer_data.qvel[:] = np.array(state.pipeline_state.qd)
            mujoco.mj_forward(viewer_model, viewer_data)
            cmd = state.info["command"]
            viewer.set_texts([(
                mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                "Command",
                f"vx={float(cmd[0]):+.2f}  vy={float(cmd[1]):+.2f}  wz={float(cmd[2]):+.2f}",
            )])
            viewer.sync()

            if state.done:
                print(f"  Episode ended at step {steps_done}")
                break

            time.sleep(eval_env.dt)

    print(f"  Steps done   : {steps_done}")
    print(f"  Mean reward  : {np.mean(rewards):.3f}")
    print(f"  Total reward : {np.sum(rewards):.3f}")

    steps = np.arange(steps_done)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(steps, rewards, color="seagreen")
    ax.set_xlabel("step")
    ax.set_ylabel("reward")
    ax.set_title("Reward per step — rollout")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("reward_rollout.png", dpi=120)
    plt.close(fig)
    print("  Reward plot  : reward_rollout.png")


def run_train(ckpt_dir: str, load_init: bool = False):
    print("=" * 60)
    print("  PPO Training  —  QuadrupedMPCEnv")
    print("=" * 60)
    print(f"  JAX backend : {jax.default_backend()}")
    print(f"  devices     : {jax.devices()}")
    print()

    restore_params = None
    if load_init:
        restore_params = load_params(ckpt_dir)
        if restore_params is None:
            print(f"  [WARN] --load requested but no checkpoint found in '{ckpt_dir}'.")
            print("  Starting training from random initialization.")
        else:
            print(f"  Initializing training from checkpoint in '{ckpt_dir}'.")

    make_inference_fn, params, _ = train_fn(
        environment=env,
        eval_env=eval_env,
        wrap_env_fn=wrap_for_brax_training,
        restore_params=restore_params,
    )

    if len(times) > 1:
        print(f"\nTime to jit:   {times[1] - times[0]}")
        print(f"Time to train: {times[-1] - times[1]}")

    save_params(params, ckpt_dir)
    return make_inference_fn, params


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--train", action="store_true", help="Run training (default)")
    mode.add_argument("--eval", action="store_true", help="Skip training and open viewer")
    parser.add_argument("--load", action="store_true", help="Load checkpoint weights before training")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints", help="Checkpoint directory")
    args = parser.parse_args()

    run_eval = args.eval

    if not run_eval:
        run_train(args.ckpt_dir, load_init=args.load)
        return

    print("=" * 60)
    print("  PPO Eval  —  QuadrupedMPCEnv")
    print("=" * 60)

    params = load_params(args.ckpt_dir)
    if params is None:
        print(f"  [WARN] No checkpoint found in '{args.ckpt_dir}', using zero action.")
        run_viewer_rollout(inference_fn=None)
    else:
        print(f"  Checkpoint loaded from '{args.ckpt_dir}'")        
        networks = ppo_networks.make_ppo_networks(
            observation_size=eval_env.observation_size,
            action_size=eval_env.action_size,
            preprocess_observations_fn=running_statistics.normalize,
        )
        inference_fn = ppo_networks.make_inference_fn(networks)
        # inference_fn is make_policy: inference_fn(params, deterministic) -> policy(obs, rng)
        policy_fn = inference_fn(params, deterministic=True)
        run_viewer_rollout(inference_fn=policy_fn)


if __name__ == "__main__":
    main()