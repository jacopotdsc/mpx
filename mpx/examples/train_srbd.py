"""
train_srbd.py
-------------
PPO training for quadruped locomotion.
Uses MuJoCo Playground environments.

Usage:
    python train_srbd.py                     # train default Go1 Playground env
    python train_srbd.py --name go1          # same as default
    python train_srbd.py --name AliengoJoystickFlatTerrain
    python train_srbd.py --eval              # eval selected Playground env
    python train_srbd.py --eval --headless   # eval without opening viewer
"""

from __future__ import annotations

import argparse
import functools
import os
import pickle
import sys
import time
from datetime import datetime

import signal

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
    num_timesteps          = 100_000_000,
    num_evals              = 5,
    reward_scaling         = 1.0,
    episode_length         = 1000,
    normalize_observations = True,
    action_repeat          = 1,
    unroll_length          = 20,
    num_minibatches        = 32,
    num_updates_per_batch  = 4,
    discounting            = 0.97,
    learning_rate          = 3e-4,
    entropy_cost           = 0.005, #1e-2,
    num_envs               = 1024,
    batch_size             = 256,
    seed                   = 0,
)
print(f"PPO_PARAMS: \n{PPO_PARAMS}")

# ─────────────────────────────────────────────────────────────────────────────
#  Environment factories
# ─────────────────────────────────────────────────────────────────────────────

def make_envs(
    env_name: str = "Go1JoystickFlatTerrain",
):
    """Return (env, eval_env, wrap_fn) with custom registration only for QuadrupedMPCEnv."""

    from go1_srbd import QuadrupedMPCEnv, default_config
    from mujoco_playground._src import locomotion
    from mujoco_playground import registry
    from mujoco_playground._src.wrapper import wrap_for_brax_training as pg_wrap

    if env_name == "QuadrupedMPCEnv":
        print(f"  [INFO] Registering custom env '{env_name}' in MuJoCo Playground registry...")
        locomotion._envs[env_name] = functools.partial(QuadrupedMPCEnv, task="flat_terrain")
        locomotion._cfgs[env_name] = default_config
        locomotion._randomizer[env_name] = locomotion._randomizer["Go1JoystickFlatTerrain"]
        locomotion.ALL_ENVS = locomotion.ALL_ENVS + (env_name,)
        registry.ALL_ENVS = registry.ALL_ENVS + (env_name,)

    env      = registry.load(env_name)
    eval_env = registry.load(env_name)
    return env, eval_env, pg_wrap

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
    eval_idx = len(x_data) - 1

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
    clock_time = times[-1].strftime("%H:%M:%S")
    print(
        f"  eval#{eval_idx:<2d} {num_steps:>12,} | "
        f"reward = {y_data[-1]:+.3f} ± {y_dataerr[-1]:.3f} | "
        f"elapsed {elapsed_m:02d}:{elapsed_s:02d} | "
        f"time {clock_time}"
    )

# ─────────────────────────────────────────────────────────────────────────────
#  Network factory
# ─────────────────────────────────────────────────────────────────────────────

POLICY_HIDDEN_LAYER_SIZES = (512, 256, 128)
DISTRIBUTION_TYPE = "tanh_normal"  # ['normal', 'tanh_normal'] — must match checkpoint

network_factory = functools.partial(
    ppo_networks.make_ppo_networks,
    policy_hidden_layer_sizes=POLICY_HIDDEN_LAYER_SIZES,
    distribution_type=DISTRIBUTION_TYPE,
)

# ─────────────────────────────────────────────────────────────────────────────
#  train_fn
# ─────────────────────────────────────────────────────────────────────────────

train_fn = functools.partial(
    ppo.train,
    **PPO_PARAMS,
    #network_factory=network_factory,
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


def run_viewer_rollout(
    eval_env,
    inference_fn=None,
    env_name: str = "Go1JoystickFlatTerrain",
    headless: bool = False,
):
    print("\n" + "=" * 60)
    print("  Rollout viewer" + (" with loaded policy" if inference_fn else " with zero action"))
    print("=" * 60)

    episode_length = PPO_PARAMS["episode_length"]

    # ── Detect env type ─────────────────────────────────────────
    is_playground = hasattr(eval_env, "mj_model")
    print(f"  [INFO] env_name: {env_name}")
    print(f"  [INFO] eval_env type: {type(eval_env)}")
    # ── For Playground (MJX): must vmap with batch ≥ 1 ──────────
    #    cuSolver crashes on single-instance MJX; we run a small
    #    batch and extract env 0 for the viewer.
    EVAL_BATCH = 1

    if is_playground:
        batched_reset = jax.jit(jax.vmap(eval_env.reset))
        batched_step  = jax.jit(jax.vmap(eval_env.step))
    else:
        batched_reset = None
        batched_step  = None
        jit_reset = jax.jit(eval_env.reset)
        jit_step  = jax.jit(eval_env.step)

    if inference_fn is not None:
        jit_infer = jax.jit(inference_fn)

    rng = jax.random.PRNGKey(42)

    # ── Reset ───────────────────────────────────────────────────
    if is_playground:
        rng, *reset_rngs = jax.random.split(rng, EVAL_BATCH + 1)
        state = batched_reset(jnp.stack(reset_rngs))
    else:
        rng, reset_rng = jax.random.split(rng)
        state = jit_reset(reset_rng)

    viewer_model = None
    viewer_data = None
    if not headless:
        # ── Viewer model ────────────────────────────────────────
        if is_playground:
            viewer_model = eval_env.mj_model
        else:
            # Extract robot type from env_name (e.g., "Go1..." -> "go1", "Aliengo..." -> "aliengo")
            robot_type = env_name.lower().split("joystick")[0].lower() if "Joystick" in env_name else env_name.lower().split("flat")[0].lower()
            viewer_model_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "data", robot_type, "scene_flat.xml")
            )
            print(f"  [INFO] Loading viewer model from: {viewer_model_path}")
            viewer_model = mujoco.MjModel.from_xml_path(viewer_model_path)
        viewer_data = mujoco.MjData(viewer_model)

    # ── Helpers ─────────────────────────────────────────────────
    def _sync_viewer(st):
        """Copy qpos/qvel of env 0 into the viewer."""
        if headless:
            return
        if is_playground:
            viewer_data.qpos[:] = np.array(st.data.qpos[0])
            viewer_data.qvel[:] = np.array(st.data.qvel[0])
        else:
            viewer_data.qpos[:] = np.array(st.pipeline_state.q)
            viewer_data.qvel[:] = np.array(st.pipeline_state.qd)
        mujoco.mj_forward(viewer_model, viewer_data)

    def _get_obs(st):
        if not is_playground:
            return st.obs
        obs = st.obs
        # obs may be a dict (e.g. {"state": ..., "privileged_state": ...})
        # or a plain array; in both cases extract batch index 0.
        if isinstance(obs, dict):
            return {k: v[0] for k, v in obs.items()}
        return obs[0]

    def _get_reward(st):
        return float(st.reward[0]) if is_playground else float(st.reward)

    def _get_done(st):
        return bool(st.done[0]) if is_playground else bool(st.done)

    def _cmd_text(st):
        cmd = st.info.get("command", None)
        if cmd is None:
            return []
        c = cmd[0] if is_playground else cmd
        return [(
            mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
            "Command",
            f"vx={float(c[0]):+.2f}  vy={float(c[1]):+.2f}  wz={float(c[2]):+.2f}",
        )]

    rewards = []
    zero_action_single = jnp.zeros((eval_env.action_size,), dtype=jnp.float32)
    if is_playground:
        zero_action = jnp.zeros((EVAL_BATCH, eval_env.action_size), dtype=jnp.float32)
    else:
        zero_action = zero_action_single
    steps_done = 0

    if headless:
        print("  [INFO] Running rollout in headless mode (no viewer).")
        for i in range(episode_length):
            if inference_fn is not None:
                rng, act_rng = jax.random.split(rng)
                obs0 = _get_obs(state)
                action0, _ = jit_infer(obs0, act_rng)
                if is_playground:
                    # broadcast env-0 action to all batch slots
                    action = jnp.broadcast_to(action0, (EVAL_BATCH, eval_env.action_size))
                else:
                    action = action0
            else:
                action = zero_action

            if is_playground:
                state = batched_step(state, action)
            else:
                state = jit_step(state, action)

            steps_done = i + 1
            rewards.append(_get_reward(state))
            print(f"  Rollout step: {steps_done}/{episode_length}", end="\r", flush=True)

            if _get_done(state):
                print(f"  Episode ended at step {steps_done}")
                break
    else:
        with mujoco.viewer.launch_passive(viewer_model, viewer_data) as viewer:
            _sync_viewer(state)
            hud = _cmd_text(state)
            if hud:
                viewer.set_texts(hud)
            viewer.sync()

            for i in range(episode_length):
                if not viewer.is_running():
                    break

                if inference_fn is not None:
                    rng, act_rng = jax.random.split(rng)
                    obs0 = _get_obs(state)
                    action0, _ = jit_infer(obs0, act_rng)
                    if is_playground:
                        # broadcast env-0 action to all batch slots
                        action = jnp.broadcast_to(action0, (EVAL_BATCH, eval_env.action_size))
                    else:
                        action = action0
                else:
                    action = zero_action

                if is_playground:
                    state = batched_step(state, action)
                else:
                    state = jit_step(state, action)

                steps_done = i + 1
                rewards.append(_get_reward(state))
                print(f"  Rollout step: {steps_done}/{episode_length}", end="\r", flush=True)

                _sync_viewer(state)
                hud = _cmd_text(state)
                if hud:
                    viewer.set_texts(hud)
                viewer.sync()

                if _get_done(state):
                    print(f"  Episode ended at step {steps_done}")
                    break

                time.sleep(eval_env.dt)

    if steps_done > 0:
        print()

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


def run_train(env, eval_env, wrap_env_fn, ckpt_dir: str,
              load_init: bool = False, env_name: str = "QuadrupedMPCEnv"):
    print("=" * 60)
    print(f"  PPO Training  —  {env_name}")
    print("=" * 60)
    print(f"  JAX backend : {jax.default_backend()}")
    print(f"  devices     : {jax.devices()}")
    print()

    restore_params = None
    selected_network_factory = network_factory
    if load_init:
        restore_params = load_params(ckpt_dir)
        if restore_params is None:
            print(f"  [WARN] --load requested but no checkpoint found in '{ckpt_dir}'.")
            print("  Starting training from random initialization.")
        else:
            print(f"  Initializing training from checkpoint in '{ckpt_dir}'.")
    else:
        print("  Fresh training: zero-initializing policy output layer.")

        def _fresh_network_factory(observation_size, action_size, **kwargs):
            # For tanh_normal, policy is an MLP with a final Dense(param_size).
            # We keep hidden layers default-init and zero only layers whose output size == param_size.
            distribution_type = kwargs.get("distribution_type", "tanh_normal")
            if distribution_type == "tanh_normal":
                param_size = 2 * action_size
            elif distribution_type == "normal":
                param_size = action_size
            else:
                raise ValueError(f"Unsupported distribution type: {distribution_type}")

            def _policy_kernel_init_factory(**init_kwargs):
                base_init = jax.nn.initializers.lecun_uniform(**init_kwargs)

                def _init(key, shape, dtype=jnp.float32):
                    if len(shape) >= 2 and shape[-1] == param_size:
                        return jnp.zeros(shape, dtype)
                    return base_init(key, shape, dtype)

                return _init

            kwargs.pop("policy_network_kernel_init_fn", None)
            return ppo_networks.make_ppo_networks(
                observation_size=observation_size,
                action_size=action_size,
                policy_hidden_layer_sizes=POLICY_HIDDEN_LAYER_SIZES,
                policy_network_kernel_init_fn=_policy_kernel_init_factory,
                distribution_type=DISTRIBUTION_TYPE,
                **kwargs,
            )

        selected_network_factory = _fresh_network_factory

        # Debug check: output-layer kernels should be exactly zero at init.
        debug_networks = selected_network_factory(
            observation_size=env.observation_size,
            action_size=env.action_size,
            preprocess_observations_fn=running_statistics.normalize,
        )
        debug_policy_params = debug_networks.policy_network.init(jax.random.PRNGKey(PPO_PARAMS["seed"]))
        flat_debug = jax.tree_util.tree_flatten_with_path(debug_policy_params)[0]
        print("  policy param shapes:")
        for path, value in flat_debug:
            if hasattr(value, "shape"):
                path_str = "/".join(str(p) for p in path)
                print(f"    {path_str}: {tuple(value.shape)}")

        out_dim = debug_networks.parametric_action_distribution.param_size
        candidate_kernels = [
            x for x in jax.tree_util.tree_leaves(debug_policy_params)
            if hasattr(x, "ndim") and x.ndim == 2 and x.shape[-1] == out_dim
        ]
        if candidate_kernels:
            norms = [float(jnp.linalg.norm(k)) for k in candidate_kernels]
            print(f"  policy output-kernel init norms: min={min(norms):.3e}, max={max(norms):.3e}, count={len(norms)}")
        else:
            print("  [WARN] Could not find output-kernel candidates for init check.")

    latest_params = restore_params
    latest_step = 0

    def _policy_params_cb(current_step, _make_policy, cb_params):
        nonlocal latest_params, latest_step
        latest_params = cb_params
        latest_step = int(current_step)

    try:
        make_inference_fn, params, _ = train_fn(
            environment=env,
            eval_env=eval_env,
            wrap_env_fn=wrap_env_fn,
            network_factory=selected_network_factory,
            restore_params=restore_params,
            policy_params_fn=_policy_params_cb,
        )
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Ctrl+C received: stopping training and saving available weights...")
        if latest_params is not None:
            save_params(latest_params, ckpt_dir)
            print(f"[INTERRUPT] Checkpoint saved at step ~{latest_step:,}.")
        else:
            print("[INTERRUPT] No parameters available to save.")
        return None, latest_params

    if len(times) > 1:
        print(f"\nTime to jit:   {times[1] - times[0]}")
        print(f"Time to train: {times[-1] - times[1]}")

    save_params(params, ckpt_dir)
    return make_inference_fn, params

def main():
    def _stop_handler(sig, frame):
        raise KeyboardInterrupt

    #signal.signal(signal.SIGINT, _stop_handler)
    #signal.signal(signal.SIGTSTP, _stop_handler)

    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--train", action="store_true", help="Run training (default)")
    mode.add_argument("--eval", action="store_true", help="Skip training and open viewer")
    parser.add_argument(
        "--name",
        type=str,
        default="AliengoJoystickFlatTerrain",
        help="Environment name. Use 'QuadrupedMPCEnv' for custom SRBD; otherwise a MuJoCo Playground env name.",
    )
    parser.add_argument("--load", action="store_true", help="Load checkpoint weights before training")
    parser.add_argument("--zero", action="store_true", help="Force zero actions (ignore policy network)")
    parser.add_argument("--headless", action="store_true", help="Eval rollout without opening the MuJoCo viewer")
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default="checkpoints",
        help="Checkpoint root directory (checkpoints are stored in <root>/<env_name>)",
    )
    args = parser.parse_args()

    # ── pick environment ────────────────────────────────────────
    env_name = args.name
    env, eval_env, wrap_fn = make_envs(env_name=env_name)
    ckpt_root = args.ckpt_dir
    ckpt_dir = os.path.join(ckpt_root, env_name)

    # ── train or eval ───────────────────────────────────────────
    if not args.eval:
        run_train(env, eval_env, wrap_fn, ckpt_dir,
                  load_init=args.load, env_name=env_name)
        return

    print("=" * 60)
    print(f"  PPO Eval  —  {env_name}")
    print("=" * 60)

    params = load_params(ckpt_dir)
    if params is None:
        print(f"  [WARN] No checkpoint found in '{ckpt_dir}', using zero action.")
        run_viewer_rollout(eval_env, inference_fn=None, env_name=env_name, headless=args.headless)
    else:
        print(f"  Checkpoint loaded from '{ckpt_dir}'")
        networks = ppo_networks.make_ppo_networks(
            observation_size=eval_env.observation_size,
            action_size=eval_env.action_size,
            policy_hidden_layer_sizes=POLICY_HIDDEN_LAYER_SIZES,
            preprocess_observations_fn=running_statistics.normalize,
            distribution_type=DISTRIBUTION_TYPE,
        )
        inference_fn = ppo_networks.make_inference_fn(networks)
        policy_fn = inference_fn(params, deterministic=True)
        run_viewer_rollout(eval_env, inference_fn=policy_fn, env_name=env_name, headless=args.headless)


if __name__ == "__main__":
    main()

