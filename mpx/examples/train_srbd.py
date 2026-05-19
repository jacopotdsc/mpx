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

import subprocess
import os
import signal
import csv

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
from plot_rollout_info import plot_llc


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
    num_timesteps          = 25_000_000,
    num_evals              = 3,
    reward_scaling         = 1.0,
    episode_length         = 500,
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
REWARD_LOG_FILE = "reward_log.txt"  # overridden at training start
CKPT_DIR = "."                       # overridden at training start


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
    plt.savefig(os.path.join(CKPT_DIR, "training_curve.png"), dpi=120)
    plt.close()

    elapsed = int((times[-1] - times[0]).total_seconds())
    elapsed_m, elapsed_s = divmod(elapsed, 60)
    clock_time = times[-1].strftime("%H:%M:%S")
    line = (
        f"  eval#{eval_idx:<2d} {num_steps:>12,} | "
        f"reward = {y_data[-1]:+.3f} ± {y_dataerr[-1]:.3f} | "
        f"elapsed {elapsed_m:02d}:{elapsed_s:02d} | "
        f"time {clock_time}"
    )
    print(line)
    with open(REWARD_LOG_FILE, "a") as _f:
        _f.write(line + "\n")

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

def save_params(params, ckpt_dir: str, suffix: str = "final"):
    os.makedirs(ckpt_dir, exist_ok=True)

    leaves = jax.tree_util.tree_leaves(params)
    np.savez(
        os.path.join(ckpt_dir, f"params_{suffix}.npz"),
        **{f"leaf_{i}": np.array(v) for i, v in enumerate(leaves)},
    )

    with open(os.path.join(ckpt_dir, f"params_{suffix}.pkl"), "wb") as f:
        pickle.dump(params, f)

    #print(f"Params saved to {ckpt_dir}/params_{suffix}.npz")
    #print(f"Params saved to {ckpt_dir}/params_{suffix}.pkl")


def _parse_load_suffix(load_arg: str) -> str:
    """Convert a --load argument (filename, basename, or suffix) to a params suffix."""
    base = os.path.basename(load_arg)
    for ext in (".npz", ".pkl"):
        if base.endswith(ext):
            base = base[: -len(ext)]
    if base.startswith("params_"):
        base = base[len("params_"):]
    return base or "best"


def load_params(ckpt_dir: str, suffix: str = "best"):
    # prefer best checkpoint, fall back to final
    pkl_path = None
    for s in (suffix, "final"):
        p = os.path.join(ckpt_dir, f"params_{s}.pkl")
        if os.path.exists(p):
            pkl_path = p
            print(f"  Loading checkpoint: {pkl_path}")
            break
    if pkl_path is None:
        return None
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def run_viewer_rollout(
    eval_env,
    inference_fn=None,
    env_name: str = "AliengoJoystickFlatTerrain",
    headless: bool = False,
    ckpt_dir: str = ".",
):
    print("\n" + "=" * 60)
    print("  Rollout viewer" + (" with loaded policy" if inference_fn else " with zero action"))
    print("=" * 60)

    episode_length = PPO_PARAMS["episode_length"]

    # cuSolver crashes on single-instance MJX; run a batch and show env 0 in viewer.
    EVAL_BATCH = 1
    print(f"  [INFO] env_name: {env_name}")

    batched_reset = jax.jit(jax.vmap(eval_env.reset))
    batched_step  = jax.jit(jax.vmap(eval_env.step))

    if inference_fn is not None:
        jit_infer = jax.jit(inference_fn)

    rng = jax.random.PRNGKey(42)
    rng, *reset_rngs = jax.random.split(rng, EVAL_BATCH + 1)
    state = batched_reset(jnp.stack(reset_rngs))

    viewer_model = None
    viewer_data = None
    if not headless:
        viewer_model = eval_env.mj_model
        viewer_data = mujoco.MjData(viewer_model)

    # ── Helpers ─────────────────────────────────────────────────
    def _sync_viewer(st):
        if headless:
            return
        viewer_data.qpos[:] = np.array(st.data.qpos[0])
        viewer_data.qvel[:] = np.array(st.data.qvel[0])
        mujoco.mj_forward(viewer_model, viewer_data)

    def _get_obs(st):
        obs = st.obs
        if isinstance(obs, dict):
            return {k: v[0] for k, v in obs.items()}
        return obs[0]

    def _get_reward(st):
        return float(st.reward[0])

    def _get_done(st):
        return bool(st.done[0])

    def _cmd_text(st):
        cmd = st.info.get("command", None)
        if cmd is None:
            return []
        c = cmd[0]
        return [(
            mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
            "Command",
            f"vx={float(c[0]):+.2f}  vy={float(c[1]):+.2f}  wz={float(c[2]):+.2f}",
        )]

    rewards = []
    action_sums = []
    info_log = []   # list of flat dicts, one per step
    zero_action_single = jnp.zeros((eval_env.action_size,), dtype=jnp.float32)
    zero_action = jnp.zeros((EVAL_BATCH, eval_env.action_size), dtype=jnp.float32)
    steps_done = 0

    def _flatten_info(info_dict, prefix=""):
        """Recursively flatten info dict; take env 0 for batched arrays."""
        out = {}
        for k, v in info_dict.items():
            full_key = f"{prefix}{k}" if not prefix else f"{prefix}/{k}"
            if isinstance(v, dict):
                out.update(_flatten_info(v, prefix=full_key))
            else:
                try:
                    arr = np.array(v)
                    if arr.ndim == 0:
                        out[full_key] = float(arr)
                    elif arr.ndim == 1 and arr.shape[0] == EVAL_BATCH:
                        # scalar per env → take env 0
                        out[full_key] = float(arr[0])
                    elif arr.ndim >= 1 and arr.shape[0] == EVAL_BATCH:
                        # vector per env → take env 0, flatten
                        row = arr[0].flatten()
                        for i, val in enumerate(row):
                            out[f"{full_key}_{i}"] = float(val)
                    else:
                        row = arr.flatten()
                        for i, val in enumerate(row):
                            out[f"{full_key}_{i}"] = float(val)
                except Exception:
                    pass
        return out

    if headless:
        print("  [INFO] Running rollout in headless mode (no viewer).")
        for i in range(episode_length):
            if inference_fn is not None:
                rng, act_rng = jax.random.split(rng)
                obs0 = _get_obs(state)
                action0, _ = jit_infer(obs0, act_rng)
                action = jnp.broadcast_to(action0, (EVAL_BATCH, eval_env.action_size))
            else:
                action = zero_action

            state = batched_step(state, action)

            steps_done = i + 1
            rewards.append(_get_reward(state))
            action_sums.append(float(jnp.sum(jnp.abs(action[0]))))
            info_log.append({"step": steps_done, **_flatten_info(state.info)})
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
                    action = jnp.broadcast_to(action0, (EVAL_BATCH, eval_env.action_size))
                else:
                    action = zero_action

                state = batched_step(state, action)

                steps_done = i + 1
                rewards.append(_get_reward(state))
                action_sums.append(float(jnp.sum(jnp.abs(action[0]))))
                info_log.append({"step": steps_done, **_flatten_info(state.info)})
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

    cumulative_rewards = np.cumsum(rewards)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axes[0].plot(steps, rewards, color="seagreen")
    axes[0].set_ylabel("reward")
    axes[0].set_title("Reward per step — rollout")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, cumulative_rewards, color="darkorange")
    axes[1].set_ylabel("cumulative reward")
    axes[1].set_title("Cumulative reward")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(steps, action_sums, color="steelblue")
    axes[2].set_xlabel("step")
    axes[2].set_ylabel("sum(|action|)")
    axes[2].set_title("Action magnitude per step")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    rollout_plot_path = os.path.join(ckpt_dir, "reward_rollout.png")
    fig.savefig(rollout_plot_path, dpi=120)
    plt.close(fig)
    print(f"  Reward plot  : {rollout_plot_path}")

    if info_log:
        csv_path = os.path.join(ckpt_dir, "rollout_info.csv")
        fieldnames = list(info_log[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(info_log)
        print(f"  Info CSV     : {csv_path}")
        plot_llc(csv_path)


def run_train(env, eval_env, wrap_env_fn, ckpt_dir: str,
              load_init=None, env_name: str = "QuadrupedMPCEnv"):
    global REWARD_LOG_FILE, CKPT_DIR
    os.makedirs(ckpt_dir, exist_ok=True)
    CKPT_DIR = ckpt_dir
    REWARD_LOG_FILE = os.path.join(ckpt_dir, "reward_log.txt")

    header_lines = [
        "=" * 60,
        f"  PPO Training  —  {env_name}",
        f"  date         : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  JAX backend  : {jax.default_backend()}",
        f"  devices      : {jax.devices()}",
        "  --- network ---",
        f"  distribution : {DISTRIBUTION_TYPE}",
        f"  hidden layers: {POLICY_HIDDEN_LAYER_SIZES}",
        f"  obs size     : {env.observation_size}",
        f"  action size  : {env.action_size}",
        "  --- ppo params ---",
        *[f"  {k:25s}: {v}" for k, v in PPO_PARAMS.items()],
        "=" * 60,
        "",
    ]
    for l in header_lines:
        print(l)
    with open(REWARD_LOG_FILE, "a") as _f:
        _f.write("\n".join(header_lines) + "\n")

    restore_params = None
    selected_network_factory = network_factory
    if load_init:
        suffix = _parse_load_suffix(load_init) if isinstance(load_init, str) else "best"
        restore_params = load_params(ckpt_dir, suffix=suffix)
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
            distribution_type = DISTRIBUTION_TYPE  # use global; brax does not pass distribution_type in kwargs
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

        print(f"    Distribution_type: {DISTRIBUTION_TYPE}")
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
    best_reward = -float("inf")

    def _policy_params_cb(current_step, _make_policy, cb_params):
        nonlocal latest_params, latest_step, best_reward
        latest_params = cb_params
        latest_step = int(current_step)
        if y_data and y_data[-1] > best_reward:
            best_reward = y_data[-1]
            save_params(cb_params, ckpt_dir, suffix="best")

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
            save_params(latest_params, ckpt_dir, suffix="final")
            print(f"[INTERRUPT] Checkpoint saved at step ~{latest_step:,}.")
        else:
            print("[INTERRUPT] No parameters available to save.")
        return None, latest_params
    except Exception as e:
        print(f"\n[ERROR] Training crashed: {e}")
        if latest_params is not None:
            save_params(latest_params, ckpt_dir, suffix="crash")
            print(f"[ERROR] Checkpoint saved at step ~{latest_step:,} → params_crash.pkl")
        else:
            print("[ERROR] No parameters available to save.")
        raise

    if len(times) > 1:
        print(f"\nTime to jit:   {times[1] - times[0]}")
        print(f"Time to train: {times[-1] - times[1]}")

    save_params(params, ckpt_dir, suffix="final")
    return make_inference_fn, params

def main():

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
    parser.add_argument("--load", nargs="?", const="best", default=None, metavar="FILE",
                        help="Load checkpoint weights. Optionally specify filename or suffix (e.g. 'params_crash.npz', 'crash'). Defaults to 'params_best.pkl'.")
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

    load_suffix = _parse_load_suffix(args.load) if args.load else "best"
    params = load_params(ckpt_dir, suffix=load_suffix)
    if params is None or args.zero:
        if args.zero:
            print("  [INFO] --zero flag set: ignoring loaded checkpoint and using zero action.")
        else:
            print(f"  [WARN] No checkpoint found in '{ckpt_dir}', using zero action.")
        run_viewer_rollout(eval_env, inference_fn=None, env_name=env_name, headless=args.headless, ckpt_dir=ckpt_dir)
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
        run_viewer_rollout(eval_env, inference_fn=policy_fn, env_name=env_name, headless=args.headless, ckpt_dir=ckpt_dir)


if __name__ == "__main__":
    def gpu_python_process_cleanup():
        """
        - Lista tutti i processi Python che usano la GPU
        - Se 1 solo: chiede conferma y/n
        - Se >1: chiede di inserire PID specifico
        - Se PID non valido: esce
        """

        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,process_name,used_memory",
                    "--format=csv,noheader"
                ],
                text=True
            )
        except Exception as e:
            print(f"[GPU CHECK] nvidia-smi failed: {e}")
            return

        processes = []

        for line in output.strip().split("\n"):
            if not line:
                continue

            try:
                pid, name, mem = [x.strip() for x in line.split(",")]
                if "python" in name.lower():
                    processes.append((int(pid), name, mem))
            except ValueError:
                continue

        if not processes:
            print("[GPU CHECK] No Python GPU processes found.")
            return

        print("\n⚠️ Python GPU processes found:\n")
        for pid, name, mem in processes:
            print(f"  PID {pid} | {name} | {mem}")

        # -------------------------
        # CASE 1: single process
        # -------------------------
        if len(processes) == 1:
            pid, name, mem = processes[0]
            ans = input(f"\nKill PID {pid}? [y/N]: ").strip().lower()

            if ans in ("y", ""):
                try:
                    print(f"Killing {pid} ...")
                    os.kill(pid, signal.SIGKILL)
                    print("Done.")
                except Exception as e:
                    print(f"[GPU CHECK] Failed to kill {pid}: {e}")
                    exit(1)
            else:
                print("Skipped. Exiting.")
                exit(1)
            return

        # -------------------------
        # CASE 2: multiple processes
        # -------------------------
        pid_list = [p[0] for p in processes]

        try:
            user_pid = input("\nMultiple processes detected. Enter PID to kill: ").strip()
            user_pid = int(user_pid)
        except Exception:
            print("Invalid input. Exiting.")
            exit(1)

        if user_pid not in pid_list:
            print("PID not in GPU Python list. Exiting.")
            return

        try:
            print(f"Killing {user_pid} ...")
            os.kill(user_pid, signal.SIGKILL)
            print("Done.")
        except Exception as e:
            print(f"[GPU CHECK] Failed to kill {user_pid}: {e}")

    gpu_python_process_cleanup()
    
    main()

