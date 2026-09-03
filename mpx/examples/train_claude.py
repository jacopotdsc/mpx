"""
train_srbd.py
-------------
PPO training and evaluation for quadruped / legged locomotion, using
JAX + Brax PPO + MuJoCo Playground environments.

Usage:
    # Train (default mode). Creates checkpoints/<env>/<timestamp>/
    python train_srbd.py --name TitaJoystickFlatTerrain

    # Evaluate: load a specific run folder and open the MuJoCo viewer
    python train_srbd.py --name TitaJoystickFlatTerrain --eval \
        --load checkpoints/TitaJoystickFlatTerrain/20260806_123456

    # Evaluation options
    python train_srbd.py --name Tita --eval --load <run_dir> --headless
    python train_srbd.py --name Tita --eval --load <run_dir> --cmd 0.5 0.0
    python train_srbd.py --name Tita --eval --random     # untrained policy
    python train_srbd.py --name Tita --eval --zero       # zero action / MPC-only

Checkpoint loading rule: prefer params_best.pkl, fall back to params_final.pkl.
"""

import os

# ── JAX / XLA runtime configuration (must run before JAX initialises) ──────────
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"
if not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.jax_cache"))
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

import argparse
import csv
import functools
import pickle
import subprocess
import sys
import time
from dataclasses import fields, is_dataclass
from datetime import datetime

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from brax.training.acme import running_statistics, specs
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
import mujoco
import mujoco.viewer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_eval import (
    plot_command_tracking,
    plot_llc,
    plot_mpc_output,
    plot_network_actions,
    plot_reward_terms,
    plot_reward_terms_separate,
    plot_rollout_rewards,
    _save_sim_video,
)

# ══════════════════════════════════════════════════════════════════════════════
#  Configuration  (edit here)
# ══════════════════════════════════════════════════════════════════════════════

# ── Network ───────────────────────────────────────────────────────────────────
POLICY_HIDDEN_LAYER_SIZES = (512, 256, 128)
DISTRIBUTION_TYPE = "tanh_normal"   # 'normal' or 'tanh_normal' — must match the checkpoint
INIT_STD = 0.03                     # initial policy standard deviation
ZERO_INIT_OUTPUT_LAYER = False      # zero the policy output layer at init (safe-exploration start)

# ── PPO ───────────────────────────────────────────────────────────────────────
# Every key here is passed verbatim to brax `ppo.train`. Anything not listed
# keeps its Brax default. This is the single source of PPO configuration.
PPO_PARAMS = dict(
      num_timesteps=100_000_000,
      num_evals=10,
      reward_scaling=1.0,
      episode_length=1000,
      normalize_observations=True,
      action_repeat=1,
      unroll_length=20,
      num_minibatches=32,
      num_updates_per_batch=4,
      discounting=0.99,
      learning_rate=3e-4,
      entropy_cost=1e-2,
      num_envs=4096,
      batch_size=512,
      max_grad_norm=1.0,
      network_factory=dict(
            policy_hidden_layer_sizes=(512, 256, 128),
            value_hidden_layer_sizes=(512, 256, 128),
            policy_obs_key="state",
            value_obs_key="privileged_state",
        ),
      num_resets_per_eval=10,
      seed = 0,
    deterministic_eval = False, # eval usa la media della policy, non un sample rumoroso
  )

SCRIPT_START_TIME = datetime.now().strftime("%Y%m%d_%H%M%S")

# Environment name shortcuts for the command line.
NAME_SHORTCUTS = {
    "go1": "Go1JoystickFlatTerrain",
    "aliengo": "AliengoJoystickE2EFlatTerrain",
    "tita": "TitaJoystickFlatTerrain",
    "titae2e": "TitaJoystickE2EFlatTerrain",
}

RECORD_VIDEO = True  # record an MP4 of the evaluation rollout


def get_gpu_name():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
        )
        return out.strip().split("\n")[0]
    except Exception:
        return "Unknown GPU"


# ══════════════════════════════════════════════════════════════════════════════
#  Environment
# ══════════════════════════════════════════════════════════════════════════════

def make_envs(env_name):
    """Return (env, eval_env, wrap_fn). Registers the custom QuadrupedMPCEnv
    in the MuJoCo Playground registry when requested."""
    from go1_srbd import QuadrupedMPCEnv, default_config
    from mujoco_playground import registry
    from mujoco_playground._src import locomotion
    from mujoco_playground._src.wrapper import wrap_for_brax_training

    if env_name == "QuadrupedMPCEnv":
        print(f"  Registering custom env '{env_name}' in the MuJoCo Playground registry.")
        locomotion._envs[env_name] = functools.partial(QuadrupedMPCEnv, task="flat_terrain")
        locomotion._cfgs[env_name] = default_config
        locomotion._randomizer[env_name] = locomotion._randomizer["Go1JoystickFlatTerrain"]
        locomotion.ALL_ENVS = locomotion.ALL_ENVS + (env_name,)
        registry.ALL_ENVS = registry.ALL_ENVS + (env_name,)

    return registry.load(env_name), registry.load(env_name), wrap_for_brax_training


# ══════════════════════════════════════════════════════════════════════════════
#  Network factory
# ══════════════════════════════════════════════════════════════════════════════

def make_policy_networks(env):
    """Build the PPO networks from the network configuration above.

    Honors POLICY_HIDDEN_LAYER_SIZES, DISTRIBUTION_TYPE, INIT_STD and
    ZERO_INIT_OUTPUT_LAYER. The same definition is used for training and for
    reloading a checkpoint at evaluation time.

    INIT_STD is passed to Brax as `init_noise_std`; ZERO_INIT_OUTPUT_LAYER is
    applied through a custom kernel initializer that zeros the policy output
    layer while leaving hidden layers at Brax's default (lecun_uniform).
    """
    param_size = 2 * env.action_size if DISTRIBUTION_TYPE == "tanh_normal" else env.action_size

    kwargs = dict(init_noise_std=INIT_STD)
    if ZERO_INIT_OUTPUT_LAYER:
        def kernel_init(init_kwargs):
            base = jax.nn.initializers.lecun_uniform(**init_kwargs)
            def init(key, shape, dtype=jnp.float32):
                if len(shape) == 2 and shape[-1] == param_size:  # policy output layer
                    return jnp.zeros(shape, dtype)
                return base(key, shape, dtype)
            return init
        kwargs["policy_network_kernel_init_fn"] = kernel_init

    base_args = dict(
        observation_size=env.observation_size,
        action_size=env.action_size,
        policy_hidden_layer_sizes=POLICY_HIDDEN_LAYER_SIZES,
        preprocess_observations_fn=running_statistics.normalize,
        distribution_type=DISTRIBUTION_TYPE,
    )
    try:
        return ppo_networks.make_ppo_networks(**base_args, **kwargs)
    except TypeError:
        # Fallback for a Brax build that lacks init_noise_std / kernel-init hooks.
        print("  [warn] Brax networks API does not accept init_noise_std / "
              "policy_network_kernel_init_fn; building with defaults.")
        return ppo_networks.make_ppo_networks(**base_args)


def make_random_params(env, networks, seed=0):
    """Random policy weights + an empty (identity) observation normalizer."""
    policy_params = networks.policy_network.init(jax.random.PRNGKey(seed))
    obs_size = env.observation_size
    if isinstance(obs_size, dict):
        proto = {
            k: specs.Array((int(np.prod(v)),) if not isinstance(v, int) else (v,), jnp.float32)
            for k, v in obs_size.items()
        }
    else:
        proto = specs.Array((obs_size,), jnp.float32)
    return running_statistics.init_state(proto), policy_params


# ══════════════════════════════════════════════════════════════════════════════
#  Checkpoints
# ══════════════════════════════════════════════════════════════════════════════

def save_params(params, run_dir, name):
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, f"params_{name}.pkl"), "wb") as f:
        pickle.dump(params, f)


def load_checkpoint(run_dir):
    """Return (params, filename), preferring params_best.pkl over params_final.pkl."""
    for name in ("params_best.pkl", "params_final.pkl"):
        path = os.path.join(run_dir, name)
        if os.path.isfile(path):
            return pickle.load(open(path, "rb")), name
    return None, None


def latest_run_dir(env_base_dir):
    """Most recent timestamped run folder under env_base_dir, or None."""
    if not os.path.isdir(env_base_dir):
        return None
    runs = sorted(
        d for d in os.listdir(env_base_dir)
        if os.path.isdir(os.path.join(env_base_dir, d))
    )
    return os.path.join(env_base_dir, runs[-1]) if runs else None


# ══════════════════════════════════════════════════════════════════════════════
#  Training progress callback
# ══════════════════════════════════════════════════════════════════════════════

_TIMES = [datetime.now()]
_STEPS, _REWARDS, _REWARD_STDS = [], [], []
_STD, _ENTROPY, _KL = [], [], []
_RUN_DIR = "."
_REWARD_LOG = "reward_log.txt"
_METRICS_LOG = "metrics_log.csv"


def progress(num_steps, metrics):
    _TIMES.append(datetime.now())
    _STEPS.append(num_steps)
    _REWARDS.append(float(metrics["eval/episode_reward"]))
    _REWARD_STDS.append(float(metrics["eval/episode_reward_std"]))
    _STD.append(metrics.get("training/policy_dist_mean_std"))
    _ENTROPY.append(metrics.get("training/entropy_loss"))
    _KL.append(metrics.get("training/kl_mean"))
    idx = len(_STEPS) - 1

    # Reward training curve.
    plt.clf()
    plt.xlim([0, max(_STEPS[-1], 1) * 1.1])
    plt.xlabel("# environment steps")
    plt.ylabel("reward per episode")
    plt.title(f"reward = {_REWARDS[-1]:.3f}")
    plt.errorbar(_STEPS, _REWARDS, yerr=_REWARD_STDS, color="blue")
    plt.savefig(os.path.join(_RUN_DIR, "training_curve.png"), dpi=120)
    plt.close()

    # Policy std / entropy / KL diagnostics (only if the algorithm reports them).
    if any(v is not None for v in _STD):
        fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
        axes[0].plot(_STEPS, _STD, color="tab:orange")
        axes[0].set_ylabel("policy std")
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(_STEPS, _ENTROPY, color="tab:green", label="entropy_loss")
        axes[1].plot(_STEPS, _KL, color="tab:red", label="kl_mean")
        axes[1].set_ylabel("entropy / KL")
        axes[1].set_xlabel("# environment steps")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(_RUN_DIR, "training_diagnostics.png"), dpi=120)
        plt.close(fig)

    elapsed = int((_TIMES[-1] - _TIMES[0]).total_seconds())
    std_s = f"{_STD[-1]:.4f}" if _STD[-1] is not None else "n/a"
    kl_s = f"{_KL[-1]:.4f}" if _KL[-1] is not None else "n/a"
    line = (
        f"  eval #{idx:<2d} | steps {num_steps:>12,} | "
        f"reward {_REWARDS[-1]:+.3f} ± {_REWARD_STDS[-1]:.3f} | "
        f"std {std_s} | kl {kl_s} | {elapsed // 60:02d}:{elapsed % 60:02d}"
    )
    print(line)
    with open(_REWARD_LOG, "a") as f:
        f.write(line + "\n")

    write_header = not os.path.exists(_METRICS_LOG)
    with open(_METRICS_LOG, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["num_steps", "episode_reward", "episode_reward_std",
                        "policy_std", "entropy_loss", "kl_mean"])
        w.writerow([num_steps, _REWARDS[-1], _REWARD_STDS[-1], _STD[-1], _ENTROPY[-1], _KL[-1]])


# ══════════════════════════════════════════════════════════════════════════════
#  Training
# ══════════════════════════════════════════════════════════════════════════════

def run_train(env, eval_env, wrap_fn, env_name, run_dir):
    global _RUN_DIR, _REWARD_LOG, _METRICS_LOG
    os.makedirs(run_dir, exist_ok=True)
    _RUN_DIR = run_dir
    _REWARD_LOG = os.path.join(run_dir, "reward_log.txt")
    _METRICS_LOG = os.path.join(run_dir, "metrics_log.csv")

    header = [
        "Mode: TRAIN",
        f"Environment: {env_name}",
        f"JAX backend: {jax.default_backend()}  |  GPU: {get_gpu_name()}",
        "",
        "Network:",
        f"  hidden layers : {POLICY_HIDDEN_LAYER_SIZES}",
        f"  distribution  : {DISTRIBUTION_TYPE}",
        f"  initial std   : {INIT_STD}",
        f"  zero init     : {ZERO_INIT_OUTPUT_LAYER}",
        f"  observation   : {env.observation_size}",
        f"  action        : {env.action_size}",
        "",
        "PPO:",
        *[f"  {k:22s}: {v}" for k, v in PPO_PARAMS.items()],
        "",
        "Run directory:",
        f"  {run_dir}",
        "",
    ]
    print("\n".join(header))
    with open(_REWARD_LOG, "a") as f:
        f.write("\n".join(header) + "\n")

    networks = make_policy_networks(env)
    latest_params, latest_step, best_reward = None, 0, -float("inf")

    def save_best(step, _make_policy, params):
        nonlocal latest_params, latest_step, best_reward
        latest_params, latest_step = params, int(step)
        if _REWARDS and _REWARDS[-1] > best_reward:
            best_reward = _REWARDS[-1]
            save_params(params, run_dir, "best")

    try:
        _, params, _ = ppo.train(
            environment=env,
            eval_env=eval_env,
            wrap_env_fn=wrap_fn,
            network_factory=lambda *a, **k: networks,
            progress_fn=progress,
            policy_params_fn=save_best,
            **PPO_PARAMS,
        )
    except KeyboardInterrupt:
        print("\n[interrupt] Ctrl+C — saving the latest parameters ...")
        if latest_params is not None:
            save_params(latest_params, run_dir, "final")
            print(f"[interrupt] saved params_final.pkl at ~{latest_step:,} steps.")
        else:
            print("[interrupt] no parameters available to save yet.")
        return

    save_params(params, run_dir, "final")
    if len(_TIMES) > 1:
        print(f"\nTraining time: {_TIMES[-1] - _TIMES[1]}")
    print(f"Checkpoints saved in {os.path.abspath(run_dir)}")


# ══════════════════════════════════════════════════════════════════════════════
#  Evaluation rollout
# ══════════════════════════════════════════════════════════════════════════════

def _flatten_info(info, prefix=""):
    """Flatten a (batched, size-1) env info dict to scalar columns for CSV/plots.
    Handles the ControlSol dataclass stored under `mpc_output` specially."""
    out = {}
    for k, v in info.items():
        key = f"{prefix}/{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_info(v, key))
        elif k == "mpc_output" and is_dataclass(v):
            for field in fields(v):
                arr = np.asarray(getattr(v, field.name))
                if arr.ndim >= 1 and arr.shape[0] == 1:
                    arr = arr[0]
                row = arr.reshape(-1)
                if row.size == 1:
                    out[f"{key}/{field.name}"] = float(row[0])
                else:
                    for i, val in enumerate(row):
                        out[f"{key}/{field.name}_{i}"] = float(val)
        else:
            try:
                arr = np.asarray(v)
            except Exception:
                continue
            if arr.ndim == 0:
                out[key] = float(arr)
            else:
                row = (arr[0] if arr.shape[0] == 1 else arr).reshape(-1)
                for i, val in enumerate(row):
                    out[f"{key}_{i}"] = float(val)
    return out


def run_rollout(eval_env, policy_fn, env_name, headless, out_dir, fixed_command=None):
    """Roll out a single environment. If policy_fn is None, use zero actions and
    force MPC-only control (the --zero baseline). Saves a rollout CSV, evaluation
    plots and (optionally) a video."""
    episode_length = PPO_PARAMS["episode_length"]
    use_policy = policy_fn is not None

    print("\n" + "=" * 60)
    print(f"  Rollout — {env_name} ({'policy' if use_policy else 'MPC-only / zero action'})")
    print(f"  GPU: {get_gpu_name()}")
    print("=" * 60)

    reset = jax.jit(jax.vmap(eval_env.reset))
    step = jax.jit(jax.vmap(eval_env.step))
    infer = jax.jit(policy_fn) if use_policy else None

    rng = jax.random.PRNGKey(42)
    state = reset(jax.random.PRNGKey(42)[None])  # batch of 1

    # Fixed command handling: command width is the env's own business.
    cmd = None
    cmd_dim = state.info["command"].shape[-1]
    if fixed_command is not None:
        fixed_command = np.asarray(fixed_command, dtype=np.float32)
        if fixed_command.shape[-1] != cmd_dim:
            raise ValueError(
                f"--cmd expects {cmd_dim} value(s) for env '{env_name}', "
                f"got {fixed_command.shape[-1]}."
            )
        cmd = jnp.broadcast_to(jnp.asarray(fixed_command), (1, cmd_dim))
        state = state.replace(info={**state.info, "command": jnp.zeros_like(cmd), "target_command": cmd})
        state = state.replace(obs=jax.jit(jax.vmap(eval_env._get_obs))(state.data, state.info))

    # Video / camera setup.
    frames, renderer, render_data, render_cam = [], None, None, None
    base_body_id = mujoco.mj_name2id(eval_env.mj_model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    if RECORD_VIDEO:
        renderer = mujoco.Renderer(eval_env.mj_model, height=480, width=640)
        render_data = mujoco.MjData(eval_env.mj_model)
        render_cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(render_cam)
        render_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        render_cam.distance, render_cam.elevation, render_cam.azimuth = 8.0, -15.0, 60.0

    def com_of(data):
        return np.asarray(data.subtree_com[base_body_id]).copy() if base_body_id >= 0 else None

    def record_frame(state):
        if not RECORD_VIDEO:
            return
        render_data.qpos[:] = np.array(state.data.qpos[0])
        render_data.qvel[:] = np.array(state.data.qvel[0])
        mujoco.mj_forward(eval_env.mj_model, render_data)
        com = com_of(render_data)
        if com is not None:
            render_cam.lookat[:] = 0.9 * render_cam.lookat + 0.1 * com
        renderer.update_scene(render_data, camera=render_cam)
        frames.append(renderer.render())

    def obs0(state):
        obs = state.obs
        return {k: v[0] for k, v in obs.items()} if isinstance(obs, dict) else obs[0]

    rewards, action_sums, info_log, network_actions = [], [], [], []
    zero_action = jnp.zeros((1, eval_env.action_size), jnp.float32)
    steps_done = [0]

    def advance(state, rng, i):
        # Choose the action.
        if use_policy:
            rng, act_rng = jax.random.split(rng)
            action0, _ = infer(obs0(state), act_rng)
            action = jnp.broadcast_to(action0, (1, eval_env.action_size))
        else:
            action = zero_action
            if "use_only_mpc" in state.info:
                state = state.replace(info={**state.info, "use_only_mpc": jnp.full((1,), True, jnp.bool_)})

        # Keep the fixed command pinned across the env's internal resampling.
        if cmd is not None:
            state = state.replace(info={**state.info, "target_command": cmd})
        state = step(state, action)
        if cmd is not None:
            state = state.replace(info={**state.info, "target_command": cmd})

        steps_done[0] = i + 1
        rewards.append(float(state.reward[0]))
        action_sums.append(float(jnp.sum(jnp.abs(action[0]))))
        info_log.append({"step": i + 1, **_flatten_info(state.info)})
        network_actions.append(np.asarray(jax.device_get(action[0]), np.float32).copy())
        print(f"  step {i + 1}/{episode_length}", end="\r", flush=True)
        record_frame(state)
        return state, rng

    if headless:
        print("  Headless mode (no viewer).")
        for i in range(episode_length):
            state, rng = advance(state, rng, i)
            if bool(state.done[0]):
                print(f"\n  Episode ended at step {i + 1}")
                break
    else:
        viewer_data = mujoco.MjData(eval_env.mj_model)
        with mujoco.viewer.launch_passive(eval_env.mj_model, viewer_data) as viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 8.0, -15.0, 60.0
            for i in range(episode_length):
                if not viewer.is_running():
                    break
                state, rng = advance(state, rng, i)
                viewer_data.qpos[:] = np.array(state.data.qpos[0])
                viewer_data.qvel[:] = np.array(state.data.qvel[0])
                mujoco.mj_forward(eval_env.mj_model, viewer_data)
                com = com_of(viewer_data)
                if com is not None:
                    viewer.cam.lookat[:] = 0.9 * viewer.cam.lookat + 0.1 * com
                viewer.sync()
                if bool(state.done[0]):
                    print(f"\n  Episode ended at step {i + 1}")
                    break
                time.sleep(eval_env.dt)

    steps = steps_done[0]
    if steps:
        print()
    print(f"  Steps done   : {steps}")
    print(f"  Mean reward  : {np.mean(rewards):.3f}")
    print(f"  Total reward : {np.sum(rewards):.3f}")

    if frames:
        _save_sim_video(out_dir, frames, video_fps=1.0 / float(eval_env.dt),
                        slowdown_factor=4.0, name_video="rollout.mp4")
        renderer.close()

    # ── Plots and CSV ──────────────────────────────────────────────────────────
    plot_dir = os.path.join(out_dir, "evaluation_plots")
    os.makedirs(plot_dir, exist_ok=True)
    step_idx = np.arange(steps)

    plot_rollout_rewards(steps=step_idx, rewards=rewards, action_sums=action_sums, ckpt_dir=plot_dir)

    if not info_log:
        return

    csv_path = os.path.join(plot_dir, "rollout_info.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(info_log[0].keys()))
        w.writeheader()
        w.writerows(info_log)
    print(f"  Info CSV     : {csv_path}")

    plot_llc(csv_path=csv_path, out_dir=plot_dir, filename="plot_llc.png")
    plot_command_tracking(info_log, plot_dir, filename="commands.png", plot_target_command=False)
    plot_reward_terms(terms=info_log, prefix="reward_terms/", out_dir=plot_dir,
                      threshold=3.0, filename="reward_terms.png")
    plot_reward_terms_separate(
        terms=info_log,
        reward_scaling=eval_env._config.reward_config.scales,
        prefix="reward_terms/",
        out_dir=plot_dir,
    )
    plot_mpc_output(info_log=info_log, prefix="mpc_output", out_dir=plot_dir, filename="mpc_output.png")

    joint_names = []
    for actuator_id in range(eval_env.action_size):
        joint_id = int(eval_env.mj_model.actuator_trnid[actuator_id, 0])
        name = mujoco.mj_id2name(eval_env.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        joint_names.append(name or f"action_{actuator_id}")
    plot_network_actions(steps=step_idx, actions=network_actions, out_dir=plot_dir,
                         joint_names=joint_names, filename="network_actions.png")


def run_eval(eval_env, env_name, env_base_dir, args):
    fixed_cmd = np.array(args.cmd) if args.cmd is not None else None
    run_dir = args.load or latest_run_dir(env_base_dir)

    if args.zero:
        policy_fn, ckpt_name = None, "(zero action / MPC-only)"
    elif args.random:
        networks = make_policy_networks(eval_env)
        params = make_random_params(eval_env, networks)
        policy_fn = ppo_networks.make_inference_fn(networks)(params, deterministic=True)
        ckpt_name = "(random policy)"
    else:
        if run_dir is None:
            raise SystemExit(
                f"No run folder found under '{env_base_dir}'. "
                f"Pass one with --load <run_dir>."
            )
        params, ckpt_name = load_checkpoint(run_dir)
        if params is None:
            raise SystemExit(f"No params_best.pkl / params_final.pkl in '{run_dir}'.")
        networks = make_policy_networks(eval_env)
        policy_fn = ppo_networks.make_inference_fn(networks)(params, deterministic=True)

    print("\n".join([
        "Mode: EVAL",
        f"Environment: {env_name}",
        f"Run directory: {run_dir}",
        f"Checkpoint: {ckpt_name}",
        "Deterministic policy: True",
        f"Command: {list(fixed_cmd) if fixed_cmd is not None else 'env default'}",
    ]))

    run_rollout(eval_env, policy_fn, env_name, args.headless,
                out_dir=run_dir or env_base_dir, fixed_command=fixed_cmd)


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="PPO training / evaluation for MuJoCo Playground locomotion."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--train", action="store_true", help="Train (default).")
    mode.add_argument("--eval", action="store_true", help="Evaluate a trained checkpoint.")
    parser.add_argument("--name", default="TitaJoystickFlatTerrain",
                        help="Environment name, or a shortcut: "
                             f"{', '.join(NAME_SHORTCUTS)}.")
    parser.add_argument("--load", default=None, metavar="RUN_DIR",
                        help="Run folder to evaluate, e.g. checkpoints/<env>/<timestamp>. "
                             "Loads params_best.pkl, falling back to params_final.pkl. "
                             "If omitted, the most recent run for the env is used.")
    parser.add_argument("--cmd", nargs="+", type=float, default=None, metavar="C",
                        help="Fixed joystick command, e.g. --cmd 0.5 0.0. Length must "
                             "match the env command (2 for Tita [vx, wz], 3 for a "
                             "quadruped [vx, vy, wz]).")
    parser.add_argument("--headless", action="store_true",
                        help="Evaluate without opening the MuJoCo viewer.")
    parser.add_argument("--random", action="store_true",
                        help="Evaluate an untrained random policy.")
    parser.add_argument("--zero", action="store_true",
                        help="Evaluate with zero actions (MPC-only baseline).")
    parser.add_argument("--ckpt-dir", default="checkpoints",
                        help="Checkpoint root; runs are stored in <root>/<env>/<timestamp>.")
    args = parser.parse_args()

    if args.eval and not args.headless and not os.environ.get("DISPLAY"):
        print("[warn] No DISPLAY detected: forcing --headless mode.")
        args.headless = True

    env_name = NAME_SHORTCUTS.get(args.name.lower(), args.name)
    env, eval_env, wrap_fn = make_envs(env_name)
    env_base_dir = os.path.join(args.ckpt_dir, env_name)

    if args.eval:
        run_eval(eval_env, env_name, env_base_dir, args)
    else:
        run_dir = os.path.join(env_base_dir, SCRIPT_START_TIME)
        run_train(env, eval_env, wrap_fn, env_name, run_dir)


if __name__ == "__main__":
    main()