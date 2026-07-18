"""
train_srbd.py
-------------
PPO training for quadruped locomotion.
Uses MuJoCo Playground environments.

Usage:
    python train_srbd.py                     # train default Go1 Playground env
    python train_srbd.py --name go1          # same as default
    python train_srbd.py --name AliengoJoystickRoughTerrain
    python train_srbd.py --eval              # eval selected Playground env
    python train_srbd.py --eval --headless   # eval without opening viewer
"""

import os
import jax

jax.config.update("jax_enable_x64", True)
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")
CACHE_DIR = os.path.expanduser("~/.jax_cache")
jax.config.update("jax_compilation_cache_dir", CACHE_DIR)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


#from __future__ import annotations

import argparse
import functools
import inspect
import os
import pickle
import sys
import time
from datetime import datetime

import subprocess
import os
import signal
import csv

# Reduce GPU memory fragmentation (must be set before JAX/XLA initialise).
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("[XLA_PYTHON_CLIENT_MEM_FRACTION]", "0.75")
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.75"

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from brax.training.acme import running_statistics
from brax.training.acme import specs
from brax.training.agents.ppo.optimizer import LRSchedule
from flax import linen
import jax.numpy as jnp
import dataclasses
from flax.core import freeze, unfreeze
from brax.envs.wrappers.training import EpisodeWrapper, VmapWrapper, AutoResetWrapper
import mujoco
import mujoco.viewer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_rollout_info import plot_llc,  _save_sim_video

def get_gpu_name():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True
        )
        return out.strip().split("\n")[0]
    except Exception:
        return "Unknown GPU"
        
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
    num_timesteps          = 50_000_000,
    num_evals              = 5,
    reward_scaling         = 1.0,
    episode_length         = 2000,
    normalize_observations = True,
    action_repeat          = 1,
    unroll_length          = 20,
    num_minibatches        = 32,
    num_updates_per_batch  = 4,
    discounting            = 0.99,
    gae_lambda             = 0.95, #default 0.95
    clipping_epsilon       = 0.3, # default 0.3
    learning_rate          = jnp.asarray(1e-5, dtype=jnp.float32),
    #learning_rate_schedule = LRSchedule.ADAPTIVE_KL,
    entropy_cost           = 0.005, #0.005, #1e-2,
    desired_kl             = 0.01, # default 0.01
    num_envs               = 4096,
    batch_size             = 1024,
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
    env_name: str = "",
    headless: bool = False,
    ckpt_dir: str = ".",
    fixed_command: np.ndarray | None = None,
    zero_command: bool = False,
):
    print("\n" + "=" * 60)
    print("  Rollout viewer" + (" with loaded policy" if inference_fn else " with zero action"))
    print("=" * 60)

    episode_length = PPO_PARAMS["episode_length"]

    # cuSolver crashes on single-instance MJX; run a batch and show env 0 in viewer.
    # Batch size must be large enough to avoid cuSolver internal errors (GPU batched LU/Cholesky).
    EVAL_BATCH = 1
    print(f"  [INFO] env_name: {env_name}")
    print(f"  GPU: {get_gpu_name()}")

    batched_reset = jax.jit(jax.vmap(eval_env.reset))
    batched_step  = jax.jit(jax.vmap(eval_env.step))

    if inference_fn is not None:
        jit_infer = jax.jit(inference_fn)

    rng = jax.random.PRNGKey(42)
    rng, *reset_rngs = jax.random.split(rng, EVAL_BATCH + 1)
    state = batched_reset(jnp.stack(reset_rngs))

    # Inject fixed command after reset if provided.
    if fixed_command is not None:
        _cmd = jnp.broadcast_to(  # (EVAL_BATCH, 3)
            jnp.array(fixed_command, dtype=jnp.float32), (EVAL_BATCH, 3)
        )
        state = state.replace(info={
            **state.info,
            "command": jnp.zeros_like(_cmd),
            "target_command": _cmd,
        })
        # Also patch the already-baked obs so the very first policy step
        # sees the correct command (command lives at indices 45:48).
        if isinstance(state.obs, dict):
            patched_obs = {
                k: (v.at[..., 45:48].set(_cmd) if v.ndim >= 2 and v.shape[-1] >= 48 else v)
                for k, v in state.obs.items()
            }
            state = state.replace(obs=patched_obs)
        elif state.obs.ndim >= 2 and state.obs.shape[-1] >= 48:
            state = state.replace(obs=state.obs.at[..., 45:48].set(_cmd))
        #print(f"  [CMD] Fixed command: vx={fixed_command[0]:+.2f}  vy={fixed_command[1]:+.2f}  wz={fixed_command[2]:+.2f}")

    viewer_model = None
    viewer_data = None

    # ── Video recording setup ───────────────────────────────
    record_video = True                 # metti a False per disattivare
    render_w, render_h = 640, 480
    render_camera = -1                  # -1 = free camera; oppure "track" se esiste nell'XML
    frames = []
    renderer = None
    render_data = None

    if not headless:
        renderer = mujoco.Renderer(eval_env.mj_model, height=render_h, width=render_w)
        render_data = mujoco.MjData(eval_env.mj_model)
        render_cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(render_cam)
        render_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        render_cam.distance = 8.0
        render_cam.elevation = -15.0
        render_cam.azimuth = 60.0

    _base_body_id = mujoco.mj_name2id(
        eval_env.mj_model,
        mujoco.mjtObj.mjOBJ_BODY,
        'base_link',
    )

    def _update_com_camera(alpha=0.10):
        mujoco.mj_subtreeVel(eval_env.mj_model, render_data)
        com = np.asarray(render_data.subtree_com[_base_body_id]).copy()
        render_cam.lookat[:] = (1.0 - alpha) * render_cam.lookat + alpha * com
        
    def _record_frame(st):
        if not record_video:
            return
        render_data.qpos[:] = np.array(st.data.qpos[0])
        render_data.qvel[:] = np.array(st.data.qvel[0])
        mujoco.mj_forward(eval_env.mj_model, render_data)
        _update_com_camera(alpha=0.10)
        renderer.update_scene(render_data, camera=render_cam)
        
        frames.append(renderer.render())

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
            if inference_fn is not None and not zero_command:
                rng, act_rng = jax.random.split(rng)
                obs0 = _get_obs(state)
                action0, _ = jit_infer(obs0, act_rng)
                action = jnp.broadcast_to(action0, (EVAL_BATCH, eval_env.action_size))
            else:
                action = zero_action
                if "use_only_mpc" in state.info:
                    state = state.replace(info={
                        **state.info,
                        "use_only_mpc": jnp.full(
                            (EVAL_BATCH,),
                            True,
                            dtype=jnp.bool_,
                        ),
                    })

            if fixed_command is not None:
                _cmd = jnp.broadcast_to(  # (EVAL_BATCH, 3)
                    jnp.array(fixed_command, dtype=jnp.float32), (EVAL_BATCH, 3)
                )
                state = state.replace(info={
                    **state.info,
                    #"command": _cmd,
                    "target_command": _cmd,
                })

            state = batched_step(state, action)

            # Re-inject after step: the env smooths/resamples command inside
            # step(), so we overwrite info again to keep the fixed value
            # visible in logs and for the next iteration's policy input.
            if fixed_command is not None:
                state = state.replace(info={
                    **state.info,
                    #"command": _cmd,
                    "target_command": _cmd,
                })

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
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.distance = 8.0
            viewer.cam.elevation = -15.0
            viewer.cam.azimuth = 60.0
            alpha = 0.1
            viewer.sync()
            #renderer.update_scene(render_data, camera=render_cam)
            for i in range(episode_length):
                if not viewer.is_running():
                    break

                if inference_fn is not None and not zero_command:
                    rng, act_rng = jax.random.split(rng)
                    obs0 = _get_obs(state)
                    action0, _ = jit_infer(obs0, act_rng)
                    action = jnp.broadcast_to(action0, (EVAL_BATCH, eval_env.action_size))
                else:
                    action = zero_action

                    if "use_only_mpc" in state.info:
                        state = state.replace(info={
                            **state.info,
                            "use_only_mpc": jnp.full(
                                (EVAL_BATCH,),
                                True,
                                dtype=jnp.bool_,
                            ),
                        })


                if fixed_command is not None:
                    _cmd = jnp.broadcast_to(  # (EVAL_BATCH, 3)
                        jnp.array(fixed_command, dtype=jnp.float32), (EVAL_BATCH, 3)
                    )
                    state = state.replace(info={
                        **state.info,
                        #"command": _cmd,
                        "target_command": _cmd,
                    })

                state = batched_step(state, action)

                # Re-inject after step: the env smooths/resamples command
                # inside step(), so overwrite to keep the fixed value in HUD.
                if fixed_command is not None:
                    state = state.replace(info={
                        **state.info,
                        #"command": _cmd,
                        "target_command": _cmd,
                    })

                steps_done = i + 1
                rewards.append(_get_reward(state))
                action_sums.append(float(jnp.sum(jnp.abs(action[0]))))
                info_log.append({"step": steps_done, **_flatten_info(state.info)})
                print(f"  Rollout step: {steps_done}/{episode_length}", end="\r", flush=True)

                _sync_viewer(state)
                hud = _cmd_text(state)
                if hud:
                    viewer.set_texts(hud)
                
                com = np.asarray(render_data.subtree_com[_base_body_id]).copy()
                viewer.cam.lookat[:] = (1.0 - alpha) * viewer.cam.lookat + alpha * com
                viewer.sync()

                if _get_done(state):
                    print(f"  Episode ended at step {steps_done}")
                    break
                
                _record_frame(state)
                time.sleep(eval_env.dt)

    if steps_done > 0:
        print()

    print(f"  Steps done   : {steps_done}")
    print(f"  Mean reward  : {np.mean(rewards):.3f}")
    print(f"  Total reward : {np.sum(rewards):.3f}")
    if frames:
        _save_sim_video(
            ckpt_dir,
            frames,
            video_fps=1.0 / float(eval_env.dt),   # 500 fps reali
            slowdown_factor=4.0,                   # x4 slow-motion
            name_video="rollout.mp4",
        )
        renderer.close()

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
    os.makedirs(ckpt_dir, exist_ok=True)
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


def _build_fresh_networks(env):
    """Return a ppo_networks with random hidden layers and zero-init output layer."""
    if DISTRIBUTION_TYPE == "tanh_normal":
        param_size = 2 * env.action_size
    elif DISTRIBUTION_TYPE == "normal":
        param_size = env.action_size
    else:
        raise ValueError(f"Unsupported distribution type: {DISTRIBUTION_TYPE}")
    
    def _policy_kernel_init_factory(**init_kwargs):
        base_init = jax.nn.initializers.lecun_uniform(**init_kwargs)
        
        def _init(key, shape, dtype=jnp.float32):
            if len(shape) == 2 and shape[-1] == param_size:
                return jnp.zeros(shape, dtype)
            #if len(shape) == 1 and shape[0] == param_size:
            #    # std_param (se il fork usa questo init anche per lui)
            #    return jnp.full(shape, jnp.log(0.1), dtype)  # o 0.1 se diretto
            return base_init(key, shape, dtype)

        return _init

    print(inspect.signature(ppo_networks.make_ppo_networks))
    network_create = ppo_networks.make_ppo_networks(
        observation_size=env.observation_size,
        action_size=env.action_size,
        policy_hidden_layer_sizes=POLICY_HIDDEN_LAYER_SIZES,
        policy_network_kernel_init_fn=_policy_kernel_init_factory,
        preprocess_observations_fn=running_statistics.normalize,
        distribution_type=DISTRIBUTION_TYPE,
        activation=linen.elu,
        mean_kernel_init_fn=lambda **kw: jax.nn.initializers.zeros,
        init_noise_std=1e-7,
    )
    if DISTRIBUTION_TYPE == "tanh_normal":
        base_policy_network = network_create.policy_network

        target_std = 0.1
        min_std = 0.0001

        # Inversa di softplus:
        # softplus(scale_raw) + min_std = target_std
        initial_scale_raw = float(
            np.log(np.expm1(target_std - min_std))
        )

        def _init_policy_with_small_std(key):
            params = unfreeze(base_policy_network.init(key))

            output_layers = [
                name
                for name, layer_params in params["params"].items()
                if (
                    "bias" in layer_params
                    and layer_params["bias"].shape
                    == (2 * env.action_size,)
                )
            ]

            if len(output_layers) != 1:
                raise RuntimeError(
                    "Impossibile identificare univocamente il layer "
                    f"finale della policy: {output_layers}"
                )

            output_name = output_layers[0]
            bias = params["params"][output_name]["bias"]

            # Prima metà: loc = 0
            bias = bias.at[:env.action_size].set(0.0)

            # Seconda metà: scale_raw ≈ -2.263
            bias = bias.at[env.action_size:].set(initial_scale_raw)

            params["params"][output_name]["bias"] = bias
            return freeze(params)

        patched_policy_network = dataclasses.replace(
            base_policy_network,
            init=_init_policy_with_small_std,
        )

        network_create = network_create.replace(
            policy_network=patched_policy_network,
        )

    self_test = True
    if self_test:
        print(f"  [SELF-TEST] Checking fresh '{DISTRIBUTION_TYPE}' policy...")

        policy_params = network_create.policy_network.init(jax.random.PRNGKey(0))

        obs_size = env.observation_size
        if isinstance(obs_size, dict):
            obs_proto = {
                k: specs.Array((int(np.prod(v)),) if not isinstance(v, int) else (v,), jnp.float32)
                for k, v in obs_size.items()
            }
            dummy_obs = {k: jnp.zeros(a.shape, dtype=jnp.float32) for k, a in obs_proto.items()}
        else:
            obs_proto = specs.Array((obs_size,), jnp.float32)
            dummy_obs = jnp.zeros((obs_size,), dtype=jnp.float32)

        normalizer_params = running_statistics.init_state(obs_proto)
        full_params = (normalizer_params, policy_params)

        # ── output grezzo della rete (robusto a tuple/array) ──
        raw_out = network_create.policy_network.apply(
            normalizer_params, policy_params, dummy_obs
        )
        if isinstance(raw_out, tuple):
            print(f"    apply() -> tuple di {len(raw_out)}: shapes "
                  f"{[np.shape(np.array(o)) for o in raw_out]}")
            for i, o in enumerate(raw_out):
                print(f"      out[{i}] = {np.array(o)}")
            loc_raw = jnp.asarray(raw_out[0])
        else:
            arr = jnp.asarray(raw_out)
            if DISTRIBUTION_TYPE == "tanh_normal":
                loc_raw, scale_raw = jnp.split(arr, 2, axis=-1)
                MIN_STD, VAR_SCALE = 0.001, 1.0
                pre_tanh_std = np.array(jax.nn.softplus(scale_raw) * VAR_SCALE + MIN_STD)
                print(f"    scale raw           : {np.array(scale_raw)}")
                print(f"    pre-tanh std        : {pre_tanh_std}")
            else:
                loc_raw = arr

        print(f"    loc raw             : {np.array(loc_raw)}  (atteso: tutti 0)")

        # ── std_param, se esiste (tipico di 'normal') ──
        params_dict = policy_params.get("params", policy_params)
        if "std_param" in params_dict:
            std_raw = np.array(params_dict["std_param"]["value"])
            print(f"    std_param raw       : {std_raw}")
            print(f"    exp(std_param)      : {np.exp(std_raw)}  (se mappatura exp)")
            print(f"    softplus(std_param) : {np.array(jax.nn.softplus(jnp.asarray(std_raw)))}  (se mappatura softplus)")
        else:
            print("    (nessun std_param nei parametri)")

        # ── azione deterministica: deve essere 0 ──
        inference_fn = ppo_networks.make_inference_fn(network_create)
        det_policy = inference_fn(full_params, deterministic=True)
        det_action, _ = det_policy(dummy_obs, jax.random.PRNGKey(0))
        det_norm = float(jnp.linalg.norm(det_action))
        print(f"    |det action|        : {det_norm:.3e}  (atteso: 0)")

        # ── std empirica: la verità, qualunque sia la mappatura interna ──
        stoch_policy = inference_fn(full_params, deterministic=False)
        keys = jax.random.split(jax.random.PRNGKey(1), 2000)
        actions = jax.vmap(lambda k: stoch_policy(dummy_obs, k)[0])(keys)
        act_mean, act_std = np.array(actions.mean(0)), np.array(actions.std(0))
        print(f"    sampled mean        : {act_mean}")
        print(f"    sampled std         : {act_std}")

        assert det_norm < 1e-8, f"[SELF-TEST FAILED] det action non-zero: {det_norm}"
        if np.any(act_std < 1e-3):
            print("    [FAIL] std ~0: niente esplorazione, PPO non imparerà "
                  "e i log-prob rischiano di esplodere. Sistemare l'init della std.")
        elif np.any(act_std > 1.0):
            print("    [WARN] std >1: esplorazione molto ampia per action_scale=0.1.")
        print("  [SELF-TEST] done.\n")

    return network_create


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
        f"  GPU name:    : "
        "  \n--- network ---",
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
            kwargs.pop("policy_network_kernel_init_fn", None)
            kwargs.pop("distribution_type", None)
            return _build_fresh_networks(env)

        selected_network_factory = _fresh_network_factory

        # Debug check: output-layer kernels should be exactly zero at init.
        debug_networks = _build_fresh_networks(env)
        debug_policy_params = debug_networks.policy_network.init(jax.random.PRNGKey(PPO_PARAMS["seed"]))
        flat_debug = jax.tree_util.tree_flatten_with_path(debug_policy_params)[0]
        print("  policy param shapes:")
        for path, value in flat_debug:
            if hasattr(value, "shape"):
                path_str = "/".join(str(p) for p in path)
                print(f"    {path_str}: {tuple(value.shape)}")

        print(f"    Distribution_type: {DISTRIBUTION_TYPE}")
        policy_module = inspect.getclosurevars(debug_networks.policy_network.apply).nonlocals["policy_module"]
        activation_fn = policy_module.activation
        print(f"    Activation (read from policy_network module): {getattr(activation_fn, '__name__', activation_fn)}")
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

        print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

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
        default="TitaJoystickFlatTerrain",
        help="Environment name. Use 'QuadrupedMPCEnv' for custom SRBD; otherwise a MuJoCo Playground env name.",
    )
    parser.add_argument("--load", nargs="?", const="best", default=None, metavar="FILE",
                        help="Load checkpoint weights. Optionally specify filename or suffix (e.g. 'params_crash.npz', 'crash'). Defaults to 'params_best.pkl'.")
    parser.add_argument("--no-load", action="store_true", help="Ignore any checkpoint and start fresh (overrides --load)")
    parser.add_argument("--zero", action="store_true", help="Force zero actions (ignore policy network)")
    parser.add_argument("--headless", action="store_true", help="Eval rollout without opening the MuJoCo viewer")
    parser.add_argument("--cmd", nargs=3, type=float, default=None, metavar=("VX", "VY", "WZ"),
                        help="Fix joystick command for eval rollout, e.g. --cmd 0.5 0.0 0.0")
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default="checkpoints",
        help="Checkpoint root directory (checkpoints are stored in <root>/<env_name>)",
    )

    args = parser.parse_args()

    # Auto-set headless if DISPLAY is missing
    if not args.train and not args.headless and (os.environ.get("DISPLAY") is None or os.environ.get("DISPLAY") == ""):
        print("[WARN] No DISPLAY detected: forcing --headless mode (no viewer)")
        args.headless = True

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

    fixed_cmd = np.array(args.cmd) if args.cmd is not None else None

    if args.zero or args.no_load:
        print("  [INFO] --zero flag: using fresh network with zero output layer.")
        networks = _build_fresh_networks(eval_env)
        policy_params = networks.policy_network.init(jax.random.PRNGKey(0))
        obs_size = eval_env.observation_size
        if isinstance(obs_size, dict):
            obs_proto = {k: specs.Array((int(np.prod(v)),) if not isinstance(v, int) else (v,), jnp.float32)
                         for k, v in obs_size.items()}
        else:
            obs_proto = specs.Array((obs_size,), jnp.float32)
        normalizer_params = running_statistics.init_state(obs_proto)
        params_zero = (normalizer_params, policy_params)
        inference_fn = ppo_networks.make_inference_fn(networks)
        policy_fn = inference_fn(params_zero, deterministic=True)
        run_viewer_rollout(eval_env, inference_fn=policy_fn, env_name=env_name, headless=args.headless, ckpt_dir=ckpt_dir, fixed_command=fixed_cmd, zero_command=True)
    elif params is None:
        print(f"  [WARN] No checkpoint found in '{ckpt_dir}', using zero action.")
        run_viewer_rollout(eval_env, inference_fn=None, env_name=env_name, headless=args.headless, ckpt_dir=ckpt_dir, fixed_command=fixed_cmd, zero_command=True)
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
        run_viewer_rollout(eval_env, inference_fn=policy_fn, env_name=env_name, headless=args.headless, ckpt_dir=ckpt_dir, fixed_command=fixed_cmd, zero_command=False)


if __name__ == "__main__":
    def _input_timeout(prompt: str, timeout: int = 5, default: str = "") -> str:
        """Read a line from stdin; return `default` if no input within `timeout` seconds."""
        def _alarm_handler(signum, frame):
            raise TimeoutError()
        old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(timeout)
        try:
            ans = input(prompt)
            signal.alarm(0)
            return ans
        except TimeoutError:
            print(f"\n[timeout] No input in {timeout}s — using default: '{default}'")
            return default
        finally:
            signal.signal(signal.SIGALRM, old_handler)
            signal.alarm(0)

    def _get_proc_state(pid: int) -> str:
        """Ritorna lo stato del processo ('R','S','D','T','Z',...) da /proc, '?' se non leggibile."""
        try:
            with open(f"/proc/{pid}/stat") as f:
                # formato: pid (comm) state ...  — comm può contenere spazi/parentesi,
                # quindi prendiamo il campo dopo l'ultima ')'
                content = f.read()
            return content.rsplit(")", 1)[1].split()[0]
        except Exception:
            return "?"


    def _mem_to_mib(mem_str: str) -> float:
        """Converte '1234 MiB' di nvidia-smi in float (MiB)."""
        try:
            return float(mem_str.split()[0])
        except Exception:
            return 0.0


    def gpu_python_process_cleanup():
        """
        - Lista i processi Python su GPU che NON sono in stato Running (S/D/T/Z)
        - Esclude il processo corrente
        - Se 1 solo: chiede conferma y/n (kill automatico dopo 5s)
        - Se >1: propone di killare quello con piu' memoria GPU (kill automatico dopo 5s)
        """
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,process_name,used_memory",
                    "--format=csv,noheader",
                ],
                text=True,
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
                pid = int(pid)
            except ValueError:
                continue

            if "python" not in name.lower():
                continue
            if pid == os.getpid():          # non proporre di killare se stessi
                continue

            state = _get_proc_state(pid)
            if state == "R":                # in running: lo lasciamo stare
                continue

            processes.append((pid, name, mem, state))

        if not processes:
            print("[GPU CHECK] No non-running Python GPU processes found.")
            return

        print("\n⚠️ Non-running Python GPU processes found:\n")
        for pid, name, mem, state in processes:
            print(f"  PID {pid} | state {state} | {mem} | {name}")

        def _kill(pid: int):
            try:
                print(f"Killing {pid} ...")
                os.kill(pid, signal.SIGKILL)
                print("Done.")
            except Exception as e:
                print(f"[GPU CHECK] Failed to kill {pid}: {e}")
                exit(1)

        # -------------------------
        # CASE 1: single process
        # -------------------------
        if len(processes) == 1:
            pid, name, mem, state = processes[0]
            ans = _input_timeout(
                f"\nKill PID {pid} (state {state}, {mem})? [Y/n]: ",
                timeout=5, default="y",
            ).strip().lower()
            if ans in ("y", ""):
                _kill(pid)
            else:
                print("Skipped. Exiting.")
                exit(1)
            return

        # -------------------------
        # CASE 2: multiple -> propone quello con piu' memoria GPU
        # -------------------------
        top = max(processes, key=lambda p: _mem_to_mib(p[2]))
        pid, name, mem, state = top
        ans = _input_timeout(
            f"\nMultiple candidates. Kill the biggest one: PID {pid} "
            f"(state {state}, {mem})? [Y/n, or enter another PID]: ",
            timeout=5, default="y",
        ).strip().lower()

        if ans in ("y", ""):
            _kill(pid)
            return
        if ans == "n":
            print("Skipped. Exiting.")
            exit(1)

        # l'utente ha digitato un PID alternativo
        try:
            user_pid = int(ans)
        except ValueError:
            print("Invalid input. Exiting.")
            exit(1)
        if user_pid not in [p[0] for p in processes]:
            print("PID not in candidate list. Exiting.")
            return
        _kill(user_pid)

    #gpu_python_process_cleanup()
    
    main()

