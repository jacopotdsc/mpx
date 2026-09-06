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


if not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


#from __future__ import annotations

import argparse
import functools
import inspect
import os
import pickle
import re
import sys
import time
from datetime import datetime

import subprocess
import os
import signal
import csv
import shutil

# Reduce GPU memory fragmentation (must be set before JAX/XLA initialise).
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ.setdefault("[XLA_PYTHON_CLIENT_MEM_FRACTION]", "0.5")
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from brax.envs.base import Wrapper
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from brax.training.agents.sac import networks as sac_networks
from brax.training.agents.sac import train as sac
from brax.training.acme import running_statistics
from brax.training.acme import specs
from brax.training.agents.ppo.optimizer import LRSchedule
from flax import linen
import jax.numpy as jnp
import dataclasses
from dataclasses import fields, is_dataclass
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from typing import Any
from flax.core import freeze, unfreeze
from brax.envs.wrappers.training import EpisodeWrapper, VmapWrapper, AutoResetWrapper
import mujoco
import mujoco.viewer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_eval import (
    plot_command_tracking,
    plot_rollout_rewards,
    plot_llc,  
    _save_sim_video, 
    plot_reward_terms, 
    plot_reward_terms_separate,
    plot_mpc_output,
    plot_network_actions
)

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
POLICY_HIDDEN_LAYER_SIZES = (512, 256, 128)
DISTRIBUTION_TYPE = "tanh_normal"  # ['normal', 'tanh_normal'] — must match checkpoint
ZERO_INIT_OUTPUT_LAYER = False # if True, init policy output layer to zero (for safe exploration)
ZERO_INIT_LOAD = True # if True, init policy output layer to zero (for safe exploration)
INIT_STD = 0.03

NUM_TIMESTEPS = 20_000_000
NUM_EVALS = 10
EPISODE_LENGTH = 1000
NUM_ENVS = 1024
DETERMINISTIC_EVAL = True  # eval usa la media della policy, non un sample rumoroso

PPO_PARAMS = dict(
      num_timesteps=NUM_TIMESTEPS,
      num_evals=NUM_EVALS,
      reward_scaling=1.0,
      episode_length=EPISODE_LENGTH,
      normalize_observations=True,
      action_repeat=1,
      unroll_length=20,
      num_minibatches=32,
      num_updates_per_batch=4,
      discounting=0.99,
      learning_rate=3e-4,
      entropy_cost=1e-2,
      num_envs=NUM_ENVS,
      batch_size=256,
      max_grad_norm=1.0,
      network_factory=dict(
            policy_hidden_layer_sizes=(512, 256, 128),
            value_hidden_layer_sizes=(512, 256, 128),
            policy_obs_key="state",
            value_obs_key="privileged_state",
        ),
      num_resets_per_eval=10,
      seed = 0,
    deterministic_eval = DETERMINISTIC_EVAL, # eval usa la media della policy, non un sample rumoroso
  )


print(f"PPO_PARAMS: \n{PPO_PARAMS}")

SAC_PARAMS = dict(
    num_timesteps          = NUM_TIMESTEPS,
    num_evals              = NUM_EVALS,
    reward_scaling         = 1.0,
    episode_length         = EPISODE_LENGTH,
    normalize_observations = True,
    action_repeat          = 1,
    discounting            = 0.997,
    learning_rate          = 3e-4,
    num_envs               = NUM_ENVS,          # SAC off-policy: molti meno env di PPO
    batch_size             = 256,
    grad_updates_per_step  = 32,
    min_replay_size        = EPISODE_LENGTH,
    max_replay_size        = EPISODE_LENGTH*NUM_ENVS,
    seed                   = 0,
    deterministic_eval     = DETERMINISTIC_EVAL, # eval usa la media della policy, non un sample rumoroso
)

print(f"SAC_PARAMS: \n{SAC_PARAMS}")

# ─────────────────────────────────────────────────────────────────────────────
#  Environment factories
# ─────────────────────────────────────────────────────────────────────────────

def make_envs(
    env_name: str = "Go1JoystickFlatTerrain",
):
    """Return (env, eval_env, wrap_fn) for a MuJoCo Playground env."""

    from mujoco_playground import registry
    from mujoco_playground._src.wrapper import wrap_for_brax_training as pg_wrap

    env      = registry.load(env_name)
    eval_env = registry.load(env_name)

    if ALGO == "sac":
        class SACStateWrapper(Wrapper):
            """Expose only the regular state observation to SAC."""

            @property
            def observation_size(self):
                return int(np.prod(self.env.observation_size["state"]))

            def reset(self, rng):
                state = self.env.reset(rng)
                return state.replace(obs=state.obs["state"])

            def step(self, state, action):
                state = self.env.step(state, action)
                return state.replace(obs=state.obs["state"])

        env = SACStateWrapper(env)
        eval_env = SACStateWrapper(eval_env)

    return env, eval_env, pg_wrap

# ─────────────────────────────────────────────────────────────────────────────
#  Progress callback
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_START_TIME = datetime.now().strftime("%Y%m%d_%H%M%S")

x_data, y_data, y_dataerr = [], [], []
reward_per_step_data = []  # storia di mean_reward_per_step, per stampare min/max tra tutti gli eval
std_mean_data, std_min_data, std_max_data = [], [], []
entropy_data, kl_data = [], []
times = [datetime.now()]
REWARD_LOG_FILE = "reward_log.txt"  # overridden at training start
METRICS_LOG_FILE = "metrics_log.csv"  # overridden at training start
CKPT_DIR = "."                       # overridden at training start

def save_source_files(target_path: str, ckpt_dir: str | None = None):
    """Copy target_path (a file or a whole directory) into <ckpt_dir>/files_save/<basename>.

    Each copied file is renamed to 'copy_<original_name>.txt' (extension
    replaced with .txt) so the snapshot can't be imported/run by mistake.

    Used to snapshot the exact source/env code a run used, so old checkpoints
    stay reproducible even if the source changes later.
    """
    dest_root = os.path.join(ckpt_dir or CKPT_DIR, "files_save")
    os.makedirs(dest_root, exist_ok=True)

    def _copy_one(src_file: str, dest_dir: str):
        os.makedirs(dest_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(src_file))[0]
        shutil.copy2(src_file, os.path.join(dest_dir, f"copy_{stem}.txt"))

    if os.path.isdir(target_path):
        dest = os.path.join(dest_root, os.path.basename(os.path.normpath(target_path)))
        for root, _, files in os.walk(target_path):
            rel = os.path.relpath(root, target_path)
            dest_dir = dest if rel == "." else os.path.join(dest, rel)
            for fname in files:
                _copy_one(os.path.join(root, fname), dest_dir)
    else:
        dest = dest_root
        _copy_one(target_path, dest)
    print(f"  [INFO] Copied files to '{target_path}' -> '{dest}'")


def save_tita_env_files(env_name: str, ckpt_dir: str | None = None):
    """Snapshot the source of the env actually being trained into files_save.

    Saves the module's folder (e.g. locomotion/tita, locomotion/go1)
    resolved from the MuJoCo Playground registry, so the snapshot always
    matches env_name instead of always saving the tita folder.
    """

    from mujoco_playground._src import locomotion
    env_cls = locomotion._envs[env_name]
    env_cls = env_cls.func if isinstance(env_cls, functools.partial) else env_cls
    env_dir = os.path.dirname(inspect.getfile(env_cls))
    save_source_files(env_dir, ckpt_dir)
    save_source_files(__file__, ckpt_dir)



def progress(num_steps, metrics):
    times.append(datetime.now())
    x_data.append(num_steps)
    y_data.append(metrics["eval/episode_reward"])
    y_dataerr.append(metrics["eval/episode_reward_std"])
    eval_idx = len(x_data) - 1

    # -- diagnostica policy: std della gaussiana, entropy, KL (chiavi 'training/*') --
    std_mean = metrics.get("training/policy_dist_mean_std")
    std_min = metrics.get("training/policy_dist_min_std")
    std_max = metrics.get("training/policy_dist_max_std")
    entropy_loss = metrics.get("training/entropy_loss")
    kl_mean = metrics.get("training/kl_mean")
    total_loss = metrics.get("training/total_loss")
    policy_loss = metrics.get("training/policy_loss")
    v_loss = metrics.get("training/v_loss")

    std_mean_data.append(std_mean)
    std_min_data.append(std_min)
    std_max_data.append(std_max)
    entropy_data.append(entropy_loss)
    kl_data.append(kl_mean)

    reward_text = f"reward: {y_data[-1]:.3f} ± {y_dataerr[-1]:.3f}"

    plt.clf()
    # x range si adatta ai dati raccolti finora, non al target finale: con
    # un training breve/interrotto l'xlim fisso a num_timesteps schiacciava
    # tutti i punti in un angolo, rendendo il grafico illeggibile.
    plt.xlim([0, max(x_data[-1], 1) * 1.1])
    plt.xlabel("# environment steps")
    plt.ylabel("reward per episode")
    plt.title(f"y={y_data[-1]:.3f} ± {y_dataerr[-1]:.3f}")
    plt.errorbar(x_data, y_data, yerr=y_dataerr, color="blue")

    # un punto rosso + etichetta reward±std per ogni eval, alternando
    # sopra/sotto la linea per non sovrapporre le scritte quando i punti
    # sono vicini tra loro
    ax = plt.gca()
    y_span = (max(y_data) - min(y_data)) if len(y_data) > 1 else 0.0
    label_offset = 0.06 * y_span if y_span > 0 else 0.05 * max(abs(y_data[-1]), 1.0)
    for i, (xi, yi, yerri) in enumerate(zip(x_data, y_data, y_dataerr)):
        above = (i % 2 == 0)
        ax.plot(xi, yi, "o", color="red", markersize=4, zorder=5)
        ax.annotate(
            f"{yi:.3f} ± {yerri:.3f}",
            xy=(xi, yi),
            xytext=(xi, yi + (label_offset if above else -label_offset)),
            fontsize=7, color="red", ha="center",
            va="bottom" if above else "top",
        )
    plt.savefig(os.path.join(CKPT_DIR, "training_curve.png"), dpi=120)
    plt.close()

    if any(v is not None for v in std_mean_data):
        fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
        axes[0].plot(x_data, std_mean_data, color="tab:orange", label="mean")
        axes[0].plot(x_data, std_min_data, color="tab:orange", alpha=0.3, linestyle="--", label="min")
        axes[0].plot(x_data, std_max_data, color="tab:orange", alpha=0.3, linestyle="--", label="max")
        axes[0].set_ylabel("policy std")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)
        std_text = (
            f"{reward_text}\n"
            f"policy std: mean={std_mean:.4f}"
            + (f" min={std_min:.4f}" if std_min is not None else "")
            + (f" max={std_max:.4f}" if std_max is not None else "")
        ) if std_mean is not None else reward_text
        axes[0].text(
            0.02, 0.95, std_text, transform=axes[0].transAxes,
            va="top", ha="left", fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        axes[1].plot(x_data, entropy_data, color="tab:green", label="entropy_loss")
        axes[1].plot(x_data, kl_data, color="tab:red", label="kl_mean")
        axes[1].set_ylabel("entropy / KL")
        axes[1].set_xlabel("# environment steps")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(CKPT_DIR, "training_diagnostics.png"), dpi=120)
        plt.close(fig)

    # reward per-step: normalizza per la lunghezza media dell'episodio, così
    # resta confrontabile con run precedenti anche se episode_length cambia.
    avg_ep_len = metrics.get("eval/avg_episode_length")
    mean_reward_per_step = y_data[-1] / avg_ep_len if avg_ep_len else None
    std_reward_per_step = y_dataerr[-1] / avg_ep_len if avg_ep_len else None

    elapsed = int((times[-1] - times[0]).total_seconds())
    elapsed_m, elapsed_s = divmod(elapsed, 60)
    clock_time = times[-1].strftime("%H:%M:%S")
    std_str = f"{std_mean:.4f}" if std_mean is not None else "n/a"
    kl_str = f"{kl_mean:.4f}" if kl_mean is not None else "n/a"
    per_step_str = (
        f"{mean_reward_per_step:+.4f} ± {std_reward_per_step:.4f}"
        if mean_reward_per_step is not None else "n/a"
    )
    line = (
        f"  eval#{eval_idx:<2d}"
        f"\n\tnum_steps = {num_steps:>12,}"
        f"\n\treward = {y_data[-1]:+.3f} ± {y_dataerr[-1]:.3f}"
        #f"\n\treward/step = {per_step_str}"
        f"\n\tstd = {std_str}"
        f"\n\tkl = {kl_str}"
        f"\n\telapsed {elapsed_m:02d}:{elapsed_s:02d}"
        f"\n\ttime {clock_time}"
        "\n-" + "-" * 30
    )
    print(line)
    with open(REWARD_LOG_FILE, "a") as _f:
        _f.write(line + "\n")

    write_header = not os.path.exists(METRICS_LOG_FILE)
    with open(METRICS_LOG_FILE, "a", newline="") as _f:
        writer = csv.writer(_f)
        if write_header:
            writer.writerow([
                "num_steps", "episode_reward", "episode_reward_std",
                "mean_reward_per_step", "std_reward_per_step", "avg_episode_length",
                "policy_std_mean", "policy_std_min", "policy_std_max",
                "entropy_loss", "kl_mean", "total_loss", "policy_loss", "v_loss",
            ])
        writer.writerow([
            num_steps, y_data[-1], y_dataerr[-1],
            mean_reward_per_step, std_reward_per_step, avg_ep_len,
            std_mean, std_min, std_max,
            entropy_loss, kl_mean, total_loss, policy_loss, v_loss,
        ])

# ─────────────────────────────────────────────────────────────────────────────
#  train_fn
# ─────────────────────────────────────────────────────────────────────────────

ALGO = "ppo"              # sovrascritto in main() dal flag --algo
ALGO_PARAMS = SAC_PARAMS if ALGO == "sac" else PPO_PARAMS  # sovrascritto in main()

#def make_train_fn(algo: str):
#    if algo == "sac":
#        return functools.partial(sac.train, **SAC_PARAMS, progress_fn=progress)
#    return functools.partial(ppo.train, **PPO_PARAMS, progress_fn=progress)

def make_train_fn(algo: str):
    train_callable = sac.train if algo == "sac" else ppo.train
    params = SAC_PARAMS if algo == "sac" else PPO_PARAMS

    sig = inspect.signature(train_callable)
    print("\n" + "=" * 70)
    print(f"  Opzioni disponibili di {algo}.train")
    print("=" * 70)
    for name, p in sig.parameters.items():
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue  # salta *args / **kwargs
        default = "(nessun default)" if p.default is inspect.Parameter.empty else p.default
        if name in params:
            print(f"  [PASSATO] {name:28s} = {params[name]!r}   (default: {default!r})")
        else:
            print(f"            {name:28s}   default: {default!r}")

    # chiavi che passi ma che train.train NON accetta (verrebbero rifiutate)
    unknown = [k for k in params if k not in sig.parameters]
    if unknown:
        print(f"\n  [ATTENZIONE] chiavi in {algo.upper()}_PARAMS non accettate da {algo}.train: {unknown}")
    print("=" * 70 + "\n")

    if algo == "sac":
        return functools.partial(sac.train, **SAC_PARAMS, progress_fn=progress)
    return functools.partial(ppo.train, **PPO_PARAMS, progress_fn=progress)

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

_RUN_DIR_RE = re.compile(r"^\d{8}_\d{6}$")  # matches SCRIPT_START_TIME's "%Y%m%d_%H%M%S"

def _list_run_dirs(env_base_dir: str) -> list[str]:
    """List timestamped run subfolders under env_base_dir, oldest first."""
    if not os.path.isdir(env_base_dir):
        return []
    return sorted(
        d for d in os.listdir(env_base_dir)
        if _RUN_DIR_RE.match(d) and os.path.isdir(os.path.join(env_base_dir, d))
    )

def _list_dir_names(path: str) -> list[str]:
    if not os.path.isdir(path):
        return []
    return sorted(
        d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))
    )

def _resolve_load(env_base_dir: str, load_arg: str):
    """Resolve a --load argument to (run_dir, suffix).

    load_arg may be a checkpoint suffix ('best'/'final'/'crash'), in which
    case the latest timestamped run under env_base_dir is used; a run
    timestamp/prefix matched under env_base_dir; a name of a run saved
    under env_base_dir/saved/; or an explicit relative path such as
    'saved/joystick_first_train'. Falls back to env_base_dir itself
    (legacy flat layout, no per-run subfolder) if no run subfolders exist.
    """
    run_dirs = _list_run_dirs(env_base_dir)
    is_suffix = load_arg in ("best", "final", "crash")
    suffix = load_arg if is_suffix else "best"

    if not is_suffix:
        direct_dir = os.path.join(env_base_dir, load_arg)
        saved_dir = os.path.join(env_base_dir, "saved", load_arg)

        if os.path.isdir(direct_dir):
            run_dir = direct_dir
        elif os.path.isdir(saved_dir):
            run_dir = saved_dir
        else:
            matches = [d for d in run_dirs if d == load_arg] or [
                d for d in run_dirs if d.startswith(load_arg)
            ]
            if not matches:
                print(
                    f"  [INFO] Available runs under '{env_base_dir}': "
                    f"{_list_dir_names(env_base_dir)}"
                )
                print(
                    f"  [INFO] Available saved runs under "
                    f"'{os.path.join(env_base_dir, 'saved')}': "
                    f"{_list_dir_names(os.path.join(env_base_dir, 'saved'))}"
                )
                raise FileNotFoundError(
                    f"No run matching '{load_arg}' found under '{env_base_dir}'."
                )
            run_dir = os.path.join(env_base_dir, matches[-1])
    elif run_dirs:
        run_dir = os.path.join(env_base_dir, run_dirs[-1])
    else:
        run_dir = env_base_dir  # legacy flat layout

    print(f"  [INFO] Resolving --load argument: {load_arg}")

    if not os.path.isdir(env_base_dir):
        raise FileNotFoundError(
            f"No checkpoints for this environment: '{env_base_dir}' does not exist."
        )
    if not any(
        os.path.isfile(os.path.join(run_dir, f"params_{s}.pkl"))
        for s in (suffix, "final")
    ):
        raise FileNotFoundError(
            f"Checkpoint not found: {os.path.join(run_dir, f'params_{suffix}.pkl')}"
        )

    print(f"  [INFO] Checkpoint directory: {run_dir}")
    return run_dir, suffix

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
    batched_step = jax.jit(jax.vmap(eval_env.step))

    rng = jax.random.PRNGKey(42)
    rng, *reset_rngs = jax.random.split(rng, EVAL_BATCH + 1)
    state = batched_reset(jnp.stack(reset_rngs))

    if inference_fn is None:
        print("  [ERROR] No inference_fn provided; creating a fresh random policy network for rollout.")
        exit(1)

    jit_infer = jax.jit(inference_fn)

    batched_reset = jax.jit(jax.vmap(eval_env.reset))
    batched_step  = jax.jit(jax.vmap(eval_env.step))

    if inference_fn is not None:
        jit_infer = jax.jit(inference_fn)

    rng = jax.random.PRNGKey(42)
    rng, *reset_rngs = jax.random.split(rng, EVAL_BATCH + 1)
    state = batched_reset(jnp.stack(reset_rngs))

    # Command dimensionality comes from the env's own command_config, not a
    # fixed assumption (e.g. 2 for Tita's [vx, wz], 3 for a quadruped's
    # [vx, vy, wz]). A trailing extra value is the target base height; base
    # height is sampled once at reset and never rewritten in step(), so if
    # omitted the env's own init/sampled height is left untouched.
    cmd_dim = state.info["command"].shape[-1]

    # Inject fixed command after reset if provided.
    if fixed_command is not None:
        fixed_command = np.asarray(fixed_command, dtype=np.float32)
        n_given = fixed_command.shape[-1]
        height_target = None
        if n_given == cmd_dim + 1:
            fixed_command, height_target = fixed_command[:cmd_dim], float(fixed_command[cmd_dim])
        elif n_given != cmd_dim:
            raise ValueError(
                f"--cmd got {n_given} value(s) but env '{env_name}' expects "
                f"{cmd_dim} (or {cmd_dim + 1} with target height as the last value)."
            )
        _cmd = jnp.broadcast_to(jnp.asarray(fixed_command), (EVAL_BATCH, cmd_dim))
        new_info = {
            **state.info,
            "command": jnp.zeros_like(_cmd),
            "target_command": _cmd,
        }
        if height_target is not None:
            new_info["base_height_target"] = jnp.full((EVAL_BATCH,), height_target, dtype=jnp.float32)
        state = state.replace(info=new_info)
        # Recompute obs from the patched info instead of poking a hardcoded
        # offset into the flat obs vector: the command's position and width
        # inside the observation are the env's own business.
        batched_get_obs = jax.jit(jax.vmap(eval_env._get_obs))
        zero_action = jnp.zeros((EVAL_BATCH, eval_env.mjx_model.nu))
        state = state.replace(obs=batched_get_obs(state.data, state.info, zero_action))

    viewer_model = None
    viewer_data = None

    # ── Video recording setup ───────────────────────────────
    record_video = True                 # metti a False per disattivare
    render_w, render_h = 640, 480
    render_camera = -1                  # -1 = free camera; oppure "track" se esiste nell'XML
    frames = []
    renderer = None
    render_data = None

    if record_video:
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
        values = "  ".join(f"{float(v):+.2f}" for v in c)
        return [(
            mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
            "Command",
            values,
        )]

    rewards = []
    action_sums = []
    info_log = []   # list of flat dicts, one per step
    network_actions = []
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

            # Special handling only for the ControlSol stored in mpc_output.
            elif k == "mpc_output" and is_dataclass(v):
                for field in fields(v):
                    field_key = f"{full_key}/{field.name}"
                    field_value = getattr(v, field.name)
                    try:
                        arr = np.asarray(field_value)

                        # Take environment 0.
                        if arr.ndim >= 1 and arr.shape[0] == EVAL_BATCH:
                            arr = arr[0]

                        row = arr.reshape(-1)

                        if row.size == 1:
                            out[field_key] = float(row[0])
                        else:
                            for i, val in enumerate(row):
                                out[f"{field_key}_{i}"] = float(val)

                    except Exception as exc:
                        print(
                            f"[flatten_info] Failed to flatten "
                            f"{field_key}: {exc}"
                        )
            else:
                try:
                    arr = np.array(v)
                    if arr.ndim == 0:
                        out[full_key] = float(arr)
                    elif arr.ndim == 1 and arr.shape[0] == EVAL_BATCH:
                        # Scalar per environment: take environment 0.
                        out[full_key] = float(arr[0])
                    elif arr.ndim >= 1 and arr.shape[0] == EVAL_BATCH:
                        # Vector per environment: take environment 0 and flatten.
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
                _cmd = jnp.broadcast_to(jnp.asarray(fixed_command), (EVAL_BATCH, cmd_dim))
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
            network_actions.append(np.asarray(jax.device_get(action[0]), dtype=np.float32).copy())
            print(f"  Rollout step: {steps_done}/{episode_length}", end="\r", flush=True)

            _record_frame(state)

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
                    _cmd = jnp.broadcast_to(jnp.asarray(fixed_command), (EVAL_BATCH, cmd_dim))
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
                network_actions.append(np.asarray(jax.device_get(action[0]), dtype=np.float32).copy())
                print(f"  Rollout step: {steps_done}/{episode_length}", end="\r", flush=True)

                _sync_viewer(state)
                hud = _cmd_text(state)
                if hud:
                    viewer.set_texts(hud)
                
                com = np.asarray(render_data.subtree_com[_base_body_id]).copy()
                viewer.cam.lookat[:] = (1.0 - alpha) * viewer.cam.lookat + alpha * com
                viewer.sync()

                _record_frame(state)

                if _get_done(state):
                    print(f"  Episode ended at step {steps_done}")
                    break
                
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
            slowdown_factor=1.0,
            name_video="rollout.mp4",
        )
        _save_sim_video(
            ckpt_dir,
            frames,
            video_fps=1.0 / float(eval_env.dt),   # 500 fps reali
            slowdown_factor=4.0,                   # x4 slow-motion
            name_video="rollout_slowed_x4.mp4",
        )
        renderer.close()

    steps = np.arange(steps_done)

    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_dir = os.path.join(ckpt_dir, "evaluation_plots")

    plot_rollout_rewards(
        steps=steps,
        rewards=rewards,
        action_sums=action_sums,
        ckpt_dir=ckpt_dir,
    )

    if info_log:
        csv_path = os.path.join(ckpt_dir, "rollout_info.csv")
        fieldnames = list(info_log[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(info_log)

        print(f"  Info CSV     : {csv_path}")

        plot_llc(
            csv_path=csv_path,
            out_dir=ckpt_dir,
            filename="plot_llc.png",
        )
        plot_command_tracking(
            info_log,
            ckpt_dir,
            filename="commands.png",
            plot_target_command=False
        )
        plot_reward_terms(
            terms=info_log,
            prefix="reward_terms/",
            out_dir=ckpt_dir,
            threshold=3.0,
            filename="reward_terms.png"
        )
        plot_reward_terms_separate(
            terms=info_log,
            reward_scaling=eval_env._config.reward_config.scales,
            prefix="reward_terms/",
            out_dir=ckpt_dir,
        )
        plot_mpc_output(
            info_log=info_log, 
            prefix="mpc_output",
            out_dir=ckpt_dir,
            filename="mpc_output.png"
        )

        joint_names = []

        for actuator_id in range(eval_env.action_size):
            joint_id = int(eval_env.mj_model.actuator_trnid[actuator_id, 0])

            joint_name = mujoco.mj_id2name(
                eval_env.mj_model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_id,
            )

            if joint_name is None:
                joint_name = f"action_{actuator_id}"

            joint_names.append(joint_name)

        plot_network_actions(
            steps=steps,
            actions=network_actions,
            out_dir=ckpt_dir,
            joint_names=joint_names,
            filename="network_actions.png",
        )


def _build_fresh_networks(env, zero_init_output_layer: bool = False):
    """Return a ppo_networks with random hidden layers and zero-init output layer."""
    if DISTRIBUTION_TYPE == "tanh_normal":
        param_size = 2 * env.action_size
    elif DISTRIBUTION_TYPE == "normal":
        param_size = env.action_size
    else:
        raise ValueError(f"Unsupported distribution type: {DISTRIBUTION_TYPE}")
    

    def _policy_kernel_init_factory(**init_kwargs):
        
        def _get_brax_default_kernel_init_fn():
            """Pull brax's default policy kernel-init factory from the signature."""
            sig = inspect.signature(ppo_networks.make_ppo_networks)
            param = sig.parameters.get("policy_network_kernel_init_fn")
            default = param.default if param is not None else inspect.Parameter.empty

            # Fallback: some forks leave it None here and set the real default deeper,
            # in make_policy_network / MLP.
            if default in (inspect.Parameter.empty, None):
                sig2 = inspect.signature(ppo_networks.networks.make_policy_network)
                p2 = sig2.parameters.get("kernel_init")
                default = p2.default if p2 is not None else jax.nn.initializers.lecun_uniform

            print(f"  [INFO] brax default kernel init: {default}")
            return default

        base_init = _get_brax_default_kernel_init_fn()(**init_kwargs)
        
        def _init(key, shape, dtype=jnp.float32):
            if zero_init_output_layer and len(shape) == 2 and shape[-1] == param_size:
                return jnp.zeros(shape, dtype)
            return base_init(key, shape, dtype)

        return _init

    if ALGO == "ppo":
        networks = ppo_networks.make_ppo_networks(
            observation_size=env.observation_size,
            action_size=env.action_size,
            preprocess_observations_fn=running_statistics.normalize,
            distribution_type=DISTRIBUTION_TYPE,
            **ALGO_PARAMS["network_factory"],
            #activation=linen.elu,
            policy_network_kernel_init_fn=_policy_kernel_init_factory,
            #init_noise_std=INIT_STD
        )
    else:
        networks = sac_networks.make_sac_networks(
            observation_size=env.observation_size,
            action_size=env.action_size,
            hidden_layer_sizes=POLICY_HIDDEN_LAYER_SIZES,
            preprocess_observations_fn=running_statistics.normalize,
            distribution_type=DISTRIBUTION_TYPE,
            activation=linen.elu,
            policy_network_kernel_init_fn=(lambda init_kwargs: _policy_kernel_init_factory(**init_kwargs)),
            init_noise_std=INIT_STD
        )

    self_test = False
    if self_test:
        print("\n" + "=" * 60)
        print(f"  [SELF-TEST] fresh '{DISTRIBUTION_TYPE}' policy")
        print("=" * 60)

        # ── 1. weights from the custom init ──
        policy_params = networks.policy_network.init(jax.random.PRNGKey(0))

        # ── 2. observation spec (flat or dict) + a RANDOM sample obs ──
        obs_size = env.observation_size
        key_obs = jax.random.PRNGKey(42)
        if isinstance(obs_size, dict):
            obs_proto = {
                k: specs.Array((int(np.prod(v)),) if not isinstance(v, int) else (v,), jnp.float32)
                for k, v in obs_size.items()
            }
            keys = jax.random.split(key_obs, len(obs_proto))
            sample_obs = {
                k: jax.random.normal(kk, a.shape, jnp.float32)
                for (k, a), kk in zip(obs_proto.items(), keys)
            }
        else:
            obs_proto = specs.Array((obs_size,), jnp.float32)
            sample_obs = jax.random.normal(key_obs, (obs_size,), jnp.float32)

        normalizer_params = running_statistics.init_state(obs_proto)
        full_params = (normalizer_params, policy_params)

        # ── 3. network / obs info ──
        print("  -- network info --")
        print(f"    ALGO                : {ALGO}")
        print(f"    distribution_type   : {DISTRIBUTION_TYPE}")
        print(f"    hidden layers       : {POLICY_HIDDEN_LAYER_SIZES}")
        print(f"    action_size         : {env.action_size}")
        print(f"    param_size (output) : {param_size}")
        print(f"    zero_init_output    : {zero_init_output_layer}")
        if isinstance(obs_size, dict):
            print(f"    observation_size    : dict -> " + ", ".join(f"{k}:{v}" for k, v in obs_size.items()))
        else:
            print(f"    observation_size    : {obs_size}")

        # ── 4. per-layer parameter shapes ──
        print("  -- policy param shapes --")
        flat = jax.tree_util.tree_flatten_with_path(policy_params)[0]
        total = 0
        for path, value in flat:
            if hasattr(value, "shape"):
                total += int(np.prod(value.shape))
                path_str = "/".join(str(getattr(p, "key", p)) for p in path)
                print(f"    {path_str:<45}: {tuple(value.shape)}")
        print(f"    {'TOTAL scalars':<45}: {total}")

        # ── 5. last-layer kernel norm (proof of zero-init) ──
        last_out = [
            x for x in jax.tree_util.tree_leaves(policy_params)
            if hasattr(x, "ndim") and x.ndim == 2 and x.shape[-1] == param_size
        ]
        if last_out:
            norms = [float(jnp.linalg.norm(k)) for k in last_out]
            print(f"    output-kernel norms : {norms}"
                  f"  ( ~0 if zero_init_output={zero_init_output_layer})")

        # ── 6. raw output on the RANDOM obs ──
        print("  -- raw apply() on a random observation --")
        raw_out = networks.policy_network.apply(normalizer_params, policy_params, sample_obs)
        if isinstance(raw_out, tuple):
            print(f"    apply() -> tuple len {len(raw_out)}, shapes "
                  f"{[np.shape(np.array(o)) for o in raw_out]}")
            loc_raw = jnp.asarray(raw_out[0])
            scale_raw = None
        else:
            arr = jnp.asarray(raw_out)
            print(f"    apply() -> array shape {arr.shape}")
            if DISTRIBUTION_TYPE == "tanh_normal":
                loc_raw, scale_raw = jnp.split(arr, 2, axis=-1)
            else:
                loc_raw, scale_raw = arr, None

        print(f"    loc   (raw)         : {np.array(loc_raw)}")
        if scale_raw is not None:
            # brax maps scale via softplus (+ min) or exp depending on noise_std_type
            print(f"    scale (raw)         : {np.array(scale_raw)}")
            print(f"    softplus(scale)     : {np.array(jax.nn.softplus(scale_raw))}  (if softplus mapping)")
            print(f"    exp(scale)          : {np.array(jnp.exp(scale_raw))}  (if exp/log mapping)")

        # ── 7. std_param, if the distribution stores it separately (typical 'normal') ──
        params_dict = policy_params.get("params", policy_params)
        if "std_param" in params_dict:
            std_raw = np.array(list(params_dict["std_param"].values())[0])
            print(f"    std_param (raw)     : {std_raw}")
        else:
            print("    std_param           : (not present in params)")

        # ── 8. deterministic action + empirical exploration std ──
        inference_fn = (ppo_networks.make_inference_fn(networks) if ALGO == "ppo"
                        else sac_networks.make_inference_fn(networks))

        det_policy = inference_fn(full_params, deterministic=True)
        det_action, _ = det_policy(sample_obs, jax.random.PRNGKey(0))
        det_norm = float(jnp.linalg.norm(det_action))
        print("  -- actions --")
        print(f"    det action          : {np.array(det_action)}")
        print(f"    |det action|        : {det_norm:.3e}")

        stoch_policy = inference_fn(full_params, deterministic=False)
        keys = jax.random.split(jax.random.PRNGKey(1), 2000)
        actions = jax.vmap(lambda k: stoch_policy(sample_obs, k)[0])(keys)
        act_mean, act_std = np.array(actions.mean(0)), np.array(actions.std(0))
        print(f"    sampled mean        : {act_mean}")
        print(f"    sampled std (empir.): {act_std}")

        # ── 9. sanity flags ──
        if zero_init_output_layer and det_norm > 1e-6:
            print(f"    [WARN] zero_init requested but |det action| = {det_norm:.2e} (expected ~0)")
        if np.any(act_std < 1e-3):
            print("    [WARN] std ~0: no exploration, the log-prob risk to explode.")
        elif np.any(act_std > 1.0):
            print("    [WARN] std >1: very wide exploration for small action_scale.")
        print("  [SELF-TEST] done.")
        print("=" * 60 + "\n")
    return networks


def run_train(env, eval_env, wrap_env_fn, ckpt_dir: str,
              resume_dir: str | None = None, resume_suffix: str = "best",
              env_name: str = "Go1JoystickFlatTerrain", algo: str = "ppo"):
    global REWARD_LOG_FILE, METRICS_LOG_FILE, CKPT_DIR
    os.makedirs(ckpt_dir, exist_ok=True)
    CKPT_DIR = ckpt_dir
    REWARD_LOG_FILE = os.path.join(ckpt_dir, "reward_log.txt")
    METRICS_LOG_FILE = os.path.join(ckpt_dir, "metrics_log.csv")
    save_tita_env_files(env_name, ckpt_dir)

    if algo == "ppo":
        policy_obs_key = ALGO_PARAMS["network_factory"]["policy_obs_key"]
        value_obs_key = ALGO_PARAMS["network_factory"]["value_obs_key"]
    else:
        # SAC: single shared obs key, no policy/value split.
        policy_obs_key = value_obs_key = "state"

    header_lines = [
        "=" * 60,
        f"  {algo.upper()} Training  —  {env_name}",
        f"  date         : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  JAX backend  : {jax.default_backend()}",
        f"  devices      : {jax.devices()}",
        f"  GPU name:    : "
        "  \n--- network ---",
        f"  distribution : {DISTRIBUTION_TYPE}",
        f"  hidden layers: {POLICY_HIDDEN_LAYER_SIZES}",
        f"  entropy cost : {ALGO_PARAMS.get('entropy_cost', 'N/A')}",
        f"  policy obs   : {policy_obs_key}",
        f"  value obs    : {value_obs_key if algo == 'ppo' else '(shared)'}",
        f"  policy net shape: {env.observation_size[policy_obs_key] if algo == 'ppo' else env.observation_size}",
        f"  value net shape : {env.observation_size[value_obs_key] if algo == 'ppo' else '(shared)'}",
        f"  obs size     : {env.observation_size}",
        f"  action size  : {env.action_size}",
        f"  --- {algo.upper()} params ---",
        *[f"  {k:25s}: {v}" for k, v in ALGO_PARAMS.items()],
        "=" * 60,
        "",
    ]
    for l in header_lines:
        print(l)
    with open(REWARD_LOG_FILE, "a") as _f:
        _f.write("\n".join(header_lines) + "\n")

    env_config_block = (
        "--- environment config ---\n"
        f"{env._config}"
        "\n" + "=" * 60 + "\n"
    )
    print(env_config_block)
    with open(REWARD_LOG_FILE, "a") as _f:
        _f.write(env_config_block + "\n")


    restore_params = None
    built_networks = _build_fresh_networks(env, zero_init_output_layer=ZERO_INIT_OUTPUT_LAYER)
    selected_network_factory = lambda *args, **kwargs: built_networks

    if resume_dir:
        restore_params = load_params(resume_dir, suffix=resume_suffix)
        if restore_params is None:
            print(f"  [WARN] --load requested but no checkpoint found in '{resume_dir}'.")
            print("  Starting training from random initialization.")
        else:
            print(f"  Initializing training from checkpoint in '{resume_dir}'.")
    else:
        print("  Fresh training.")
    
    def _extract_actor_critic(restore_params):
        """Return (policy_params, value_params) from a brax restore tuple, or (None, None)."""
        if restore_params is None:
            return None, None
        # brax PPO save format: (normalizer_params, policy_params, value_params)
        # older/other formats may pack differently; guard defensively.
        try:
            _, policy_params, value_params = restore_params
            return policy_params, value_params
        except (ValueError, TypeError):
            return None, None

    loaded_policy, loaded_value = _extract_actor_critic(restore_params)

    def _dump_param_shapes(label, net, params):
        # Fall back to a fresh init only if no loaded params are available.
        if params is None:
            params = net.init(jax.random.PRNGKey(ALGO_PARAMS["seed"]))
        flat = jax.tree_util.tree_flatten_with_path(params)[0]
        print(f"  {label} param shapes:")
        total = 0
        for path, value in flat:
            if hasattr(value, "shape"):
                n = int(np.prod(value.shape))
                total += n
                path_str = "/".join(str(getattr(p, "key", p)) for p in path)
                norm = float(jnp.linalg.norm(value))
                print(f"    {path_str:<45}: {str(tuple(value.shape)):<15} ({n:,})  norm={norm:.3e}")
        print(f"    {'TOTAL scalars':<45}: {'':<15} ({total:,})")
        return params

    src = "LOADED" if restore_params is not None else "FRESH INIT"
    print(f"  [network params shown from: {src}]")
    network_policy_params = _dump_param_shapes("actor (policy)", built_networks.policy_network, loaded_policy)
    if ALGO == "ppo":
        _dump_param_shapes("critic (value)", built_networks.value_network, loaded_value)

    print(f"    Distribution_type: {DISTRIBUTION_TYPE}")
    policy_module = inspect.getclosurevars(built_networks.policy_network.apply).nonlocals["policy_module"]
    activation_fn = policy_module.activation
    print(f"    Activation (read from policy_network module): {getattr(activation_fn, '__name__', activation_fn)}")
    out_dim = built_networks.parametric_action_distribution.param_size
    candidate_kernels = [
        x for x in jax.tree_util.tree_leaves(network_policy_params)
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

    train_fn = make_train_fn(algo)
    train_kwargs = dict(
        environment=env,
        eval_env=eval_env,
        wrap_env_fn=wrap_env_fn,
        network_factory=selected_network_factory,
        restore_params=restore_params,
        policy_params_fn=_policy_params_cb,
    )

    accepted = inspect.signature(train_fn.func).parameters
    dropped = [k for k in train_kwargs if k not in accepted]
    if dropped:
        print(f"  [WARN] {algo}.train does not support: {dropped} — ignored.")
    train_kwargs = {k: v for k, v in train_kwargs.items() if k in accepted}

    try:
        make_inference_fn, params, _ = train_fn(**train_kwargs)
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Ctrl+C received: stopping training and saving available weights...")
        if latest_params is not None:
            save_params(latest_params, ckpt_dir, suffix="final")
            print(f"[INTERRUPT] Checkpoint saved at step ~{latest_step:,}.")
        else:
            print("[INTERRUPT] No parameters available to save.")
        print(f"[INFO] Saved training info in {os.path.abspath(ckpt_dir)}")
        return None, latest_params
    except Exception as e:
        print(f"\n[ERROR] Training crashed: {e}")
        if latest_params is not None:
            save_params(latest_params, ckpt_dir, suffix="crash")
            print(f"[ERROR] Checkpoint saved at step ~{latest_step:,} → params_crash.pkl")
        else:
            print("[ERROR] No parameters available to save.")
        print(f"[INFO] Saved training info in {os.path.abspath(ckpt_dir)}")
        raise

    if len(times) > 1:
        print(f"\nTime to jit:   {times[1] - times[0]}")
        print(f"Time to train: {times[-1] - times[1]}")

    save_params(params, ckpt_dir, suffix="final")
    print(f"[INFO] Saved training info in {os.path.abspath(ckpt_dir)}")
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
        help="MuJoCo Playground environment name.",
    )
    parser.add_argument("--algo", type=str, choices=["ppo", "sac"], default="ppo",
                        help="RL algorithm: 'ppo' (default) o 'sac'")
    parser.add_argument("--load", nargs="?", const="best", default=None, metavar="RUN_OR_SUFFIX",
                        help="Load checkpoint weights. Checkpoints are stored per-run under "
                             "<ckpt-dir>/<name>/<run_timestamp>/. Bare '--load' loads the latest "
                             "run's best checkpoint; '--load <run_timestamp>' loads that specific "
                             "run (exact or prefix match); '--load crash'/'final' loads that "
                             "suffix from the latest run.")
    parser.add_argument("--zero", action="store_true", help="Force zero actions (ignore policy network)")
    parser.add_argument("--headless", action="store_true", help="Eval rollout without opening the MuJoCo viewer")
    parser.add_argument("--random", action="store_true", help="Use a random network for evaluation")
    parser.add_argument("--cmd", nargs="+", type=float, default=None, metavar="CMD_I",
                        help="Fix joystick command for eval rollout. Number of values must match "
                             "the env's command_config (e.g. 2 values for Tita's [vx, wz], "
                             "3 for a quadruped's [vx, vy, wz]), e.g. --cmd 0.5 0.0. One extra "
                             "trailing value sets the target base height (envs that support it, "
                             "e.g. Tita); if omitted, the env's own init height is used, "
                             "e.g. --cmd 0.5 0.0 0.35")
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default="checkpoints",
        help="Checkpoint root directory (checkpoints are stored in <root>/<env_name>)",
    )

    args = parser.parse_args()

    global ALGO, ALGO_PARAMS
    ALGO = args.algo
    ALGO_PARAMS = SAC_PARAMS if args.algo == "sac" else PPO_PARAMS

    # Auto-set headless if DISPLAY is missing
    if not args.train and not args.headless and (os.environ.get("DISPLAY") is None or os.environ.get("DISPLAY") == ""):
        print("[WARN] No DISPLAY detected: forcing --headless mode (no viewer)")
        args.headless = True

    # ── pick environment ────────────────────────────────────────
    _NAME_SHORTCUTS = {
        "go1": "Go1JoystickFlatTerrain",
        "aliengo" : "AliengoJoystickE2EFlatTerrain",
        "tita": "TitaJoystickFlatTerrain",
        "titae2e": "TitaJoystickE2EFlatTerrain",
        "litee2e": "LiteE2EJoystickFlatTerrain",
        "lite": "LiteJoystickFlatTerrain",
    }
    env_name = _NAME_SHORTCUTS.get(args.name.lower(), args.name)
    env, eval_env, wrap_fn = make_envs(env_name=env_name)
    env_base_dir = os.path.join(args.ckpt_dir, env_name)


    # ── train or eval ───────────────────────────────────────────
    if not args.eval:
        resume_dir, resume_suffix = (
            _resolve_load(env_base_dir, args.load) if args.load else (None, "best")
        )
        ckpt_dir = os.path.join(env_base_dir, SCRIPT_START_TIME)
        run_train(env, eval_env, wrap_fn, ckpt_dir,
                  resume_dir=resume_dir, resume_suffix=resume_suffix,
                  env_name=env_name, algo=ALGO)
        return

    print("=" * 60)
    print(f"  {ALGO.upper()} Eval  —  {env_name}")
    print("=" * 60)


    fixed_cmd = np.array(args.cmd) if args.cmd is not None else None

    def _make_fresh_params(env, networks, seed=0, tag=""):

        # --- weights ---
        policy_params = networks.policy_network.init(jax.random.PRNGKey(seed))

        obs_size = env.observation_size
        if isinstance(obs_size, dict):
            obs_proto = {
                k: specs.Array((int(np.prod(v)),) if not isinstance(v, int) else (v,), jnp.float32)
                for k, v in obs_size.items()
            }
        else:
            obs_proto = specs.Array((obs_size,), jnp.float32)

        # --- normalizer state ---
        normalizer_params = running_statistics.init_state(obs_proto)

        # --- diagnostics ---
        print(f"  [fresh_params{(' ' + tag) if tag else ''}] seed={seed}")
        print(f"    obs_size type      : {type(obs_size).__name__} -> {obs_size}")
        n_leaves = len(jax.tree_util.tree_leaves(policy_params))
        total = int(sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(policy_params)))
        print(f"    policy params      : {n_leaves} leaves, {total} scalars")
        print(f"    normalizer         : {'dict' if isinstance(obs_proto, dict) else obs_proto.shape} (empty: mean 0 / var 1 -> obs NOT normalized)")

        return (normalizer_params, policy_params)


    run_dir, load_suffix = _resolve_load(
        env_base_dir,
        args.load if args.load else "best",
    )
    ckpt_dir = run_dir if run_dir else env_base_dir
    params_inference_fn = load_params(ckpt_dir, suffix=load_suffix)
    zero_command = False

    if args.random or params_inference_fn is None:

        if params_inference_fn is None:
            print(f"  [NET-INIT] No checkpoint found in '{ckpt_dir}' — using random network.")
        elif args.random:
            print("  [NET-INIT] --random flag: using random network.")
        else:
            print("  [NET-INIT] Using random network: no flag detected.")

        networks = _build_fresh_networks(eval_env)
        normalizer_params, policy_params = _make_fresh_params(eval_env, networks, seed=0, tag="random")
        params_inference_fn = (normalizer_params, policy_params)  

    elif args.zero:
        print("  [NET-INIT] --zero flag: using fresh network with zero output layer.")
        networks = _build_fresh_networks(eval_env, zero_init_output_layer=True)
        normalizer_params, policy_params = _make_fresh_params(eval_env, networks, seed=0, tag="zero")
        params_inference_fn = (normalizer_params, policy_params)

        zero_command = True
    else:
        print(f"  [NET-INIT] Checkpoint will be loaded from '{ckpt_dir}'")

        networks = _build_fresh_networks(eval_env)

    inference_fn = ppo_networks.make_inference_fn(networks) if args.algo == "ppo" else sac_networks.make_inference_fn(networks)

    policy_fn = inference_fn(params_inference_fn, deterministic=True)

    run_viewer_rollout(
        eval_env=eval_env, 
        inference_fn=policy_fn, 
        env_name=env_name, 
        headless=args.headless, 
        ckpt_dir=ckpt_dir, 
        fixed_command=fixed_cmd, 
        zero_command=zero_command
    )
    

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

