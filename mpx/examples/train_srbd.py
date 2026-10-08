"""
train_srbd.py
-------------
PPO / SAC training for MuJoCo Playground locomotion.
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

# Compatibility shim: brax 0.14.2 calls jax.device_put_replicated (train.py:756),
# which jax 0.10.2 removed (along with device_put_sharded). Restore it with the
# correct semantics: replicate the pytree across `devices`, adding a leading
# device axis so pmap can map over it.
if not hasattr(jax, "device_put_replicated"):
    import jax.numpy as _jnp

    def _device_put_replicated(x, devices):
        n = len(devices)
        try:
            from jax.sharding import PositionalSharding
            sharding = PositionalSharding(devices)

            def _rep(leaf):
                leaf = _jnp.asarray(leaf)
                stacked = _jnp.broadcast_to(leaf, (n,) + leaf.shape)
                return jax.device_put(stacked, sharding.reshape((n,) + (1,) * leaf.ndim))
        except Exception:
            def _rep(leaf):
                leaf = _jnp.asarray(leaf)
                return _jnp.broadcast_to(leaf, (n,) + leaf.shape)
        return jax.tree_util.tree_map(_rep, x)

    jax.device_put_replicated = _device_put_replicated

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
import json
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

from algorithm_initialization import (
    initialize_residual,
    install_configured_ppo_loss,
    make_sac_networks,
    restore_ppo_loss,
)

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
POLICY_HIDDEN_LAYER_SIZES = (512, 256, 128)  # Dimensioni dei layer nascosti dell'actor.
CRITIC_HIDDEN_LAYER_SIZES = (512, 256, 128)  # Dimensioni dei layer nascosti del critic/value.
DISTRIBUTION_TYPE = "tanh_normal"  # Gaussiana compressa con tanh: azioni in [-1, 1].
ZERO_INIT_OUTPUT_LAYER = True  # True: l'azione deterministica iniziale è esattamente zero.
INIT_STD = 0.1  # Std iniziale della policy; il test verifica se 0.03 limitava l'attivazione.
LOSS_NN_ACTION = True  # Aggiunge mean(action_deterministica**2) alla loss dell'actor.
LOSS_NN_ACTION_COST = 1.0  # Prior sul mean action ridotto da 2.0 per consentire correzioni utili.
CRITIC_WARMUP = True  # Prima adatta il critic al mix flat/rough con actor congelato.
PPO_CRITIC_WARMUP_TIMESTEPS = 1_000_000  # Step richiesti per il burn-in PPO della value network.
PPO_CRITIC_WARMUP_ON_RESTORE = False  # Resume dal checkpoint warm-up: actor e critic riprendono insieme.

NUM_TIMESTEPS = 20_000_000  # Transizioni totali raccolte durante il training.
NUM_EVALS = 10  # Numero di valutazioni distribuite lungo il training.
EPISODE_LENGTH = 1000  # Durata massima di un episodio, in step dell'environment.
NUM_ENVS = 1024  # Environment paralleli usati da PPO.
DETERMINISTIC_EVAL = True  # Usa la media della policy durante l'eval, senza campionare.
DEFAULT_NUM_EVAL_ENVS = inspect.signature(ppo.train).parameters["num_eval_envs"].default  # Default Brax, non duplicato localmente.

PPO_PARAMS = dict(
      leg_only_actions=False,  # 8 azioni: offset gambe e velocità residuale delle ruote.
      num_timesteps=NUM_TIMESTEPS,  # Numero totale di transizioni dell'environment.
      num_evals=NUM_EVALS,  # Numero di valutazioni durante il training.
      reward_scaling=1.0,  # Moltiplicatore delle reward usato per return e value target.
      episode_length=EPISODE_LENGTH,  # Step massimi prima del reset dell'episodio.
      normalize_observations=True,  # Normalizza le obs con media e varianza correnti.
      action_repeat=1,  # Step dell'environment eseguiti per ogni azione della policy.
      unroll_length=20,  # Step consecutivi per environment in ogni segmento di rollout.
      num_minibatches=32,  # Suddivisioni del batch per ogni epoca di ottimizzazione.
      num_updates_per_batch=2,  # Epoche di ottimizzazione su ogni batch raccolto.
      discounting=0.99,  # Gamma: fattore di sconto delle reward future.
      # La baseline ad azione zero è già quasi ottima nel task a regime:
      # update piccoli evitano che PPO se ne allontani alla prima epoca.
      learning_rate=1e-5,  # Learning rate Adam condiviso da actor e value network.
      # Adaptive KL unilaterale: riduce il LR se l'update è troppo grande,
      # senza superare il valore massimo conservativo.
      learning_rate_schedule=LRSchedule.ADAPTIVE_KL,  # Riduce il LR quando il KL è troppo alto.
      #desired_kl=0.01,  # KL desiderato per update; usato dalla schedule adaptive KL.
      learning_rate_schedule_min_lr=1e-6,  # Limite inferiore del learning rate adattivo.
      learning_rate_schedule_max_lr=1e-5,  # Limite superiore del learning rate adattivo.
      entropy_cost=0.0,  # Peso del bonus di entropia; zero non incentiva maggiore varianza.
      num_envs=NUM_ENVS,  # Environment paralleli per raccogliere i rollout PPO.
      batch_size=256,  # Campioni raggruppati in ogni batch dell'ottimizzatore PPO.
      max_grad_norm=1.0,  # Soglia di clipping della norma globale del gradiente.
      network_factory=dict(
            activation=linen.elu,  # Attivazione dei layer nascosti di actor e critic.
            policy_hidden_layer_sizes=POLICY_HIDDEN_LAYER_SIZES,  # Layer nascosti dell'actor.
            value_hidden_layer_sizes=CRITIC_HIDDEN_LAYER_SIZES,  # Layer nascosti del critic.
            policy_obs_key="state",  # Parte dell'osservazione visibile all'actor.
            value_obs_key="privileged_state",  # Informazioni privilegiate visibili al critic.
        ),
      num_resets_per_eval=1,  # Episodi completi ripetuti per ogni valutazione.
      seed=0,  # Seed per inizializzazione, rollout e aggiornamenti.
      deterministic_eval=DETERMINISTIC_EVAL,  # Usa il modo della policy durante l'eval.
  )

print(f"PPO_PARAMS: \n{PPO_PARAMS}")

# SAC usato nello studio Tita. Questo blocco si modifica come PPO_PARAMS.
# RPL usa DDPG/HER e non definisce reward_scaling=10: rete, learning rate,
# update per blocco e prior sull'azione restano scelte esplicite dello studio.
SAC_PARAMS = dict(
    num_timesteps=NUM_TIMESTEPS,  # Transizioni totali inserite nel replay buffer.
    num_evals=NUM_EVALS,  # Numero di valutazioni durante il training.
    episode_length=EPISODE_LENGTH,  # Step massimi prima del reset dell'episodio.
    num_envs=10,  # Collector paralleli; deve dividere update_every_transitions.
    seed=0,  # Seed JAX e del campionamento dal replay buffer.
    deterministic_eval=DETERMINISTIC_EVAL,  # Usa il modo dell'actor durante l'eval.
    # num_eval_envs omesso: usa DEFAULT_NUM_EVAL_ENVS letto dalla firma Brax.
    num_resets_per_eval=1,  # Episodi completi per ogni environment di valutazione.
    leg_only_actions=False,  # 8 azioni come PPO: gambe e ruote, per un confronto controllato.
    # SAC può terminare automaticamente il warm-up perché actor e critic hanno
    # optimizer separati. CRITIC_WARMUP è l'interruttore condiviso con PPO.
    burn_in_min_blocks=20,  # Blocchi critic-only minimi prima dello sblocco dell'actor.
    burn_in_max_blocks=200,  # Sblocco forzato dell'actor se il critic non converge prima.
    burn_in_window=10,  # Blocchi per finestra usati nel confronto delle stime Q.
    burn_in_rel_tol=0.01,  # Variazione relativa di Q sotto cui il critic è stabile.
    burn_in_explore_std=0.1,  # Rumore extra sulle azioni nei rollout esplorativi del warm-up.
    burn_in_on_restore=False,  # Ripete il warm-up caricando un learner SAC esistente.
    actor_learning_rate=1e-4,  # Learning rate Adam della policy SAC.
    critic_learning_rate=1e-3,  # Learning rate Adam delle due reti Q.
    alpha_learning_rate=1e-3,  # Learning rate Adam della temperatura di entropia alpha.
    batch_size=512,  # Transizioni campionate dal replay per ogni gradient update.
    reward_scaling=1.0,  # Scala naturale: le reward Tita producono già target Q di ordine unitario.
    discounting=0.99,  # Gamma usato nei target di Bellman.
    initial_alpha=0.1,  # Valore iniziale della temperatura di entropia.
    tau=0.005,  # Coefficiente Polyak per il soft update delle target Q.
    replay_capacity=1_000_000,  # Numero massimo di transizioni conservate nel replay.
    update_every_transitions=1000,  # Nuove transizioni raccolte prima di ogni blocco update.
    updates_per_block=64,  # Gradient update eseguiti dopo ogni blocco di raccolta.
    init_std=INIT_STD,  # Deviazione standard iniziale dell'actor prima della tanh.
    network_factory=dict(
        policy_hidden_layer_sizes=POLICY_HIDDEN_LAYER_SIZES,  # Layer nascosti dell'actor SAC.
        critic_hidden_layer_sizes=CRITIC_HIDDEN_LAYER_SIZES,  # Layer nascosti delle reti Q.
        activation=linen.elu,  # Attivazione dei layer nascosti di actor e reti Q.
        distribution_type=DISTRIBUTION_TYPE,  # Famiglia della distribuzione della policy.
    ),
    # Decadimento opzionale dopo lo sblocco dell'actor SAC dal critic warm-up.
    zero_action_prior_final=None,  # Peso finale del prior; None mantiene il costo globale.
    prior_decay_steps=0,  # Transizioni usate per interpolare verso il peso finale.
)

print(f"SAC_PARAMS: \n{SAC_PARAMS}")

# ─────────────────────────────────────────────────────────────────────────────
#  Environment factories
# ─────────────────────────────────────────────────────────────────────────────

class SACStateWrapper(Wrapper):
    """Actor and critic share the current joystick state, with no study augmentation."""
    @property
    def observation_size(self):
        return int(np.prod(self.env.observation_size["state"]))

    def _get_obs(self, data, info, action):
        return self.env._get_obs(data, info, action)["state"].astype(jnp.float32)

    def reset(self, rng):
        state = self.env.reset(rng)
        return state.replace(obs=state.obs["state"].astype(jnp.float32))

    def step(self, state, action):
        state = self.env.step(state, action)
        return state.replace(obs=state.obs["state"].astype(jnp.float32))


def make_envs(
    env_name: str = "Go1JoystickFlatTerrain",
    reward_scale_overrides: dict | None = None,
    fixed_target: list | None = None,
    obstacle_mode: str | None = None,
    obstacle_height: float | None = None,
    terrain_overlay: str | None = None,
):
    """Return (env, eval_env, wrap_fn) for a MuJoCo Playground env.

    reward_scale_overrides: optional {term_name: value} dict merged into
    config.reward_config.scales via registry.load's config_overrides. Used
    by the analysis_training reward-effects study to vary only the reward
    coefficients while every other config field (command, PPO, network,
    episode length, ...) stays exactly as the env's default_config().
    """

    from mujoco_playground import registry
    from mujoco_playground._src.wrapper import wrap_for_brax_training as pg_wrap

    import mujoco_playground
    print(f"  [INFO] mujoco_playground loaded from: {mujoco_playground.__file__}")
    print(f"  [INFO] mujoco_playground._src.wrapper loaded from: {pg_wrap.__module__} -> {sys.modules[pg_wrap.__module__].__file__}")

    config_overrides = {}
    if env_name.startswith("TitaJoystick") and "E2E" not in env_name:
        config_overrides["leg_only_actions"] = ALGO_PARAMS["leg_only_actions"]
    if fixed_target is not None:
        if not env_name.startswith("TitaJoystick") or "E2E" in env_name:
            raise ValueError("--train-cmd currently supports the MPC Tita environment")
        config_overrides["command_config.fixed_target"] = fixed_target
    if obstacle_mode is not None:
        if not env_name.startswith("TitaJoystick") or "E2E" in env_name:
            raise ValueError("--obstacle currently supports the MPC Tita environment")
        config_overrides["sparse_obstacles.enabled"] = True
        config_overrides["sparse_obstacles.mode"] = obstacle_mode
        if obstacle_height is not None:
            if obstacle_height <= 0:
                raise ValueError("--obstacle-height must be positive")
            config_overrides["sparse_obstacles.height"] = obstacle_height
    elif obstacle_height is not None:
        raise ValueError("--obstacle-height requires --obstacle")
    if terrain_overlay is not None:
        if env_name != "TitaJoystickFlatTerrain":
            raise ValueError("--terrain-overlay currently uses TitaJoystickFlatTerrain")
        if terrain_overlay != "rough":
            raise ValueError("Supported terrain overlay: rough")
        terrain_xml = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "data", "tita", "scene_rough.xml"))
        if not os.path.isfile(terrain_xml):
            raise FileNotFoundError(terrain_xml)
        if obstacle_mode is not None:
            raise ValueError("Use either --terrain-overlay or --obstacle, not both")
        config_overrides["sparse_obstacles.enabled"] = True
        config_overrides["sparse_obstacles.mode"] = "terrain_overlay"
        config_overrides["sparse_obstacles.terrain_xml"] = terrain_xml
        print(f"  [INFO] Rough XML overlaid on flat task: {terrain_xml}")
    if reward_scale_overrides:
        config_overrides.update({
            f"reward_config.scales.{k}": v for k, v in reward_scale_overrides.items()
        })
        print(f"  [INFO] reward_config.scales overrides: {reward_scale_overrides}")

    env      = registry.load(env_name, config_overrides=config_overrides)
    eval_env = registry.load(env_name, config_overrides=config_overrides)

    if ALGO == "sac":
        env, eval_env = SACStateWrapper(env), SACStateWrapper(eval_env)

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
    nn_action_mean_l2 = metrics.get("training/nn_action_mean_l2")
    nn_action_prior_loss = metrics.get("training/nn_action_prior_loss")
    nn_action_abs_max = metrics.get("training/nn_action_abs_max")
    nn_action_means = [
        metrics.get(f"training/nn_action_mean_{i}") for i in range(8)
    ]
    nn_action_rms = [
        metrics.get(f"training/nn_action_rms_{i}") for i in range(8)
    ]
    policy_obs_norm_rms = metrics.get("training/policy_obs_norm_rms")
    policy_obs_norm_abs_max = metrics.get("training/policy_obs_norm_abs_max")
    policy_obs_norm_clip_fraction = metrics.get(
        "training/policy_obs_norm_clip_fraction"
    )
    critic_warmup_active = metrics.get("training/critic_warmup_active")

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
    nn_action_str = (
        f"{nn_action_mean_l2:.6f} (loss={nn_action_prior_loss:.6f})"
        if nn_action_mean_l2 is not None and nn_action_prior_loss is not None
        else "n/a"
    )
    def _format_joint_values(values):
        present = [value for value in values if value is not None]
        return (
            "[" + ", ".join(f"{float(value):+.4f}" for value in present) + "]"
            if present else "n/a"
        )

    action_mean_str = _format_joint_values(nn_action_means)
    action_rms_str = _format_joint_values(nn_action_rms)
    action_max_str = (
        f"{float(nn_action_abs_max):.4f}"
        if nn_action_abs_max is not None else "n/a"
    )
    obs_norm_str = (
        f"rms={float(policy_obs_norm_rms):.3f}, "
        f"max={float(policy_obs_norm_abs_max):.3f}, "
        f"clip={100.0 * float(policy_obs_norm_clip_fraction):.3f}%"
        if policy_obs_norm_rms is not None
        else "n/a"
    )
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
        f"\n\tnn action L2 = {nn_action_str}"
        f"\n\taction mean = {action_mean_str}"
        f"\n\taction rms  = {action_rms_str}"
        f"\n\taction |max| = {action_max_str}"
        f"\n\tnormalized policy obs: {obs_norm_str}"
        f"\n\tcritic warmup = {critic_warmup_active if critic_warmup_active is not None else 'n/a'}"
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
                "nn_action_mean_l2", "nn_action_prior_loss",
                *[f"nn_action_mean_{i}" for i in range(8)],
                *[f"nn_action_rms_{i}" for i in range(8)],
                "nn_action_abs_max", "policy_obs_norm_rms",
                "policy_obs_norm_abs_max", "policy_obs_norm_clip_fraction",
                "critic_warmup_active",
            ])
        writer.writerow([
            num_steps, y_data[-1], y_dataerr[-1],
            mean_reward_per_step, std_reward_per_step, avg_ep_len,
            std_mean, std_min, std_max,
            entropy_loss, kl_mean, total_loss, policy_loss, v_loss,
            nn_action_mean_l2, nn_action_prior_loss,
            *nn_action_means, *nn_action_rms,
            nn_action_abs_max, policy_obs_norm_rms,
            policy_obs_norm_abs_max, policy_obs_norm_clip_fraction,
            critic_warmup_active,
        ])

# ─────────────────────────────────────────────────────────────────────────────
#  train_fn
# ─────────────────────────────────────────────────────────────────────────────

ALGO = "ppo"              # sovrascritto in main() dal flag --algo
ALGO_PARAMS = SAC_PARAMS if ALGO == "sac" else PPO_PARAMS  # sovrascritto in main()

#def make_train_fn(algo: str):
#    if algo == "sac":
#        return functools.partial(sac.train, **SAC_PARAMS, progress_fn=progress)
#    return functools.partial(ppo.train, **{k: v for k, v in PPO_PARAMS.items() if k != "leg_only_actions"}, progress_fn=progress)

def make_train_fn(algo: str, progress_fn=progress, params_override=None):
    if algo == "sac":
        raise ValueError("SAC is dispatched through run_sac_training with SAC_PARAMS")
    train_callable = ppo.train
    params = dict(params_override or (SAC_PARAMS if algo == "sac" else PPO_PARAMS))

    sig = inspect.signature(train_callable)

    # chiavi che passi ma che train.train NON accetta (verrebbero rifiutate)
    unknown = [k for k in params if k not in sig.parameters and k != "leg_only_actions"]
    if unknown:
        print(f"\n  [ATTENZIONE] chiavi in {algo.upper()}_PARAMS non accettate da {algo}.train: {unknown}")
    print("=" * 70 + "\n")

    return functools.partial(
        ppo.train,
        **{k: v for k, v in params.items() if k != "leg_only_actions"},
        progress_fn=progress_fn,
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
    # Load the requested selection, falling back to the completed/interrupted run.
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


def build_eval_inference_fn(env, load_arg, env_name="TitaJoystickFlatTerrain",
                            ckpt_root=None):
    """Return a deterministic inference_fn(obs, key) -> (action, extras) built
    from a saved checkpoint. Reuses the training network construction and the
    (normalizer_params, policy_params) pickle. Used by the residual_eval sweeps."""
    if ckpt_root is None:
        ckpt_root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "checkpoints")
    env_base_dir = os.path.join(ckpt_root, env_name)
    run_dir, load_suffix = _resolve_load(env_base_dir, load_arg if load_arg else "best")
    ckpt_dir = run_dir if run_dir else env_base_dir
    params = load_params(ckpt_dir, suffix=load_suffix)
    if params is None:
        raise FileNotFoundError(f"No checkpoint under {ckpt_dir}")
    networks = _build_fresh_networks(env)
    make_inf = ppo_networks.make_inference_fn(networks)
    print(f"  [build_eval_inference_fn] loaded {ckpt_dir} suffix={load_suffix}")
    return make_inf(params, deterministic=True)


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

    load_arg may be a checkpoint suffix ('best', 'best_return', 'latest',
    'final', 'interrupted', or 'crash'), in which
    case the latest timestamped run under env_base_dir is used; a run
    timestamp/prefix matched under env_base_dir; a name of a run saved
    under env_base_dir/saved/; or an explicit relative path such as
    'saved/joystick_first_train'. Falls back to env_base_dir itself
    (legacy flat layout, no per-run subfolder) if no run subfolders exist.
    """
    run_dirs = _list_run_dirs(env_base_dir)
    is_suffix = load_arg in {
        "best", "best_return", "latest",
        "final", "interrupted", "crash",
    }
    suffix = load_arg if is_suffix else "best"

    if not is_suffix:
        # Permit transfer/fine-tuning from another task's run directory,
        # e.g. the flat/single-left checkpoint when starting rough terrain.
        if os.path.isabs(load_arg) and os.path.isdir(load_arg):
            return os.path.abspath(load_arg), "best"
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

    episode_length = ALGO_PARAMS["episode_length"]

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
    """Build PPO or SAC networks from the selected network_factory config."""
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
            # Clip the NORMALIZED observation to +-10. Without this a near-constant
            # obs component (e.g. a contact force at ~const during normal walking,
            # so tiny running std) that JUMPS at a divergence produces
            # (jump)/(tiny_std) = a huge normalized value -> NaN in the networks.
            # brax's own locomotion configs set max_abs_value; it was omitted here
            # and was the cause of the training NaNs in the violent step regime.
            preprocess_observations_fn=functools.partial(
                running_statistics.normalize, max_abs_value=10.0),
            distribution_type=DISTRIBUTION_TYPE,
            **ALGO_PARAMS["network_factory"],
            policy_network_kernel_init_fn=_policy_kernel_init_factory,
            #init_noise_std=INIT_STD
        )
        if zero_init_output_layer and DISTRIBUTION_TYPE == "tanh_normal":
            networks = initialize_residual(networks, env.action_size, std=INIT_STD)
    elif ALGO == "sac":
        networks = make_sac_networks(
            observation_size=env.observation_size,
            action_size=env.action_size,
            preprocess_observations_fn=functools.partial(
                running_statistics.normalize, max_abs_value=10.0),
            **ALGO_PARAMS["network_factory"],
        )
        if zero_init_output_layer:
            # Brax's tanh-normal head needs explicit initialization of mean AND
            # std.  With the flag disabled, retain Brax's default random head.
            networks = initialize_residual(
                networks, env.action_size, std=INIT_STD
            )
    else:
        raise ValueError(f"Unsupported algorithm: {ALGO}")

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


def _install_training_signal_handlers():
    """Turn SIGINT/SIGTERM into a catchable stop request and report its name."""
    received = {"signum": None}
    previous = {}

    def _handler(signum, _frame):
        received["signum"] = signum
        raise KeyboardInterrupt

    for sig in (signal.SIGINT, signal.SIGTERM):
        previous[sig] = signal.getsignal(sig)
        signal.signal(sig, _handler)
    return received, previous


def _restore_training_signal_handlers(previous):
    for sig, handler in previous.items():
        signal.signal(sig, handler)


def _signal_name(received):
    signum = received.get("signum")
    return signal.Signals(signum).name if signum is not None else "KeyboardInterrupt"


def run_sac_training(env, eval_env, ckpt_dir, resume_dir=None, resume_suffix="best"):
    from pathlib import Path
    import sac_training
    out = Path(ckpt_dir)
    cfg = dict(SAC_PARAMS)
    num_eval_envs = cfg.get("num_eval_envs", DEFAULT_NUM_EVAL_ENVS)
    for module_path in (sac_training.__file__,
                        os.path.join(os.path.dirname(__file__), "algorithm_initialization.py")):
        save_source_files(module_path, ckpt_dir)
    factory_config = cfg["network_factory"]
    serialized_cfg = dict(cfg, network_factory=dict(
        factory_config, activation=factory_config["activation"].__name__))
    metadata = dict(algorithm="sac", parameters=serialized_cfg,
                    zero_init_output_layer=ZERO_INIT_OUTPUT_LAYER,
                    loss_nn_action=LOSS_NN_ACTION,
                    loss_nn_action_cost=LOSS_NN_ACTION_COST,
                    critic_warmup=CRITIC_WARMUP,
                    observation_size=env.observation_size,
                    action_size=env.action_size,
                    num_eval_envs=num_eval_envs,
                    policy_hidden_layer_sizes=list(factory_config["policy_hidden_layer_sizes"]),
                    critic_hidden_layer_sizes=list(factory_config["critic_hidden_layer_sizes"]),
                    activation=factory_config["activation"].__name__,
                    observation="joystick state", normalize_observations=True,
                    resume_mode="weights only; replay and optimizers restart")
    (out / "sac_config.json").write_text(json.dumps(metadata, indent=2))
    (out / "environment_config.json").write_text(json.dumps(env._config.to_dict(), indent=2))
    restore = None
    if resume_dir:
        path = Path(resume_dir) / f"learner_{resume_suffix}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"SAC warm start requires its matching learner checkpoint: {path}")
        old_meta = json.loads((Path(resume_dir) / "sac_config.json").read_text())
        if (old_meta['observation_size'], old_meta['action_size']) != (env.observation_size, env.action_size):
            raise ValueError("Checkpoint observation/action dimensions differ from the current environment")
        with path.open("rb") as f: restore = pickle.load(f)
    eval_reset = jax.jit(jax.vmap(eval_env.reset))
    eval_step = jax.jit(jax.vmap(eval_env.step))
    best_return = -float("inf")
    latest = None

    def checkpoint(step, make_policy, params, learner):
        nonlocal best_return, latest
        latest = (params, learner)
        policy = jax.jit(make_policy(params, deterministic=cfg['deterministic_eval']))
        episode_returns, episode_lengths = [], []
        for episode in range(cfg['num_resets_per_eval']):
            reset_key = jax.random.fold_in(jax.random.PRNGKey(cfg['seed'] + 100), episode)
            state = eval_reset(jax.random.split(reset_key, num_eval_envs))
            alive = np.ones(num_eval_envs, dtype=bool)
            returns = np.zeros(num_eval_envs); lengths = np.zeros(num_eval_envs)
            key = jax.random.fold_in(jax.random.PRNGKey(cfg['seed'] + 101), episode)
            for _ in range(cfg['episode_length']):
                key, action_key = jax.random.split(key)
                action, _ = policy(state.obs, action_key)
                state = eval_step(state, action)
                reward, done = jax.device_get((state.reward, state.done))
                returns += np.where(alive, reward, 0.); lengths += alive
                alive &= ~np.asarray(done, dtype=bool)
                if not alive.any(): break
            episode_returns.append(returns); episode_lengths.append(lengths)
        returns, lengths = np.concatenate(episode_returns), np.concatenate(episode_lengths)
        if not np.isfinite(returns).all():
            raise FloatingPointError("Nonfinite SAC evaluation return")
        metrics = {'eval/episode_reward': float(returns.mean()),
                   'eval/episode_reward_std': float(returns.std()),
                   'eval/avg_episode_length': float(lengths.mean()),
                   'eval/num_episodes': int(len(returns))}
        progress(step, metrics)
        save_params(params, ckpt_dir, "latest")
        save_params(params, ckpt_dir, "final")
        (out / "latest.json").write_text(json.dumps({"step": int(step)}, indent=2))
        with (out / "learner_latest.pkl").open("wb") as f:
            pickle.dump(jax.device_get(learner), f)
        with (out / "learner_final.pkl").open("wb") as f: pickle.dump(jax.device_get(learner), f)
        record = dict(
            step=int(step), episode_reward=metrics['eval/episode_reward'],
            episode_reward_std=metrics['eval/episode_reward_std'],
            avg_episode_length=float(lengths.mean()),
            num_episodes=int(len(returns)),
        )
        if metrics['eval/episode_reward'] > best_return:
            best_return = metrics['eval/episode_reward']
            save_params(params, ckpt_dir, "best_return")
            with (out / "learner_best_return.pkl").open("wb") as f:
                pickle.dump(jax.device_get(learner), f)
            (out / "best_return.json").write_text(json.dumps(record, indent=2))
            save_params(params, ckpt_dir, "best")
            with (out / "learner_best.pkl").open("wb") as f:
                pickle.dump(jax.device_get(learner), f)
            (out / "best.json").write_text(json.dumps(
                {"criterion": "episode_return", **record}, indent=2))

    train_options = {k: v for k, v in cfg.items()
                     if k not in ('num_evals', 'deterministic_eval', 'num_eval_envs', 'num_resets_per_eval', 'leg_only_actions', 'network_factory')}
    train_options['critic_burn_in'] = CRITIC_WARMUP
    train_options['zero_action_prior'] = (
        LOSS_NN_ACTION_COST if LOSS_NN_ACTION else 0.0
    )
    if not LOSS_NN_ACTION:
        train_options['zero_action_prior_final'] = None
    if cfg['num_evals'] < 1 or num_eval_envs < 1 or cfg['num_resets_per_eval'] < 1:
        raise ValueError("SAC num_evals, num_eval_envs and num_resets_per_eval must be positive")
    train_options['eval_every'] = max(cfg['update_every_transitions'],
        int(np.ceil(cfg['num_timesteps'] / cfg['num_evals'] / cfg['update_every_transitions'])) * cfg['update_every_transitions'])
    built_networks = _build_fresh_networks(env, zero_init_output_layer=ZERO_INIT_OUTPUT_LAYER)
    selected_network_factory = lambda *args, **kwargs: built_networks
    received_signal, previous_handlers = _install_training_signal_handlers()
    try:
        make_policy, params, learner = sac_training.train(
            env, output=ckpt_dir, callback=checkpoint, restore=restore,
            network_factory=selected_network_factory, **train_options)
    except KeyboardInterrupt:
        if latest is not None:
            save_params(latest[0], ckpt_dir, "interrupted")
            save_params(latest[0], ckpt_dir, "final")
            with (out / "learner_interrupted.pkl").open("wb") as f:
                pickle.dump(jax.device_get(latest[1]), f)
            (out / "interrupted.json").write_text(json.dumps({
                "signal": _signal_name(received_signal),
                "checkpoint": "params_interrupted.pkl",
            }, indent=2))
            print(f"[INTERRUPT] SAC weights saved after {_signal_name(received_signal)}.")
            return None, latest[0]
        print(f"[INTERRUPT] {_signal_name(received_signal)} received before SAC exposed weights.")
        return None, None
    except Exception:
        if latest is not None:
            save_params(latest[0], ckpt_dir, "crash")
            with (out / "learner_crash.pkl").open("wb") as f:
                pickle.dump(jax.device_get(latest[1]), f)
        raise
    finally:
        _restore_training_signal_handlers(previous_handlers)
    return make_policy, params


def run_train(env, eval_env, wrap_env_fn, ckpt_dir: str,
              resume_dir: str | None = None, resume_suffix: str = "best",
              env_name: str = "Go1JoystickFlatTerrain", algo: str = "ppo"):
    global REWARD_LOG_FILE, METRICS_LOG_FILE, CKPT_DIR
    os.makedirs(ckpt_dir, exist_ok=True)
    CKPT_DIR = ckpt_dir
    REWARD_LOG_FILE = os.path.join(ckpt_dir, "reward_log.txt")
    METRICS_LOG_FILE = os.path.join(ckpt_dir, "metrics_log.csv")
    save_tita_env_files(env_name, ckpt_dir)
    if (env._config.sparse_obstacles.enabled
            and env._config.sparse_obstacles.mode == "terrain_overlay"):
        save_source_files(env._config.sparse_obstacles.terrain_xml, ckpt_dir)

    if algo == "ppo":
        policy_obs_key = ALGO_PARAMS["network_factory"]["policy_obs_key"]
        value_obs_key = ALGO_PARAMS["network_factory"]["value_obs_key"]
    else:
        # SAC: single shared obs key, no policy/value split.
        policy_obs_key = value_obs_key = "state"

    header_lines = [
        "=" * 60,
        f"  {algo.upper()} Training  —  {env_name}",
        f"  date         : {SCRIPT_START_TIME}",
        f"  JAX backend  : {jax.default_backend()}",
        f"  devices      : {jax.devices()}",
        f"  GPU name:    : "
        "  \n--- network ---",
        f"  distribution : {DISTRIBUTION_TYPE}",
        f"  policy hidden layers: {ALGO_PARAMS['network_factory']['policy_hidden_layer_sizes']}",
        f"  critic hidden layers: {ALGO_PARAMS['network_factory']['value_hidden_layer_sizes'] if algo == 'ppo' else ALGO_PARAMS['network_factory']['critic_hidden_layer_sizes']}",
        f"  entropy cost : {ALGO_PARAMS.get('entropy_cost', 'N/A')}",
        f"  zero output init   : {ZERO_INIT_OUTPUT_LAYER}",
        f"  initial policy std : {INIT_STD}",
        f"  NN-action prior    : {LOSS_NN_ACTION}",
        f"  NN-action cost     : {LOSS_NN_ACTION_COST}",
        f"  critic warmup      : {CRITIC_WARMUP}",
        f"  PPO warmup steps   : {PPO_CRITIC_WARMUP_TIMESTEPS if algo == 'ppo' and CRITIC_WARMUP else 0}",
        f"  PPO warmup restore : {PPO_CRITIC_WARMUP_ON_RESTORE if algo == 'ppo' else 'n/a'}",
        f"  eval environments  : {ALGO_PARAMS.get('num_eval_envs', DEFAULT_NUM_EVAL_ENVS)}",
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


    if algo == "sac":
        return run_sac_training(env, eval_env, ckpt_dir, resume_dir, resume_suffix)

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
            # Fresh Brax normalizers are initialized as float64 while
            # jax_enable_x64 is on. Older/restored checkpoints may contain
            # float32 statistics; the first update then promotes them to
            # float64, which makes PPO's lax.scan carry types inconsistent.
            # Cast only the running statistics (not policy/value weights or
            # the integer count) so restored outputs remain numerically equal.
            try:
                normalizer, policy, value = restore_params
                cast_stats = lambda tree: jax.tree_util.tree_map(
                    lambda x: jnp.asarray(x, dtype=jnp.float64), tree
                )
                normalizer = normalizer.replace(
                    mean=cast_stats(normalizer.mean),
                    std=cast_stats(normalizer.std),
                    summed_variance=cast_stats(normalizer.summed_variance),
                )
                restore_params = (normalizer, policy, value)
                print("  [INFO] Restored observation-normalizer stats cast to float64.")
            except (AttributeError, TypeError, ValueError):
                # Preserve compatibility with nonstandard/legacy checkpoint
                # layouts; Brax will report any unsupported format itself.
                pass
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
    best_return = -float("inf")
    print("  [checkpoint selection] best=highest deterministic eval return")

    def _progress_and_checkpoint(current_step, metrics):
        """Bind deterministic eval return to the parameters just evaluated."""
        nonlocal best_return
        progress(current_step, metrics)
        reward = float(metrics["eval/episode_reward"])
        step = int(current_step)

        # policy_params_fn runs immediately before this evaluation, so
        # latest_params is exactly the policy that produced `reward`.
        if latest_params is None:
            return

        eval_record = {
            "step": step,
            "episode_reward": reward,
            "avg_episode_length": float(
                metrics.get("eval/avg_episode_length", 0.0)
            ),
        }

        if reward > best_return:
            best_return = reward
            save_params(latest_params, ckpt_dir, suffix="best_return")
            save_params(latest_params, ckpt_dir, suffix="best")
            with open(os.path.join(ckpt_dir, "best_return.json"), "w") as f:
                json.dump(eval_record, f, indent=2)
            with open(os.path.join(ckpt_dir, "best.json"), "w") as f:
                json.dump({"criterion": "episode_return", **eval_record}, f, indent=2)

    def _policy_params_cb(current_step, _make_policy, cb_params):
        nonlocal latest_params, latest_step, best_return
        latest_params = cb_params
        latest_step = int(current_step)
        # This callback precedes evaluation.  Always persist it as latest;
        # `_progress_and_checkpoint` selects best only after seeing its reward.
        save_params(cb_params, ckpt_dir, suffix="latest")
        with open(os.path.join(ckpt_dir, "latest.json"), "w") as f:
            json.dump({"step": latest_step}, f, indent=2)
        # Brax evaluates step 0 immediately before exposing its parameters.
        # Bind that return to the exact initial zero-output policy.
        if latest_step == 0 and y_data:
            initial_reward = float(y_data[-1])
            best_return = initial_reward
            initial_record = {
                "criterion": "episode_return",
                "step": 0,
                "episode_reward": initial_reward,
            }
            save_params(cb_params, ckpt_dir, suffix="best_return")
            save_params(cb_params, ckpt_dir, suffix="best")
            for filename in ("best_return.json", "best.json"):
                with open(os.path.join(ckpt_dir, filename), "w") as f:
                    json.dump(initial_record, f, indent=2)

    base_train_kwargs = dict(
        environment=env,
        eval_env=eval_env,
        wrap_env_fn=wrap_env_fn,
        network_factory=selected_network_factory,
    )

    accepted = inspect.signature(ppo.train).parameters
    dropped = [k for k in base_train_kwargs if k not in accepted]
    if dropped:
        print(f"  [WARN] {algo}.train does not support: {dropped} — ignored.")
    base_train_kwargs = {
        k: v for k, v in base_train_kwargs.items() if k in accepted
    }

    def _run_ppo_phase(
        phase_params, phase_restore, step_offset, critic_warmup, label
    ):
        """Runs one Brax PPO phase with globally monotonic callback steps."""
        def phase_progress(current_step, metrics):
            _progress_and_checkpoint(step_offset + int(current_step), metrics)

        def phase_policy_params(current_step, make_policy, cb_params):
            _policy_params_cb(
                step_offset + int(current_step), make_policy, cb_params
            )

        restore_ppo_loss()
        if LOSS_NN_ACTION or critic_warmup:
            install_configured_ppo_loss(
                LOSS_NN_ACTION, LOSS_NN_ACTION_COST, critic_warmup
            )
        print(
            f"  [PPO phase] {label}: critic_warmup={critic_warmup}, "
            f"action_prior={LOSS_NN_ACTION}, "
            f"requested_steps={phase_params['num_timesteps']:,}"
        )
        train_fn = make_train_fn(
            algo,
            progress_fn=phase_progress,
            params_override=phase_params,
        )
        return train_fn(
            **base_train_kwargs,
            restore_params=phase_restore,
            policy_params_fn=phase_policy_params,
        )

    received_signal, previous_handlers = _install_training_signal_handlers()
    if LOSS_NN_ACTION:
        print(
            "  [PPO loss] zero-action network prior enabled: "
            f"cost={LOSS_NN_ACTION_COST:g}"
        )
    try:
        run_critic_warmup = CRITIC_WARMUP and (
            restore_params is None or PPO_CRITIC_WARMUP_ON_RESTORE
        )
        if CRITIC_WARMUP and restore_params is not None and not run_critic_warmup:
            print(
                "  [PPO warmup] skipped on restore: updating the observation "
                "normalizer would change the loaded policy even with frozen "
                "actor weights"
            )
        if run_critic_warmup:
            total_steps = int(PPO_PARAMS["num_timesteps"])
            requested_warmup = min(
                int(PPO_CRITIC_WARMUP_TIMESTEPS), total_steps
            )
            print(
                "  [PPO warmup] phase 1: actor frozen; value network and "
                "observation normalizer train"
            )
            warmup_params = dict(PPO_PARAMS)
            warmup_params.update(
                num_timesteps=requested_warmup,
                # Brax needs one post-initial-eval epoch; two evals make the
                # warm-up boundary explicit and checkpointable.
                num_evals=2,
            )
            make_inference_fn, params, _ = _run_ppo_phase(
                warmup_params,
                restore_params,
                step_offset=0,
                critic_warmup=True,
                label="critic warm-up",
            )
            warmup_actual_steps = latest_step
            save_params(params, ckpt_dir, suffix="warmup")
            with open(os.path.join(ckpt_dir, "warmup.json"), "w") as f:
                json.dump(
                    {
                        "requested_steps": requested_warmup,
                        "actual_steps": warmup_actual_steps,
                        "actor_frozen": True,
                    },
                    f,
                    indent=2,
                )
            remaining_steps = max(0, total_steps - warmup_actual_steps)
            print(
                "[PPO critic warmup] finished: "
                f"{warmup_actual_steps:,} steps. Actor unlocked; continuing "
                "with the standard Brax PPO gradients plus the configured "
                "action prior."
            )
            if remaining_steps:
                training_params = dict(PPO_PARAMS)
                training_params.update(
                    num_timesteps=remaining_steps,
                    # Avoid replaying the exact warm-up PRNG stream after the
                    # second Brax train call reinitializes its optimizer.
                    seed=int(PPO_PARAMS["seed"]) + 1,
                )
                make_inference_fn, params, _ = _run_ppo_phase(
                    training_params,
                    params,
                    step_offset=warmup_actual_steps,
                    critic_warmup=False,
                    label="actor + value",
                )
        else:
            make_inference_fn, params, _ = _run_ppo_phase(
                dict(PPO_PARAMS),
                restore_params,
                step_offset=0,
                critic_warmup=False,
                label="actor + value",
            )
    except KeyboardInterrupt:
        stop_name = _signal_name(received_signal)
        print(f"\n[INTERRUPT] {stop_name} received: saving available weights...")
        if latest_params is not None:
            save_params(latest_params, ckpt_dir, suffix="interrupted")
            save_params(latest_params, ckpt_dir, suffix="final")
            with open(os.path.join(ckpt_dir, "interrupted.json"), "w") as f:
                json.dump({
                    "signal": stop_name,
                    "step": latest_step,
                    "checkpoint": "params_interrupted.pkl",
                }, f, indent=2)
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
    finally:
        restore_ppo_loss()
        _restore_training_signal_handlers(previous_handlers)

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
                             "run's best checkpoint; suffixes include best, best_return, "
                             "latest, final, interrupted and crash; "
                             "'--load <run_timestamp>' loads that specific "
                             "run (exact or prefix match); an absolute run directory loads "
                             "a checkpoint from another task; '--load crash'/'final' loads that "
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
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Override num_timesteps (e.g. 1_000_000 for a smoke run)")
    parser.add_argument("--num-envs", type=int, default=None,
                        help="Override num_envs (e.g. 256 for a fast smoke run)")
    parser.add_argument("--num-evals", type=int, default=None,
                        help="Override num_evals")
    parser.add_argument("--train-cmd", nargs=2, type=float, default=None, metavar=("VX", "WZ"),
                        help="Fixed Tita target for training/evaluation; keeps the zero-start command LPF")
    parser.add_argument(
        "--obstacle", choices=("single_left", "double", "multi"), default=None,
        help="Enable the selected fixed Tita obstacle layout. Omit for flat terrain.")
    parser.add_argument(
        "--terrain-overlay", choices=("rough",), default=None,
        help="Overlay terrain geoms from the project's rough XML on Tita flat terrain.")
    parser.add_argument(
        "--obstacle-height", type=float, default=None, metavar="METERS",
        help="Height for single_left/double (default: environment config, 0.015 m).")
    parser.add_argument("--seed", type=int, default=None, help="Override RNG seed")
    parser.add_argument(
        "--reward-override", type=str, default=None, metavar="JSON_PATH",
        help="Path to a JSON file of {reward_term: scale} overrides merged into "
             "config.reward_config.scales. Only the listed terms are changed; "
             "everything else in the env config stays at its default. Used by "
             "analysis_training's reward-effects study.")

    args = parser.parse_args()

    global ALGO, ALGO_PARAMS
    ALGO = args.algo
    ALGO_PARAMS = SAC_PARAMS if args.algo == "sac" else PPO_PARAMS

    # Runtime overrides for smoke vs full training (mutate the params dict used
    # by make_train_fn). Keep everything else identical across runs.
    if args.timesteps is not None:
        ALGO_PARAMS["num_timesteps"] = args.timesteps
    if args.num_envs is not None:
        ALGO_PARAMS["num_envs"] = args.num_envs
    if args.num_evals is not None:
        ALGO_PARAMS["num_evals"] = args.num_evals
    if args.seed is not None:
        ALGO_PARAMS["seed"] = args.seed
    print(f"  [config] num_timesteps={ALGO_PARAMS['num_timesteps']} "
          f"num_envs={ALGO_PARAMS['num_envs']} num_evals={ALGO_PARAMS['num_evals']} "
          f"seed={ALGO_PARAMS['seed']}")

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
    }
    env_name = _NAME_SHORTCUTS.get(args.name.lower(), args.name)
    reward_scale_overrides = None
    if args.reward_override:
        with open(args.reward_override) as f:
            reward_scale_overrides = json.load(f)
    env, eval_env, wrap_fn = make_envs(
        env_name=env_name,
        reward_scale_overrides=reward_scale_overrides,
        fixed_target=args.train_cmd,
        obstacle_mode=args.obstacle,
        obstacle_height=args.obstacle_height,
        terrain_overlay=args.terrain_overlay,
    )
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
