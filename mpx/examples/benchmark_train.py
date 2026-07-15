"""
benchmark_envs.py
-----------------
For each environment in ENV_NAMES, measures:

  1. JIT compile time (full training-step rollout + update)
  2. rollout time for ONE FULL PPO TRAINING STEP, replicating brax's
     schedule as a single jitted graph (nested lax.scan):
     (batch_size * num_minibatches // num_envs) unrolls of
     unroll_length steps each -> with default params: 16 unrolls x
     20 steps x 512 envs = 163,840 env-steps
  3. one network update: num_minibatches x num_updates_per_batch
     gradient steps (forward + backward + Adam) over the full batch,
     with the same minibatch layout as brax PPO
  4. physics-model diagnostics (substeps, geoms, hfield, solver, obs)

Prints a final recap table comparing all envs (sim_dt, ctrl_dt, etc.)
and a projection of the full training time for each.

NOTE ON COMPILE TIME: the first rollout call compiles the whole MJX
training-step graph — this can take SEVERAL MINUTES per env on the
very first run. It is cached in ~/.jax_cache, so subsequent runs
start almost instantly. Do NOT Ctrl+C during "compiling..." — wait.

Fidelity vs. real ppo.train:
  - the whole training-step rollout is one jitted graph, like brax
  - update loss is a surrogate (same networks / minibatch schedule,
    no GAE / log-prob / entropy, no normalizer update) -> ~optimistic
  - eval episodes (num_evals) and checkpointing are not included
  The RELATIVE comparison between envs is what this script is for.

Usage:
    python benchmark_envs.py
    python benchmark_envs.py --envs Go1JoystickFlatTerrain AliengoJoystickRoughTerrain
    python benchmark_envs.py --num-envs 512 --unroll 20 --repeats 3
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import jax

CACHE_DIR = os.path.expanduser("~/.jax_cache")
jax.config.update("jax_compilation_cache_dir", CACHE_DIR)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")

import argparse
import functools
import sys
import time

import jax.numpy as jnp
import numpy as np
import optax

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics
from brax.training.acme import specs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────────────────────────────────────────
#  Environments to compare (edit here).
#  Key   = env name
#  Value = dict of config overrides applied on top of the env's default
#          config (may be empty). Supports nested keys with dot notation.
#  Example:
#      "Go1JoystickFlatTerrain": {"sim_dt": 0.002, "ctrl_dt": 0.02},
# ─────────────────────────────────────────────────────────────────────────────
ENV_CONFIGS = {
    "Go1JoystickFlatTerrain": {},
    "AliengoJoystickFlatTerrain": {},
    "TitaJoystickFlatTerrain": {},
    "AliengoJoystickE2EFlatTerrain": {},
    "TitaJoystickE2EFlatTerrain": {},
}

# Same parameters as train_srbd.py (PPO_PARAMS)
DEFAULTS = dict(
    num_envs=512,
    unroll_length=20,
    num_minibatches=32,
    num_updates_per_batch=4,
    batch_size=256,
    episode_length=500,
    action_repeat=1,
    num_timesteps=100_000_000,
    learning_rate=3e-4,
    seed=0,
)

POLICY_HIDDEN_LAYER_SIZES = (512, 256, 128)
DISTRIBUTION_TYPE = "tanh_normal"


# ─────────────────────────────────────────────────────────────────────────────
#  Env loading (same logic as train_srbd.py)
# ─────────────────────────────────────────────────────────────────────────────
def _set_nested(config, key, value):
    """Set config['a.b.c'] = value on an ml_collections ConfigDict."""
    parts = key.split(".")
    node = config
    for p in parts[:-1]:
        node = node[p]
    last = parts[-1]
    try:
        node[last] = value
    except (KeyError, AttributeError):
        # key does not exist in the default config -> add it explicitly
        with node.unlocked():
            node[last] = value


def load_env(env_name: str, overrides: dict | None = None):
    from mujoco_playground._src import locomotion
    from mujoco_playground import registry
    from mujoco_playground._src.wrapper import wrap_for_brax_training as pg_wrap

    if env_name == "QuadrupedMPCEnv" and env_name not in locomotion._envs:
        try:
            from go1_srbd import QuadrupedMPCEnv, default_config
            locomotion._envs[env_name] = functools.partial(QuadrupedMPCEnv, task="flat_terrain")
            locomotion._cfgs[env_name] = default_config
            locomotion._randomizer[env_name] = locomotion._randomizer["Go1JoystickFlatTerrain"]
            locomotion.ALL_ENVS = locomotion.ALL_ENVS + (env_name,)
            registry.ALL_ENVS = registry.ALL_ENVS + (env_name,)
        except ImportError as e:
            print(f"  [SKIP] Could not register {env_name}: {e}")
            return None, None

    try:
        config = registry.get_default_config(env_name)
    except Exception as e:
        print(f"  [SKIP] Could not load config for {env_name}: {e}")
        return None, None

    if overrides:
        print("  --- config overrides ---")
        for k, v in overrides.items():
            _set_nested(config, k, v)
            print(f"    {k} = {v}")

    env = registry.load(env_name, config=config)
    return env, pg_wrap


# ─────────────────────────────────────────────────────────────────────────────
#  Physics-model diagnostics — explains the WHY
# ─────────────────────────────────────────────────────────────────────────────
def env_diagnostics(env):
    d = {}
    try:
        m = env.mj_model
        sim_dt = float(m.opt.timestep)
        ctrl_dt = float(env.dt)
        d["sim_dt"] = sim_dt
        d["ctrl_dt"] = ctrl_dt
        d["substeps"] = int(round(ctrl_dt / sim_dt))
        d["nq/nv"] = f"{m.nq}/{m.nv}"
        d["nbody"] = int(m.nbody)
        d["ngeom"] = int(m.ngeom)
        d["nhfield"] = int(m.nhfield)
        d["npair"] = int(m.npair)
        d["solver_iter"] = int(m.opt.iterations)
        d["ls_iter"] = int(m.opt.ls_iterations)
    except Exception as e:
        d["mj_model"] = f"unavailable ({e})"

    d["obs_size"] = env.observation_size
    d["action_size"] = env.action_size
    return d


def print_diagnostics(d):
    labels = {
        "sim_dt": "sim_dt (physics)",
        "ctrl_dt": "ctrl_dt (policy)",
        "substeps": "physics substeps per env.step",
        "nq/nv": "nq / nv (DoF)",
        "nhfield": "nhfield (heightfield/terrain)",
        "npair": "npair (explicit contact pairs)",
        "solver_iter": "solver iterations",
        "ls_iter": "ls iterations",
    }
    for k, v in d.items():
        print(f"    {labels.get(k, k):35s}: {v}")


# ─────────────────────────────────────────────────────────────────────────────
#  PPO networks (same sizes as train_srbd.py)
# ─────────────────────────────────────────────────────────────────────────────
def obs_spec_from_env(env):
    obs_size = env.observation_size
    if isinstance(obs_size, dict):
        return {
            k: specs.Array(v if isinstance(v, tuple) else (int(v),), jnp.float32)
            for k, v in obs_size.items()
        }
    return specs.Array((int(obs_size),), jnp.float32)


def build_networks_and_policy(env, seed):
    networks = ppo_networks.make_ppo_networks(
        observation_size=env.observation_size,
        action_size=env.action_size,
        policy_hidden_layer_sizes=POLICY_HIDDEN_LAYER_SIZES,
        preprocess_observations_fn=running_statistics.normalize,
        distribution_type=DISTRIBUTION_TYPE,
    )
    key = jax.random.PRNGKey(seed)
    k_pol, k_val = jax.random.split(key)
    policy_params = networks.policy_network.init(k_pol)
    value_params = networks.value_network.init(k_val)
    normalizer = running_statistics.init_state(obs_spec_from_env(env))

    make_policy = ppo_networks.make_inference_fn(networks)
    policy_fn = make_policy((normalizer, policy_params), deterministic=False)
    return networks, policy_fn, (policy_params, value_params), normalizer


# ─────────────────────────────────────────────────────────────────────────────
#  Rollout for ONE FULL TRAINING STEP as a single jitted graph,
#  replicating brax's PPO schedule with a nested lax.scan:
#  num_unrolls = batch_size * num_minibatches // num_envs consecutive
#  unrolls of unroll_length steps each.
# ─────────────────────────────────────────────────────────────────────────────
def make_training_step_rollout_fn(wrapped_env, policy_fn, unroll_length, num_unrolls):
    def step_fn(carry, _):
        state, rng = carry
        rng, act_key = jax.random.split(rng)
        action, _ = policy_fn(state.obs, act_key)
        next_state = wrapped_env.step(state, action)
        return (next_state, rng), state.obs

    def one_unroll(carry, _):
        carry, obs_seq = jax.lax.scan(step_fn, carry, None, length=unroll_length)
        return carry, obs_seq  # obs_seq: (unroll_length, num_envs, ...)

    def rollout(state, rng):
        (state, rng), obs_all = jax.lax.scan(
            one_unroll, (state, rng), None, length=num_unrolls
        )
        # obs_all: (num_unrolls, unroll_length, num_envs, ...)
        return state, obs_all

    return jax.jit(rollout)


# ─────────────────────────────────────────────────────────────────────────────
#  Surrogate update: same compute profile as 1 PPO update
#  (forward policy+value, backward, Adam) with brax's minibatch layout.
# ─────────────────────────────────────────────────────────────────────────────
def make_update_fn(networks, normalizer, num_minibatches, num_updates_per_batch, lr):
    optimizer = optax.adam(lr)

    def loss_fn(params, obs):
        policy_params, value_params = params
        logits = networks.policy_network.apply(normalizer, policy_params, obs)
        values = networks.value_network.apply(normalizer, value_params, obs)
        # dummy loss with the same forward/backward cost profile as PPO
        return jnp.mean(jnp.square(logits)) + jnp.mean(jnp.square(values))

    grad_fn = jax.value_and_grad(loss_fn)

    def minibatch_step(carry, obs_mb):
        params, opt_state = carry
        loss, grads = grad_fn(params, obs_mb)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), loss

    def update(params, opt_state, obs_flat):
        # obs_flat: pytree with leading dim = total transitions of the batch
        def split_mb(x):
            n = x.shape[0] - (x.shape[0] % num_minibatches)
            return x[:n].reshape(num_minibatches, n // num_minibatches, *x.shape[1:])

        obs_mb = jax.tree_util.tree_map(split_mb, obs_flat)

        def epoch(carry, _):
            carry, losses = jax.lax.scan(minibatch_step, carry, obs_mb)
            return carry, jnp.mean(losses)

        (params, opt_state), losses = jax.lax.scan(
            epoch, (params, opt_state), None, length=num_updates_per_batch
        )
        return params, opt_state, jnp.mean(losses)

    return jax.jit(update), optimizer


# ─────────────────────────────────────────────────────────────────────────────
#  Timing helpers
# ─────────────────────────────────────────────────────────────────────────────
def timed(fn, *args):
    t0 = time.perf_counter()
    out = fn(*args)
    jax.block_until_ready(out)
    return time.perf_counter() - t0, out


def benchmark_env(env_name, cfg, overrides=None):
    print("\n" + "=" * 70)
    print(f"  BENCHMARK — {env_name}")
    print("=" * 70)

    env, pg_wrap = load_env(env_name, overrides)
    if env is None:
        return None

    diag = env_diagnostics(env)
    print("  --- model diagnostics ---")
    print_diagnostics(diag)

    num_unrolls = max(1, (cfg["batch_size"] * cfg["num_minibatches"]) // cfg["num_envs"])
    env_steps_per_training_step = (
        num_unrolls * cfg["unroll_length"] * cfg["num_envs"] * cfg["action_repeat"]
    )
    print(f"\n  brax schedule: {num_unrolls} unrolls x {cfg['unroll_length']} steps "
          f"x {cfg['num_envs']} envs = {env_steps_per_training_step:,} env-steps / training step")

    wrapped = pg_wrap(
        env,
        episode_length=cfg["episode_length"],
        action_repeat=cfg["action_repeat"],
    )

    networks, policy_fn, params, normalizer = build_networks_and_policy(env, cfg["seed"])

    rng = jax.random.PRNGKey(cfg["seed"])
    rng, reset_rng = jax.random.split(rng)
    reset_keys = jax.random.split(reset_rng, cfg["num_envs"])

    jit_reset = jax.jit(wrapped.reset)
    rollout_fn = make_training_step_rollout_fn(
        wrapped, policy_fn, cfg["unroll_length"], num_unrolls
    )
    update_fn, optimizer = make_update_fn(
        networks, normalizer,
        cfg["num_minibatches"], cfg["num_updates_per_batch"], cfg["learning_rate"],
    )
    opt_state = optimizer.init(params)

    # ── reset + compile ──────────────────────────────────────────
    print("  compiling reset... (do NOT Ctrl+C)", flush=True)
    t_reset, state = timed(jit_reset, reset_keys)
    print(f"  reset ({cfg['num_envs']} envs) + compile   : {t_reset:8.2f} s")

    print("  compiling full training-step rollout... this can take SEVERAL", flush=True)
    print("  MINUTES on the first run (cached in ~/.jax_cache afterwards).", flush=True)
    rng, roll_rng = jax.random.split(rng)
    t_compile_roll, (state_after, obs_all) = timed(rollout_fn, state, roll_rng)
    print(f"  compile rollout (1st call)          : {t_compile_roll:8.2f} s")

    # flatten obs: (num_unrolls, unroll, num_envs, ...) -> (N, ...)
    def flatten(x):
        return x.reshape(-1, *x.shape[3:])
    obs_flat = jax.tree_util.tree_map(flatten, obs_all)

    print("  compiling update...", flush=True)
    t_compile_upd, (params2, opt_state2, _) = timed(update_fn, params, opt_state, obs_flat)
    print(f"  compile update (1st call)           : {t_compile_upd:8.2f} s")

    # ── steady-state timing ──────────────────────────────────────
    roll_times, upd_times = [], []
    cur_state = state_after
    for rep in range(cfg["repeats"]):
        rng, roll_rng = jax.random.split(rng)
        t_r, (cur_state, obs_all) = timed(rollout_fn, cur_state, roll_rng)
        roll_times.append(t_r)

        obs_flat = jax.tree_util.tree_map(flatten, obs_all)
        t_u, (params2, opt_state2, _) = timed(update_fn, params2, opt_state2, obs_flat)
        upd_times.append(t_u)
        print(f"  repeat {rep+1}/{cfg['repeats']}: rollout {t_r*1000:.0f} ms | "
              f"update {t_u*1000:.0f} ms", flush=True)

    roll_med = float(np.median(roll_times))
    upd_med = float(np.median(upd_times))
    iter_time = roll_med + upd_med
    sps = env_steps_per_training_step / iter_time
    n_iters = cfg["num_timesteps"] / env_steps_per_training_step
    eta_h = n_iters * iter_time / 3600.0

    total_steps = num_unrolls * cfg["unroll_length"]
    print(f"\n  --- steady-state timing (median over {cfg['repeats']} repeats) ---")
    print(f"  rollout (1 training step, {env_steps_per_training_step:,} env-steps): "
          f"{roll_med*1000:10.1f} ms   ({roll_med*1000/total_steps:.2f} ms/step batched)")
    print(f"  1 network update ({cfg['num_minibatches']}x{cfg['num_updates_per_batch']} minibatches): "
          f"{upd_med*1000:10.1f} ms")
    print(f"  env-steps/second                    : {sps:12,.0f}")
    print(f"  full-training estimate ({cfg['num_timesteps']:,} steps): {eta_h:.2f} h")

    return dict(
        name=env_name, diag=diag, overrides=overrides or {},
        compile_rollout=t_compile_roll, compile_update=t_compile_upd,
        rollout_ms=roll_med * 1000, update_ms=upd_med * 1000,
        sps=sps, eta_h=eta_h,
    )


def print_summary(results, cfg):
    results = [r for r in results if r is not None]
    if not results:
        return

    print("\n\n" + "=" * 112)
    print("  FINAL RECAP — all environments")
    print("=" * 112)
    header = (f"  {'env':30s} {'sim_dt':>8s} {'ctrl_dt':>8s} {'sub':>4s} "
              f"{'rollout(ms)':>12s} {'update(ms)':>11s} {'steps/s':>11s} {'ETA(h)':>7s} {'ratio':>7s}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    baseline = min(r["rollout_ms"] for r in results)
    for r in sorted(results, key=lambda r: r["rollout_ms"]):
        d = r["diag"]
        sim_dt = d.get("sim_dt", float("nan"))
        ctrl_dt = d.get("ctrl_dt", float("nan"))
        sub = d.get("substeps", "-")
        ratio = r["rollout_ms"] / baseline
        print(f"  {r['name']:30s} {sim_dt:8.4f} {ctrl_dt:8.4f} {str(sub):>4s} "
              f"{r['rollout_ms']:12.1f} {r['update_ms']:11.1f} "
              f"{r['sps']:11,.0f} {r['eta_h']:7.2f} {'x'+format(ratio, '.1f'):>7s}")

    print("\n  Compile times (1st call, empty cache):")
    for r in results:
        print(f"  {r['name']:30s} rollout: {r['compile_rollout']:7.1f} s | "
              f"update: {r['compile_update']:6.1f} s")

    if any(r["overrides"] for r in results):
        print("\n  Config overrides used:")
        for r in results:
            if r["overrides"]:
                print(f"  {r['name']:30s} {r['overrides']}")

    print("\n  Extended diagnostics:")
    print(f"  {'env':30s} {'nq/nv':>8s} {'ngeom':>6s} {'nhfield':>8s} {'npair':>6s} "
          f"{'solver':>7s} {'obs_size':>24s}")
    for r in results:
        d = r["diag"]
        obs = str(d.get("obs_size", "?"))
        if len(obs) > 24:
            obs = obs[:21] + "..."
        print(f"  {r['name']:30s} {str(d.get('nq/nv','?')):>8s} {str(d.get('ngeom','?')):>6s} "
              f"{str(d.get('nhfield','?')):>8s} {str(d.get('npair','?')):>6s} "
              f"{str(d.get('solver_iter','?')):>7s} {obs:>24s}")

    print("\n  How to read this:")
    print("  - If update(ms) is similar across envs but rollout(ms) diverges, the")
    print("    bottleneck is PHYSICS (substeps, contacts, heightfield), not the network.")
    print("  - More substeps = proportionally more physics work per env.step.")
    print("  - nhfield > 0 (rough terrain) makes MJX collision much more expensive")
    print("    than an infinite flat plane.")
    print("  - Large obs_size mainly inflates update(ms) (bigger first layer).")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--envs", nargs="+", default=None,
                   help="Env names (empty overrides). If omitted, uses ENV_CONFIGS "
                        "defined at the top of the script, including its overrides.")
    p.add_argument("--num-envs", type=int, default=DEFAULTS["num_envs"])
    p.add_argument("--unroll", type=int, default=DEFAULTS["unroll_length"])
    p.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()

    cfg = dict(DEFAULTS)
    cfg["num_envs"] = args.num_envs
    cfg["unroll_length"] = args.unroll
    cfg["batch_size"] = args.batch_size
    cfg["repeats"] = args.repeats

    backend = jax.default_backend()
    print(f"JAX backend: {backend} | devices: {jax.devices()}")
    if backend == "cpu":
        print("[WARN] Running on CPU! Everything (compile AND training) will be")
        print("       extremely slow. Check your CUDA-enabled jax installation.")
    print(f"Config: num_envs={cfg['num_envs']}, unroll={cfg['unroll_length']}, "
          f"batch_size={cfg['batch_size']}, num_minibatches={cfg['num_minibatches']}, "
          f"repeats={cfg['repeats']}")

    if args.envs is not None:
        env_configs = {name: {} for name in args.envs}
    else:
        env_configs = ENV_CONFIGS

    results = [benchmark_env(name, cfg, overrides)
               for name, overrides in env_configs.items()]
    print_summary(results, cfg)


if __name__ == "__main__":
    main()