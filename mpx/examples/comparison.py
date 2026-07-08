"""
compare_rollouts.py
-------------------
Runs TWO INDEPENDENT simulations in parallel (separate processes):

  1. "nn"   : eval with the trained policy network (loaded from checkpoint)
  2. "zero" : eval with zero-output network (same as `--zero` in train_srbd.py)

Each process is fully independent: if one finishes (or crashes) first, the
other keeps running. Each one writes its own CSV (flushed incrementally, so
data survives crashes/Ctrl+C) with everything useful for metrics:

  - step / sim time
  - base position + quaternion, CoM position/velocity (from subtree sensors
    if available, otherwise from qpos/subtree_com)
  - base linear/angular velocity in WORLD and BODY frame
  - joint positions / velocities
  - commanded velocity (command + target_command)
  - policy action
  - applied motor torques (data.actuator_force), ctrl, mpc_tau
  - the whole flattened `state.info` (including low_level_controller/tau_p,
    tau_d, q_des, dq_des, kp, kd, ...)
  - reward, done

At the end (or with --plot-only) it produces the plots:
  * commanded vs actual velocity (vx, vy, wz)  — per run + comparison NN vs ZERO
  * applied motor torques vs network torque (tau_net = tau_p + tau_d) per motor

Usage:
    python compare_rollouts.py                                 # default env
    python compare_rollouts.py --name TitaJoystickFlatTerrain
    python compare_rollouts.py --cmd 0.5 0.0 0.0               # fixed command
    python compare_rollouts.py --steps 2000 --outdir results/run1
    python compare_rollouts.py --plot-only --outdir results/run1
    python compare_rollouts.py --only nn                       # single run
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import multiprocessing as mp

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
#  Small utils (no jax here: main process stays GPU-free)
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# info keys that are useless/huge in the CSV
INFO_SKIP_KEYS = {"rng", "mpc_state"}
# max number of elements per flattened info entry
INFO_MAX_ELEMS = 64


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """MuJoCo quaternion (w, x, y, z) -> rotation matrix (world_R_body)."""
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array([
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)],
    ])


def flatten_info(info_dict, batch_size: int, prefix: str = "") -> dict:
    """Recursively flatten state.info taking env 0 of the batch (same spirit
    as _flatten_info in train_srbd.py, but with skip-list and size cap)."""
    out = {}
    for k, v in info_dict.items():
        if k in INFO_SKIP_KEYS:
            continue
        full_key = f"{prefix}/{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_info(v, batch_size, prefix=full_key))
            continue
        try:
            arr = np.asarray(v)
            if arr.dtype == object:
                continue
            if arr.ndim >= 1 and arr.shape[0] == batch_size:
                arr = arr[0]  # take env 0
            arr = np.atleast_1d(np.asarray(arr, dtype=np.float64)).flatten()
            if arr.size > INFO_MAX_ELEMS:
                continue
            if arr.size == 1:
                out[full_key] = float(arr[0])
            else:
                for i, val in enumerate(arr):
                    out[f"{full_key}_{i}"] = float(val)
        except Exception:
            pass
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Worker: one independent simulation (runs in its own process)
# ─────────────────────────────────────────────────────────────────────────────

def rollout_worker(*args, **kwargs):
    """Thin wrapper: run the real worker and, on any crash, dump the FULL
    traceback both to stdout and to <outdir>/error_<mode>.log so it never
    gets lost/interleaved between the two processes."""
    import traceback
    mode, outdir = args[0], args[4]
    try:
        _rollout_worker(*args, **kwargs)
    except KeyboardInterrupt:
        raise
    except BaseException:
        tb = traceback.format_exc()
        print(f"[{mode.upper():>4s}] CRASHED:\n{tb}", flush=True)
        try:
            os.makedirs(outdir, exist_ok=True)
            with open(os.path.join(outdir, f"error_{mode}.log"), "w") as f:
                f.write(tb)
            print(f"[{mode.upper():>4s}] Traceback saved to "
                  f"{os.path.join(outdir, f'error_{mode}.log')}", flush=True)
        except Exception:
            pass
        raise


def _rollout_worker(
    mode: str,               # "nn" | "zero"
    env_name: str,
    ckpt_dir: str,
    load_suffix: str,
    outdir: str,
    steps: int | None,
    fixed_cmd: list | None,
    seed: int,
    mem_fraction: float,
):
    tag = f"[{mode.upper():>4s}]"

    def log(msg):
        print(f"{tag} {msg}", flush=True)

    # ── GPU memory: set BEFORE any jax initialization ────────────────────
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(mem_fraction)
    os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
    # Two JAX processes on one GPU: CUDA graph instantiation happens OUTSIDE
    # the XLA memory pool and easily OOMs (and makes cuSolver fail with
    # "internal error"). Disable command buffers as suggested by XLA itself.
    xla_flags = os.environ.get("XLA_FLAGS", "")
    if "--xla_gpu_enable_command_buffer" not in xla_flags:
        os.environ["XLA_FLAGS"] = (xla_flags + " --xla_gpu_enable_command_buffer=").strip()

    # Disable CUDA-graph command buffers: with two JAX processes on the same
    # GPU their instantiation exhausts driver memory (RESOURCE_EXHAUSTED:
    # "Failed to instantiate CUDA graph ... CUDA_ERROR_OUT_OF_MEMORY").
    xla_flags = os.environ.get("XLA_FLAGS", "")
    if "--xla_gpu_enable_command_buffer" not in xla_flags:
        os.environ["XLA_FLAGS"] = (
            xla_flags + " --xla_gpu_enable_command_buffer="
        ).strip()

    sys.path.insert(0, SCRIPT_DIR)

    # train_srbd sets XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 at import time:
    # re-override AFTER the import, before the JAX backend initializes.
    import train_srbd as ts  # noqa: E402
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(mem_fraction)

    import jax                     # noqa: E402
    import jax.numpy as jnp        # noqa: E402
    import mujoco                  # noqa: E402
    from brax.training.agents.ppo import networks as ppo_networks  # noqa: E402
    from brax.training.acme import running_statistics, specs       # noqa: E402

    log(f"env={env_name}  mode={mode}  mem_fraction={mem_fraction}")

    # ── Environment ──────────────────────────────────────────────────────
    _env, eval_env, _wrap = ts.make_envs(env_name=env_name)
    del _env

    episode_length = int(steps) if steps else int(ts.PPO_PARAMS["episode_length"])
    dt = float(eval_env.dt)
    nu = int(eval_env.mj_model.nu)
    nq = int(eval_env.mj_model.nq)
    nv = int(eval_env.mj_model.nv)

    # ── Policy ───────────────────────────────────────────────────────────
    if mode == "zero":
        log("Using fresh network with zero output layer (like --zero).")
        networks = ts._build_fresh_networks(eval_env)
        policy_params = networks.policy_network.init(jax.random.PRNGKey(0))
        obs_size = eval_env.observation_size
        if isinstance(obs_size, dict):
            obs_proto = {
                k: specs.Array(
                    (int(np.prod(v)),) if not isinstance(v, int) else (v,),
                    jnp.float32,
                )
                for k, v in obs_size.items()
            }
        else:
            obs_proto = specs.Array((obs_size,), jnp.float32)
        normalizer_params = running_statistics.init_state(obs_proto)
        params = (normalizer_params, policy_params)
        inference_fn = ppo_networks.make_inference_fn(networks)
        policy_fn = inference_fn(params, deterministic=True)
    else:
        params = ts.load_params(ckpt_dir, suffix=load_suffix)
        if params is None:
            log(f"[ERROR] No checkpoint found in '{ckpt_dir}' — aborting NN run.")
            return
        networks = ppo_networks.make_ppo_networks(
            observation_size=eval_env.observation_size,
            action_size=eval_env.action_size,
            policy_hidden_layer_sizes=ts.POLICY_HIDDEN_LAYER_SIZES,
            preprocess_observations_fn=running_statistics.normalize,
            distribution_type=ts.DISTRIBUTION_TYPE,
        )
        inference_fn = ppo_networks.make_inference_fn(networks)
        policy_fn = inference_fn(params, deterministic=True)
        log(f"Checkpoint '{load_suffix}' loaded from '{ckpt_dir}'.")

    jit_infer = jax.jit(policy_fn)

    # ── Batched env (cuSolver workaround, same as train_srbd) ────────────
    EVAL_BATCH = 2
    batched_reset = jax.jit(jax.vmap(eval_env.reset))
    batched_step = jax.jit(jax.vmap(eval_env.step))

    rng = jax.random.PRNGKey(seed)
    rng, *reset_rngs = jax.random.split(rng, EVAL_BATCH + 1)
    state = batched_reset(jnp.stack(reset_rngs))

    cmd_b = None
    if fixed_cmd is not None:
        cmd_b = jnp.broadcast_to(
            jnp.array(fixed_cmd, dtype=state.info["command"].dtype),
            (EVAL_BATCH, 3),
        )

    def inject_cmd(st):
        """Force both command and target_command so the command is exactly
        fixed (the env smooths command toward target inside step())."""
        if cmd_b is None:
            return st
        return st.replace(info={
            **st.info,
            "command": cmd_b,
            "target_command": cmd_b,
        })

    state = inject_cmd(state)

    # ── CSV setup (incremental, flushed) ─────────────────────────────────
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, f"rollout_{mode}.csv")

    # optional CoM sensors (present in the Tita env)
    def _sensor_adr(name):
        try:
            sid = eval_env.mj_model.sensor(name).id
            adr = eval_env.mj_model.sensor_adr[sid]
            dim = eval_env.mj_model.sensor_dim[sid]
            return list(range(adr, adr + dim))
        except Exception:
            return None

    com_adr = _sensor_adr("base_subtree_com")
    com_vel_adr = _sensor_adr("base_subtree_linvel")

    def build_row(step_idx, st, action_np):
        d = st.data
        qpos = np.asarray(d.qpos[0])
        qvel = np.asarray(d.qvel[0])
        row = {
            "step": step_idx,
            "time_s": step_idx * dt,
            "reward": float(st.reward[0]),
            "done": float(st.done[0]),
        }
        # base pose
        row["base_pos_x"], row["base_pos_y"], row["base_pos_z"] = qpos[0:3]
        row["base_quat_w"], row["base_quat_x"], row["base_quat_y"], row["base_quat_z"] = qpos[3:7]
        # CoM (sensors if available, fallback to subtree_com/base pos)
        sdata = np.asarray(d.sensordata[0]) if hasattr(d, "sensordata") else None
        if com_adr is not None and sdata is not None:
            com = sdata[com_adr]
        else:
            try:
                com = np.asarray(d.subtree_com[0][1])  # body 1 = base subtree
            except Exception:
                com = qpos[0:3]
        row["com_x"], row["com_y"], row["com_z"] = com[:3]
        if com_vel_adr is not None and sdata is not None:
            vcom = sdata[com_vel_adr]
            row["com_vx"], row["com_vy"], row["com_vz"] = vcom[:3]
        # velocities: world + body frame
        R = quat_to_rotmat(qpos[3:7])
        v_world = qvel[0:3]
        w_body = qvel[3:6]              # MuJoCo free joint: angvel is body-frame
        v_body = R.T @ v_world
        w_world = R @ w_body
        row["linvel_world_x"], row["linvel_world_y"], row["linvel_world_z"] = v_world
        row["linvel_body_x"], row["linvel_body_y"], row["linvel_body_z"] = v_body
        row["angvel_body_x"], row["angvel_body_y"], row["angvel_body_z"] = w_body
        row["angvel_world_x"], row["angvel_world_y"], row["angvel_world_z"] = w_world
        # joints
        for i, v in enumerate(qpos[7:]):
            row[f"joint_pos_{i}"] = float(v)
        for i, v in enumerate(qvel[6:]):
            row[f"joint_vel_{i}"] = float(v)
        # commanded velocity (explicit columns, also in flattened info)
        cmd = np.asarray(st.info["command"][0])
        tcmd = np.asarray(st.info["target_command"][0])
        row["cmd_vx"], row["cmd_vy"], row["cmd_wz"] = cmd[:3]
        row["target_cmd_vx"], row["target_cmd_vy"], row["target_cmd_wz"] = tcmd[:3]
        # action
        for i, v in enumerate(action_np):
            row[f"action_{i}"] = float(v)
        # applied torques
        try:
            tau_applied = np.asarray(d.actuator_force[0])
            for i, v in enumerate(tau_applied):
                row[f"tau_applied_{i}"] = float(v)
        except Exception:
            pass
        try:
            ctrl = np.asarray(d.ctrl[0])
            for i, v in enumerate(ctrl):
                row[f"ctrl_{i}"] = float(v)
        except Exception:
            pass
        # everything else in info (mpc_tau, low_level_controller/tau_p, tau_d, ...)
        row.update(flatten_info(state.info, EVAL_BATCH))
        return row

    log(f"Rollout: {episode_length} steps  (nq={nq}, nv={nv}, nu={nu}, dt={dt})")
    log(f"CSV → {csv_path}")

    zero_action = jnp.zeros((EVAL_BATCH, eval_env.action_size), dtype=jnp.float32)

    csv_file = None
    writer = None
    steps_done = 0
    rewards = []
    t0 = time.time()

    try:
        csv_file = open(csv_path, "w", newline="")
        for i in range(episode_length):
            # policy
            if mode == "nn":
                rng, act_rng = jax.random.split(rng)
                obs0 = ({k: v[0] for k, v in state.obs.items()}
                        if isinstance(state.obs, dict) else state.obs[0])
                action0, _ = jit_infer(obs0, act_rng)
                action = jnp.broadcast_to(action0, (EVAL_BATCH, eval_env.action_size))
            else:
                # zero-output network == zero action deterministically; skip
                # the forward pass to save time (identical result)
                action = zero_action

            state = inject_cmd(state)
            state = batched_step(state, action)
            state = inject_cmd(state)   # re-inject after step (env resamples/smooths)

            steps_done = i + 1
            rewards.append(float(state.reward[0]))

            row = build_row(steps_done, state, np.asarray(action[0]))
            if writer is None:
                writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()),
                                        extrasaction="ignore", restval="")
                writer.writeheader()
            writer.writerow(row)

            if steps_done % 50 == 0:
                csv_file.flush()
                el = time.time() - t0
                log(f"step {steps_done}/{episode_length}  "
                    f"({steps_done / max(el, 1e-9):.1f} it/s, elapsed {el:.0f}s)")

            if bool(state.done[0]):
                log(f"Episode ended (done=True) at step {steps_done}")
                break

    except KeyboardInterrupt:
        log(f"Interrupted at step {steps_done} — CSV kept.")
    finally:
        if csv_file is not None:
            csv_file.flush()
            csv_file.close()

    log(f"DONE — steps: {steps_done}  mean reward: {np.mean(rewards) if rewards else float('nan'):.3f}  "
        f"total: {np.sum(rewards) if rewards else 0.0:.3f}")
    log(f"CSV saved: {csv_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Plotting (main process, no jax needed)
# ─────────────────────────────────────────────────────────────────────────────

def _load_csv(path):
    import pandas as pd
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        return df if len(df) > 0 else None
    except Exception as e:
        print(f"[PLOT] Could not read {path}: {e}")
        return None


def _tau_columns(df, prefix):
    cols = []
    i = 0
    while f"{prefix}_{i}" in df.columns:
        cols.append(f"{prefix}_{i}")
        i += 1
    return cols


def plot_velocity_tracking(df, label, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = df["time_s"].to_numpy()
    pairs = [
        ("cmd_vx", "linvel_body_x", "vx  [m/s]"),
        ("cmd_vy", "linvel_body_y", "vy  [m/s]"),
        ("cmd_wz", "angvel_body_z", "wz  [rad/s]"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for ax, (c_cmd, c_act, ylab) in zip(axes, pairs):
        ax.plot(t, df[c_cmd], "k--", lw=1.6, label="commanded")
        ax.plot(t, df[c_act], lw=1.2, label="actual")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")
    axes[0].set_title(f"Commanded vs actual velocity — {label}")
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    path = os.path.join(outdir, f"velocity_tracking_{label}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[PLOT] {path}")


def plot_velocity_comparison(df_nn, df_zero, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairs = [
        ("cmd_vx", "linvel_body_x", "vx  [m/s]"),
        ("cmd_vy", "linvel_body_y", "vy  [m/s]"),
        ("cmd_wz", "angvel_body_z", "wz  [rad/s]"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for ax, (c_cmd, c_act, ylab) in zip(axes, pairs):
        # command from the NN run (identical if --cmd is fixed)
        ax.plot(df_nn["time_s"], df_nn[c_cmd], "k--", lw=1.6, label="commanded")
        ax.plot(df_nn["time_s"], df_nn[c_act], lw=1.2, label="actual (NN)",
                color="tab:blue")
        ax.plot(df_zero["time_s"], df_zero[c_act], lw=1.2, label="actual (zero)",
                color="tab:red", alpha=0.85)
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")
    axes[0].set_title("Commanded vs actual velocity — NN vs ZERO")
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    path = os.path.join(outdir, "velocity_tracking_comparison.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[PLOT] {path}")


def plot_torques(df, label, outdir):
    """Applied motor torque vs network torque (tau_net = tau_p + tau_d)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = df["time_s"].to_numpy()
    tau_app_cols = _tau_columns(df, "tau_applied")
    tau_p_cols = _tau_columns(df, "low_level_controller/tau_p")
    tau_d_cols = _tau_columns(df, "low_level_controller/tau_d")
    if not tau_app_cols:
        tau_app_cols = _tau_columns(df, "mpc_tau")  # fallback

    nu = max(len(tau_app_cols), len(tau_p_cols))
    if nu == 0:
        print(f"[PLOT] No torque columns found for '{label}', skipping torque plot.")
        return

    ncols = 2
    nrows = int(np.ceil(nu / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 2.6 * nrows),
                             sharex=True, squeeze=False)
    for m in range(nu):
        ax = axes[m // ncols][m % ncols]
        if m < len(tau_app_cols):
            ax.plot(t, df[tau_app_cols[m]], lw=1.1, color="tab:blue",
                    label="tau applied")
        if m < len(tau_p_cols) and m < len(tau_d_cols):
            tau_net = df[tau_p_cols[m]].to_numpy() + df[tau_d_cols[m]].to_numpy()
            ax.plot(t, tau_net, lw=1.1, color="tab:orange", alpha=0.9,
                    label="tau net (tau_p + tau_d)")
        ax.set_title(f"motor {m}", fontsize=9)
        ax.grid(alpha=0.3)
        if m == 0:
            ax.legend(loc="upper right", fontsize=8)
    for m in range(nu, nrows * ncols):
        axes[m // ncols][m % ncols].axis("off")
    fig.suptitle(f"Applied vs network torque — {label}")
    fig.supxlabel("time [s]")
    fig.supylabel("torque [Nm]")
    fig.tight_layout(rect=[0.02, 0.02, 1, 0.97])
    path = os.path.join(outdir, f"torques_{label}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[PLOT] {path}")


def make_all_plots(outdir):
    df_nn = _load_csv(os.path.join(outdir, "rollout_nn.csv"))
    df_zero = _load_csv(os.path.join(outdir, "rollout_zero.csv"))

    if df_nn is not None:
        plot_velocity_tracking(df_nn, "nn", outdir)
        plot_torques(df_nn, "nn", outdir)
    if df_zero is not None:
        plot_velocity_tracking(df_zero, "zero", outdir)
        plot_torques(df_zero, "zero", outdir)
    if df_nn is not None and df_zero is not None:
        plot_velocity_comparison(df_nn, df_zero, outdir)
    if df_nn is None and df_zero is None:
        print(f"[PLOT] No CSV found in '{outdir}'.")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", type=str, default="TitaJoystickFlatTerrain",
                        help="Environment name (default: TitaJoystickFlatTerrain)")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints",
                        help="Checkpoint root dir (checkpoints in <root>/<env_name>)")
    parser.add_argument("--load", type=str, default="best",
                        help="Checkpoint suffix for the NN run (default: best)")
    parser.add_argument("--outdir", type=str, default=None,
                        help="Output dir for CSVs and plots "
                             "(default: <ckpt-dir>/<env_name>/compare)")
    parser.add_argument("--steps", type=int, default=None,
                        help="Rollout steps (default: PPO_PARAMS episode_length)")
    parser.add_argument("--cmd", nargs=3, type=float, default=None,
                        metavar=("VX", "VY", "WZ"),
                        help="Fixed joystick command, e.g. --cmd 0.5 0.0 0.0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mem-fraction", type=float, default=0.30,
                        help="XLA GPU memory fraction PER PROCESS (2 procs run "
                             "together: keep 2*x well below 1 to leave room "
                             "for cuSolver/CUDA-graph overhead, default 0.30)")
    parser.add_argument("--stagger", type=float, default=10.0,
                        help="Seconds between starting the two processes, so "
                             "they don't hit peak compilation memory together "
                             "(default: 10)")
    parser.add_argument("--serial", action="store_true",
                        help="Run the two simulations one AFTER the other, each "
                             "with more GPU memory (use if parallel OOMs)")
    parser.add_argument("--only", choices=["nn", "zero"], default=None,
                        help="Run only one of the two simulations")
    parser.add_argument("--plot", "--plot-only", dest="plot_only",
                        action="store_true",
                        help="Skip simulations: look for the CSVs in --outdir "
                             "and only regenerate the plots")
    args = parser.parse_args()

    ckpt_dir = os.path.join(args.ckpt_dir, args.name)
    outdir = args.outdir or os.path.join(ckpt_dir, "compare")
    os.makedirs(outdir, exist_ok=True)

    if args.plot_only:
        make_all_plots(outdir)
        return

    modes = [args.only] if args.only else ["nn", "zero"]
    if args.only or args.serial:
        # single process at a time can take more GPU memory
        mem_fraction = max(args.mem_fraction, 0.5)
    else:
        mem_fraction = args.mem_fraction

    ctx = mp.get_context("spawn")

    def _launch(mode):
        p = ctx.Process(
            target=rollout_worker,
            name=f"rollout-{mode}",
            args=(mode, args.name, ckpt_dir, args.load, outdir,
                  args.steps, args.cmd, args.seed, mem_fraction),
        )
        p.start()
        print(f"[MAIN] Started '{mode}' simulation (pid {p.pid})")
        return p

    try:
        if args.serial:
            # one after the other: still separate processes and separate CSVs,
            # a crash of the first never prevents the second from running
            for mode in modes:
                p = _launch(mode)
                p.join()
                status = "OK" if p.exitcode == 0 else f"exitcode {p.exitcode}"
                print(f"[MAIN] '{mode}' finished ({status})")
        else:
            procs = {}
            for idx, mode in enumerate(modes):
                if idx > 0 and args.stagger > 0:
                    print(f"[MAIN] Waiting {args.stagger:.0f}s before starting "
                          f"the next process (--stagger)...")
                    time.sleep(args.stagger)
                procs[mode] = _launch(mode)

            # Wait for BOTH independently: one finishing/crashing never
            # stops the other.
            for mode, p in procs.items():
                p.join()
                status = "OK" if p.exitcode == 0 else f"exitcode {p.exitcode}"
                print(f"[MAIN] '{mode}' finished ({status})")
    except KeyboardInterrupt:
        print("[MAIN] Ctrl+C: letting workers flush their CSVs...")
        for p in mp.active_children():
            p.join(timeout=15)
            if p.is_alive():
                p.terminate()

    print("[MAIN] Generating plots...")
    make_all_plots(outdir)
    print(f"[MAIN] All outputs in: {outdir}")


if __name__ == "__main__":
    main()