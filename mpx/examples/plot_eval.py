"""
plot_rollout_info.py
--------------------
Plot low-level controller signals from a rollout_info.csv saved by train_srbd.py.

Usage:
    python plot_rollout_info.py                              # looks for checkpoints/AliengoJoystickFlatTerrain/rollout_info.csv
    python plot_rollout_info.py --csv path/to/rollout_info.csv
    python plot_rollout_info.py --ckpt-dir checkpoints/AliengoJoystickFlatTerrain
"""

import argparse
from array import array
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mpx.config.config_dfcip as config

_DEFAULT_DIR = os.path.join(os.path.dirname(__file__), "plots", "test")
_MPC_DIR = os.path.join(_DEFAULT_DIR, "mpc_prediction")

def get_cols(df, prefix):
    """Return sorted list of columns that start with prefix."""
    return sorted([c for c in df.columns if c.startswith(prefix)])

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def plot_signal(ax, steps, df, cols, title, ylabel="torque [Nm]", cmap="tab20"):
    colors = plt.get_cmap(cmap)(np.linspace(0, 1, max(len(cols), 1)))
    for i, col in enumerate(cols):
        label = col.split("_")[-1]  # joint index
        ax.plot(steps, df[col].values, label=f"j{label}", color=colors[i], linewidth=0.8)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6, ncol=4, loc="upper right")

def plot_network_actions(
    steps,
    actions,
    out_dir,
    joint_names,
    filename="network_actions.png",
):
    """Plot each network action in a two-column figure."""
    os.makedirs(out_dir, exist_ok=True)

    steps = np.asarray(steps)
    actions = np.asarray(actions, dtype=np.float32)

    if actions.ndim != 2:
        raise ValueError(
            f"Expected actions with shape (num_steps, num_actions), "
            f"got {actions.shape}."
        )

    num_actions = actions.shape[1]
    num_columns = 2
    num_rows = int(np.ceil(num_actions / num_columns))

    fig, axes = plt.subplots(
        nrows=num_rows,
        ncols=num_columns,
        figsize=(14, 2.8 * num_rows),
        sharex=True,
    )

    axes = np.asarray(axes).reshape(-1)

    for action_index in range(num_actions):
        ax = axes[action_index]

        joint_name = (
            joint_names[action_index]
            if action_index < len(joint_names)
            else f"action_{action_index}"
        )

        ax.plot(
            steps,
            actions[:, action_index],
            linewidth=1.0,
        )

        ax.axhline(
            0.0,
            linewidth=0.8,
            linestyle="--",
            alpha=0.5,
        )

        ax.set_title(joint_name)
        ax.set_ylabel("Network action")
        ax.grid(True, alpha=0.3)

    # Disable unused plots when the number of actions is odd.
    for axis_index in range(num_actions, len(axes)):
        axes[axis_index].axis("off")

    # Add the x label only to the last active row.
    first_last_row_index = max(0, (num_rows - 1) * num_columns)

    for axis_index in range(first_last_row_index, num_actions):
        axes[axis_index].set_xlabel("Step")

    fig.suptitle("Network actions", fontsize=15)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))

    output_path = os.path.join(out_dir, filename)

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"  Network actions plot: {output_path}")

def plot_command_tracking(
    info_log,
    out_dir,
    filename="command_tracking.png",
    plot_target_command=False,
):
    """Plot commands and measured DFCIP signals (all in BODY frame)."""
    if not info_log:
        print("[plot] info_log is empty.")
        return None

    required = ("dfcip_state_3", "dfcip_state_4", "dfcip_state_10", "dfcip_state_12")
    if not all(k in info_log[0] for k in required):
        print(f"[plot] plot_command_tracking skipped: no {required} in info "
              "(env has no DFCIP/MPC state, e.g. an E2E env).")
        return None

    os.makedirs(out_dir, exist_ok=True)
    command_path = os.path.join(out_dir, filename)

    steps = np.asarray([
        row.get("step", i)
        for i, row in enumerate(info_log)
    ])

    # Command is in BODY frame [vx, vy, wz]. dfcip_state_3/4 are WORLD, so
    # rotate them into BODY with theta (dfcip_state_10): v_body = R(theta)^T @ v_world.
    vx_world = np.asarray([row["dfcip_state_3"] for row in info_log])   # WORLD
    vy_world = np.asarray([row["dfcip_state_4"] for row in info_log])   # WORLD
    theta    = np.asarray([row["dfcip_state_10"] for row in info_log])  # yaw [rad]
    w_meas   = np.asarray([row["dfcip_state_12"] for row in info_log])  # BODY yaw rate

    cos_t, sin_t = np.cos(theta), np.sin(theta)
    vx_body =  cos_t * vx_world + sin_t * vy_world   # forward (BODY)
    vy_body = -sin_t * vx_world + cos_t * vy_world   # lateral (BODY, ~0)

    measured_series = [vx_body, vy_body, w_meas]

    titles = [
        "Linear velocity X",
        "Linear velocity Y",
        "Angular velocity Z",
    ]

    y_labels = [
        "Velocity [m/s]",
        "Velocity [m/s]",
        "Angular velocity [rad/s]",
    ]

    measured_labels = [
        "Measured CoM velocity X (body)",
        "Measured CoM velocity Y (body)",
        "Measured yaw rate (body)",
    ]

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12, 9),
        sharex=True,
    )

    for i, axis in enumerate(axes):
        command_key = f"command_{i}"
        target_key = f"target_command_{i}"

        required_keys = [command_key]

        if plot_target_command:
            required_keys.append(target_key)

        missing_keys = [
            key
            for key in required_keys
            if any(key not in row for row in info_log)
        ]

        if missing_keys:
            print(f"[plot] Missing command fields: {missing_keys}")
            plt.close(fig)
            return None

        command = np.asarray([
            row[command_key]
            for row in info_log
        ])

        measured_velocity = measured_series[i]   # already BODY frame

        axis.plot(
            steps,
            command,
            label="Command",
        )

        if plot_target_command:
            target_command = np.asarray([
                row[target_key]
                for row in info_log
            ])

            axis.plot(
                steps,
                target_command,
                linestyle="--",
                label="Target command",
            )

        axis.plot(
            steps,
            measured_velocity,
            label=measured_labels[i],
        )

        axis.set_title(titles[i])
        axis.set_ylabel(y_labels[i])
        axis.grid(True)
        axis.legend()

    axes[-1].set_xlabel("Step")

    fig.tight_layout()
    fig.savefig(
        command_path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"[plot] Command tracking saved → {command_path}")

def plot_rollout_rewards(
    steps,
    rewards,
    action_sums,
    ckpt_dir,
    filename="reward_rollout.png",
):
    steps = np.asarray(steps)
    rewards = np.asarray(rewards)
    action_sums = np.asarray(action_sums)

    cumulative_rewards = np.cumsum(rewards)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10, 9),
        sharex=True,
    )

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
    rollout_plot_path = os.path.join(ckpt_dir, filename)

    fig.savefig(
        rollout_plot_path,
        dpi=120,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"  Reward plot  : {rollout_plot_path}")

def _save_sim_video(
        video_dir,
        frames,
        video_fps=30,
        slowdown_factor=1.0,
        name_video="simulation_video.mp4",
    ) -> None:
        print("Saving simulation video... ", end="\n", flush=True)
        try:
            import imageio
        except ImportError:
            print("imageio not installed, skipping video saving.")
            return
        if not frames:
            print("No frames captured, skipping video saving.")
            return
        print(f"Captured {len(frames)} frames at {video_fps} fps.")
        video_dir = os.path.join(video_dir)
        os.makedirs(video_dir, exist_ok=True)
        video_path = os.path.join(video_dir, name_video)
        try:
            imageio.mimwrite(video_path, frames, fps=video_fps/slowdown_factor, macro_block_size=1)
            print(
                f"[sim_video] saved ({len(frames)} frames, {video_fps} fps, slowdown x{slowdown_factor:.2f}): {video_path}"
            )
        except Exception as e:
            print(f"[sim_video] failed to save video: {e}")

def plot_llc(
    csv_path: str,
    filename: str = "plot_llc.png",
    out_dir: str = _DEFAULT_DIR,
) -> str | None:
    """
    Load csv_path and save the LLC plot.

    Returns the output path, or None if the CSV does not exist.
    Can be called programmatically from train_srbd.py.
    """
    if not os.path.exists(csv_path):
        print(f"[plot_llc] CSV not found: {csv_path}")
        return None

    _ensure_dir(out_dir)

    print(f"  Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    steps = df["step"].values if "step" in df.columns else np.arange(len(df))

    LLC = "low_level_controller/"

    tau_ff_cols = get_cols(df, LLC + "tau_ff_")
    tau_p_cols = get_cols(df, LLC + "tau_p_")
    tau_d_cols = get_cols(df, LLC + "tau_d_")
    action_cols = get_cols(df, LLC + "action_")

    kp = (
        df[LLC + "kp"].iloc[0]
        if (LLC + "kp") in df.columns
        else float("nan")
    )
    kd = (
        df[LLC + "kd"].iloc[0]
        if (LLC + "kd") in df.columns
        else float("nan")
    )
    action_scale = (
        df[LLC + "action_scale"].iloc[0]
        if (LLC + "action_scale") in df.columns
        else float("nan")
    )

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(14, 10),
        sharex=False,
        gridspec_kw={"width_ratios": [1, 1]},
    )

    ax_tff = fig.add_subplot(3, 1, 1)
    plot_signal(
        ax_tff,
        steps,
        df,
        tau_ff_cols,
        "tau_ff  (feed-forward torque)",
    )
    ax_tff.set_xlabel("")

    ax_tp = fig.add_subplot(3, 2, 3)
    ax_td = fig.add_subplot(3, 2, 4, sharex=ax_tp)

    tp_vals = (
        df[tau_p_cols].values
        if tau_p_cols
        else np.array([0.0])
    )
    td_vals = (
        df[tau_d_cols].values
        if tau_d_cols
        else np.array([0.0])
    )

    tp_min, tp_max = tp_vals.min(), tp_vals.max()
    td_min, td_max = td_vals.min(), td_vals.max()

    plot_signal(
        ax_tp,
        steps,
        df,
        tau_p_cols,
        f"tau_p   (Kp={kp:.3g}  |  min={tp_min:.2f}  max={tp_max:.2f})",
    )
    plot_signal(
        ax_td,
        steps,
        df,
        tau_d_cols,
        f"tau_d   (Kd={kd:.3g}  |  min={td_min:.2f}  max={td_max:.2f})",
    )

    ax_act = fig.add_subplot(3, 1, 3)
    plot_signal(
        ax_act,
        steps,
        df,
        action_cols,
        "action  (policy output)",
        ylabel="action []",
    )
    ax_act.set_xlabel("step")

    for ax in axes.flat:
        ax.set_visible(False)

    fig.suptitle(
        f"Rollout — Low-Level Controller  |  "
        f"action_scale={action_scale:.4g}\n{csv_path}",
        fontsize=10,
        y=1.01,
    )
    fig.tight_layout()

    out_path = os.path.join(out_dir, filename)

    fig.savefig(
        out_path,
        dpi=130,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"  Saved: {out_path}")


def plot_reward_terms(
    terms,
    prefix: str = "reward_terms/",
    threshold: float = 10.0,
    filename: str = "reward_terms.png",
    out_dir: str = _DEFAULT_DIR,
    title: str = "Reward terms",
    verbose: bool = False,
):
    """Plot dei termini di reward su due subplot.

    terms     : dict {nome: array (T,)} oppure lista di dict (uno per step).
    prefix    : prefisso delle chiavi da selezionare ("" o None = tutte).
    threshold : chi ha almeno un |valore| > threshold va sopra, gli altri sotto.
    verbose   : stampa lunghezza / min / max / NaN di ogni serie.
    """
    _ensure_dir(out_dir)

    # ── lista di dict (uno per step) -> dict di serie temporali ───────────
    if isinstance(terms, (list, tuple)):
        if len(terms) == 0:
            print("[plot] info log vuoto")
            return None
        terms = {
            k: np.asarray([np.asarray(d[k]).reshape(-1)[0] for d in terms])
            for k in terms[0]
        }

    if prefix:
        terms = {k[len(prefix):]: v for k, v in terms.items() if k.startswith(prefix)}
        if not terms:
            print(f"[plot] nessuna chiave con prefisso '{prefix}', salto (l'env non ha reward terms loggati).")
            return None

    data = {k: np.asarray(v, dtype=float).reshape(-1) for k, v in terms.items()}

    if not data:
        print(f"[plot] nessuna chiave con prefisso '{prefix}'")
        return None

    # ── diagnostica ───────────────────────────────────────────────────────
    if verbose:
        print(f"[plot] {len(data)} serie:")
        for k, v in sorted(data.items()):
            finite = v[np.isfinite(v)]
            if finite.size:
                print(f"       {k:<28} len={v.size:<6} "
                      f"min={finite.min():<12.4g} max={finite.max():<12.4g} "
                      f"non-finite={v.size - finite.size}")
            else:
                print(f"       {k:<28} len={v.size:<6} ALL non-finite (NaN/inf)")

    # ── scarta le serie non plottabili ────────────────────────────────────
    dropped = [k for k, v in data.items() if not np.isfinite(v).any()]
    for k in dropped:
        data.pop(k)

    if dropped:
        print(f"[plot] serie senza valori finiti, escluse: {dropped}")

    if not data:
        print("[plot] nessuna serie plottabile")
        return None

    n_pts = max(len(v) for v in data.values())
    if n_pts < 2:
        print(f"[plot] attenzione: solo {n_pts} punto per serie "
              f"(passa la lista completa info_log, non info_log[0])")

    big = sorted([k for k, v in data.items() if np.nanmax(np.abs(v[np.isfinite(v)])) > threshold])
    small = sorted([k for k in data if k not in big])

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    fig.suptitle(title, fontsize=13)

    cmap = plt.get_cmap("tab20")
    marker = "o" if n_pts < 2 else None

    for ax, names, subtitle in (
        (ax_top, big, f"|max| > {threshold:g}"),
        (ax_bot, small, f"|max| <= {threshold:g}"),
    ):
        for i, k in enumerate(names):
            v = np.where(np.isfinite(data[k]), data[k], np.nan)  # inf -> gap
            ax.plot(v, linewidth=1.2, marker=marker, color=cmap(i % 20), label=k)

        ax.set_title(subtitle)
        ax.set_ylabel("value")
        ax.grid(True, alpha=0.3)
        ax.axhline(0.0, color="k", linewidth=0.6, alpha=0.4)

        if names:
            ax.legend(fontsize=8, ncol=2, loc="upper right")

    ax_bot.set_xlabel("Step")

    fig.tight_layout()
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[plot] saved → {path}")

    return path

def plot_reward_terms_separate(
    terms,
    prefix: str = "reward_terms/",
    dir_name: str = None,
    out_dir: str = _DEFAULT_DIR,
    verbose: bool = False,
):
    """Come plot_reward_terms, ma salva un PNG separato per ogni termine.

    terms    : dict {nome: array (T,)} oppure lista di dict (uno per step).
    prefix   : prefisso delle chiavi da selezionare ("" o None = tutte).
               Da qui viene ricavato anche il nome della cartella.
    dir_name : per forzare un nome cartella diverso da quello del prefix.
    verbose  : stampa lunghezza / min / max / NaN di ogni serie.
    """
    # ── nome cartella dal prefix ──────────────────────────────────────────
    if dir_name is None:
        dir_name = prefix.strip("/").replace("/", "_") if prefix else "plots"

    plots_dir = os.path.join(out_dir, dir_name)
    _ensure_dir(plots_dir)

    # ── lista di dict (uno per step) -> dict di serie temporali ───────────
    if isinstance(terms, (list, tuple)):
        if len(terms) == 0:
            print("[plot] info log vuoto")
            return None
        terms = {
            k: np.asarray([np.asarray(d[k]).reshape(-1)[0] for d in terms])
            for k in terms[0]
        }

    # terms resta con TUTTE le chiavi (command_*, robot/..., reward_terms/...):
    # serve intatto piu' sotto per i grafici extra di tracking/altezza.
    if prefix:
        filtered = {k[len(prefix):]: v for k, v in terms.items() if k.startswith(prefix)}
        if not filtered:
            print(f"[plot] nessuna chiave con prefisso '{prefix}', salto (l'env non ha reward terms loggati).")
            return None
    else:
        filtered = terms

    data = {k: np.asarray(v, dtype=float).reshape(-1) for k, v in filtered.items()}

    if not data:
        print(f"[plot] nessuna chiave con prefisso '{prefix}'")
        return None

    # ── diagnostica ───────────────────────────────────────────────────────
    if verbose:
        print(f"[plot] {len(data)} serie:")
        for k, v in sorted(data.items()):
            finite = v[np.isfinite(v)]
            if finite.size:
                print(f"       {k:<28} len={v.size:<6} "
                      f"min={finite.min():<12.4g} max={finite.max():<12.4g} "
                      f"non-finiti={v.size - finite.size}")
            else:
                print(f"       {k:<28} len={v.size:<6} TUTTI non-finiti (NaN/inf)")

    # ── una figura per termine ────────────────────────────────────────────
    paths = {}

    for k in sorted(data):
        v = data[k]

        if not np.isfinite(v).any():
            print(f"[plot] '{k}' senza valori finiti, saltato")
            continue

        # ── grafici extra: comando vs misurato, per tracking e altezza ─────
        if k == "tracking_lin_vel":
            # Comandi: numero di componenti variabile a seconda del robot
            # (2 per Tita [vx, wz], 3 per un quadrupede [vx, vy, wz], ...).
            # L'ultima componente è sempre quella angolare (coerente con
            # "commands[-1]" usato dagli env in joystickE2E.py).
            command_keys = sorted(
                (ck for ck in terms if re.fullmatch(r"command_\d+", ck)),
                key=lambda ck: int(ck.split("_")[1]),
            )
            ang_key = command_keys[-1] if command_keys else None
            required = ("robot/local_linvel_0", "robot/gyro_2")
            if command_keys and all(rk in terms for rk in required):
                steps_t = np.arange(len(terms[command_keys[0]]))
                cmd_lin = np.asarray(terms[command_keys[0]], dtype=float)
                cmd_ang = np.asarray(terms[ang_key], dtype=float)
                meas_lin = np.asarray(terms["robot/local_linvel_0"], dtype=float)
                meas_ang = np.asarray(terms["robot/gyro_2"], dtype=float)

                fig_v, (ax_reward, ax_lin, ax_ang) = plt.subplots(
                    3, 1, figsize=(10, 10), sharex=True
                )

                # Reward term "tracking_lin_vel" nella stessa figura, invece
                # che nel solito PNG separato (vedi blocco generico sotto).
                reward_v = np.where(np.isfinite(v), v, np.nan)
                ax_reward.plot(steps_t, reward_v, linewidth=1.2, color="steelblue")
                ax_reward.set_title(f"{k} (reward term)")
                ax_reward.set_ylabel("value")
                ax_reward.grid(True, alpha=0.3)
                ax_reward.axhline(0.0, color="k", linewidth=0.6, alpha=0.4)
                finite_reward = reward_v[np.isfinite(reward_v)]
                if finite_reward.size:
                    ax_reward.text(
                        0.01, 0.02,
                        f"mean={finite_reward.mean():.4g}   min={finite_reward.min():.4g}   "
                        f"max={finite_reward.max():.4g}   sum={finite_reward.sum():.4g}",
                        transform=ax_reward.transAxes, fontsize=8, color="gray",
                    )

                ax_lin.plot(steps_t, cmd_lin, label="Command vx")
                ax_lin.plot(steps_t, meas_lin, label="Measured vx (local_linvel)")
                ax_lin.set_title("Linear velocity tracking")
                ax_lin.set_ylabel("m/s")
                ax_lin.grid(True, alpha=0.3)
                ax_lin.legend()

                ax_ang.plot(steps_t, cmd_ang, label=f"Command omega ({ang_key})")
                ax_ang.plot(steps_t, meas_ang, label="Measured omega (gyro_z)")
                ax_ang.set_title("Angular velocity tracking")
                ax_ang.set_ylabel("rad/s")
                ax_ang.set_xlabel("Step")
                ax_ang.grid(True, alpha=0.3)
                ax_ang.legend()

                fig_v.tight_layout()
                path_v = os.path.join(plots_dir, "tracking_lin_vel.png")
                fig_v.savefig(path_v, dpi=120)
                plt.close(fig_v)
                paths["velocity_tracking"] = path_v
                print(f"[plot] velocity tracking saved -> {path_v}")
                # Il reward term è già incluso sopra (ax_reward): non generare
                # anche il solito tracking_lin_vel.png separato.
                continue
            else:
                print(f"[plot] velocity tracking skipped: missing command_* or {required}")

        elif k == "base_height":
            if "robot/com_height" in terms and "robot/base_height_target" in terms:
                steps_h = np.arange(len(terms["robot/com_height"]))
                com_height = np.asarray(terms["robot/com_height"], dtype=float)
                target = float(np.asarray(terms["robot/base_height_target"]).reshape(-1)[0])

                fig_h, (ax_reward, ax_h) = plt.subplots(
                    2, 1, figsize=(10, 7), sharex=True
                )

                # Reward term "base_height" nella stessa figura, invece che
                # nel solito PNG separato (vedi blocco generico sotto).
                reward_h = np.where(np.isfinite(v), v, np.nan)
                ax_reward.plot(steps_h, reward_h, linewidth=1.2, color="steelblue")
                ax_reward.set_title(f"{k} (reward term)")
                ax_reward.set_ylabel("value")
                ax_reward.grid(True, alpha=0.3)
                ax_reward.axhline(0.0, color="k", linewidth=0.6, alpha=0.4)
                finite_reward = reward_h[np.isfinite(reward_h)]
                if finite_reward.size:
                    ax_reward.text(
                        0.01, 0.02,
                        f"mean={finite_reward.mean():.4g}   min={finite_reward.min():.4g}   "
                        f"max={finite_reward.max():.4g}   sum={finite_reward.sum():.4g}",
                        transform=ax_reward.transAxes, fontsize=8, color="gray",
                    )

                ax_h.axhline(target, color="k", linestyle="--", label="Target height")
                ax_h.plot(steps_h, com_height, label="Measured CoM height")
                ax_h.set_title("Base height tracking")
                ax_h.set_xlabel("Step")
                ax_h.set_ylabel("Height [m]")
                ax_h.grid(True, alpha=0.3)
                ax_h.legend()

                fig_h.tight_layout()
                path_h = os.path.join(plots_dir, "base_height.png")
                fig_h.savefig(path_h, dpi=120)
                plt.close(fig_h)
                paths["height_tracking"] = path_h
                print(f"[plot] height tracking saved -> {path_h}")
                continue
            else:
                print("[plot] height tracking skipped: missing 'robot/com_height' or 'robot/base_height_target'")

        elif k == "wheel_track":
            # feet_pos è (2, 3) -> appiattito in robot/feet_pos_0..5
            # (0:3 = ruota sinistra, 3:6 = ruota destra).
            feet_pos_keys = [f"robot/feet_pos_{i}" for i in range(6)]
            if all(fk in terms for fk in feet_pos_keys):
                steps_w = np.arange(len(terms[feet_pos_keys[0]]))
                feet_pos = np.stack(
                    [np.asarray(terms[fk], dtype=float) for fk in feet_pos_keys],
                    axis=-1,
                )  # (T, 6)
                left = feet_pos[:, 0:3]
                right = feet_pos[:, 3:6]
                wheel_dist = np.linalg.norm(left - right, axis=-1)
                target = float(config.d)

                fig_w, (ax_reward, ax_w) = plt.subplots(
                    2, 1, figsize=(10, 7), sharex=True
                )

                # Reward term "wheel_track" nella stessa figura, invece che
                # nel solito PNG separato (vedi blocco generico sotto).
                reward_w = np.where(np.isfinite(v), v, np.nan)
                ax_reward.plot(steps_w, reward_w, linewidth=1.2, color="steelblue")
                ax_reward.set_title(f"{k} (reward term)")
                ax_reward.set_ylabel("value")
                ax_reward.grid(True, alpha=0.3)
                ax_reward.axhline(0.0, color="k", linewidth=0.6, alpha=0.4)
                finite_reward = reward_w[np.isfinite(reward_w)]
                if finite_reward.size:
                    ax_reward.text(
                        0.01, 0.02,
                        f"mean={finite_reward.mean():.4g}   min={finite_reward.min():.4g}   "
                        f"max={finite_reward.max():.4g}   sum={finite_reward.sum():.4g}",
                        transform=ax_reward.transAxes, fontsize=8, color="gray",
                    )

                ax_w.axhline(target, color="k", linestyle="--", label="Target track width")
                ax_w.plot(steps_w, wheel_dist, label="Measured wheel-to-wheel distance")
                ax_w.set_title("Wheel track tracking")
                ax_w.set_xlabel("Step")
                ax_w.set_ylabel("Distance [m]")
                ax_w.grid(True, alpha=0.3)
                ax_w.legend()

                fig_w.tight_layout()
                path_w = os.path.join(plots_dir, "wheel_track.png")
                fig_w.savefig(path_w, dpi=120)
                plt.close(fig_w)
                paths["wheel_track"] = path_w
                print(f"[plot] wheel track saved -> {path_w}")
                continue
            else:
                print(f"[plot] wheel track skipped: missing one of {feet_pos_keys}")

        v = np.where(np.isfinite(v), v, np.nan)  # inf -> gap
        marker = "o" if v.size < 2 else None

        fig, ax = plt.subplots(figsize=(10, 4))

        ax.plot(v, linewidth=1.2, marker=marker, color="steelblue")
        ax.set_title(k)
        ax.set_xlabel("Step")
        ax.set_ylabel("value")
        ax.grid(True, alpha=0.3)
        ax.axhline(0.0, color="k", linewidth=0.6, alpha=0.4)

        finite = v[np.isfinite(v)]
        ax.text(
            0.01, 0.02,
            f"mean={finite.mean():.4g}   min={finite.min():.4g}   "
            f"max={finite.max():.4g}   sum={finite.sum():.4g}",
            transform=ax.transAxes, fontsize=8, color="gray",
        )

        fig.tight_layout()

        safe = k.replace("/", "_")
        path = os.path.join(plots_dir, f"{safe}.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)

        paths[k] = path

    print(f"[plot] {len(paths)} figure salvate → {plots_dir}")

    return paths

    
def plot_mpc_output(
    info_log,
    prefix: str = "mpc_output",
    filename: str = "mpc_output.png",
    out_dir: str = _DEFAULT_DIR,
    title: str = "MPC output",
    verbose: bool = False,
):
    """Plot the 9 MPC output values in a single 3x3 figure.

    Expected output order:
        0: a
        1: ac_z
        2: alpha
        3: left wheel GRF x
        4: left wheel GRF y
        5: left wheel GRF z
        6: right wheel GRF x
        7: right wheel GRF y
        8: right wheel GRF z

    Supported input formats
    -----------------------
    1. A list of dictionaries containing a single 9-element vector:

        info_log[t]["mpc_output"] = array(shape=(9,))

    2. A dictionary containing the complete time series:

        info_log["mpc_output"] = array(shape=(T, 9))

    3. Separate fields:

        info_log[t]["mpc_output/a"]
        info_log[t]["mpc_output/ac_z"]
        info_log[t]["mpc_output/alpha"]
        info_log[t]["mpc_output/grf"] = array(shape=(6,))

    Args:
        info_log:
            A list of dictionaries, one per simulation step, or a dictionary
            containing complete time series.

        prefix:
            Prefix used to locate MPC output data.

        filename:
            Name of the output image.

        out_dir:
            Directory where the image is saved.

        title:
            Figure title.

        verbose:
            Print statistics for every MPC output component.

    Returns:
        The saved image path, or None if no valid data is found.
    """

    def _extract_flat_field(
        step_info: dict,
        base_key: str,
        expected_size: int | None = None,
    ) -> np.ndarray:
        """
        Extract a scalar or indexed vector from a flattened info dictionary.

        Examples:
        mpc_output/a       -> array([value])
        mpc_output/grf_0
        ...
        mpc_output/grf_5   -> array with 6 values
        """

        # Scalar stored directly.
        if base_key in step_info:
            values = np.asarray(step_info[base_key], dtype=float).reshape(-1)

        else:
            prefix = base_key + "_"

            indexed_items = []

            for key, value in step_info.items():
                if not key.startswith(prefix):
                    continue

                suffix = key[len(prefix):]

                if suffix.isdigit():
                    indexed_items.append((int(suffix), value))

            indexed_items.sort(key=lambda item: item[0])

            values = np.asarray(
                [value for _, value in indexed_items],
                dtype=float,
            )

        if expected_size is not None and values.size != expected_size:
            raise ValueError(
                f"'{base_key}' contains {values.size} values; "
                f"expected {expected_size}. "
                f"Available keys: "
                f"{[key for key in step_info if key.startswith(base_key)]}"
            )

        return values
    
    os.makedirs(out_dir, exist_ok=True)

    prefix = prefix.rstrip("/")

    scalar_keys = [
        f"{prefix}/a",
        f"{prefix}/ac_z",
        f"{prefix}/alpha",
    ]
    grf_key = [
        f"{prefix}/grf_{i}" for i in range(6)
    ]

    # ------------------------------------------------------------------
    # Extract data from a list of dictionaries.
    # ------------------------------------------------------------------
    if isinstance(info_log, (list, tuple)):
        if len(info_log) == 0:
            print("[plot] Empty info log")
            return None

        rows = []

        for step_index, step_info in enumerate(info_log):
            # Single flattened vector:
            # [a, ac_z, alpha, left_grf_xyz, right_grf_xyz]
            if prefix in step_info:
                output = np.asarray(
                    step_info[prefix],
                    dtype=float,
                ).reshape(-1)

                if output.size != 9:
                    print(
                        f"[plot] '{prefix}' contains {output.size} values "
                        f"at step {step_index}; expected 9"
                    )
                    return None

                rows.append(output)
                continue

            # Separate ControlSol fields.
            missing_keys_scalar = [
                key for key in scalar_keys
                if key not in step_info
            ]

            missing_keys_grf = [
                f"{prefix}/grf_{i}"
                for i in range(len(grf_key))
                if f"{prefix}/grf_{i}" not in step_info
            ]

            missing_keys = missing_keys_scalar + missing_keys_grf

            if missing_keys:
                print(
                    f"[plot] Missing MPC output fields at step "
                    f"{step_index}: {missing_keys}"
                )
                return None

            a = _extract_flat_field(
                step_info,
                "mpc_output/a",
                expected_size=1,
            )

            ac_z = _extract_flat_field(
                step_info,
                "mpc_output/ac_z",
                expected_size=1,
            )

            alpha = _extract_flat_field(
                step_info,
                "mpc_output/alpha",
                expected_size=1,
            )

            grf = _extract_flat_field(
                step_info,
                "mpc_output/grf",
                expected_size=6,
            )

            if a.size != 1 or ac_z.size != 1 or alpha.size != 1:
                print(
                    f"[plot] a, ac_z, and alpha must be scalar values "
                    f"at step {step_index}"
                )
                return None

            if grf.size != 6:
                print(
                    f"[plot] '{grf_key}' contains {grf.size} values "
                    f"at step {step_index}; expected 6"
                )
                return None

            output = np.concatenate(
                [
                    a,
                    ac_z,
                    alpha,
                    grf[0:3],  # Left wheel contact force.
                    grf[3:6],  # Right wheel contact force.
                ]
            )

            rows.append(output)

        data = np.stack(rows, axis=0)

    # ------------------------------------------------------------------
    # Extract data from a dictionary of complete time series.
    # ------------------------------------------------------------------
    elif isinstance(info_log, dict):
        # Single array with shape (T, 9).
        if prefix in info_log:
            data = np.asarray(
                info_log[prefix],
                dtype=float,
            )

            if data.ndim == 1:
                if data.size != 9:
                    print(
                        f"[plot] '{prefix}' contains {data.size} values; "
                        "expected 9"
                    )
                    return None

                data = data.reshape(1, 9)

            elif data.ndim == 2:
                if data.shape[1] == 9:
                    pass
                elif data.shape[0] == 9:
                    data = data.T
                else:
                    print(
                        f"[plot] Invalid '{prefix}' shape: {data.shape}. "
                        "Expected (T, 9) or (9, T)"
                    )
                    return None
            else:
                print(
                    f"[plot] Invalid '{prefix}' shape: {data.shape}"
                )
                return None

        # Separate ControlSol time series.
        else:
            required_keys = scalar_keys + [grf_key]
            missing_keys = [
                key for key in required_keys
                if key not in info_log
            ]

            if missing_keys:
                print(
                    f"[plot] Missing MPC output fields: {missing_keys}"
                )
                return None

            a = np.asarray(
                info_log[scalar_keys[0]],
                dtype=float,
            ).reshape(-1)

            ac_z = np.asarray(
                info_log[scalar_keys[1]],
                dtype=float,
            ).reshape(-1)

            alpha = np.asarray(
                info_log[scalar_keys[2]],
                dtype=float,
            ).reshape(-1)

            grf = np.asarray(
                info_log[grf_key],
                dtype=float,
            )

            if grf.ndim == 1:
                if grf.size != 6:
                    print(
                        f"[plot] '{grf_key}' contains {grf.size} values; "
                        "expected 6"
                    )
                    return None

                grf = grf.reshape(1, 6)

            elif grf.ndim == 2:
                if grf.shape[1] == 6:
                    pass
                elif grf.shape[0] == 6:
                    grf = grf.T
                else:
                    print(
                        f"[plot] Invalid '{grf_key}' shape: {grf.shape}. "
                        "Expected (T, 6) or (6, T)"
                    )
                    return None
            else:
                print(
                    f"[plot] Invalid '{grf_key}' shape: {grf.shape}"
                )
                return None

            lengths = {
                a.size,
                ac_z.size,
                alpha.size,
                grf.shape[0],
            }

            if len(lengths) != 1:
                print(
                    "[plot] MPC output series have different lengths: "
                    f"a={a.size}, ac_z={ac_z.size}, "
                    f"alpha={alpha.size}, grf={grf.shape[0]}"
                )
                return None

            data = np.column_stack(
                [
                    a,
                    ac_z,
                    alpha,
                    grf[:, 0:3],
                    grf[:, 3:6],
                ]
            )

    else:
        print(
            "[plot] info_log must be a dictionary or a list of dictionaries"
        )
        return None

    if data.shape[1] != 9:
        print(
            f"[plot] Extracted {data.shape[1]} MPC outputs; expected 9"
        )
        return None

    plot_titles = [
        "a",
        "ac_z",
        "alpha",
        "Left wheel GRF — Fx",
        "Left wheel GRF — Fy",
        "Left wheel GRF — Fz",
        "Right wheel GRF — Fx",
        "Right wheel GRF — Fy",
        "Right wheel GRF — Fz",
    ]

    y_labels = [
        "value",
        "value",
        "value",
        "force [N]",
        "force [N]",
        "force [N]",
        "force [N]",
        "force [N]",
        "force [N]",
    ]

    if verbose:
        print(
            f"[plot] MPC output: {data.shape[0]} steps, "
            f"{data.shape[1]} components"
        )

        for index, name in enumerate(plot_titles):
            values = data[:, index]
            finite_values = values[np.isfinite(values)]

            if finite_values.size == 0:
                print(f"       {name:<28} all values are non-finite")
                continue

            print(
                f"       {name:<28} "
                f"min={finite_values.min():<12.4g} "
                f"max={finite_values.max():<12.4g} "
                f"mean={finite_values.mean():<12.4g} "
                f"non-finite={values.size - finite_values.size}"
            )

    steps = np.arange(data.shape[0])

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(15, 10),
        sharex=True,
    )

    fig.suptitle(title, fontsize=14)

    for index, ax in enumerate(axes.flat):
        values = np.where(
            np.isfinite(data[:, index]),
            data[:, index],
            np.nan,
        )

        marker = "o" if values.size < 2 else None

        ax.plot(
            steps,
            values,
            linewidth=1.2,
            marker=marker,
        )

        ax.set_title(plot_titles[index], fontsize=10)
        ax.set_ylabel(y_labels[index])
        ax.grid(True, alpha=0.3)
        ax.axhline(
            0.0,
            linewidth=0.6,
            alpha=0.4,
        )

        if index >= 6:
            ax.set_xlabel("Step")

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])

    path = os.path.join(out_dir, filename)

    fig.savefig(
        path,
        dpi=120,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"[plot] Saved MPC output plot -> {path}")
