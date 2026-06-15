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

_DEFAULT_DIR = os.path.join(os.path.dirname(__file__), "plots", "test")
_MPC_DIR = os.path.join(_DEFAULT_DIR, "mpc_prediction")


def _extract_step_idx(path: str) -> int:
    name = os.path.basename(path)
    match = re.search(r"_t(\d+)\.png$", name)
    return int(match.group(1)) if match else -1


def get_cols(df, prefix):
    """Return sorted list of columns that start with prefix."""
    return sorted([c for c in df.columns if c.startswith(prefix)])

import os
import numpy as np
import matplotlib.pyplot as plt

import os
import numpy as np
import matplotlib.pyplot as plt


import os
import numpy as np
import matplotlib.pyplot as plt


def plot_mpc_state_and_output(
    x0_list,
    u0_list,
    x_ref_list=None,
    u_ref_list=None,
    filename_state: str = "mpc_input.png",
    filename_u0: str = "mpc_output.png",
    out_dir: str = "./",
    title_state: str = "MPC State (x0)",
    title_u0: str = "MPC Control (U0)",
    save_csv: bool = True,
):
    os.makedirs(out_dir, exist_ok=True)

    x0 = np.asarray(x0_list, dtype=float)
    u0 = np.asarray(u0_list, dtype=float)

    if x0.ndim == 1:
        x0 = x0[None, :]

    if u0.ndim == 1:
        u0 = u0[None, :]

    x_ref = np.asarray(x_ref_list, dtype=float) if x_ref_list is not None else None
    u_ref = np.asarray(u_ref_list, dtype=float) if u_ref_list is not None else None

    if x_ref is not None and x_ref.ndim == 1:
        x_ref = x_ref[None, :]

    if u_ref is not None and u_ref.ndim == 1:
        u_ref = u_ref[None, :]

    state_names = [
        "pcom_x", "pcom_y", "pcom_z",
        "vcom_x", "vcom_y", "vcom_z",
        "c_world_x", "c_world_y", "c_world_z",
        "vcz",
        "theta",
        "v",
        "omega",
    ]

    control_names = [
        "a", "acz", "alpha",
        "flx", "fly", "flz",
        "frx", "fry", "frz",
    ]

    nx = x0.shape[1]
    nu = u0.shape[1]

    # ============================================================
    # CSV SAVE
    # ============================================================
    if save_csv:
        df_x = pd.DataFrame({"step": np.arange(x0.shape[0])})

        for i in range(nx):
            name = state_names[i] if i < len(state_names) else f"x{i}"
            df_x[name] = x0[:, i]

        if x_ref is not None:
            n_ref = min(x_ref.shape[0], x0.shape[0])
            for i in range(x_ref.shape[1]):
                name = state_names[i] if i < len(state_names) else f"x{i}"
                df_x.loc[:n_ref - 1, f"{name}_ref"] = x_ref[:n_ref, i]

        _save_df_csv(df_x, out_dir, filename_state)

        df_u = pd.DataFrame({"step": np.arange(u0.shape[0])})

        for i in range(nu):
            name = control_names[i] if i < len(control_names) else f"u{i}"
            df_u[name] = u0[:, i]

        if u_ref is not None:
            n_ref = min(u_ref.shape[0], u0.shape[0])
            for i in range(u_ref.shape[1]):
                name = control_names[i] if i < len(control_names) else f"u{i}"
                df_u.loc[:n_ref - 1, f"{name}_ref"] = u_ref[:n_ref, i]

        _save_df_csv(df_u, out_dir, filename_u0)

    # ============================================================
    # STATES FIGURE — 5 x 3
    # ============================================================
    fig1, axes1 = plt.subplots(5, 3, figsize=(18, 12), sharex=True)
    axes1 = axes1.flatten()

    for i in range(15):
        ax = axes1[i]

        if i < nx:
            ax.plot(x0[:, i], label="x")

            if x_ref is not None and i < x_ref.shape[1]:
                n = min(x0.shape[0], x_ref.shape[0])
                ax.plot(np.arange(n), x_ref[:n, i], "--", label="ref")

            ax.set_title(state_names[i] if i < len(state_names) else f"x{i}")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        else:
            ax.axis("off")

    fig1.suptitle(title_state)
    fig1.tight_layout()

    state_path = os.path.join(out_dir, filename_state)
    fig1.savefig(state_path, dpi=120)
    plt.close(fig1)

    print(f"[plot] saved → {state_path}")

    # ============================================================
    # CONTROL FIGURE — 3 x 3
    # ============================================================
    fig2, axes2 = plt.subplots(3, 3, figsize=(12, 10), sharex=True)
    axes2 = axes2.flatten()

    for i in range(9):
        ax = axes2[i]

        if i < nu:
            ax.plot(u0[:, i], label="u")

            if u_ref is not None and i < u_ref.shape[1]:
                n = min(u0.shape[0], u_ref.shape[0])
                ax.plot(np.arange(n), u_ref[:n, i], "--", label="ref")

            ax.set_title(control_names[i] if i < len(control_names) else f"u{i}")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        else:
            ax.axis("off")

    fig2.suptitle(title_u0)
    fig2.tight_layout()

    u_path = os.path.join(out_dir, filename_u0)
    fig2.savefig(u_path, dpi=120)
    plt.close(fig2)

    print(f"[plot] saved → {u_path}")

def plot_wbc_desired(
    desired_list,
    current_list=None,
    mpc_utils=None,
    nj=None,
    out_dir=".",
    filename_com_base="wbc_desired_com_base.png",
    filename_wheels="wbc_desired_wheels.png",
    filename_joints="wbc_desired_joints.png",
    save_csv: bool = True,
):
    """
    Plot WBC desired references, optionally against current values.

    Layout:
      - COM/base: 3 rows x 2 cols
      - wheels:   3 rows x 2 cols
      - joints:   4 rows x 2 cols

    Also saves one CSV inside:
        out_dir/csv_data/wbc_desired.csv
    """

    if mpc_utils is None:
        raise ValueError("mpc_utils must be passed to plot_wbc_desired(...).")

    if nj is None:
        raise ValueError("nj must be passed to plot_wbc_desired(...).")

    os.makedirs(out_dir, exist_ok=True)

    desired = np.asarray(desired_list, dtype=float)

    if desired.size == 0:
        print("[plot_wbc_desired] empty desired_list, skipping.")
        return

    if desired.ndim == 1:
        desired = desired[None, :]

    current = None

    if current_list is not None and len(current_list) > 0:
        current = np.asarray(current_list, dtype=float)

        if current.ndim == 1:
            current = current[None, :]

        n = min(desired.shape[0], current.shape[0])
        desired = desired[:n]
        current = current[:n]

    t = np.arange(desired.shape[0])

    # ============================================================
    # CSV SAVE
    # ============================================================
    if save_csv:
        df = pd.DataFrame({"step": t})

        for i in range(desired.shape[1]):
            df[f"desired_{i}"] = desired[:, i]

        if current is not None:
            for i in range(current.shape[1]):
                df[f"current_{i}"] = current[:, i]

        _save_df_csv(df, out_dir, "wbc_desired.csv")

    def _save(fig, filename):
        path = os.path.join(out_dir, filename)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        print(f"[plot] saved → {path}")

    def _setup_ax(ax, title, ylabel=None):
        ax.set_title(title)

        if ylabel is not None:
            ax.set_ylabel(ylabel)

        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2)

    def _plot_scalar(ax, idx, name, ylabel=None):
        if idx >= desired.shape[1]:
            ax.axis("off")
            return

        ax.plot(t, desired[:, idx], label=f"{name} desired")

        if current is not None and idx < current.shape[1]:
            ax.plot(t, current[:, idx], "--", label=f"{name} current")

        _setup_ax(ax, name, ylabel)

    def _rotmat_to_rpy(R9):
        """
        R9: array (T, 9) con le entry della matrice di rotazione (row-major).
        Ritorna (T, 3) -> roll, pitch, yaw  (convenzione ZYX).
        """
        R00 = R9[:, 0]
        R10 = R9[:, 3]
        R20 = R9[:, 6]
        R21 = R9[:, 7]
        R22 = R9[:, 8]

        roll  = np.arctan2(R21, R22)
        pitch = np.arctan2(-R20, np.sqrt(R21**2 + R22**2))
        yaw   = np.arctan2(R10, R00)

        return np.column_stack([roll, pitch, yaw])

    def _plot_rpy(ax, base_idx, name, ylabel="[rad]"):
        end = base_idx + 9

        if end > desired.shape[1]:
            ax.axis("off")
            return

        rpy_d = _rotmat_to_rpy(desired[:, base_idx:end])

        rpy_c = None
        if current is not None and end <= current.shape[1]:
            rpy_c = _rotmat_to_rpy(current[:, base_idx:end])

        for k, lab in enumerate(["roll", "pitch", "yaw"]):
            ax.plot(t, rpy_d[:, k], label=f"{name}_{lab} desired")

            if rpy_c is not None:
                ax.plot(t, rpy_c[:, k], "--", label=f"{name}_{lab} current")

        _setup_ax(ax, name, ylabel)

    def _plot_xyz(ax, base_idx, name, ylabel=None):
        for k, lab in enumerate(["x", "y", "z"]):
            idx = base_idx + k

            if idx >= desired.shape[1]:
                continue

            ax.plot(t, desired[:, idx], label=f"{name}_{lab} desired")

            if current is not None and idx < current.shape[1]:
                ax.plot(t, current[:, idx], "--", label=f"{name}_{lab} current")

        _setup_ax(ax, name, ylabel)

    def _plot_matrix_diag_or_entries(ax, base_idx, name):
        """
        Base rotation is stored as 9 entries.
        Plot only useful entries to keep the figure readable.
        """

        entries = [
            (0, "R00"),
            (1, "R01"),
            (3, "R10"),
            (4, "R11"),
            (8, "R22"),
        ]

        for off, lab in entries:
            idx = base_idx + off

            if idx >= desired.shape[1]:
                continue

            ax.plot(t, desired[:, idx], label=f"{name}_{lab} desired")

            if current is not None and idx < current.shape[1]:
                ax.plot(t, current[:, idx], "--", label=f"{name}_{lab} current")

        _setup_ax(ax, name, "")

    # ============================================================
    # FIGURE 1 — COM / BASE — 3 rows x 2 cols
    # ============================================================
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
    axes = axes.reshape(3, 2)

    _plot_xyz(
        axes[0, 0],
        mpc_utils._REF_COM_POS,
        "com_pos",
        ylabel="[m]",
    )

    _plot_xyz(
        axes[0, 1],
        mpc_utils._REF_COM_VEL,
        "com_vel",
        ylabel="[m/s]",
    )

    _plot_xyz(
        axes[1, 0],
        mpc_utils._REF_COM_ACC,
        "com_acc",
        ylabel="[m/s²]",
    )

    _plot_rpy(
        axes[1, 1],
        mpc_utils._REF_BASE_ROT,
        "base_rot",
        ylabel="[rad]",
    )
    _plot_xyz(
        axes[2, 0],
        mpc_utils._REF_BASE_OMG,
        "base_omega",
        ylabel="[rad/s]",
    )

    _plot_xyz(
        axes[2, 1],
        mpc_utils._REF_BASE_ALP,
        "base_alpha",
        ylabel="[rad/s²]",
    )

    fig.suptitle("WBC desired vs current — COM / base", fontsize=14)

    for ax in axes[-1, :]:
        ax.set_xlabel("WBC step")

    _save(fig, filename_com_base)

    # ============================================================
    # FIGURE 2 — WHEELS — 3 rows x 2 cols
    # ============================================================
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
    axes = axes.reshape(3, 2)

    _plot_xyz(
        axes[0, 0],
        mpc_utils._REF_LW_POS,
        "lwheel_pos",
        ylabel="[m]",
    )

    _plot_xyz(
        axes[0, 1],
        mpc_utils._REF_RW_POS,
        "rwheel_pos",
        ylabel="[m]",
    )

    _plot_xyz(
        axes[1, 0],
        mpc_utils._REF_LW_VEL,
        "lwheel_vel",
        ylabel="[m/s]",
    )

    _plot_xyz(
        axes[1, 1],
        mpc_utils._REF_RW_VEL,
        "rwheel_vel",
        ylabel="[m/s]",
    )

    _plot_xyz(
        axes[2, 0],
        mpc_utils._REF_LW_ACC,
        "lwheel_acc",
        ylabel="[m/s²]",
    )

    _plot_xyz(
        axes[2, 1],
        mpc_utils._REF_RW_ACC,
        "rwheel_acc",
        ylabel="[m/s²]",
    )

    fig.suptitle("WBC desired vs current — wheels", fontsize=14)

    for ax in axes[-1, :]:
        ax.set_xlabel("WBC step")

    _save(fig, filename_wheels)

    # ============================================================
    # FIGURE 3 — JOINTS — 4 rows x 2 cols
    # ============================================================
    q_start = mpc_utils._REF_JOINTS
    dq_start = mpc_utils._REF_JOINTS + nj
    ddq_start = mpc_utils._REF_JOINTS + 2 * nj

    fig, axes = plt.subplots(4, 2, figsize=(16, 14), sharex=True)
    axes = axes.reshape(4, 2)

    for j in range(nj):
        row = j // 2
        col = j % 2

        if row >= 4:
            break

        ax = axes[row, col]

        q_idx = q_start + j
        dq_idx = dq_start + j
        ddq_idx = ddq_start + j

        if q_idx < desired.shape[1]:
            ax.plot(t, desired[:, q_idx], label=f"q{j} desired")

            if current is not None and q_idx < current.shape[1]:
                ax.plot(t, current[:, q_idx], "--", label=f"q{j} current")

        if dq_idx < desired.shape[1]:
            ax.plot(t, desired[:, dq_idx], label=f"dq{j} desired")

            if current is not None and dq_idx < current.shape[1]:
                ax.plot(t, current[:, dq_idx], "--", label=f"dq{j} current")

        if ddq_idx < desired.shape[1]:
            ax.plot(t, desired[:, ddq_idx], label=f"ddq{j} desired")

            if current is not None and ddq_idx < current.shape[1]:
                ax.plot(t, current[:, ddq_idx], "--", label=f"ddq{j} current")

        ax.set_title(f"joint {j}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2)

    for j in range(nj, 8):
        row = j // 2
        col = j % 2

        if row < 4:
            axes[row, col].axis("off")

    fig.suptitle("WBC desired vs current — joints", fontsize=14)

    for ax in axes[-1, :]:
        ax.set_xlabel("WBC step")

    _save(fig, filename_joints)

def plot_torques_and_contacts(
    torques: list,
    contact_forces: list,
    filename_torques: str = "wbc_torques.png",
    filename_contacts: str = "wbc_contacts.png",
    out_dir: str = _DEFAULT_DIR,
    title_torques: str = "Torques",
    title_contacts: str = "Contact Forces",
    save_csv: bool = True,
):
    """
    Plot:
    1) Torques, usually 8 joints -> 2 rows
    2) Contact forces, usually 2 feet -> each with 3D force components

    Also saves CSVs inside:
        out_dir/csv_data/
    """

    _ensure_dir(out_dir)

    U = np.asarray(torques, dtype=float)
    F = np.asarray(contact_forces, dtype=float)

    if U.ndim == 1:
        U = U[:, None]

    if F.ndim == 2:
        # fallback: assumes flat contact forces, e.g. (T, 6)
        if F.shape[1] == 6:
            F = F.reshape(F.shape[0], 2, 3)
        else:
            raise ValueError(
                f"contact_forces has shape {F.shape}; expected (T, 2, 3) or (T, 6)."
            )

    if F.ndim != 3:
        raise ValueError(
            f"contact_forces must be 3D, expected (T, n_feet, 3), got {F.shape}."
        )

    # ============================================================
    # CSV SAVE
    # ============================================================
    if save_csv:
        df_tau = pd.DataFrame({"step": np.arange(U.shape[0])})

        for j in range(U.shape[1]):
            df_tau[f"tau_j{j}"] = U[:, j]

        _save_df_csv(df_tau, out_dir, filename_torques)

        df_f = pd.DataFrame({"step": np.arange(F.shape[0])})

        for foot in range(F.shape[1]):
            for k, axis in enumerate(["x", "y", "z"]):
                df_f[f"foot_{foot}_f{axis}"] = F[:, foot, k]

        _save_df_csv(df_f, out_dir, filename_contacts)

    # ============================================================
    # TORQUES FIGURE
    # ============================================================
    frames = np.arange(U.shape[0])

    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

    cmap = plt.get_cmap("tab10")

    for j in range(min(4, U.shape[1])):
        ax1.plot(
            frames,
            U[:, j],
            label=f"j{j}",
            color=cmap(j),
            linewidth=1.0,
        )

    ax1.set_title("Torques 0–3")
    ax1.set_ylabel("Torque [Nm]")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8, ncol=4)

    for j in range(4, min(8, U.shape[1])):
        ax2.plot(
            frames,
            U[:, j],
            label=f"j{j}",
            color=cmap(j),
            linewidth=1.0,
        )

    ax2.set_title("Torques 4–7")
    ax2.set_ylabel("Torque [Nm]")
    ax2.set_xlabel("Frame")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8, ncol=4)

    fig1.suptitle(title_torques)
    fig1.tight_layout()

    path1 = os.path.join(out_dir, filename_torques)
    fig1.savefig(path1, dpi=120)
    plt.close(fig1)

    print(f"[plot] saved → {path1}")

    # ============================================================
    # CONTACT FORCES FIGURE
    # ============================================================
    T = min(F.shape[0], U.shape[0])
    F_plot = F[:T]
    frames = np.arange(T)

    n_feet = F_plot.shape[1]

    fig2, axes = plt.subplots(
        n_feet,
        1,
        figsize=(11, 3 * n_feet),
        sharex=True,
    )

    if n_feet == 1:
        axes = [axes]

    for foot in range(n_feet):
        ax = axes[foot]

        fx = F_plot[:, foot, 0]
        fy = F_plot[:, foot, 1]
        fz = F_plot[:, foot, 2]

        ax.plot(frames, fx, label="Fx")
        ax.plot(frames, fy, label="Fy")
        ax.plot(frames, fz, label="Fz")

        ax.set_title(f"Contact force - foot_{foot}")
        ax.set_ylabel("Force [N]")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=3)

    axes[-1].set_xlabel("Frame")

    fig2.suptitle(title_contacts)
    fig2.tight_layout()

    path2 = os.path.join(out_dir, filename_contacts)
    fig2.savefig(path2, dpi=120)
    plt.close(fig2)

    print(f"[plot] saved → {path2}")

def plot_signal(ax, steps, df, cols, title, ylabel="torque [Nm]", cmap="tab20"):
    colors = plt.get_cmap(cmap)(np.linspace(0, 1, max(len(cols), 1)))
    for i, col in enumerate(cols):
        label = col.split("_")[-1]  # joint index
        ax.plot(steps, df[col].values, label=f"j{label}", color=colors[i], linewidth=0.8)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6, ncol=4, loc="upper right")

def plot_llc(csv_path: str, out_path: str | None = None) -> str | None:
    """
    Load csv_path and save the LLC plot.
    Returns the output path, or None if the CSV does not exist.
    Can be called programmatically from train_srbd.py.
    """
    if not os.path.exists(csv_path):
        print(f"[plot_llc] CSV not found: {csv_path}")
        return None

    print(f"  Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    steps = df["step"].values if "step" in df.columns else np.arange(len(df))

    LLC = "low_level_controller/"

    tau_ff_cols = get_cols(df, LLC + "tau_ff_")
    tau_p_cols  = get_cols(df, LLC + "tau_p_")
    tau_d_cols  = get_cols(df, LLC + "tau_d_")
    action_cols = get_cols(df, LLC + "action_")

    kp = df[LLC + "kp"].iloc[0] if (LLC + "kp") in df.columns else float("nan")
    kd = df[LLC + "kd"].iloc[0] if (LLC + "kd") in df.columns else float("nan")
    action_scale = df[LLC + "action_scale"].iloc[0] if (LLC + "action_scale") in df.columns else float("nan")

    fig, axes = plt.subplots(
        3, 2,
        figsize=(14, 10),
        sharex=True,
        gridspec_kw={"width_ratios": [1, 1]},
    )
    ax_tff = fig.add_subplot(3, 1, 1)
    plot_signal(ax_tff, steps, df, tau_ff_cols, "tau_ff  (feed-forward torque)")
    ax_tff.set_xlabel("")

    ax_tp = fig.add_subplot(3, 2, 3)
    ax_td = fig.add_subplot(3, 2, 4, sharex=ax_tp)
    tp_vals = df[tau_p_cols].values if tau_p_cols else np.array([0.0])
    td_vals = df[tau_d_cols].values if tau_d_cols else np.array([0.0])
    tp_min, tp_max = tp_vals.min(), tp_vals.max()
    td_min, td_max = td_vals.min(), td_vals.max()
    plot_signal(ax_tp, steps, df, tau_p_cols,
                f"tau_p   (Kp={kp:.3g}  |  min={tp_min:.2f}  max={tp_max:.2f})")
    plot_signal(ax_td, steps, df, tau_d_cols,
                f"tau_d   (Kd={kd:.3g}  |  min={td_min:.2f}  max={td_max:.2f})")

    ax_act = fig.add_subplot(3, 1, 3)
    plot_signal(ax_act, steps, df, action_cols, "action  (policy output)", ylabel="action []")
    ax_act.set_xlabel("step")

    for ax in axes.flat:
        ax.set_visible(False)

    fig.suptitle(
        f"Rollout — Low-Level Controller  |  action_scale={action_scale:.4g}\n{csv_path}",
        fontsize=10, y=1.01,
    )
    fig.tight_layout()

    if out_path is None:
        out_path = os.path.join(os.path.dirname(csv_path), "plots", "plot_llc.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _csv_data_dir(out_dir: str) -> str:
    path = os.path.join(out_dir, "csv_data")
    os.makedirs(path, exist_ok=True)
    return path

def _stem(filename: str) -> str:
    return os.path.splitext(os.path.basename(filename))[0]

def _save_df_csv(df: pd.DataFrame, out_dir: str, filename: str):
    csv_dir = _csv_data_dir(out_dir)
    csv_path = os.path.join(csv_dir, f"{_stem(filename)}.csv")
    df.to_csv(csv_path, index=False)
    print(f"[csv]  saved → {csv_path}")
    return csv_path

def _matrix_to_df(arr, prefix: str):
    arr = np.asarray(arr, dtype=float)

    if arr.ndim == 1:
        arr = arr[:, None]

    data = {"step": np.arange(arr.shape[0])}
    for i in range(arr.shape[1]):
        data[f"{prefix}_{i}"] = arr[:, i]

    return pd.DataFrame(data)

def _read_matrix_cols(df: pd.DataFrame, prefix: str):
    cols = sorted(
        [c for c in df.columns if c.startswith(prefix + "_")],
        key=lambda c: int(c.split("_")[-1]),
    )

    if not cols:
        return None

    return df[cols].to_numpy(dtype=float)

# ── MPC prediction plots ──────────────────────────────────────────────────────

def plot_mpc_prediction_state(
    X_pred,
    timestep: int = 0,
    filename: str = "mpc_state_t0.png",
    out_dir: str = _MPC_DIR,
):
    """Plot MPC predicted state trajectory (nx=13) at a given timestep.

    X_pred: array of shape (N+1, 13)
    State layout: [pcom(3), dpcom(3), c(3), vcz(1), θ(1), v(1), ω(1)]
    """
    _ensure_dir(out_dir)
    X = np.asarray(X_pred)
    steps = np.arange(X.shape[0])

    labels_pcom  = ["pcom_x", "pcom_y", "pcom_z"]
    labels_dpcom = ["ṗcom_x", "ṗcom_y", "ṗcom_z"]
    labels_c     = ["c_x", "c_y", "c_z"]

    fig, axes = plt.subplots(4, 3, figsize=(14, 10))
    fig.suptitle(f"MPC prediction – state  (sim step {timestep})", fontsize=13)

    def _title(label, col):
        return f"{label}\n[{col[0]:.3f} → {col[-1]:.3f}]"

    for j in range(3):
        axes[0, j].plot(steps, X[:, j])
        axes[0, j].set_title(_title(labels_pcom[j], X[:, j]))
        axes[0, j].set_xlabel("horizon step")
    for j in range(3):
        axes[1, j].plot(steps, X[:, 3 + j])
        axes[1, j].set_title(_title(labels_dpcom[j], X[:, 3 + j]))
        axes[1, j].set_xlabel("horizon step")
    for j in range(3):
        axes[2, j].plot(steps, X[:, 6 + j])
        axes[2, j].set_title(_title(labels_c[j], X[:, 6 + j]))
        axes[2, j].set_xlabel("horizon step")

    scalar_names   = ["vcz", "θ (rad)", "v (m/s)", "ω (rad/s)"]
    scalar_indices = [9, 10, 11, 12]
    for j, (name, idx) in enumerate(zip(scalar_names[:3], scalar_indices[:3])):
        axes[3, j].plot(steps, X[:, idx])
        axes[3, j].set_title(_title(name, X[:, idx]))
        axes[3, j].set_xlabel("horizon step")
    axes[3, 2].plot(steps, X[:, 11], label="v")
    axes[3, 2].plot(steps, X[:, 12], label="ω", linestyle="--")
    axes[3, 2].set_title(
        f"v / ω\n[v: {X[0,11]:.3f}→{X[-1,11]:.3f}  ω: {X[0,12]:.3f}→{X[-1,12]:.3f}]"
    )
    axes[3, 2].legend(fontsize=8)

    fig.tight_layout()
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[plot] saved → {path}")

def plot_mpc_prediction_control(
    U_pred,
    timestep: int = 0,
    filename: str = "mpc_control_t0.png",
    out_dir: str = _MPC_DIR,
):
    """Plot MPC predicted control trajectory (nu=9) at a given timestep.

    U_pred: array of shape (N, 9)
    Control layout: [a(1), acz(1), α(1), Fl(3), Fr(3)]
    """
    _ensure_dir(out_dir)
    U = np.asarray(U_pred)
    steps = np.arange(U.shape[0])

    fig, axes = plt.subplots(3, 3, figsize=(14, 8))
    fig.suptitle(f"MPC prediction – control  (sim step {timestep})", fontsize=13)

    ctrl_labels = [
        "a (lin acc)", "acz (z acc)", "α (ang acc)",
        "Fl_x", "Fl_y", "Fl_z",
        "Fr_x", "Fr_y", "Fr_z",
    ]
    for idx, (ax, lbl) in enumerate(zip(axes.flat, ctrl_labels)):
        col = U[:, idx]
        ax.plot(steps, col)
        ax.set_title(f"{lbl}\n[{col[0]:.3f} → {col[-1]:.3f}]")
        ax.set_xlabel("horizon step")

    fig.tight_layout()
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[plot] saved → {path}")

def render_mpc_prediction_video(
    mpc_dir: str,
    out_path: str | None = None,
    fps: int = 10,
) -> str | None:
    """Build a single MPC video by stacking state/control plots side-by-side."""
    state_paths = sorted(
        glob.glob(os.path.join(mpc_dir, "state_mpc_t*.png")),
        key=_extract_step_idx,
    )
    control_paths = sorted(
        glob.glob(os.path.join(mpc_dir, "control_mpc_t*.png")),
        key=_extract_step_idx,
    )

    if not state_paths or not control_paths:
        print(f"[mpc_video] Missing MPC frames in: {mpc_dir}")
        return None

    state_map = {_extract_step_idx(p): p for p in state_paths}
    control_map = {_extract_step_idx(p): p for p in control_paths}
    common_steps = sorted(set(state_map.keys()) & set(control_map.keys()))

    if not common_steps:
        print(f"[mpc_video] No matching state/control timesteps in: {mpc_dir}")
        return None

    frames = []
    for step in common_steps:
        left = plt.imread(state_map[step])
        right = plt.imread(control_map[step])

        if left.ndim == 2:
            left = np.stack([left, left, left], axis=-1)
        if right.ndim == 2:
            right = np.stack([right, right, right], axis=-1)

        h = max(left.shape[0], right.shape[0])

        def _pad_to_height(img, target_h):
            if img.shape[0] == target_h:
                return img
            pad_h = target_h - img.shape[0]
            pad = np.ones((pad_h, img.shape[1], img.shape[2]), dtype=img.dtype)
            return np.concatenate([img, pad], axis=0)

        left = _pad_to_height(left, h)
        right = _pad_to_height(right, h)
        frame = np.concatenate([left, right], axis=1)
        # imageio requires uint8; plt.imread returns float32 [0,1] for PNG
        if frame.dtype != np.uint8:
            frame = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)
        # drop alpha channel if present
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = frame[:, :, :3]
        frames.append(frame)

    os.makedirs(mpc_dir, exist_ok=True)
    if out_path is None:
        out_path = os.path.join(mpc_dir, "mpc_prediction_video.mp4")

    try:
        import imageio.v2 as imageio

        imageio.mimsave(out_path, frames, fps=fps)
        print(f"[mpc_video] saved → {out_path}")
        return out_path
    except Exception as exc:
        fallback = os.path.splitext(out_path)[0] + ".gif"
        try:
            import imageio.v2 as imageio

            imageio.mimsave(fallback, frames, fps=fps)
            print(f"[mpc_video] mp4 failed ({exc}); saved gif → {fallback}")
            return fallback
        except Exception as exc2:
            print(f"[mpc_video] failed to save video: {exc2}")
            return None


# ── 1. Reward ─────────────────────────────────────────────────────────────────
def plot_rewards(
    rewards: np.ndarray,
    filename: str = "rewards.png",
    out_dir: str = _DEFAULT_DIR,
    title: str = "Reward",
):
    """Plot per-step reward and cumulative reward on the same figure.

    Parameters
    ----------
    rewards  : 1-D array of per-step scalar rewards.
    filename : output filename (saved inside out_dir).
    out_dir  : directory where the figure is saved.
    title    : figure suptitle.
    """
    _ensure_dir(out_dir)
    rewards = np.asarray(rewards, dtype=float)
    steps = np.arange(len(rewards))
    cumulative = np.cumsum(rewards)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(title, fontsize=13)

    ax1.plot(steps, rewards, color="steelblue", linewidth=1.2)
    ax1.set_ylabel("Reward / step")
    ax1.set_title(f"Per-step  [total={cumulative[-1]:.3f}]")
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps, cumulative, color="darkorange", linewidth=1.5)
    ax2.set_ylabel("Cumulative reward")
    ax2.set_xlabel("Step")
    ax2.set_title("Cumulative")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[plot] saved → {path}")

    csv_path = os.path.join(out_dir, os.path.splitext(filename)[0] + ".csv")
    pd.DataFrame({"step": steps, "reward": rewards, "cumulative": cumulative}).to_csv(
        csv_path, index=False
    )
    print(f"[csv]  saved → {csv_path}")


# ── 2. Commanded vs actual velocity ──────────────────────────────────────────

def plot_velocity(
    v_cmd: np.ndarray,
    v_actual: np.ndarray,
    dt: float = 1.0,
    labels_cmd: tuple = ("vx_cmd", "vy_cmd", "wz_cmd"),
    labels_actual: tuple = ("vx", "vy", "wz"),
    filename: str = "velocity.png",
    out_dir: str = _DEFAULT_DIR,
    title: str = "Commanded vs Actual Velocity",
):
    """Plot commanded velocity vs actual velocity over time.

    Parameters
    ----------
    v_cmd    : array (T, D) or (T,) of commanded velocities.
    v_actual : array (T, D) or (T,) of actual velocities.
    dt       : time step in seconds (used for x-axis).
    labels_cmd    : legend labels for commanded signals.
    labels_actual : legend labels for actual signals.
    filename : output filename.
    out_dir  : directory where the figure is saved.
    title    : figure suptitle.
    """
    _ensure_dir(out_dir)
    v_cmd    = np.asarray(v_cmd,    dtype=float)
    v_actual = np.asarray(v_actual, dtype=float)

    if v_cmd.ndim == 1:
        v_cmd    = v_cmd[:, None]
        v_actual = v_actual[:, None]
        labels_cmd    = (labels_cmd[0],)
        labels_actual = (labels_actual[0],)

    T, D = v_cmd.shape
    time = np.arange(T) * dt

    fig, axes = plt.subplots(D, 1, figsize=(10, 3 * D), sharex=True)
    if D == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=13)

    colors_cmd    = ["steelblue",  "seagreen",  "mediumpurple"]
    colors_actual = ["darkorange", "firebrick", "goldenrod"]

    for i, ax in enumerate(axes):
        lc = colors_cmd[i % len(colors_cmd)]
        la = colors_actual[i % len(colors_actual)]
        ax.plot(time, v_cmd[:, i],    color=lc, linewidth=1.0, linestyle="--",
                label=labels_cmd[i] if i < len(labels_cmd) else f"cmd_{i}")
        ax.plot(time, v_actual[:, i], color=la, linewidth=1.2,
                label=labels_actual[i] if i < len(labels_actual) else f"actual_{i}")
        ax.set_ylabel("velocity")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[plot] saved → {path}")

    csv_data = {"time": time}
    for i in range(D):
        lc = labels_cmd[i]    if i < len(labels_cmd)    else f"cmd_{i}"
        la = labels_actual[i] if i < len(labels_actual) else f"actual_{i}"
        csv_data[lc] = v_cmd[:, i]
        csv_data[la] = v_actual[:, i]
    csv_path = os.path.join(out_dir, os.path.splitext(filename)[0] + ".csv")
    pd.DataFrame(csv_data).to_csv(csv_path, index=False)
    print(f"[csv]  saved → {csv_path}")


# ── 3. XY position (top) + Z position (bottom) ───────────────────────────────

def plot_position(
    positions: np.ndarray,
    dt: float = 1.0,
    filename: str = "position.png",
    out_dir: str = _DEFAULT_DIR,
    title: str = "CoM Position",
):
    """Plot x/y position over time (top row) and z position over time (bottom).

    Parameters
    ----------
    positions : array (T, 3) — columns are [x, y, z].
    dt        : time step in seconds.
    filename  : output filename.
    out_dir   : directory where the figure is saved.
    title     : figure suptitle.
    """
    _ensure_dir(out_dir)
    positions = np.asarray(positions, dtype=float)
    assert positions.ndim == 2 and positions.shape[1] >= 3, \
        "positions must be (T, 3+)"

    T = positions.shape[0]
    time = np.arange(T) * dt
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1,
        figsize=(10, 7),
        gridspec_kw={"height_ratios": [2, 1]},
    )
    fig.suptitle(title, fontsize=13)

    # ── top: x and y over time ────────────────────────────────────────────
    ax_top.plot(time, x, color="steelblue",  linewidth=1.3, label="x")
    ax_top.plot(time, y, color="darkorange", linewidth=1.3, label="y")
    ax_top.set_ylabel("Position (m)")
    ax_top.set_title("x / y")
    ax_top.legend(fontsize=9)
    ax_top.grid(True, alpha=0.3)

    # ── bottom: z over time ───────────────────────────────────────────────
    ax_bot.plot(time, z, color="seagreen", linewidth=1.3, label="z")
    ax_bot.set_ylabel("Height (m)")
    ax_bot.set_xlabel("Time (s)")
    ax_bot.set_title("z")
    ax_bot.legend(fontsize=9)
    ax_bot.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[plot] saved → {path}")

    csv_path = os.path.join(out_dir, os.path.splitext(filename)[0] + ".csv")
    pd.DataFrame({"time": time, "x": x, "y": y, "z": z}).to_csv(csv_path, index=False)
    print(f"[csv]  saved → {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot LLC signals from rollout_info.csv and/or run demo plots."
    )
    parser.add_argument("--csv", type=str, default=None, help="Path to rollout_info.csv")
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default=os.path.join("checkpoints", "AliengoJoystickFlatTerrain"),
        help="Checkpoint directory containing rollout_info.csv",
    )
    parser.add_argument("--out", type=str, default=None, help="Output PNG path for LLC plot (default: next to CSV)")
    parser.add_argument(
        "--demo-dir",
        type=str,
        default=None,
        help="If set, generate synthetic demo plots (rewards/velocity/position) in this directory.",
    )
    args = parser.parse_args()

    # ── LLC plot ──────────────────────────────────────────────────────────
    csv_path = args.csv or os.path.join(args.ckpt_dir, "rollout_info.csv")
    plot_llc(csv_path, out_path=args.out)

    # ── Demo plots for all functions ──────────────────────────────────────
    demo_dir     = args.demo_dir or _DEFAULT_DIR
    demo_mpc_dir = os.path.join(demo_dir, "mpc_prediction")
    T   = 300
    N   = 20    # MPC horizon
    rng = np.random.default_rng(42)

    print(f"\n[demo] Generating synthetic plots in: {demo_dir}")

    # 1. Rewards
    rewards = np.clip(rng.normal(loc=0.6, scale=0.25, size=T), -1, 2)
    plot_rewards(rewards, filename="demo_rewards.png", out_dir=demo_dir, title="Demo – Reward")

    # 2. Velocity (vx, vy, wz)
    v_cmd = np.zeros((T, 3))
    v_cmd[:, 0] = 0.5
    v_cmd[T // 2:, 2] = 0.3
    v_actual = v_cmd + rng.normal(0, 0.04, (T, 3))
    plot_velocity(
        v_cmd, v_actual,
        dt=0.01,
        filename="demo_velocity.png",
        out_dir=demo_dir,
        title="Demo – Commanded vs Actual Velocity",
    )

    # 3. Position
    t_arr = np.linspace(0, 3 * np.pi, T)
    pos = np.stack([
        np.cumsum(np.full(T, 0.015)),
        0.08 * np.sin(t_arr),
        0.35 + 0.015 * np.sin(2 * t_arr),
    ], axis=1)
    plot_position(pos, dt=0.01, filename="demo_position.png", out_dir=demo_dir, title="Demo – CoM Position")

    # 4. MPC state prediction  (N+1, 13)
    h = np.linspace(0, 1, N + 1)
    X_demo = np.zeros((N + 1, 13))
    X_demo[:, 0] = 0.0 + h * 0.5          # pcom_x drift
    X_demo[:, 2] = 0.35 + 0.02 * np.sin(np.pi * h)   # pcom_z
    X_demo[:, 3] = 0.5                    # dpcom_x constant
    X_demo[:, 10] = 0.05 * np.sin(2 * np.pi * h)      # theta
    X_demo[:, 11] = 0.5 + rng.normal(0, 0.01, N + 1)  # v
    X_demo[:, 12] = rng.normal(0, 0.02, N + 1)         # omega
    plot_mpc_prediction_state(
        X_demo, timestep=0, filename="demo_mpc_state.png", out_dir=demo_mpc_dir
    )

    # 5. MPC control prediction  (N, 9)
    U_demo = np.zeros((N, 9))
    U_demo[:, 0] = rng.normal(0.0, 0.1, N)    # a
    U_demo[:, 3:6] = rng.normal(0, 5.0, (N, 3))  # F
    U_demo[:, 6:9] = rng.normal(0, 5.0, (N, 3))  # Fr
    plot_mpc_prediction_control(
        U_demo, timestep=0, filename="demo_mpc_control.png", out_dir=demo_mpc_dir
    )

def main_replot_from_csv():
    from tita import TITA_PATH, dir_path
    import mpx.utils.mpc_utils as mpc_utils
    import mujoco

    out_dir = TITA_PATH

    parser = argparse.ArgumentParser(
        description="Regenerate plots from CSV files saved in out_dir/csv_data."
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default=out_dir,
        help="Directory containing csv_data/ and where regenerated plots are saved.",
    )

    parser.add_argument(
        "--csv-dir",
        type=str,
        default=None,
        help="CSV directory. Default: out_dir/csv_data",
    )

    args = parser.parse_args()

    out_dir = args.out_dir
    csv_dir = args.csv_dir or os.path.join(out_dir, "csv_data")

    if not os.path.isdir(csv_dir):
        raise FileNotFoundError(f"CSV directory not found: {csv_dir}")

    print(f"[replot] reading CSV from: {csv_dir}")
    print(f"[replot] saving plots to:   {out_dir}")

    model = mujoco.MjModel.from_xml_path(
        dir_path + "/../data/tita/tita_world.xml"
    )
    nj = model.nv - 6

    # ------------------------------------------------------------
    # MPC input/output
    # ------------------------------------------------------------
    state_csv = os.path.join(csv_dir, "mpc_input.csv")
    control_csv = os.path.join(csv_dir, "mpc_output.csv")

    if os.path.exists(state_csv) and os.path.exists(control_csv):
        df_x = pd.read_csv(state_csv)
        df_u = pd.read_csv(control_csv)

        # Hardcoded: colonne generate da plot_mpc_state_and_output
        x_cols = [
            "pcom_x", "pcom_y", "pcom_z",
            "vcom_x", "vcom_y", "vcom_z",
            "c_world_x", "c_world_y", "c_world_z",
            "vcz",
            "theta",
            "v",
            "omega",
        ]

        u_cols = [
            "a", "acz", "alpha",
            "flx", "fly", "flz",
            "frx", "fry", "frz",
        ]

        x0 = df_x[x_cols].to_numpy(dtype=float)
        u0 = df_u[u_cols].to_numpy(dtype=float)

        x_ref_cols = [f"{c}_ref" for c in x_cols if f"{c}_ref" in df_x.columns]
        u_ref_cols = [f"{c}_ref" for c in u_cols if f"{c}_ref" in df_u.columns]

        x_ref = df_x[x_ref_cols].to_numpy(dtype=float) if len(x_ref_cols) > 0 else None
        u_ref = df_u[u_ref_cols].to_numpy(dtype=float) if len(u_ref_cols) > 0 else None

        plot_mpc_state_and_output(
            x0,
            u0,
            x_ref_list=x_ref,
            u_ref_list=u_ref,
            out_dir=out_dir,
            save_csv=False,
        )
    else:
        print(f"[replot] MPC input/output CSVs not found in: {csv_dir}")

    # ------------------------------------------------------------
    # WBC torques / contacts
    # ------------------------------------------------------------
    torques_csv = os.path.join(csv_dir, "wbc_torques.csv")
    contacts_csv = os.path.join(csv_dir, "wbc_contacts.csv")

    if os.path.exists(torques_csv) and os.path.exists(contacts_csv):
        df_tau = pd.read_csv(torques_csv)
        df_f = pd.read_csv(contacts_csv)

        # Hardcoded: colonne generate da plot_torques_and_contacts
        tau_cols = [f"tau_j{j}" for j in range(nj)]
        torques = df_tau[tau_cols].to_numpy(dtype=float)

        contact_cols = [
            "foot_0_fx", "foot_0_fy", "foot_0_fz",
            "foot_1_fx", "foot_1_fy", "foot_1_fz",
        ]

        contacts_flat = df_f[contact_cols].to_numpy(dtype=float)
        contact_forces = contacts_flat.reshape(contacts_flat.shape[0], 2, 3)

        plot_torques_and_contacts(
            torques,
            contact_forces,
            out_dir=out_dir,
            save_csv=False,
        )
    else:
        print(f"[replot] WBC torques/contacts CSVs not found in: {csv_dir}")

    # ------------------------------------------------------------
    # WBC desired
    # ------------------------------------------------------------
    desired_csv = os.path.join(csv_dir, "wbc_desired.csv")

    if os.path.exists(desired_csv):
        df_des = pd.read_csv(desired_csv)

        desired_cols = [f"desired_{i}" for i in range(len(df_des.columns)) if f"desired_{i}" in df_des.columns]
        current_cols = [f"current_{i}" for i in range(len(df_des.columns)) if f"current_{i}" in df_des.columns]

        desired = df_des[desired_cols].to_numpy(dtype=float)
        current = df_des[current_cols].to_numpy(dtype=float) if len(current_cols) > 0 else None

        plot_wbc_desired(
            desired_list=desired,
            current_list=current,
            mpc_utils=mpc_utils,
            nj=nj,
            out_dir=out_dir,
            save_csv=False,
        )
    else:
        print(f"[replot] WBC desired CSV not found in: {csv_dir}")

    print("[replot] done.")

if __name__ == "__main__":
    main_replot_from_csv()
