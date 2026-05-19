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
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_cols(df, prefix):
    """Return sorted list of columns that start with prefix."""
    return sorted([c for c in df.columns if c.startswith(prefix)])


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None, help="Path to rollout_info.csv")
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default=os.path.join("checkpoints", "AliengoJoystickFlatTerrain"),
        help="Checkpoint directory containing rollout_info.csv",
    )
    parser.add_argument("--out", type=str, default=None, help="Output PNG path (default: next to CSV)")
    args = parser.parse_args()

    csv_path = args.csv or os.path.join(args.ckpt_dir, "rollout_info.csv")
    plot_llc(csv_path, out_path=args.out)


if __name__ == "__main__":
    main()
