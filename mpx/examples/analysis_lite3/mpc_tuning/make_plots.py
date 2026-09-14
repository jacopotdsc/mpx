#!/usr/bin/env python3
"""Grafici baseline vs controller corretto (slew-rate limiter)."""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.join(os.path.dirname(__file__))
DT = 0.005
OUT = os.path.join(BASE, "plots")
os.makedirs(OUT, exist_ok=True)


def load(path):
    rows = list(csv.DictReader(open(path)))
    t = [i * DT for i in range(len(rows))]
    def c(name):
        return [float(r[name]) for r in rows]
    return t, {k: c(k) for k in rows[0].keys()}


def fig_fall_vs_fix():
    tb, b = load(os.path.join(BASE, "baseline/vx1p0.csv"))
    tf, f = load(os.path.join(BASE, "fix1_acc0p8/vx1p0.csv"))
    fig, ax = plt.subplots(2, 2, figsize=(11, 6.5))
    for a, key, lab, cmdv in [
        (ax[0, 0], "vx", "vx [m/s]", 1.0),
        (ax[0, 1], "base_z", "base z [m]", 0.31),
        (ax[1, 0], "roll", "roll [rad]", 0.0),
        (ax[1, 1], "pitch", "pitch [rad]", 0.0),
    ]:
        a.plot(tb, b[key], color="tab:red", lw=1.0, label="baseline (step)")
        a.plot(tf, f[key], color="tab:blue", lw=1.2, label="con slew-limiter")
        if key in ("vx",):
            a.axhline(cmdv, color="k", ls="--", lw=0.8, label="comando")
        a.set_ylabel(lab); a.set_xlabel("t [s]"); a.grid(alpha=0.3)
    ax[0, 0].legend(fontsize=8, loc="best")
    fig.suptitle("Lite3 SRBD MPC — comando [vx,vy,wz]=[1.0,0,0]  (baseline cade, limiter stabile)")
    fig.tight_layout()
    p = os.path.join(OUT, "vx1p0_baseline_vs_fix.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    return p


def _mean_ss(path, col, tss=2.5):
    t, d = load(path)
    i0 = int(tss / DT)
    seg = d[col][i0:]
    return sum(seg) / len(seg)


def fig_gain_tracking():
    """vx misurato a regime vs comando, con guadagno OFF (deficit ~15%) e ON (1.2)."""
    cmd_off = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    off = [_mean_ss(os.path.join(BASE, "fix1_acc0p8/vx1p0.csv"), "vx")]  # 1.0 (gain off)
    off += [_mean_ss(os.path.join(BASE, f"diag_hi/vx{('%.1f' % c).replace('.', 'p')}.csv"), "vx")
            for c in cmd_off[1:]]
    cmd_on = [1.0, 1.2]
    on = [_mean_ss(os.path.join(BASE, "fix2_gain120/vx1p0.csv"), "vx"),
          _mean_ss(os.path.join(BASE, "fix2_gain120/vx1p2.csv"), "vx")]
    fig, ax = plt.subplots(figsize=(7, 5.2))
    ax.plot([0.9, 1.6], [0.9, 1.6], "k--", lw=0.8, label="ideale (misurato = comando)")
    ax.plot(cmd_off, off, "o-", color="tab:red", label="guadagno OFF (deficit ~15%)")
    ax.plot(cmd_on, on, "s-", color="tab:blue", ms=9, label="guadagno ON (vel_ref_gain=1.2)")
    for c in (1.0, 1.2):
        ax.axvline(c, color="gray", lw=0.5, ls=":")
    ax.set_xlabel("vx comandato [m/s]"); ax.set_ylabel("vx misurato a regime [m/s]")
    ax.set_title("Compensazione del deficit di velocità (feedforward gain)")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout()
    p = os.path.join(OUT, "vx_tracking_gain.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    return p


def fig_cmd_vs_meas():
    cases = [
        ("fix2_gain120/vx1p0.csv", "[1.0, 0, 0]"),
        ("fix2_gain120/vx1p2.csv", "[1.2, 0, 0]"),
        ("fix2_gain120/vy0p4.csv", "[0, 0.4, 0]"),
        ("fix2_gain120/vx1p0_vy0p4.csv", "[1.0, 0.4, 0]"),
    ]
    fig, axs = plt.subplots(len(cases), 1, figsize=(10, 11), sharex=True)
    for ax, (rel, title) in zip(axs, cases):
        t, d = load(os.path.join(BASE, rel))
        for key, cmdkey, col in [("vx", "cmd_vx", "tab:blue"),
                                 ("vy", "cmd_vy", "tab:green"),
                                 ("wz", "cmd_wz", "tab:orange")]:
            ax.plot(t, d[key], color=col, lw=1.1, label=f"{key} misurato")
            ax.plot(t, d[cmdkey], color=col, ls="--", lw=0.9, alpha=0.7, label=f"{key} comando")
        ax.set_ylabel("v [m/s] / w [rad/s]"); ax.set_title(f"comando {title}", fontsize=9)
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=7, ncol=3, loc="upper right")
    axs[-1].set_xlabel("t [s]")
    fig.suptitle("Lite3 SRBD MPC con slew-limiter — comandato vs misurato (comandi a step)")
    fig.tight_layout()
    p = os.path.join(OUT, "cmd_vs_measured_fix.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    return p


def fig_diag():
    """Diagnostica: coppie e contatti (pianificati vs reali) per vx=1.0 col fix."""
    t, d = load(os.path.join(BASE, "fix1_acc0p8/vx1p0.csv"))
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for j in range(12):
        ax[0].plot(t, d[f"tau{j}"], lw=0.5)
    ax[0].axhline(30, color="k", ls="--", lw=0.8); ax[0].axhline(-30, color="k", ls="--", lw=0.8)
    ax[0].set_ylabel("coppie giunti [Nm]"); ax[0].set_title("Coppie (12 giunti) vs limiti +-30 Nm", fontsize=9)
    ax[0].grid(alpha=0.3)
    for i, foot in enumerate(["FL", "FR", "HL", "HR"]):
        ax[1].plot(t, [v + i * 1.2 for v in d[f"{foot}_planned"]], color="tab:gray", lw=0.8)
        ax[1].plot(t, [v + i * 1.2 for v in d[f"{foot}_contact"]], color="tab:red", lw=0.8, alpha=0.7)
    ax[1].set_ylabel("contatti (grigio=pianif, rosso=reale)"); ax[1].set_xlabel("t [s]")
    ax[1].set_yticks([i * 1.2 + 0.5 for i in range(4)]); ax[1].set_yticklabels(["FL", "FR", "HL", "HR"])
    ax[1].set_title("Contatti pianificati vs reali (trot)", fontsize=9); ax[1].grid(alpha=0.3)
    fig.suptitle("Lite3 SRBD MPC con slew-limiter — diagnostica vx=1.0")
    fig.tight_layout()
    p = os.path.join(OUT, "diag_vx1p0_fix.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    return p


if __name__ == "__main__":
    for fn in (fig_fall_vs_fix, fig_gain_tracking, fig_cmd_vs_meas, fig_diag):
        print("scritto", fn())
