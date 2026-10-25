"""Export velocity tracking CSVs and plots for the saved Tita video rollouts.

Run from anywhere with:
    python export_video_metrics.py

The script reads both tita_hybrid_plan/videos/rollouts and
tita_hybrid_plan_v2/videos/rollouts. It does not rerun the simulations.
"""
from __future__ import annotations

import csv
import json
import sys
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

EXAMPLES = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLES))
from mujoco_playground import registry  # noqa: E402

PLANS = {
    "tita_hybrid_plan": EXAMPLES / "tita_hybrid_plan",
    "tita_hybrid_plan_v2": EXAMPLES / "tita_hybrid_plan_v2",
}


def _model_for(env_name: str, extra: dict):
    overrides = {"command_config.mode": "joystick", "episode_length": 1000,
                 "randomize_reset": 1.0, "residual_config.enabled": False, **extra}
    return registry.load(env_name, config_overrides=overrides).mj_model


def _velocity(model, qpos: np.ndarray, dt: float):
    """Differentiate generalized positions, then express free-root velocity locally."""
    free_joints = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if not len(free_joints):
        raise ValueError("Rollout model has no free root joint")
    qadr = int(model.jnt_qposadr[free_joints[0]])
    vadr = int(model.jnt_dofadr[free_joints[0]])
    qvel = np.empty((len(qpos) - 1, model.nv), dtype=np.float64)
    for i in range(len(qvel)):
        mujoco.mj_differentiatePos(model, qvel[i], dt, qpos[i], qpos[i + 1])
    # The free joint's translational velocity is world aligned. Rotate it into
    # the root frame, matching the local linear velocity used by compare.py.
    local = np.empty((len(qvel), 3), dtype=np.float64)
    for i, v in enumerate(qvel):
        quat = qpos[i, qadr + 3:qadr + 7]
        inv = np.empty(4, dtype=np.float64)
        mujoco.mju_negQuat(inv, quat)
        mujoco.mju_rotVecQuat(local[i], v[vadr:vadr + 3], inv)
    # Free-joint dofs are ordered linear x/y/z, angular x/y/z.
    # Heading/yaw rate is the angular z component (index +5), not roll (+3).
    wz = qvel[:, vadr + 5]
    return local, wz


def _rmse(measured, target, valid):
    error = measured[valid] - target[valid]
    return float(np.sqrt(np.mean(error ** 2))) if error.size else float("nan")


def _plot(path: Path, clip: dict, time_s, target_vx, vx_mpc, vx_res, target_wz,
          wz_mpc, wz_res, valid_mpc, valid_res, plan_name):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for axis, target, baseline, residual, name, unit, vm, vr in (
        (axes[0], target_vx, vx_mpc, vx_res, "vx", "m/s", valid_mpc, valid_res),
        (axes[1], target_wz, wz_mpc, wz_res, "wz", "rad/s", valid_mpc, valid_res),
    ):
        axis.plot(time_s, target, color="gray", linestyle="--", linewidth=1.5, label="command")
        axis.plot(time_s, np.where(vm, baseline, np.nan), color="tab:blue", label="MPC measured")
        axis.plot(time_s, np.where(vr, residual, np.nan), color="tab:orange", label="MPC + residual measured")
        e_mpc = baseline[vm] - target[vm]
        e_res = residual[vr] - target[vr]
        rmse_mpc = float(np.sqrt(np.mean(e_mpc ** 2))) if e_mpc.size else float("nan")
        rmse_res = float(np.sqrt(np.mean(e_res ** 2))) if e_res.size else float("nan")
        me_mpc = float(np.mean(e_mpc)) if e_mpc.size else float("nan")
        me_res = float(np.mean(e_res)) if e_res.size else float("nan")
        std_mpc = float(np.std(e_mpc)) if e_mpc.size else float("nan")
        std_res = float(np.std(e_res)) if e_res.size else float("nan")
        axis.set_ylabel(f"{name} [{unit}]")
        axis.set_title(
            f"{name} tracking - RMSE MPC {rmse_mpc:.3f} {unit} | "
            f"ME {me_mpc:+.3f} ± std {std_mpc:.3f} {unit}; "
            f"MPC + residual RMSE {rmse_res:.3f} {unit} | "
            f"ME {me_res:+.3f} ± std {std_res:.3f} {unit}"
        )
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best")
    axes[-1].set_xlabel("Time [s]")
    cmd = clip["command"]
    fig.suptitle(f"{plan_name} | {clip['terrain']} | command vx={cmd['vx']:+.1f} m/s, wz={cmd['wz']:+.1f} rad/s")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def export(plan_name: str, folder: Path, output_dir: Path | None = None):
    videos = output_dir if output_dir else folder / "videos"
    manifest = json.loads((videos / "clips.json").read_text(encoding="utf-8"))
    data_dir = videos / "data_csv"
    plots_dir = videos / "plots"
    data_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)
    models = {}
    summaries = []

    for clip in manifest["clips"]:
        model_key = (clip["env_name"], json.dumps(clip["extra"], sort_keys=True))
        if model_key not in models:
            models[model_key] = _model_for(clip["env_name"], clip["extra"])
        model = models[model_key]
        rollout_path = videos / clip["rollout"]
        with np.load(rollout_path) as rollout:
            q_mpc = rollout["qpos_mpc"]
            q_res = rollout["qpos_res"]
            done_mpc = rollout["done_mpc"].astype(bool)
            done_res = rollout["done_res"].astype(bool)
        dt = float(clip["dt"])
        vel_mpc, wz_mpc = _velocity(model, q_mpc, dt)
        vel_res, wz_res = _velocity(model, q_res, dt)
        n = min(len(q_mpc), len(q_res))
        nv = min(len(vel_mpc), len(vel_res), n)
        t = np.arange(n) * dt
        # qpos[0] is the state after the first environment step, not the reset
        # state. Velocities are between consecutive saved post-step states.
        tv = (np.arange(nv) + 1.5) * dt
        target_vx = np.full(nv, float(clip["command"]["vx"]))
        target_wz = np.full(nv, float(clip["command"]["wz"]))
        vm = np.ones(nv, dtype=bool)
        vr = np.ones(nv, dtype=bool)
        if done_mpc.any():
            vm[np.argmax(done_mpc):] = False
        if done_res.any():
            vr[np.argmax(done_res):] = False
        csv_path = data_dir / f"{clip['id']}_frames.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            qpos_cols = [f"qpos_mpc_{j}" for j in range(q_mpc.shape[1])] + \
                        [f"qpos_residual_{j}" for j in range(q_res.shape[1])]
            writer.writerow(["frame", "time_s", "velocity_time_s", "cmd_vx", "cmd_wz", "vx_mpc", "wz_mpc",
                             "vx_residual", "wz_residual", "valid_mpc", "valid_residual",
                             "done_mpc", "done_residual", *qpos_cols])
            for i in range(n):
                has_velocity = i < nv
                qpos_time = (i + 1) * dt
                velocity_time = (i + 1.5) * dt if has_velocity else ""
                writer.writerow([i, f"{qpos_time:.6f}", f"{velocity_time:.6f}" if has_velocity else "",
                                 clip["command"]["vx"], clip["command"]["wz"],
                                 f"{vel_mpc[i, 0]:.6f}" if has_velocity else "",
                                 f"{wz_mpc[i]:.6f}" if has_velocity else "",
                                 f"{vel_res[i, 0]:.6f}" if has_velocity else "",
                                 f"{wz_res[i]:.6f}" if has_velocity else "",
                                 int(vm[i]) if has_velocity else 0, int(vr[i]) if has_velocity else 0,
                                 int(done_mpc[i]), int(done_res[i]), *q_mpc[i].tolist(), *q_res[i].tolist()])
        clip["frames_csv"] = str(csv_path.relative_to(videos))
        _plot(plots_dir / f"{clip['id']}_tracking.png", clip, tv, target_vx, vel_mpc[:nv, 0],
              vel_res[:nv, 0], target_wz, wz_mpc[:nv], wz_res[:nv], vm, vr, plan_name)
        summaries.append({
            "experiment": clip["id"], "terrain": clip["terrain"], "cmd_vx_mps": clip["command"]["vx"],
            "cmd_wz_rad_s": clip["command"]["wz"], "mpc_vx_rmse_mps": _rmse(vel_mpc[:nv, 0], target_vx, vm),
            "residual_vx_rmse_mps": _rmse(vel_res[:nv, 0], target_vx, vr),
            "mpc_wz_rmse_rad_s": _rmse(wz_mpc[:nv], target_wz, vm),
            "residual_wz_rmse_rad_s": _rmse(wz_res[:nv], target_wz, vr),
            "mpc_fall_time_s": clip["mpc_fall_time_s"], "residual_fall_time_s": clip["residual_fall_time_s"],
            "frames_csv": str(csv_path.relative_to(videos)),
        })
    summary_path = data_dir / "metrics_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    (videos / "clips.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (data_dir / "metrics_summary.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[{plan_name}] exported {len(summaries)} experiments: {data_dir}, {plots_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", choices=list(PLANS))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    names = [args.plan] if args.plan else list(PLANS)
    for name in names:
        export(name, PLANS[name], args.output_dir)
