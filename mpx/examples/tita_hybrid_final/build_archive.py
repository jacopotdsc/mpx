"""Build a per-episode, paired MPC/residual archive for final Tita policies.

Run from any directory with the mjpl interpreter:
  python build_archive.py --simulate
  python build_archive.py --render
  python build_archive.py --simulate --render

All command/spawn pairs share the same reset keys across MPC and all policies.
Simulation writes every episode to rollouts/, data_csv/, plots/, and the index
files before moving to the next terrain so an interrupted run can be inspected.
"""
from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jp
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

EXAMPLES = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(EXAMPLES))
from mujoco_playground import registry  # noqa: E402

SECONDS = 10.0
SPAWNS_PER_COMMAND = 5
SEED = 20261019
COMMANDS = [(0.5, 0.0), (1.0, 0.0), (1.5, 0.0), (2.0, 0.0),
            (1.0, 0.5), (1.0, -0.8), (2.0, 0.8), (-1.0, 0.0)]
SLOPE_COMMANDS = [(0.25, 0.0), (0.5, 0.0), (0.75, 0.0), (1.0, 0.0),
                  (-0.25, 0.0), (-0.5, 0.0), (-1.0, 0.0), (0.0, 0.0)]
TERRAINS = {
    "flat": ("TitaJoystickFlatTerrain", {}),
    "rough": ("TitaJoystickRoughTerrain", {}),
    "perlin": ("TitaJoystickPerlinTerrain", {}),
    "slope10": ("TitaJoystickFlatTerrain", {
        "terrain_config.scene": "tilt10", "terrain_config.tilt_deg": 10.0}),
}
POLICIES = {
    "RA3": {"load": "20261019_003649", "plan": "tita_hybrid_plan"},
    "RS4": {"load": "20261018_221525", "plan": "tita_hybrid_plan"},
    "RA4": {"load": "20261019_065047", "plan": "tita_hybrid_plan_v2"},
    "RA5": {"load": "20261019_084903", "plan": "tita_hybrid_plan_v2"},
}
# Re-run only the policy/terrain combinations that can add useful comparisons
# beyond the saved aggregate results. Flat is already collected in full.
POLICIES_BY_TERRAIN = {
    "flat": ("RA3", "RS4", "RA4", "RA5"),
    "rough": ("RA4",),
    "perlin": ("RA4",),
    "slope10": ("RA5",),
}
POLICY_CKPT_ROOT = EXAMPLES / "checkpoints"
FPS, WIDTH, HEIGHT = 25, 640, 480


def commands_for(terrain: str):
    return SLOPE_COMMANDS if terrain == "slope10" else COMMANDS


def command_id(vx: float, wz: float) -> str:
    return f"vx{vx:+.1f}_wz{wz:+.1f}".replace("+", "p").replace("-", "m").replace(".", "p")


def _overrides(extra: dict, residual: bool):
    return {"command_config.mode": "joystick", "episode_length": 1000,
            "randomize_reset": 1.0, "residual_config.enabled": residual, **extra}


def _rpy(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    roll = jp.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = jp.arcsin(jp.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = jp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def _make_runner(env, policy, zero: bool, batch_size: int):
    n = batch_size
    steps = int(round(SECONDS / float(env.dt)))

    @jax.jit
    def run(reset_keys, cmds):
        state = jax.vmap(env.reset)(reset_keys)
        yaw0 = _rpy(state.data.qpos[:, 3:7])[2]

        def body(carry, _):
            state, alive, rng, yaw_command = carry
            state.info["target_command"] = cmds
            if "steps_until_next_cmd" in state.info:
                state.info["steps_until_next_cmd"] = jp.full((n,), 10**6, dtype=jp.int32)
            # Match eval_robust.py: integrate the actually ramped command.
            yaw_command = yaw_command + state.info["command"][:, 1] * env.dt
            rng, action_key = jax.random.split(rng)
            if zero:
                action = jp.zeros((n, env.action_size))
            else:
                action, _ = jax.vmap(policy)(state.obs, jax.random.split(action_key, n))
            state = jax.vmap(env.step)(state, action)
            roll, pitch, yaw = _rpy(state.data.qpos[:, 3:7])
            drift = jp.arctan2(jp.sin(yaw - yaw0 - yaw_command),
                               jp.cos(yaw - yaw0 - yaw_command))
            vx = jax.vmap(env.get_local_linvel)(state.data)[:, 0]
            wz = jax.vmap(env.get_gyro)(state.data)[:, 2]
            command = state.info["command"]
            tn = jp.linalg.norm(state.info["tau_nominal"], axis=-1)
            tr = jp.linalg.norm(state.info["tau_residual"], axis=-1)
            ok = alive > 0
            values = (state.data.qpos, state.done, alive, command[:, 0], command[:, 1],
                      vx, wz, roll, pitch, yaw, drift,
                      jp.where(ok, tr / jp.maximum(tn, 1e-6), 0.0))
            alive = alive * (1.0 - state.done)
            return (state, alive, rng, yaw_command), values

        _, values = jax.lax.scan(
            body, (state, jp.ones(n), jax.random.PRNGKey(1), jp.zeros(n)),
            None, length=steps)
        return values

    return run


def _to_numpy(values):
    return tuple(np.asarray(v) for v in values)


def _write_pair_csv(path: Path, mpc: dict, res: dict, dt: float):
    n = len(mpc["cmd_vx"])
    fields = ["time_s", "cmd_vx", "cmd_wz"]
    for controller in ("mpc", "residual"):
        fields += [f"{controller}_{key}" for key in
                   ("alive", "done", "vx_mps", "wz_rad_s", "roll_rad", "pitch_rad",
                    "yaw_rad", "heading_drift_rad", "residual_ratio")]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i in range(n):
            row = {"time_s": f"{(i + 1) * dt:.6f}",
                   "cmd_vx": f"{mpc['cmd_vx'][i]:.6f}",
                   "cmd_wz": f"{mpc['cmd_wz'][i]:.6f}"}
            for controller, data in (("mpc", mpc), ("residual", res)):
                for key in ("alive", "done", "vx_mps", "wz_rad_s", "roll_rad", "pitch_rad",
                            "yaw_rad", "heading_drift_rad", "residual_ratio"):
                    value = data[key][i]
                    row[f"{controller}_{key}"] = (int(value) if key in ("alive", "done") else
                                                   (f"{float(value):.7g}" if np.isfinite(value) else "nan"))
            writer.writerow(row)


def _episode_metrics(data: dict, dt: float):
    alive = np.asarray(data["alive"], dtype=bool)
    valid = alive & np.isfinite(data["vx_mps"]) & np.isfinite(data["wz_rad_s"])
    if not valid.any():
        valid = np.isfinite(data["vx_mps"]) & np.isfinite(data["wz_rad_s"])
    evx = np.abs(data["vx_mps"] - data["cmd_vx"])
    ewz = np.abs(data["wz_rad_s"] - data["cmd_wz"])
    safe = lambda x: float(np.nanmean(np.asarray(x)[valid])) if valid.any() else float("nan")
    fall_indices = np.flatnonzero(np.asarray(data["done"], dtype=bool))
    fell = bool(len(fall_indices))
    fall_time = float((fall_indices[0] + 1) * dt) if fell else None
    drift = float(abs(data["heading_drift_rad"][np.flatnonzero(valid)[-1]])) if valid.any() else float("nan")
    mean_vx, mean_wz = safe(evx), safe(ewz)
    mean_roll, mean_pitch = safe(np.abs(data["roll_rad"])), safe(np.abs(data["pitch_rad"]))
    flags = []
    if not fell and mean_vx > 0.5: flags.append("linear_tracking_review")
    if not fell and mean_wz > 0.5: flags.append("yaw_tracking_review")
    if not fell and drift > math.pi / 2: flags.append("heading_drift_review")
    if not fell and (mean_roll > 0.2 or mean_pitch > 0.2): flags.append("body_tilt_review")
    return {"fell": fell, "fall_time_s": fall_time, "survived_s": float(alive.sum() * dt),
            "mean_abs_vx_err_mps": mean_vx, "mean_abs_wz_err_rad_s": mean_wz,
            "mean_abs_roll_rad": mean_roll, "mean_abs_pitch_rad": mean_pitch,
            "final_heading_drift_rad": drift,
            "mean_residual_torque_ratio": safe(data["residual_ratio"]),
            "quality_flags": flags}


def _tracking_plot(path: Path, meta: dict, mpc: dict, res: dict, dt: float):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    t = np.arange(len(mpc["cmd_vx"])) * dt + dt
    for ax, signal, label, unit in (
        (axes[0], "vx_mps", "vx", "m/s"),
        (axes[1], "wz_rad_s", "wz", "rad/s"),
    ):
        cmd_name = "cmd_vx" if signal == "vx_mps" else "cmd_wz"
        target = mpc[cmd_name]
        ax.plot(t, target, color="gray", linestyle="--", linewidth=1.5, label="command")
        for dataset, tag, color in ((mpc, "MPC measured", "tab:blue"),
                                    (res, "MPC + residual measured", "tab:orange")):
            alive = np.asarray(dataset["alive"], dtype=bool)
            measured = np.where(alive, dataset[signal], np.nan)
            ax.plot(t, measured, color=color, label=tag)
            valid = alive & np.isfinite(dataset[signal])
            err = dataset[signal][valid] - target[valid]
            rmse = float(np.sqrt(np.mean(err ** 2))) if len(err) else float("nan")
            me = float(np.mean(err)) if len(err) else float("nan")
            std = float(np.std(err)) if len(err) else float("nan")
            if tag == "MPC measured":
                mpc_stats = (rmse, me, std)
            else:
                res_stats = (rmse, me, std)
        ax.set_ylabel(f"{label} [{unit}]")
        ax.set_title(
            f"{label} tracking - RMSE MPC {mpc_stats[0]:.3f} {unit} | "
            f"ME {mpc_stats[1]:+.3f} ± std {mpc_stats[2]:.3f}; "
            f"residual RMSE {res_stats[0]:.3f} {unit} | "
            f"ME {res_stats[1]:+.3f} ± std {res_stats[2]:.3f}"
        )
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle(f"{meta['terrain']} | {meta['policy']} | {meta['command_id']} | spawn {meta['spawn_id']:02d}")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _save_json(path: Path, obj):
    def clean(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        return value
    path.write_text(json.dumps(clean(obj), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _cache_path(terrain: str, controller: str, cmd_i: int, seed: int) -> Path:
    path = OUT / "cache" / f"{terrain}__{controller}__cmd{cmd_i:02d}__seed{seed}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _load_cached_command(path: Path):
    with np.load(path) as cache:
        return {key: cache[key] for key in cache.files}


def _cache_command(path: Path, batch: dict):
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **batch)
    tmp.replace(path)


def simulate(terrains: list[str], seed: int):
    rollouts = OUT / "rollouts"
    data_csv = OUT / "data_csv"
    plots = OUT / "plots"
    for p in (rollouts, data_csv, plots): p.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "manifest.json"
    summary_path = data_csv / "episode_metrics.csv"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        prior_seed = manifest.get("protocol", {}).get("seed")
        if prior_seed != seed:
            raise ValueError(f"Archive already uses seed={prior_seed}; use the same seed or move it aside")
        manifest["protocol"]["jax_backend"] = jax.default_backend()
        manifest["protocol"]["devices"] = [str(device) for device in jax.devices()]
        selected = set(terrains)
        manifest["clips"] = [c for c in manifest.get("clips", []) if c["terrain"] not in selected]
        all_rows = []
        if summary_path.exists():
            with summary_path.open(newline="", encoding="utf-8") as f:
                all_rows = [row for row in csv.DictReader(f) if row["terrain"] not in selected]
    else:
        all_rows = []
        manifest = {"archive": OUT.name, "protocol": {"seconds": SECONDS,
        "spawns_per_command": SPAWNS_PER_COMMAND, "seed": seed,
        "reset_keys": "jax.random.split(jax.random.PRNGKey(seed), n), reused unchanged for every controller",
        "commands": {t: [list(c) for c in commands_for(t)] for t in TERRAINS},
        "jax_backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()]}, "clips": []}

    import train_srbd
    for terrain in terrains:
        env_name, extra = TERRAINS[terrain]
        commands = commands_for(terrain)
        n = len(commands) * SPAWNS_PER_COMMAND
        reset_keys = jax.random.split(jax.random.PRNGKey(seed), n)
        key_words = np.asarray(reset_keys)

        mpc_env = registry.load(env_name, config_overrides=_overrides(extra, False))
        mpc_runner = _make_runner(mpc_env, None, True, SPAWNS_PER_COMMAND)
        mpc_by_command = []
        for cmd_i, (vx, wz) in enumerate(commands):
            cache_path = _cache_path(terrain, "mpc", cmd_i, seed)
            if cache_path.exists():
                batch = _load_cached_command(cache_path)
                print(f"[{terrain}] MPC command {cmd_i + 1}/{len(commands)} cache", flush=True)
            else:
                cmd_keys = reset_keys[cmd_i * SPAWNS_PER_COMMAND:(cmd_i + 1) * SPAWNS_PER_COMMAND]
                cmd_batch = jp.asarray([[vx, wz]] * SPAWNS_PER_COMMAND, dtype=jp.float64)
                batch = _unpack(_to_numpy(mpc_runner(cmd_keys, cmd_batch)))
                _cache_command(cache_path, batch)
                print(f"[{terrain}] MPC command {cmd_i + 1}/{len(commands)} simulated", flush=True)
            mpc_by_command.append(batch)
        dt = float(mpc_env.dt)
        del mpc_env

        for policy_name in POLICIES_BY_TERRAIN[terrain]:
            spec = POLICIES[policy_name]
            env = registry.load(env_name, config_overrides=_overrides(extra, True))
            policy = train_srbd.build_eval_inference_fn(
                env, spec["load"], env_name="TitaJoystickFlatTerrain", ckpt_root=str(POLICY_CKPT_ROOT))
            if abs(float(env.dt) - dt) > 1e-12:
                raise RuntimeError(f"dt mismatch for {policy_name}/{terrain}")
            res_runner = _make_runner(env, policy, False, SPAWNS_PER_COMMAND)
            res_by_command = []
            for cmd_i, (vx, wz) in enumerate(commands):
                cache_path = _cache_path(terrain, policy_name, cmd_i, seed)
                if cache_path.exists():
                    batch = _load_cached_command(cache_path)
                    print(f"[{terrain}] {policy_name} command {cmd_i + 1}/{len(commands)} cache", flush=True)
                else:
                    cmd_keys = reset_keys[cmd_i * SPAWNS_PER_COMMAND:(cmd_i + 1) * SPAWNS_PER_COMMAND]
                    cmd_batch = jp.asarray([[vx, wz]] * SPAWNS_PER_COMMAND, dtype=jp.float64)
                    batch = _unpack(_to_numpy(res_runner(cmd_keys, cmd_batch)))
                    _cache_command(cache_path, batch)
                    print(f"[{terrain}] {policy_name} command {cmd_i + 1}/{len(commands)} simulated", flush=True)
                res_by_command.append(batch)
            del env, policy

            for cmd_i, (vx, wz) in enumerate(commands):
                for spawn in range(SPAWNS_PER_COMMAND):
                    episode_i = cmd_i * SPAWNS_PER_COMMAND + spawn
                    cid = f"{terrain}__{policy_name}__{command_id(vx,wz)}__spawn{spawn:02d}"
                    mpc = _episode(mpc_by_command[cmd_i], spawn)
                    res = _episode(res_by_command[cmd_i], spawn)
                    mpc_metrics = _episode_metrics(mpc, dt)
                    res_metrics = _episode_metrics(res, dt)
                    mpc_fell, res_fell = mpc_metrics["fell"], res_metrics["fell"]
                    case = ("mpc_fell_residual_survived" if mpc_fell and not res_fell else
                            "both_fell" if mpc_fell and res_fell else
                            "residual_fell_mpc_survived" if res_fell else "both_survived")
                    meta = {"id": cid, "terrain": terrain, "policy": policy_name,
                            "checkpoint": spec["load"], "plan": spec["plan"],
                            "command": {"vx": vx, "wz": wz}, "command_id": command_id(vx, wz),
                            "command_index": cmd_i, "spawn_id": spawn,
                            "reset_key": [int(x) for x in key_words[episode_i]],
                            "dt": dt, "seconds": SECONDS, "case": case,
                            "mpc": mpc_metrics, "residual": res_metrics,
                            "rollout": f"rollouts/{terrain}/{cid}.npz",
                            "csv": f"data_csv/{cid}.csv", "plot": f"plots/{cid}.png"}
                    (rollouts / terrain).mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(OUT / meta["rollout"],
                        qpos_mpc=mpc["qpos"], done_mpc=mpc["done"], alive_mpc=mpc["alive"],
                        cmd_vx=mpc["cmd_vx"], cmd_wz=mpc["cmd_wz"], vx_mpc=mpc["vx_mps"],
                        wz_mpc=mpc["wz_rad_s"], roll_mpc=mpc["roll_rad"], pitch_mpc=mpc["pitch_rad"],
                        yaw_mpc=mpc["yaw_rad"], heading_drift_mpc=mpc["heading_drift_rad"],
                        residual_ratio_mpc=mpc["residual_ratio"], qpos_residual=res["qpos"],
                        done_residual=res["done"], alive_residual=res["alive"], vx_residual=res["vx_mps"],
                        wz_residual=res["wz_rad_s"], roll_residual=res["roll_rad"],
                        pitch_residual=res["pitch_rad"], yaw_residual=res["yaw_rad"],
                        heading_drift_residual=res["heading_drift_rad"],
                        residual_ratio=res["residual_ratio"], reset_key=key_words[episode_i])
                    _write_pair_csv(OUT / meta["csv"], mpc, res, dt)
                    _tracking_plot(OUT / meta["plot"], meta, mpc, res, dt)
                    all_rows.append(_summary_row(meta))
                    manifest["clips"].append(meta)
            _write_summary(summary_path, all_rows)
            _save_json(OUT / "manifest.json", manifest)
            print(f"[{terrain}] {policy_name}: saved {n} paired episodes", flush=True)
    _write_summary(summary_path, all_rows)
    _save_json(OUT / "manifest.json", manifest)
    print(f"Saved {len(all_rows)} policy episodes under {OUT}", flush=True)


def _unpack(values):
    keys = ("qpos", "done", "alive", "cmd_vx", "cmd_wz", "vx_mps", "wz_rad_s",
            "roll_rad", "pitch_rad", "yaw_rad", "heading_drift_rad", "residual_ratio")
    return {k: v for k, v in zip(keys, values)}


def _episode(batch, i):
    return {k: np.asarray(v[:, i]) for k, v in batch.items()}


def _summary_row(meta):
    row = {"id": meta["id"], "terrain": meta["terrain"], "policy": meta["policy"],
           "checkpoint": meta["checkpoint"], "command_id": meta["command_id"],
           "cmd_vx_mps": meta["command"]["vx"], "cmd_wz_rad_s": meta["command"]["wz"],
           "spawn_id": meta["spawn_id"], "reset_key": ":".join(map(str, meta["reset_key"])),
           "case": meta["case"], "rollout": meta["rollout"], "csv": meta["csv"], "plot": meta["plot"]}
    for controller in ("mpc", "residual"):
        for key, value in meta[controller].items():
            if key == "quality_flags":
                row[f"{controller}_quality_flags"] = ";".join(value)
            else:
                row[f"{controller}_{key}"] = value
    return row


def _write_summary(path: Path, rows: list[dict]):
    if not rows: return
    cols = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _render_track(renderer, model, qpos, done, dt, label, colour, cam_cfg):
    data = mujoco.MjData(model)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.distance, cam.elevation, cam.azimuth = cam_cfg
    stride = max(1, int(round(1.0 / (FPS * dt))))
    fell_at = int(np.argmax(done > 0)) if np.any(done) else None
    for k in range(0, len(qpos), stride):
        kk = k if fell_at is None else min(k, fell_at)
        data.qpos[:] = qpos[kk]
        mujoco.mj_forward(model, data)
        cam.lookat[:] = data.qpos[:3]
        renderer.update_scene(data, camera=cam)
        img = renderer.render()
        from PIL import Image, ImageDraw, ImageFont
        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, WIDTH, 22], fill=(0, 0, 0))
        draw.text((6, 5), label, fill=colour, font=ImageFont.load_default())
        draw.text((WIDTH - 70, 5), f"t={k * dt:4.1f}s", fill=(255, 255, 255), font=ImageFont.load_default())
        if fell_at is not None and k >= fell_at:
            draw.rectangle([WIDTH // 2 - 70, HEIGHT // 2 - 18, WIDTH // 2 + 70, HEIGHT // 2 + 18], fill=(200, 0, 0))
            draw.text((WIDTH // 2 - 55, HEIGHT // 2 - 6), f"CADUTO a {fell_at * dt:.1f}s", fill=(255, 255, 255), font=ImageFont.load_default())
        yield np.asarray(img)


def render_candidates(terrains: list[str], cam_cfg=(6.5, -25.0, 135.0)):
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    (OUT / "videos").mkdir(exist_ok=True)
    index_rows = []
    by_group = {}
    for clip in manifest["clips"]:
        if clip["terrain"] not in terrains:
            continue
        residual = clip["residual"]
        include = (clip["case"] == "mpc_fell_residual_survived" or
                   bool(residual["quality_flags"]) or
                   (clip["terrain"] == "flat" and clip["command"]["vx"] == 1.0 and clip["command"]["wz"] == 0.0))
        if include:
            by_group.setdefault((clip["terrain"], clip["policy"]), []).append(clip)
    for (terrain, policy), clips in sorted(by_group.items()):
        env_name, extra = TERRAINS[terrain]
        model = registry.load(env_name, config_overrides=_overrides(extra, False)).mj_model
        renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
        outpath = OUT / "videos" / f"{terrain}__{policy}_candidates.mp4"
        writer = imageio.get_writer(str(outpath), fps=FPS, codec="libx264",
                                    quality=8, macro_block_size=None)
        for clip in clips:
            rollout_path = OUT / clip["rollout"]
            with np.load(rollout_path) as r:
                left = _render_track(renderer, model, r["qpos_mpc"], r["done_mpc"], clip["dt"],
                                     f"MPC | {clip['id']}", (255, 200, 0), cam_cfg)
                right = _render_track(renderer, model, r["qpos_residual"], r["done_residual"], clip["dt"],
                                      f"{policy} | {clip['id']}", (0, 255, 120), cam_cfg)
            title = np.zeros((HEIGHT, 2 * WIDTH, 3), dtype=np.uint8)
            title[:] = 20
            from PIL import Image, ImageDraw, ImageFont
            title_image = Image.fromarray(title)
            draw = ImageDraw.Draw(title_image)
            text = f"{clip['id']} | caso: {clip['case']}\n" \
                   f"vx={clip['command']['vx']:+.2f} m/s, wz={clip['command']['wz']:+.2f} rad/s | " \
                   f"flag: {', '.join(clip['residual']['quality_flags']) or 'nessuno'}"
            draw.multiline_text((40, HEIGHT // 2 - 20), text, fill="white", font=ImageFont.load_default())
            title_frame = np.asarray(title_image)
            for _ in range(int(1.5 * FPS)):
                writer.append_data(title_frame)
            for a, b in zip(left, right):
                writer.append_data(np.concatenate((a, b), axis=1))
            index_rows.append({"id": clip["id"], "terrain": terrain, "policy": policy,
                               "case": clip["case"], "quality_flags": ";".join(clip["residual"]["quality_flags"])})
            print(f"[render] {clip['id']}", flush=True)
        renderer.close()
        writer.close()
        print(f"[render] {outpath} ({len(clips)} episodi)", flush=True)
    with (OUT / "videos" / "video_index.csv").open("w", newline="", encoding="utf-8") as f:
        cols = ["id", "terrain", "policy", "case", "quality_flags"]
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(index_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--terrains", default=",".join(TERRAINS),
                        help="comma-separated subset: flat,rough,perlin,slope10")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--distance", type=float, default=6.5)
    parser.add_argument("--elevation", type=float, default=-25.0)
    parser.add_argument("--azimuth", type=float, default=135.0)
    args = parser.parse_args()
    if not (args.simulate or args.render):
        parser.error("use --simulate and/or --render")
    terrains = [x.strip() for x in args.terrains.split(",") if x.strip()]
    unknown = sorted(set(terrains) - set(TERRAINS))
    if unknown: parser.error(f"unknown terrains: {unknown}")
    if args.simulate: simulate(terrains, args.seed)
    if args.render: render_candidates(terrains, (args.distance, args.elevation, args.azimuth))


if __name__ == "__main__":
    main()
