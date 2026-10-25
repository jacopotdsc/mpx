import os
os.environ.setdefault("MUJOCO_GL", "egl")
import jax; jax.config.update("jax_enable_x64", True)  # MPC/WBC need float64
"""Side-by-side videos of the evaluation experiments: left MPC-only, right
MPC + residual policy, same terrain / spawn / command.

Two steps:
  1) simulate all experiments and save them (trajectories + clips.json):
       python tita_hybrid_plan_v2/make_videos.py --plan v1 --simulate
  2) render the video from clips.json, only the clips with "keep": true
     (no simulation, a few minutes; the camera can be changed):
       python tita_hybrid_plan_v2/make_videos.py --plan v1 --render [--distance 5 --elevation -25 --azimuth 135]
  --simulate --render does both. --plan v2 uses RA4 (RA5 on the slope).

Files: <plan folder>/videos/clips.json, videos/rollouts/*.npz,
       videos/<plan folder>_mpc_vs_residual.mp4
"""
import argparse
import json
import sys
from pathlib import Path

import jax.numpy as jp
import mediapy
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

EXAMPLES = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLES))
from mujoco_playground import registry  # noqa: E402

SECONDS = 10.0
FPS = 25
W, H = 640, 480
TERRAINS = {
    "piano": ("TitaJoystickFlatTerrain", {}),
    "rough (ciottoli)": ("TitaJoystickRoughTerrain", {}),
    "perlin": ("TitaJoystickPerlinTerrain", {}),
    "pendenza 10°": ("TitaJoystickFlatTerrain",
                     {"terrain_config.scene": "tilt10", "terrain_config.tilt_deg": 10.0,
                      # World +x is uphill in scene_tilt10.xml. A fixed heading
                      # keeps every positive vx candidate pointed uphill.
                      "randomize_reset": 0.0}),
}
COMMANDS = {
    "piano": [(1.0, 0.0), (2.0, 0.8)],
    "rough (ciottoli)": [(1.0, 0.0), (2.0, 0.8)],
    "perlin": [(1.0, 0.0), (1.0, -0.8)],
    "pendenza 10°": [(0.5, 0.0), (-0.5, 0.0)],
}
CANDIDATE_COMMANDS = {
    # Ten clips for each requested terrain, each paired MPC vs residual.
    "piano": [(0.5, 0.0), (1.0, 0.0), (1.5, 0.0), (2.0, 0.0),
              (1.0, 0.5), (1.0, -0.8), (2.0, 0.8), (-1.0, 0.0),
              (0.0, 0.0), (0.5, -0.5)],
    "rough (ciottoli)": [(0.5, 0.0), (1.0, 0.0), (1.5, 0.0), (2.0, 0.0),
                         (1.0, 0.5), (1.0, -0.8), (2.0, 0.8), (-1.0, 0.0),
                         (0.0, 0.0), (0.5, 0.5)],
    "perlin": [(0.5, 0.0), (1.0, 0.0), (1.5, 0.0), (2.0, 0.0),
               (1.0, 0.5), (1.0, -0.8), (2.0, 0.8), (-1.0, 0.0),
               (0.0, 0.0), (0.5, -0.5)],
    "pendenza 10°": [(0.1, 0.0), (0.2, 0.0), (0.3, 0.0), (0.4, 0.0),
                     (0.5, 0.0), (0.6, 0.0), (0.7, 0.0), (0.8, 0.0),
                     (0.9, 0.0), (1.0, 0.0)],
}
PLANS = {
    "v1": {"folder": EXAMPLES / "tita_hybrid_plan",
           "models": {t: ("RA3", "20261019_003649") for t in TERRAINS}},
    "v2": {"folder": EXAMPLES / "tita_hybrid_plan_v2",
           "models": {"piano": ("RA4", "20261019_065047"),
                      "rough (ciottoli)": ("RA4", "20261019_065047"),
                      "perlin": ("RA4", "20261019_065047"),
                      "pendenza 10°": ("RA5", "20261019_084903")}},
}


def env_overrides(extra, residual):
    return {"command_config.mode": "joystick", "episode_length": 1000,
            "randomize_reset": 1.0, "residual_config.enabled": residual, **extra}


def rollout(env_name, extra, residual, load, commands):
    """Batched rollout of all commands; returns qpos (T, N, nq), done (T, N), dt."""
    import train_srbd  # only needed to simulate (loads the policy)
    env = registry.load(env_name, config_overrides=env_overrides(extra, residual))
    policy = None
    if residual:
        policy = train_srbd.build_eval_inference_fn(
            env, load, env_name="TitaJoystickFlatTerrain", ckpt_root=str(EXAMPLES / "checkpoints"))
    n = len(commands)
    cmds = jp.array(commands)
    steps = int(SECONDS / env.dt)

    @jax.jit
    def run(keys):
        state = jax.vmap(env.reset)(keys)

        def body(carry, _):
            state, key = carry
            state.info["target_command"] = cmds
            state.info["steps_until_next_cmd"] = jp.full((n,), 10**6, dtype=jp.int32)
            key, k = jax.random.split(key)
            if policy is None:
                action = jp.zeros((n, env.action_size))
            else:
                action, _ = jax.vmap(policy)(state.obs, jax.random.split(k, n))
            state = jax.vmap(env.step)(state, action)
            return (state, key), (state.data.qpos, state.done)

        _, (qpos, done) = jax.lax.scan(body, (state, jax.random.PRNGKey(1)), None, length=steps)
        return qpos, done

    # Same keys as the robust evaluation's first spawns: identical for both controllers.
    keys = jax.random.split(jax.random.PRNGKey(0), n)
    qpos, done = run(keys)
    return np.asarray(qpos), np.asarray(done), float(env.dt)


def fall_time(done, dt):
    # The scan records the state after each step; done[k] therefore occurs at
    # (k + 1) * dt rather than at k * dt.
    return float((np.argmax(done > 0) + 1) * dt) if done.any() else None


def simulate(plan_name, output_dir=None, candidate_set=False, slope_only=False, terrain_only=None):
    plan = PLANS[plan_name]
    vdir = Path(output_dir) if output_dir else plan["folder"] / "videos"
    (vdir / "rollouts").mkdir(parents=True, exist_ok=True)
    manifest_path = vdir / "clips.json"
    if slope_only and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        clips = [c for c in manifest["clips"] if c["terrain"] != "pendenza 10°"]
    else:
        clips = []
    for t_idx, (terrain, (env_name, extra)) in enumerate(TERRAINS.items()):
        if slope_only and terrain != "pendenza 10°":
            continue
        if terrain_only is not None and terrain != terrain_only:
            continue
        name, load = plan["models"][terrain]
        cmds = (CANDIDATE_COMMANDS if candidate_set else COMMANDS)[terrain]
        q_mpc, d_mpc, dt = rollout(env_name, extra, False, None, cmds)
        q_res, d_res, _ = rollout(env_name, extra, True, load, cmds)
        for i, (vx, wz) in enumerate(cmds):
            cid = f"{t_idx}{i}_{env_name.replace('TitaJoystick', '').replace('Terrain', '').lower()}" \
                  f"{'_tilt10' if extra else ''}_vx{vx:+.1f}_wz{wz:+.1f}"
            f = vdir / "rollouts" / f"{cid}.npz"
            np.savez_compressed(f, qpos_mpc=q_mpc[:, i], done_mpc=d_mpc[:, i],
                                qpos_res=q_res[:, i], done_res=d_res[:, i])
            clip = dict(id=cid, keep=True, terrain=terrain, env_name=env_name, extra=extra,
                        command={"vx": vx, "wz": wz}, residual_model=name, checkpoint=load,
                        run_seed=0, reset_key_index=i,
                        mpc_fell=bool(d_mpc[:, i].any()), mpc_fall_time_s=fall_time(d_mpc[:, i], dt),
                        residual_fell=bool(d_res[:, i].any()), residual_fall_time_s=fall_time(d_res[:, i], dt),
                        dt=dt, rollout=str(f.relative_to(vdir)))
            clips.append(clip)
            print(f"[{plan_name}] {cid}: MPC fell={clip['mpc_fell']} {name} fell={clip['residual_fell']}",
                  flush=True)
    if terrain_only is not None:
        fixed_zcom_only = os.environ.get("TITA_MPC_FIXED_ZCOM_REFERENCE", "0") == "1"
        flat_reference = os.environ.get("TITA_MPC_FLAT_Z_REFERENCE", "0") == "1"
        if fixed_zcom_only:
            candidate_set = f"10_{terrain_only}_zcomref_0.4m_only"
            note = ("Fixed COM-height reference ablation: z_com_ref=0.4 m absolute; "
                    "terrain/contact reference remains measured as z_contact_ref=x0[8]. "
                    "Residual policy/checkpoint unchanged; MPC and MPC+residual share "
                    "commands and reset keys.")
        elif flat_reference:
            candidate_set = f"10_{terrain_only}_flat_z_contact_0_z_com_0.4m"
            note = ("Flat-terrain reference ablation: z_contact_ref=0 and absolute "
                    "z_com_ref=0.4 m; same policy, commands and reset keys for "
                    "MPC and MPC+residual.")
        else:
            candidate_set = f"10_{terrain_only}_standard_z_reference"
            note = ("Standard reference: z_contact_ref=x0[8] and z_com_ref="
                    "z_contact_ref+0.4 m; same commands and reset keys for MPC "
                    "and MPC+residual.")
        manifest = dict(
            plan=plan["folder"].name, seconds=SECONDS, fps=FPS,
            candidate_set=candidate_set,
            note=note,
            clips=clips)
    elif slope_only and manifest_path.exists():
        manifest["clips"] = clips
        manifest["candidate_set"] = "10_per_terrain"
        manifest["note"] = "Pendio: heading fissato verso monte (+x), 10 velocità positive."
    else:
        manifest = dict(
            plan=plan["folder"].name, seconds=SECONDS, fps=FPS,
            candidate_set="10_per_terrain" if candidate_set else "legacy",
            note="Metti \"keep\": false sugli esperimenti da togliere, poi rilancia con --render.",
            clips=clips)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"[{plan_name}] saved {vdir / 'clips.json'}", flush=True)


def render_track(model, qpos, done, dt, label, colour, cam_cfg):
    data = mujoco.MjData(model)
    # Render the URDF visual meshes cleanly: hide the red collision proxies
    # (geom group 1), keep the environment and robot visual meshes (group 2),
    # and brighten the black visual material so mesh details are legible.
    black_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "black")
    if black_id >= 0:
        model.mat_rgba[black_id] = [0.40, 0.43, 0.48, 1.0]
    scene_option = mujoco.MjvOption()
    scene_option.geomgroup[1] = 0
    scene_option.geomgroup[2] = 1
    renderer = mujoco.Renderer(model, height=H, width=W)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.distance, cam.elevation, cam.azimuth = cam_cfg
    stride = int(round(1.0 / (FPS * dt)))
    fell_at = np.argmax(done > 0) if done.any() else None
    font = ImageFont.load_default()
    frames = []
    for k in range(0, len(qpos), stride):
        kk = k if fell_at is None else min(k, fell_at)
        data.qpos[:] = qpos[kk]
        mujoco.mj_forward(model, data)
        cam.lookat[:] = data.qpos[:3]
        renderer.update_scene(data, camera=cam, scene_option=scene_option)
        img = Image.fromarray(renderer.render())
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, W, 22], fill=(0, 0, 0))
        d.text((6, 5), label, fill=colour, font=font)
        d.text((W - 70, 5), f"t={(k + 1) * dt:4.1f}s", fill=(255, 255, 255), font=font)
        if fell_at is not None and k >= fell_at:
            d.rectangle([W // 2 - 70, H // 2 - 18, W // 2 + 70, H // 2 + 18], fill=(200, 0, 0))
            d.text((W // 2 - 55, H // 2 - 6), f"CADUTO a {(fell_at + 1) * dt:.1f}s", fill=(255, 255, 255), font=font)
        frames.append(np.asarray(img))
    renderer.close()
    return frames


def title_card(text, seconds=1.5):
    img = Image.new("RGB", (2 * W, H), (20, 20, 20))
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    for i, line in enumerate(text.split("\n")):
        d.text((40, H // 2 - 30 + 18 * i), line, fill=(255, 255, 255), font=font)
    return [np.asarray(img)] * int(seconds * FPS)


def render(plan_name, cam_cfg, output_dir=None):
    plan = PLANS[plan_name]
    vdir = Path(output_dir) if output_dir else plan["folder"] / "videos"
    manifest = json.loads((vdir / "clips.json").read_text())
    models = {}
    video = title_card(f"{manifest['plan']}: SINISTRA solo MPC  |  DESTRA MPC + rete residua\n"
                       f"stesso terreno, spawn e comando; {manifest['seconds']:.0f} s per esperimento", 3.0)
    kept = [c for c in manifest["clips"] if c.get("keep", True)]
    for c in kept:
        key = (c["env_name"], json.dumps(c["extra"], sort_keys=True))
        if key not in models:  # the rendering model only (no policy)
            models[key] = registry.load(c["env_name"], config_overrides=env_overrides(c["extra"], False)).mj_model
        r = np.load(vdir / c["rollout"])
        tag = f"{c['terrain']}, comando vx={c['command']['vx']:+.1f} m/s wz={c['command']['wz']:+.1f} rad/s"
        video += title_card(f"{tag}\nsinistra: solo MPC   destra: MPC + {c['residual_model']}")
        left = render_track(models[key], r["qpos_mpc"], r["done_mpc"], c["dt"], f"solo MPC | {tag}",
                            (255, 200, 0), cam_cfg)
        right = render_track(models[key], r["qpos_res"], r["done_res"], c["dt"],
                             f"MPC + {c['residual_model']} | {tag}", (0, 255, 120), cam_cfg)
        video += [np.concatenate([a, b], axis=1) for a, b in zip(left, right)]
        print(f"[{plan_name}] rendered {c['id']}", flush=True)
    out = vdir / f"{manifest['plan']}_mpc_vs_residual.mp4"
    mediapy.write_video(str(out), video, fps=FPS)
    print(f"[{plan_name}] written {out} ({len(video) / FPS:.0f} s, {len(kept)} esperimenti)", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plan", choices=list(PLANS), required=True)
    p.add_argument("--simulate", action="store_true")
    p.add_argument("--render", action="store_true")
    p.add_argument("--output-dir", type=Path, help="alternate output folder; existing videos remain untouched")
    p.add_argument("--candidates-10", action="store_true", help="generate ten command/spawn candidates per terrain")
    p.add_argument("--slope-only", action="store_true", help="replace only the ten uphill slope rollouts in the existing manifest")
    p.add_argument("--terrain-only", choices=list(TERRAINS), help="simulate only one terrain into a separate output folder")
    p.add_argument("--distance", type=float, default=6.5)
    p.add_argument("--elevation", type=float, default=-25.0)
    p.add_argument("--azimuth", type=float, default=135.0)
    args = p.parse_args()
    if not (args.simulate or args.render):
        p.error("use --simulate and/or --render")
    if args.simulate:
        simulate(args.plan, args.output_dir, args.candidates_10, args.slope_only, args.terrain_only)
    if args.render:
        render(args.plan, (args.distance, args.elevation, args.azimuth), args.output_dir)


if __name__ == "__main__":
    main()
