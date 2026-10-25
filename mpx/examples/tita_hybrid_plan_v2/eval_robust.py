import jax; jax.config.update("jax_enable_x64", True)  # MPC/WBC need float64
"""Robustness evaluation: our MPC-only, our HRC policy, tita_v1, tita_v1 MPC-only.

Same protocol for every controller: terrains flat / rough (cobblestones) /
perlin (+ slopes 5 and 10 deg for our env only), fixed joystick commands
applied through the env's ramp, 5 random spawns (position and heading) per
command, 10 s episodes, batched and scanned on the GPU.

Metrics per terrain: fall rate, mean time to fall, mean |vx error| and
|wz error| while alive, residual/nominal torque ratio ||tau_res|| / ||tau_nom||,
and orientation (v2): mean |roll|, |pitch| while alive and the heading drift at
the end of surviving episodes, |yaw - yaw0 - integral of the wz command|.

Usage (from mpx/examples):
  python tita_hybrid_plan/eval_robust.py --controller mpc
  python tita_hybrid_plan/eval_robust.py --controller hrc --load <run_dir_name>[/best]
  # tita_v1 runs on its own code (residual_tita worktrees first on PYTHONPATH):
  PYTHONPATH=<mjpg_residual_tita>:<mpx_residual_tita> \
    python tita_hybrid_plan/eval_robust.py --controller v1 --v1-examples <mpx_residual_tita>/mpx/examples
  (add --zero for the tita_v1 MPC-only baseline)
Results: tita_hybrid_plan_v2/robust_results/<controller>.json
"""
import argparse
import json
import sys
from pathlib import Path

import jax.numpy as jp
import numpy as np

COMMANDS = [(0.5, 0.0), (1.0, 0.0), (1.5, 0.0), (2.0, 0.0),
            (1.0, 0.5), (1.0, -0.8), (2.0, 0.8), (-1.0, 0.0)]
# 10 deg slope: linear commands only, |vx| <= 1 m/s (user, 18/10).
COMMANDS_SLOPE = [(0.25, 0.0), (0.5, 0.0), (0.75, 0.0), (1.0, 0.0),
                  (-0.25, 0.0), (-0.5, 0.0), (-1.0, 0.0), (0.0, 0.0)]
SEEDS = 5
SECONDS = 10.0
OUT = Path(__file__).resolve().parent / "robust_results"


def terrains(controller):
    t = {"flat": ("TitaJoystickFlatTerrain", {}),
         "rough": ("TitaJoystickRoughTerrain", {}),
         "perlin": ("TitaJoystickPerlinTerrain", {})}
    if controller in ("mpc", "hrc"):
        # 10 deg slope only (user): an infinite tilted plane, so that random
        # headings give uphill / downhill / cross-slope with no ramp edges.
        t["slope10"] = ("TitaJoystickFlatTerrain", {"terrain_config.scene": "tilt10",
                                                    "terrain_config.tilt_deg": 10.0})
    return t


def ours_overrides(controller):
    # Joystick task, random spawn heading as tita_v1, no noise, residual off
    # for MPC-only.
    return {"command_config.mode": "joystick", "episode_length": 1000,
            "randomize_reset": 1.0, "residual_config.enabled": controller == "hrc"}


def _roll_pitch_yaw(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    roll = jp.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = jp.arcsin(jp.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = jp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def evaluate(env, policy, zero, seeds=SEEDS, commands=COMMANDS):
    n = len(commands) * seeds
    cmds = jp.array([c for c in commands for _ in range(seeds)])
    steps = int(SECONDS / env.dt)
    keys = jax.random.split(jax.random.PRNGKey(0), n)

    @jax.jit
    def run(keys):
        state = jax.vmap(env.reset)(keys)
        yaw0 = _roll_pitch_yaw(state.data.qpos[:, 3:7])[2]

        def body(carry, _):
            state, alive, key, yaw_cmd = carry
            state.info["target_command"] = cmds
            state.info["steps_until_next_cmd"] = jp.full((n,), 10**6, dtype=jp.int32)
            key, k = jax.random.split(key)
            if zero:
                action = jp.zeros((n, env.action_size))
            else:
                obs = state.obs
                action, _ = jax.vmap(policy)(obs, jax.random.split(k, n))
            yaw_cmd = yaw_cmd + state.info["command"][:, 1] * env.dt
            state = jax.vmap(env.step)(state, action)
            roll, pitch, yaw = _roll_pitch_yaw(state.data.qpos[:, 3:7])
            dyaw = yaw - yaw0 - yaw_cmd
            drift = jp.abs(jp.arctan2(jp.sin(dyaw), jp.cos(dyaw)))
            vx = jax.vmap(env.get_local_linvel)(state.data)[:, 0]
            wz = jax.vmap(env.get_gyro)(state.data)[:, 2]
            tn = jp.linalg.norm(state.info["tau_nominal"], axis=-1)
            tr = jp.linalg.norm(state.info["tau_residual"], axis=-1)
            # where, not *: after a fall the state can be NaN and NaN*0 = NaN.
            ok = alive > 0
            out = (alive, jp.where(ok, jp.abs(vx - state.info["command"][:, 0]), 0.0),
                   jp.where(ok, jp.abs(wz - state.info["command"][:, 1]), 0.0),
                   jp.where(alive > 0, tr / jp.maximum(tn, 1e-6), 0.0),
                   jp.where(ok, jp.abs(roll), 0.0), jp.where(ok, jp.abs(pitch), 0.0),
                   jp.where(ok, drift, 0.0))
            alive = alive * (1.0 - state.done)
            return (state, alive, key, yaw_cmd), out

        _, (alive, evx, ewz, ratio, aroll, apitch, drift) = jax.lax.scan(
            body, (state, jp.ones(n), jax.random.PRNGKey(1), jp.zeros(n)), None, length=steps)
        return alive, evx, ewz, ratio, aroll, apitch, drift

    alive, evx, ewz, ratio, aroll, apitch, drift = [np.asarray(x) for x in run(keys)]
    life = alive.sum(axis=0)                          # steps alive per episode
    fell = life < steps
    per_cmd = {}
    for i, c in enumerate(commands):
        sl = slice(i * seeds, (i + 1) * seeds)
        per_cmd[f"{c[0]:+.1f},{c[1]:+.1f}"] = int(fell[sl].sum())
    alive_steps = max(alive.sum(), 1.0)
    return dict(
        episodes=int(n), falls=int(fell.sum()), fall_rate=float(fell.mean()),
        mean_time_to_fall_s=float(life[fell].mean() * env.dt) if fell.any() else None,
        mean_abs_vx_err=float(evx.sum() / alive_steps),
        mean_abs_wz_err=float(ewz.sum() / alive_steps),
        residual_ratio=float(ratio.sum() / alive_steps),
        mean_abs_roll=float(aroll.sum() / alive_steps),
        mean_abs_pitch=float(apitch.sum() / alive_steps),
        final_heading_drift=float(drift[-1][~fell].mean()) if (~fell).any() else None,
        falls_per_command=per_cmd)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--controller", choices=["mpc", "hrc", "v1"], required=True)
    p.add_argument("--load", default=None, help="run dir name (and /suffix) under checkpoints")
    p.add_argument("--v1-examples", default=None, help="tita_v1 mpx/examples directory")
    p.add_argument("--zero", action="store_true", help="v1 only: MPC-only (zero action)")
    p.add_argument("--only", default=None, help="comma-separated terrains")
    p.add_argument("--seeds", type=int, default=SEEDS, help="spawns per command")
    p.add_argument("--rough-seeds", type=int, default=2,
                   help="spawns per command on rough (2590 boxes: slow in MJX)")
    args = p.parse_args()

    examples = Path(args.v1_examples) if args.controller == "v1" else Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(examples))
    import train_srbd  # the controller's own network construction and loader
    from mujoco_playground import registry

    name = args.controller + ("_mpc" if args.zero else "")
    results = {}
    for tname, (env_name, extra) in terrains(args.controller).items():
        if args.only and tname not in args.only.split(","):
            continue
        overrides = ({} if args.controller == "v1" else {**ours_overrides(args.controller), **extra})
        if args.controller == "v1":
            overrides = {"noise_config.level": 0.0}
        env = registry.load(env_name, config_overrides=overrides or None)
        zero = args.controller == "mpc" or args.zero
        policy = None
        if not zero:
            policy = train_srbd.build_eval_inference_fn(env, args.load, env_name="TitaJoystickFlatTerrain",
                                                        ckpt_root=str(examples / "checkpoints"))
        seeds = args.rough_seeds if tname == "rough" else args.seeds
        commands = COMMANDS_SLOPE if tname.startswith("slope") else COMMANDS
        results[tname] = evaluate(env, policy, zero, seeds, commands)
        print(f"[{name}] {tname}: {results[tname]}", flush=True)
        OUT.mkdir(exist_ok=True)
        (OUT / f"{name}.json").write_text(json.dumps(results, indent=2))  # partial


if __name__ == "__main__":
    main()
