"""Gait diagnosis for a trained joystick policy.

Runs deterministic rollouts at several fixed commands, records per-step signals,
and writes one CSV per command plus a summary CSV and a few plots.

    python diagnose_gait.py --name litee2e --load

Reuses train_srbd.py for checkpoint loading and network construction; the env
itself is the single source of truth for physics, obs and rewards.
"""

import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mujoco_playground import registry
from train_srbd import load_params, _build_fresh_networks, _resolve_load
from brax.training.agents.ppo import networks as ppo_networks


# [vx, vy, wz] for a quadruped; the last two are dropped for a 2D command env.
COMMANDS = [
    ("stand",      [0.0, 0.0, 0.0]),
    ("vx_low",     [0.3, 0.0, 0.0]),
    ("vx_mid",     [0.8, 0.0, 0.0]),
    ("vx_high",    [1.4, 0.0, 0.0]),
    ("vy_only",    [0.0, 0.6, 0.0]),
    ("yaw_only",   [0.0, 0.0, 1.0]),
    ("vx_and_yaw", [0.8, 0.0, 0.8]),
]

FEET = ["FL", "FR", "HL", "HR"]


def rollout(env, policy, command, steps):
    """One deterministic rollout with a pinned command. Returns a dict of arrays."""
    reset = jax.jit(jax.vmap(env.reset))
    step = jax.jit(jax.vmap(env.step))

    state = reset(jnp.stack([jax.random.PRNGKey(42)]))
    cmd = jnp.asarray(command[: state.info["command"].shape[-1]], dtype=jnp.float32)
    cmd = jnp.broadcast_to(cmd, state.info["command"].shape)
    # Start at the commanded value instead of ramping from zero, and keep the
    # env's resampler from overwriting it.
    state = state.replace(info={**state.info, "command": cmd, "target_command": cmd})

    rec = {k: [] for k in (
        "command", "local_linvel", "gyro", "base_z", "rpy", "qpos", "qvel",
        "action", "target_pos", "torque", "foot_pos", "foot_vel", "contact",
        "feet_air_time", "swing_peak", "reward", "done")}
    rec["reward_terms"] = {}

    for _ in range(steps):
        act = policy(jax.tree_util.tree_map(lambda x: x[0], state.obs), jax.random.PRNGKey(0))[0]
        action = jnp.broadcast_to(act, (1, env.action_size))
        state = state.replace(info={**state.info, "target_command": cmd})
        state = step(state, action)
        state = state.replace(info={**state.info, "target_command": cmd})

        d0 = jax.tree_util.tree_map(lambda x: x[0], state.data)
        quat = np.asarray(d0.qpos[3:7])
        w, x, y, z = quat
        rec["rpy"].append([
            np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
            np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)),
            np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)),
        ])
        rec["command"].append(np.asarray(state.info["command"][0]))
        rec["local_linvel"].append(np.asarray(env.get_local_linvel(d0)))
        rec["gyro"].append(np.asarray(env.get_gyro(d0)))
        rec["base_z"].append(float(d0.qpos[2]))
        rec["qpos"].append(np.asarray(d0.qpos[7:]))
        rec["qvel"].append(np.asarray(d0.qvel[6:]))
        rec["action"].append(np.asarray(act))
        rec["target_pos"].append(
            np.asarray(env._default_pose + act * env._config.action_scale))
        rec["torque"].append(np.asarray(d0.actuator_force))
        rec["foot_pos"].append(np.asarray(d0.site_xpos[env._feet_site_id]))
        rec["foot_vel"].append(
            np.asarray(d0.sensordata[env._foot_linvel_sensor_adr]))
        rec["contact"].append(np.asarray(
            [d0.sensordata[env._mj_model.sensor_adr[s]] > 0
             for s in env._feet_floor_found_sensor]))
        rec["feet_air_time"].append(np.asarray(state.info["feet_air_time"][0]))
        rec["swing_peak"].append(np.asarray(state.info["swing_peak"][0]))
        rec["reward"].append(float(state.reward[0]))
        rec["done"].append(float(state.done[0]))
        for k, v in state.info["reward_terms"].items():
            rec["reward_terms"].setdefault(k, []).append(float(np.asarray(v)[0]))
        if float(state.done[0]) > 0:
            break

    out = {k: np.asarray(v) for k, v in rec.items() if k != "reward_terms"}
    out["reward_terms"] = {k: np.asarray(v) for k, v in rec["reward_terms"].items()}
    return out


def foot_stats(contact, foot_pos, foot_vel, dt):
    """Per-foot gait statistics. contact: (T, 4) bool."""
    stats = {}
    for i, name in enumerate(FEET):
        c = contact[:, i].astype(bool)
        td = np.flatnonzero((~c[:-1]) & c[1:]) + 1     # touchdown
        lo = np.flatnonzero(c[:-1] & (~c[1:])) + 1     # liftoff

        # run lengths of stance / swing
        stance, swing, cur, curval = [], [], 0, c[0]
        for v in c:
            if v == curval:
                cur += 1
            else:
                (stance if curval else swing).append(cur * dt)
                cur, curval = 1, v
        (stance if curval else swing).append(cur * dt)

        z = foot_pos[:, i, 2]
        vxy = np.linalg.norm(foot_vel[:, i, :2], axis=-1)
        # longest consecutive stretch with no contact
        gaps = [len(g) for g in "".join("1" if v else "0" for v in c).split("1")]

        stats[name] = dict(
            contact_pct=100.0 * c.mean(),
            duty_factor=c.mean(),
            n_touchdown=len(td),
            n_liftoff=len(lo),
            step_freq_hz=len(td) / (len(c) * dt) if len(c) else 0.0,
            stance_mean=np.mean(stance) if stance else 0.0,
            stance_std=np.std(stance) if stance else 0.0,
            swing_mean=np.mean(swing) if swing else 0.0,
            swing_std=np.std(swing) if swing else 0.0,
            foot_z_min=z.min(), foot_z_mean=z.mean(), foot_z_max=z.max(),
            slip_stance_mean=vxy[c].mean() if c.any() else float("nan"),
            max_air_gap_s=max(gaps) * dt if gaps else 0.0,
        )
    return stats


def smoothness(rec, dt):
    a = rec["action"]
    d1 = np.diff(a, axis=0)
    d2 = np.diff(a, n=2, axis=0)
    qacc = np.diff(rec["qvel"], axis=0) / dt
    dtau = np.diff(rec["torque"], axis=0)
    return dict(
        act_d1_mean=np.abs(d1).mean(), act_d1_rms=np.sqrt((d1 ** 2).mean()),
        act_d1_std=d1.std(), act_d1_peak=np.abs(d1).max(),
        act_d2_rms=np.sqrt((d2 ** 2).mean()), act_d2_peak=np.abs(d2).max(),
        qacc_rms=np.sqrt((qacc ** 2).mean()), qacc_peak=np.abs(qacc).max(),
        dtau_rms=np.sqrt((dtau ** 2).mean()), dtau_peak=np.abs(dtau).max(),
        base_z_std=rec["base_z"].std(), base_z_ptp=np.ptp(rec["base_z"]),
        roll_std=rec["rpy"][:, 0].std(), pitch_std=rec["rpy"][:, 1].std(),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="litee2e")
    ap.add_argument("--load", nargs="?", const="best", default="best")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    shortcuts = {"litee2e": "Lite3JoystickE2EFlatTerrain",
                 "lite3": "Lite3JoystickFlatTerrain",
                 "go1": "Go1JoystickFlatTerrain",
                 "aliengo": "AliengoJoystickE2EFlatTerrain"}
    env_name = shortcuts.get(args.name.lower(), args.name)
    env = registry.load(env_name)
    dt = env.dt

    base = os.path.join(args.ckpt_dir, env_name)
    run_dir, suffix = _resolve_load(base, args.load)
    params = load_params(run_dir, suffix=suffix)
    networks = _build_fresh_networks(env)
    policy = ppo_networks.make_inference_fn(networks)(params, deterministic=True)

    out_dir = args.out or os.path.join(run_dir, "gait_diagnosis")
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n  env {env_name} | dt {dt:.3f} | out {out_dir}\n")

    summary = []
    for tag, cmd in COMMANDS:
        rec = rollout(env, policy, cmd, args.steps)
        T = len(rec["base_z"])
        st = foot_stats(rec["contact"], rec["foot_pos"], rec["foot_vel"], dt)
        sm = smoothness(rec, dt)

        # per-step CSV
        cols = {"step": np.arange(T)}
        for j, n in enumerate(["cmd_vx", "cmd_vy", "cmd_wz"][: rec["command"].shape[1]]):
            cols[n] = rec["command"][:, j]
        for j, n in enumerate(["vx", "vy", "vz"]):
            cols["base_" + n] = rec["local_linvel"][:, j]
        for j, n in enumerate(["roll", "pitch", "yaw"]):
            cols[n] = rec["rpy"][:, j]
        for j, n in enumerate(["wx", "wy", "wz"]):
            cols["gyro_" + n] = rec["gyro"][:, j]
        cols["base_z"] = rec["base_z"]
        for j in range(rec["qpos"].shape[1]):
            cols[f"q{j}"] = rec["qpos"][:, j]
            cols[f"dq{j}"] = rec["qvel"][:, j]
            cols[f"act{j}"] = rec["action"][:, j]
            cols[f"tgt{j}"] = rec["target_pos"][:, j]
            cols[f"tau{j}"] = rec["torque"][:, j]
        for i, f in enumerate(FEET):
            for j, n in enumerate("xyz"):
                cols[f"{f}_p{n}"] = rec["foot_pos"][:, i, j]
                cols[f"{f}_v{n}"] = rec["foot_vel"][:, i, j]
            cols[f"{f}_contact"] = rec["contact"][:, i].astype(int)
            cols[f"{f}_air_time"] = rec["feet_air_time"][:, i]
            cols[f"{f}_swing_peak"] = rec["swing_peak"][:, i]
        cols["reward"] = rec["reward"]
        for k, v in rec["reward_terms"].items():
            cols["rew_" + k] = v
        csv_path = os.path.join(out_dir, f"rollout_{tag}.csv")
        np.savetxt(csv_path, np.column_stack(list(cols.values())), delimiter=",",
                   header=",".join(cols.keys()), comments="")

        # reward clipping analysis (terms are already scaled, before *dt and clip)
        raw = np.sum(np.column_stack(list(rec["reward_terms"].values())), axis=1)
        neg_pct = 100.0 * (raw < 0).mean()
        clipped_pct = 100.0 * ((raw * dt) < 0).mean()

        row = dict(command=tag, steps=T,
                   cmd_vx=cmd[0], cmd_vy=cmd[1], cmd_wz=cmd[2],
                   vx_mean=rec["local_linvel"][:, 0].mean(),
                   vy_mean=rec["local_linvel"][:, 1].mean(),
                   wz_mean=rec["gyro"][:, 2].mean(),
                   err_vx=abs(cmd[0] - rec["local_linvel"][:, 0].mean()),
                   err_wz=abs(cmd[2] - rec["gyro"][:, 2].mean()),
                   reward_mean=rec["reward"].mean(),
                   raw_neg_pct=neg_pct, clipped_pct=clipped_pct, **sm)
        for f in FEET:
            for k, v in st[f].items():
                row[f"{f}_{k}"] = v
        summary.append(row)

        print(f"--- {tag:11s} T={T:4d}  vx {rec['local_linvel'][:,0].mean():+.2f}"
              f"/{cmd[0]:+.2f}  wz {rec['gyro'][:,2].mean():+.2f}/{cmd[2]:+.2f}"
              f"  rew {rec['reward'].mean():.4f}")
        print("     duty  " + "  ".join(
            f"{f} {st[f]['duty_factor']:.3f}" for f in FEET)
            + f"   | reward grezza <0 in {neg_pct:.1f}% degli step")
        print("     td#   " + "  ".join(
            f"{f} {st[f]['n_touchdown']:3d}" for f in FEET)
            + f"   | act_d1_rms {sm['act_d1_rms']:.4f} act_d2_rms {sm['act_d2_rms']:.4f}")

        # plots
        fig, ax = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
        for i, f in enumerate(FEET):
            ax[0].fill_between(np.arange(T), i, i + rec["contact"][:, i] * 0.85,
                               step="pre", alpha=.8, label=f)
            ax[1].plot(rec["foot_pos"][:, i, 2], label=f)
        ax[0].set_yticks(np.arange(4) + .4); ax[0].set_yticklabels(FEET)
        ax[0].set_ylabel("contatti")
        ax[1].set_ylabel("z piede [m]"); ax[1].legend(ncol=4, fontsize=7)
        ax[2].plot(rec["action"]); ax[2].set_ylabel("action")
        ax[3].plot(rec["base_z"], label="base z")
        ax[3].plot(rec["rpy"][:, 0], label="roll")
        ax[3].plot(rec["rpy"][:, 1], label="pitch")
        ax[3].set_ylabel("base"); ax[3].legend(fontsize=7); ax[3].set_xlabel("step")
        fig.suptitle(f"{env_name} — {tag} — cmd {cmd}")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"gait_{tag}.png"), dpi=110)
        plt.close(fig)

    keys = list(summary[0].keys())
    with open(os.path.join(out_dir, "summary.csv"), "w") as f:
        f.write(",".join(keys) + "\n")
        for r in summary:
            f.write(",".join(str(r[k]) for k in keys) + "\n")

    # duty factor comparison across commands
    fig, ax = plt.subplots(figsize=(9, 4))
    w = 0.2
    for i, foot in enumerate(FEET):
        ax.bar(np.arange(len(summary)) + i * w,
               [r[f"{foot}_duty_factor"] for r in summary], w, label=foot)
    ax.set_xticks(np.arange(len(summary)) + 1.5 * w)
    ax.set_xticklabels([r["command"] for r in summary], rotation=20)
    ax.set_ylabel("duty factor"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "duty_factor.png"), dpi=110)
    plt.close(fig)

    print(f"\n  scritto {out_dir}/summary.csv + rollout_*.csv + png\n")


if __name__ == "__main__":
    main()
