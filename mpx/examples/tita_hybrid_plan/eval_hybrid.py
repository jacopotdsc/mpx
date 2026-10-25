import jax; jax.config.update("jax_enable_x64", True)  # MPC/WBC need float64
"""Paper evaluation (Phase 4): MPC-only vs HRC (MPC + SAC residual) on Tita.

E1 (Tab. I, Fig. 4): Task 1, 2.5 m at constant CoM height, on 0 / 5 / 10 deg.
E2 (Fig. 7): Task 1 on flat ground, pushes of 10 / 20 / 30 N at 12 / 20 / 32 s.
E3 (Fig. 8): 5th-order polynomial random-point trajectory and high-dynamic
multi-target trajectory, given to the MPC as velocity commands.
E5 (not in the paper): rough cobblestone terrain (TitaJoystickRoughTerrain,
never seen in training), distances 2.5 / 4 m at cruise speed 0.5 / 1.0 /
1.5 m/s. The criterion is survival (falls, time to fall), not tracking.

Usage (from mpx/examples):
  python tita_hybrid_plan/eval_hybrid.py --run checkpoints/TitaJoystickFlatTerrain/<run> [--suffix best]
Outputs go to <run>/eval_hybrid/ (CSV per rollout, plots, results.json/.md).
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import jax.numpy as jp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from brax.training.agents.sac import networks as sac_networks

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # mpx/examples
import sac_training  # noqa: E402
from train_srbd import SAC_PARAMS  # noqa: E402
from mujoco_playground import registry  # noqa: E402

ENV_NAME = "TitaJoystickFlatTerrain"
ERR = ["x_err", "v_err", "theta_p_dot", "z_err", "roll", "pitch", "yaw"]
PAPER_TAB1 = {  # Task 1 RMSE [x_w, v, theta_p, z_c, beta]: BMC, HRC
    0: ([0.2475, 0.0533, 0.0157, 0.0168, 0.0065], [0.0134, 0.0119, 0.0106, 0.0032, 0.0057]),
    5: ([0.1397, 0.0400, 0.0177, 0.0167, 0.0064], [0.0401, 0.0099, 0.0114, 0.0015, 0.0042]),
    10: ([0.3436, 0.0619, 0.0461, 0.0170, 0.0106], [0.0366, 0.0086, 0.0289, 0.0012, 0.0103]),
}


def load_policy(run_dir, suffix, env):
    with (Path(run_dir) / f"params_{suffix}.pkl").open("rb") as f:
        params = pickle.load(f)
    net = sac_training.make_network(env.observation_size["state"][0], env.action_size,
                                    SAC_PARAMS["init_std"])
    return jax.jit(sac_networks.make_inference_fn(net)(params, deterministic=True))


def rollout(overrides, policy, seconds, reference=None, env_name=ENV_NAME):
    """One deterministic episode. reference(t) -> (x_ref, v_ref) when external."""
    overrides = dict(overrides)
    overrides["residual_config.enabled"] = policy is not None
    if reference is not None:
        overrides["command_config.external_reference"] = True
    env = registry.load(env_name, config_overrides=overrides)
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    state = reset(jax.random.PRNGKey(0))
    key = jax.random.PRNGKey(1)
    rows = []
    for i in range(int(seconds / env.dt)):
        if reference is not None:
            x_ref, v_ref = reference((i + 1) * env.dt)
            state.info["x_ref"] = jp.asarray(x_ref)
            state.info["command"] = jp.array([v_ref, 0.0])
        action = (jp.zeros(env.action_size) if policy is None
                  else policy(state.obs["state"], key)[0])
        state = step(state, action)
        q_b, p_c, _ = env._balance_state(state.data, state.info)
        e = np.asarray(state.info["task_errors"])
        rows.append(dict(
            t=(i + 1) * env.dt, x_w=float(q_b[0]), x_ref=float(state.info["x_ref"]),
            v=float(q_b[1]), v_ref=float(state.info["command"][0]),
            theta_p=float(q_b[2]), theta_p_dot=float(q_b[3]),
            z_c=float(e[3] + state.info["base_height_target"]),
            pitch=float(e[5]), roll=float(e[4]), yaw=float(e[6]),
            tau_res=np.abs(np.asarray(state.info["tau_residual"])).max(),
            fell=float(state.done)))
        if state.done:
            break
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


def rmse(log):
    """Paper Tab. I metrics. theta_p has no target in the paper (theta_pd is
    removed): we report its RMS deviation from the episode mean."""
    r = lambda a: float(np.sqrt(np.mean(np.square(a))))
    return dict(
        x_w=r(log["x_w"] - log["x_ref"]), v=r(log["v"] - log["v_ref"]),
        theta_p=r(log["theta_p"] - log["theta_p"].mean()), z_c=r(log["z_c"] - 0.4),
        beta=r(log["pitch"]), final_x_err=float(log["x_w"][-1] - log["x_ref"][-1]),
        fell=bool(log["fell"].max() > 0), steps=int(len(log["t"])))


def save_log(log, path):
    keys = list(log)
    np.savetxt(path, np.column_stack([log[k] for k in keys]), delimiter=",",
               header=",".join(keys), comments="")


def plot(logs, title, path, keys=("x_w", "v", "theta_p", "z_c", "pitch")):
    fig, axes = plt.subplots(len(keys), 1, figsize=(8, 2.2 * len(keys)), sharex=True)
    for ax, k in zip(axes, keys):
        for name, log in logs.items():
            ax.plot(log["t"], log[k], label=name)
        if k in ("x_w", "v"):
            ref = next(iter(logs.values()))
            ax.plot(ref["t"], ref["x_ref" if k == "x_w" else "v_ref"], "k--", lw=1, label="target")
        ax.set_ylabel(k)
    axes[0].legend(loc="best")
    axes[0].set_title(title)
    axes[-1].set_xlabel("t [s]")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def min_jerk_reference(points, durations, hold):
    """Piecewise 5th-order (min-jerk) position profile through `points`,
    each segment lasting durations[i], followed by `hold` seconds."""
    times = np.concatenate([[0.0], np.cumsum(durations)])

    def ref(t):
        k = int(np.searchsorted(times, t, side="right") - 1)
        if k >= len(durations):
            return points[-1], 0.0
        T = durations[k]
        s = (t - times[k]) / T
        dx = points[k + 1] - points[k]
        x = points[k] + dx * (10 * s**3 - 15 * s**4 + 6 * s**5)
        v = dx * (30 * s**2 - 60 * s**3 + 30 * s**4) / T
        return x, v
    return ref, float(times[-1] + hold)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--suffix", default="best")
    p.add_argument("--only", default="E1,E2,E3,E5")
    args = p.parse_args()
    out = Path(args.run) / "eval_hybrid"
    out.mkdir(exist_ok=True)
    env = registry.load(ENV_NAME)
    hrc = load_policy(args.run, args.suffix, env)
    controllers = {"MPC": None, "HRC": hrc}
    results = {}

    if "E1" in args.only:
        for lane, deg in [(0, 0), (1, 5), (2, 10)]:
            ov = {"command_config.distance": [2.5, 2.5], "terrain_config.scene": "slopes",
                  "terrain_config.lanes": [lane]}
            logs = {n: rollout(ov, c, 25.0) for n, c in controllers.items()}
            for n, log in logs.items():
                save_log(log, out / f"E1_{deg}deg_{n}.csv")
                results[f"E1_{deg}deg_{n}"] = rmse(log)
                print(f"E1 {deg}deg {n}: {results[f'E1_{deg}deg_{n}']}", flush=True)
            plot(logs, f"E1 Task 1, {deg} deg", out / f"E1_{deg}deg.png")

    if "E2" in args.only:
        ov = {"command_config.distance": [2.5, 2.5],
              "push_config.times": [12.0, 20.0, 32.0], "push_config.forces": [10.0, 20.0, 30.0]}
        logs = {n: rollout(ov, c, 45.0) for n, c in controllers.items()}
        for n, log in logs.items():
            save_log(log, out / f"E2_push_{n}.csv")
            res = rmse(log)
            for t0 in (12.0, 20.0, 32.0):
                w = (log["t"] >= t0) & (log["t"] < t0 + 8.0)
                err = np.abs(log["x_w"][w] - log["x_ref"][w])
                calm = (err < 0.02) & (np.abs(log["theta_p_dot"][w]) < 0.05)
                # recovery: first time after which the state stays calm
                idx = np.where(~calm)[0]
                rec = float(log["t"][w][idx[-1] + 1] - t0) if len(idx) and idx[-1] + 1 < w.sum() else (0.0 if not len(idx) else float("nan"))
                res[f"push{int(t0)}_max_dx"] = float(err.max())
                res[f"push{int(t0)}_recovery_s"] = rec
            results[f"E2_{n}"] = res
            print(f"E2 {n}: {res}", flush=True)
        plot(logs, "E2 pushes 10/20/30 N at 12/20/32 s", out / "E2_push.png", keys=("x_w", "theta_p"))

    if "E3" in args.only:
        rng = np.random.default_rng(0)
        poly_pts = np.concatenate([[0.0], np.cumsum(rng.uniform(-1.0, 1.5, 6))])
        trajs = {
            "poly5": min_jerk_reference(poly_pts, [4.0] * 6, 5.0),
            "multi_target": min_jerk_reference(
                np.array([0.0, 1.5, 0.5, 3.0, 1.0, 2.5]), [2.0, 1.5, 2.5, 2.0, 2.0], 5.0),
        }
        for name, (ref, T) in trajs.items():
            logs = {n: rollout({}, c, T, reference=ref) for n, c in controllers.items()}
            for n, log in logs.items():
                save_log(log, out / f"E3_{name}_{n}.csv")
                results[f"E3_{name}_{n}"] = rmse(log)
                print(f"E3 {name} {n}: {results[f'E3_{name}_{n}']}", flush=True)
            plot(logs, f"E3 {name}", out / f"E3_{name}.png", keys=("x_w", "v", "theta_p"))

    if "E5" in args.only:
        rough = {}
        for dist in (2.5, 4.0):
            for vmax in (0.5, 1.0, 1.5):
                ov = {"command_config.distance": [dist, dist], "command_config.v_max": vmax,
                      "command_config.a_max": max(0.5, vmax)}
                for n, c in controllers.items():
                    log = rollout(ov, c, 20.0, env_name="TitaJoystickRoughTerrain")
                    tag = f"E5_rough_D{dist}_v{vmax}_{n}"
                    save_log(log, out / f"{tag}.csv")
                    r = rmse(log)
                    r["time_to_fall_s"] = float(log["t"][-1]) if r["fell"] else None
                    results[tag] = r
                    rough.setdefault(n, []).append(r["fell"])
                    print(f"{tag}: fell={r['fell']} t_end={log['t'][-1]:.2f} x_rmse={r['x_w']:.3f}", flush=True)
        for n, falls in rough.items():
            results[f"E5_rough_summary_{n}"] = {"falls": int(sum(falls)), "episodes": len(falls)}
            print(f"E5 rough {n}: {sum(falls)}/{len(falls)} falls", flush=True)

    (out / "results.json").write_text(json.dumps(results, indent=2))
    lines = ["| Scenario | Controller | x_w | v | theta_p | z_c | beta | final x err | fell |",
             "|---|---|---|---|---|---|---|---|---|"]
    for k, r in results.items():
        if "x_w" in r:
            lines.append(f"| {k.rsplit('_', 1)[0]} | {k.rsplit('_', 1)[1]} | {r['x_w']:.4f} | {r['v']:.4f} | "
                         f"{r['theta_p']:.4f} | {r['z_c']:.4f} | {r['beta']:.4f} | {r['final_x_err']:.4f} | {r['fell']} |")
    lines += ["", "Paper Tab. I (Task 1): BMC / HRC"]
    for deg, (bmc, hrc_p) in PAPER_TAB1.items():
        lines.append(f"- {deg} deg: BMC {bmc}  HRC {hrc_p}")
    (out / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
