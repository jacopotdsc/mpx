"""Headless validation of the standalone SRBD controller.

Runs a fixed set of joystick scenarios, computes tracking / posture / contact /
torque metrics, writes them to JSON and prints a readable summary.

    python lite3_srbd_validate.py                  # Lite3
    python lite3_srbd_validate.py --robot aliengo  # Aliengo baseline
    python lite3_srbd_validate.py --robot both --out results/

Both robots run the same scenarios, so the two JSON files can be diffed
directly. Lite3 is not expected to match Aliengo numerically -- mass, inertia
and geometry differ -- but the qualitative behaviour must be equivalent.
"""

import argparse
import json
import os

import numpy as np

FEET = ["FL", "FR", "HL", "HR"]

# steps are at the controller rate (200 Hz), so 1200 steps = 6 s.
SCENARIOS = [
    ("stand",        (0.0,  0.0,  0.0), 1200),
    ("vx_pos",       (0.3,  0.0,  0.0), 1200),
    ("vx_neg",      (-0.3,  0.0,  0.0), 1200),
    ("vy_pos",       (0.0,  0.2,  0.0), 1200),
    ("wz_pos",       (0.0,  0.0,  0.5), 1200),
    ("wz_neg",       (0.0,  0.0, -0.5), 1200),
    ("vx_wz",        (0.3,  0.0,  0.3), 1200),
    ("long_run",     (0.3,  0.0,  0.0), 4000),
]

# Command transition: still -> forward -> turn -> still, 400 steps (2 s) each.
def _transition(k):
    if k < 400:
        return (0.0, 0.0, 0.0)
    if k < 800:
        return (0.3, 0.0, 0.0)
    if k < 1200:
        return (0.3, 0.0, 0.4)
    return (0.0, 0.0, 0.0)


def analyse(rows, cmd, mass, nominal_h, dt):
    """All metrics for one rollout. `rows` is the list of dicts from the runner."""
    d = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    n = len(rows)
    mg = mass * 9.81
    tau = np.column_stack([d[f"tau{j}"] for j in range(12)])
    settled = slice(n // 2, None)     # steady state = second half

    finite = all(np.isfinite(v).all() for v in d.values())
    fall_idx = np.flatnonzero(d["base_z"] < 0.5 * nominal_h)
    cvx, cvy, cwz = cmd if not callable(cmd) else (d["cmd_vx"], d["cmd_vy"], d["cmd_wz"])

    def track(actual, target):
        err = actual - target
        return dict(
            mean=float(np.mean(actual[settled])),
            rmse=float(np.sqrt(np.mean(err[settled] ** 2))),
            mae=float(np.mean(np.abs(err[settled]))),
            steady_err=float(np.mean(err[settled])),
        )

    out = dict(
        steps=n, duration_s=n * dt, finite=bool(finite),
        fell=bool(len(fall_idx) > 0),
        fall_step=int(fall_idx[0]) if len(fall_idx) else -1,
        vx=track(d["vx"], cvx), vy=track(d["vy"], cvy), wz=track(d["wz"], cwz),
        base_z_mean=float(d["base_z"].mean()), base_z_std=float(d["base_z"].std()),
        base_z_err=float(d["base_z"].mean() - nominal_h),
        roll_max=float(np.abs(d["roll"]).max()),
        roll_rms=float(np.sqrt((d["roll"] ** 2).mean())),
        pitch_max=float(np.abs(d["pitch"]).max()),
        pitch_rms=float(np.sqrt((d["pitch"] ** 2).mean())),
        grf_z_mean=float(d["grf_z_sum"].mean()), weight_N=float(mg),
        grf_err_pct=float(100 * (d["grf_z_sum"].mean() - mg) / mg),
        tau_max=float(np.abs(tau).max()), tau_rms=float(np.sqrt((tau ** 2).mean())),
        tau_max_per_joint=[float(v) for v in np.abs(tau).max(axis=0)],
    )

    # Overshoot and settling time on the dominant commanded axis.
    axis, target = ("vx", cvx) if abs(np.mean(cvx)) >= abs(np.mean(cwz)) else ("wz", cwz)
    tgt = float(np.mean(target))
    if abs(tgt) > 1e-6:
        sig = d[axis]
        out["overshoot_pct"] = float(100 * (np.max(sig * np.sign(tgt)) - abs(tgt)) / abs(tgt))
        within = np.abs(sig - tgt) <= 0.1 * abs(tgt)
        idx = n
        for i in range(n):                      # first index that stays inside the band
            if within[i:].all():
                idx = i
                break
        out["settling_s"] = float(idx * dt)
    else:
        out["overshoot_pct"] = None
        out["settling_s"] = None

    # Per-foot contact statistics.
    feet = {}
    for f in FEET:
        c = d[f"{f}_contact"].astype(bool)
        td = int(np.sum((~c[:-1]) & c[1:]))
        runs, cur, val = [], 0, c[0]
        for v in c:
            if v == val:
                cur += 1
            else:
                runs.append((val, cur * dt))
                cur, val = 1, v
        runs.append((val, cur * dt))
        st = [t for on, t in runs if on]
        sw = [t for on, t in runs if not on]
        feet[f] = dict(
            duty=float(c.mean()), touchdowns=td,
            stance_mean=float(np.mean(st)) if st else 0.0,
            swing_mean=float(np.mean(sw)) if sw else 0.0,
            max_air_s=float(max(sw)) if sw else 0.0,
            grf_z_mean=float(d[f"{f}_grf_z"].mean()),
            mismatch_pct=float(100 * np.mean(c != d[f"{f}_planned"].astype(bool))),
        )
    out["feet"] = feet
    out["contact_mismatch_pct"] = float(np.mean([feet[f]["mismatch_pct"] for f in FEET]))
    out["duty_spread"] = float(max(feet[f]["duty"] for f in FEET)
                               - min(feet[f]["duty"] for f in FEET))
    # Symmetry: left vs right and front vs hind duty factors.
    out["sym_left_right"] = float(abs((feet["FL"]["duty"] + feet["HL"]["duty"])
                                      - (feet["FR"]["duty"] + feet["HR"]["duty"])) / 2)
    out["sym_front_hind"] = float(abs((feet["FL"]["duty"] + feet["FR"]["duty"])
                                      - (feet["HL"]["duty"] + feet["HR"]["duty"])) / 2)
    return out


def run(robot, out_dir):
    if robot == "lite3":
        import lite3_srbd as runner
        import mpx.config.config_srbd_lite3 as cfg
        nominal_h = float(cfg.robot_height)
    else:
        import srbd_quad as runner
        import mpx.config.config_srbd as cfg
        nominal_h = float(cfg.robot_height)
    dt = 1.0 / float(cfg.whole_body_frequency)
    mass = float(cfg.mass)

    results = {}
    scenarios = SCENARIOS + [("transition", _transition, 1600)]
    for name, cmd, steps in scenarios:
        print(f"\n=== {robot} / {name} ===")
        rows = runner.main(headless=True, steps=steps, metrics=os.devnull, cmd=cmd)
        results[name] = analyse(rows, cmd, mass, nominal_h, dt)
        r = results[name]
        print(f"  vx {r['vx']['mean']:+.3f} (rmse {r['vx']['rmse']:.3f}) | "
              f"wz {r['wz']['mean']:+.3f} (rmse {r['wz']['rmse']:.3f}) | "
              f"h {r['base_z_mean']:.3f} | GRF err {r['grf_err_pct']:+.1f}% | "
              f"tau max {r['tau_max']:.1f} | mismatch {r['contact_mismatch_pct']:.1f}% | "
              f"fell {r['fell']} | finite {r['finite']}")

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{robot}_srbd_validation.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  -> {path}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", choices=["lite3", "aliengo", "both"], default="lite3")
    ap.add_argument("--out", default="validation")
    args = ap.parse_args()
    for rb in (["lite3", "aliengo"] if args.robot == "both" else [args.robot]):
        run(rb, args.out)
