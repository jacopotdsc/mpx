#!/usr/bin/env python3
"""Metriche quantitative da un CSV prodotto da lite3_srbd.py --metrics.

Regime (steady-state): finestra t >= T_SS (default 2.0 s) fino a fine episodio,
per escludere il transitorio di partenza. dt_log = 1/whole_body_frequency = 0.005 s.

Uso:
    python analyze_srbd.py <csv> [--json out.json] [--tss 2.0]
    python analyze_srbd.py --table <dir_con_csv> [--json out.json]

Solo stdlib.
"""
import csv
import json
import math
import os
import sys

DT = 0.005            # 200 Hz log
T_SS = 2.0            # inizio finestra regime (s)
TAU_LIMIT = 30.0      # Nm, soglia di saturazione
FEET = ["FL", "FR", "HL", "HR"]


def read_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows


def col(rows, name):
    out = []
    for r in rows:
        try:
            out.append(float(r[name]))
        except (KeyError, ValueError):
            out.append(float("nan"))
    return out


def _rmse(meas, cmd):
    xs = [(m - c) for m, c in zip(meas, cmd) if not (math.isnan(m) or math.isnan(c))]
    if not xs:
        return None
    return math.sqrt(sum(x * x for x in xs) / len(xs))


def _mean(xs):
    v = [x for x in xs if not math.isnan(x)]
    return sum(v) / len(v) if v else None


def _maxabs(xs):
    v = [abs(x) for x in xs if not math.isnan(x)]
    return max(v) if v else None


def analyze(path, tss=T_SS):
    rows = read_csv(path)
    n = len(rows)
    i0 = min(int(tss / DT), n - 1) if n else 0
    ss = rows[i0:]

    def c(name, sub=None):
        data = col(rows, name)
        return col_slice(data, i0) if sub == "ss" else data

    vx = col(rows, "vx"); vy = col(rows, "vy"); wz = col(rows, "wz")
    cvx = col(rows, "cmd_vx"); cvy = col(rows, "cmd_vy"); cwz = col(rows, "cmd_wz")
    bz = col(rows, "base_z"); roll = col(rows, "roll"); pitch = col(rows, "pitch")

    def sl(a):
        return a[i0:]

    out = {}
    out["rows"] = n
    out["cmd"] = [_mean(cvx), _mean(cvy), _mean(cwz)]
    # velocita' misurate a regime
    out["mean_vx_ss"] = _mean(sl(vx))
    out["mean_vy_ss"] = _mean(sl(vy))
    out["mean_wz_ss"] = _mean(sl(wz))
    # RMSE su regime
    out["rmse_vx_ss"] = _rmse(sl(vx), sl(cvx))
    out["rmse_vy_ss"] = _rmse(sl(vy), sl(cvy))
    out["rmse_wz_ss"] = _rmse(sl(wz), sl(cwz))
    # errore medio a regime (signed)
    out["err_vx_ss"] = (_mean(sl(vx)) - _mean(sl(cvx))) if _mean(sl(vx)) is not None else None
    out["err_vy_ss"] = (_mean(sl(vy)) - _mean(sl(cvy))) if _mean(sl(vy)) is not None else None
    out["err_wz_ss"] = (_mean(sl(wz)) - _mean(sl(cwz))) if _mean(sl(wz)) is not None else None
    # assetto / altezza
    out["roll_maxabs"] = _maxabs(roll)
    out["pitch_maxabs"] = _maxabs(pitch)
    out["roll_mean_ss"] = _mean(sl(roll))
    out["pitch_mean_ss"] = _mean(sl(pitch))
    out["base_z_mean_ss"] = _mean(sl(bz))
    out["base_z_min"] = min([x for x in bz if not math.isnan(x)], default=None)
    out["base_z_max"] = max([x for x in bz if not math.isnan(x)], default=None)
    # coppie: max abs, RMS, % saturazione (su tutti i 12 giunti, tutta la traiettoria)
    all_tau = []
    tau_max_per = []
    for j in range(12):
        tj = col(rows, f"tau{j}")
        tj = [x for x in tj if not math.isnan(x)]
        all_tau.extend(tj)
        tau_max_per.append(max((abs(x) for x in tj), default=0.0))
    if all_tau:
        out["tau_maxabs"] = max(abs(x) for x in all_tau)
        out["tau_rms"] = math.sqrt(sum(x * x for x in all_tau) / len(all_tau))
        out["tau_sat_pct"] = 100.0 * sum(1 for x in all_tau if abs(x) >= TAU_LIMIT) / len(all_tau)
    else:
        out["tau_maxabs"] = out["tau_rms"] = out["tau_sat_pct"] = None
    # contatti: duty reale, mismatch pianificato-reale per piede
    duty = {}
    mism = {}
    for f in FEET:
        real = col(rows, f"{f}_contact")
        plan = col(rows, f"{f}_planned")
        real_ss = [x for x in sl(real) if not math.isnan(x)]
        duty[f] = (sum(real_ss) / len(real_ss)) if real_ss else None
        pairs = [(p, r) for p, r in zip(sl(plan), sl(real)) if not (math.isnan(p) or math.isnan(r))]
        mism[f] = (sum(1 for p, r in pairs if (p > 0.5) != (r > 0.5)) / len(pairs)) if pairs else None
    out["duty_real_ss"] = duty
    out["contact_mismatch_ss"] = mism
    # simmetria gait: duty medio coppie diagonali
    def avg(keys):
        vals = [duty[k] for k in keys if duty[k] is not None]
        return sum(vals) / len(vals) if vals else None
    out["duty_diag_FL_HR"] = avg(["FL", "HR"])
    out["duty_diag_FR_HL"] = avg(["FR", "HL"])
    # caduta / instabilita'
    nan_any = any(math.isnan(x) for x in vx + vy + wz + bz)
    fell = (out["base_z_min"] is not None and out["base_z_min"] < 0.15) or \
           (out["roll_maxabs"] is not None and out["roll_maxabs"] > 0.8) or \
           (out["pitch_maxabs"] is not None and out["pitch_maxabs"] > 0.8) or nan_any
    out["nan_or_inf"] = nan_any
    out["fell"] = bool(fell)
    return out


def col_slice(a, i0):
    return a[i0:]


def fmt(v, p=4):
    if v is None:
        return "  -  "
    if isinstance(v, float) and math.isnan(v):
        return " nan "
    return f"{v:.{p}f}"


def table(dir_path, tss=T_SS):
    files = sorted(f for f in os.listdir(dir_path) if f.endswith(".csv"))
    res = {}
    for f in files:
        res[f[:-4]] = analyze(os.path.join(dir_path, f), tss)
    # stampa tabella
    hdr = ("test", "cmd", "mvx", "mvy", "mwz",
           "rvx", "rvy", "rwz", "roll_mx", "pit_mx", "bz_min",
           "tau_mx", "tau_rms", "sat%", "fell")
    print("| " + " | ".join(hdr) + " |")
    print("|" + "|".join(["---"] * len(hdr)) + "|")
    for name, m in res.items():
        cmd = m["cmd"]
        cmds = f"[{cmd[0]:.1f},{cmd[1]:.1f},{cmd[2]:.1f}]"
        print("| " + " | ".join([
            name, cmds,
            fmt(m["mean_vx_ss"], 3), fmt(m["mean_vy_ss"], 3), fmt(m["mean_wz_ss"], 3),
            fmt(m["rmse_vx_ss"], 3), fmt(m["rmse_vy_ss"], 3), fmt(m["rmse_wz_ss"], 3),
            fmt(m["roll_maxabs"], 3), fmt(m["pitch_maxabs"], 3), fmt(m["base_z_min"], 3),
            fmt(m["tau_maxabs"], 1), fmt(m["tau_rms"], 1), fmt(m["tau_sat_pct"], 1),
            "SI" if m["fell"] else "no",
        ]) + " |")
    return res


if __name__ == "__main__":
    args = sys.argv[1:]
    tss = T_SS
    if "--tss" in args:
        k = args.index("--tss"); tss = float(args[k + 1]); del args[k:k + 2]
    out_json = None
    if "--json" in args:
        k = args.index("--json"); out_json = args[k + 1]; del args[k:k + 2]
    if args and args[0] == "--table":
        res = table(args[1], tss)
    else:
        res = {os.path.basename(args[0])[:-4]: analyze(args[0], tss)}
        print(json.dumps(res, indent=2))
    if out_json:
        with open(out_json, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\nscritto {out_json}")
