"""Valuta UN set di parametri gait/MPC del Lite3 su una SUITE di comandi (flat, CPU).

Serve a trovare un set coerente per il Lite3: un cambio a singolo parametro che
va bene su vx puro puo' rompere vx+vy (coupling). Qui ogni set e' giudicato sul
caso peggiore della suite, non su un comando solo.

GPU occupata -> CPU forzata prima degli import. 1 env, reset fisso (home),
comando rampato 0->target in 1.5 s poi tenuto (lo step puro cade nel transitorio).
Metriche a regime t>=2.0 s.

Uso:
    python eval_set.py <tag> [key=val ...]
    key: qz (=Qomega_z, default 200), duty_factor, step_freq, step_height,
         clearence_speed. Es:
    python eval_set.py base                         # set attuale su disco
    python eval_set.py s1 qz=200 step_height=0.045 duty_factor=0.62
"""
import os, sys, csv, math, json

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import jax
import jax.numpy as jnp
assert jax.default_backend() == "cpu"
import mpx.config.config_lite3 as config

tag = sys.argv[1]
qz = 200.0
overrides = {}
for kv in sys.argv[2:]:
    k, v = kv.split("=")
    if k == "qz":
        qz = float(v)
    else:
        overrides[k] = float(v)

config.Qomega = jnp.diag(jnp.array([1.0, 1.0, qz / 10.0])) * 1e1
# matrix-weight scales (for peak-error tuning): qdpy=Qdp[vy], qrotxy=Qrot[roll,pitch]
qdpy = overrides.pop("qdpy", 1.0)
qrotxy = overrides.pop("qrotxy", 1.0)
config.Qdp = jnp.diag(jnp.array([1.0, qdpy, 1.0])) * 1e3
config.Qrot = jnp.diag(jnp.array([1e3 * qrotxy, 1e3 * qrotxy, 0.0]))
for k, v in overrides.items():
    setattr(config, k, v)
config.W = jax.scipy.linalg.block_diag(
    config.Qp, config.Qrot, config.Qdp, config.Qomega, config.Qgrf
)

import lite3_srbd

DT = 1.0 / 200.0
ramp_steps = int(1.5 / DT)
SUITE = {
    "vx1":      (1.0, 0.0, 0.0),
    "vx1vy0p4": (1.0, 0.4, 0.0),
    "vy0p4":    (0.0, 0.4, 0.0),
    "wz0p6":    (0.0, 0.0, 0.6),
    "vx1wz0p6": (1.0, 0.0, 0.6),
}
outdir = os.path.join(os.path.dirname(__file__), "sets", tag)
os.makedirs(outdir, exist_ok=True)


def metrics(path):
    rows = list(csv.DictReader(open(path)))
    ss = [r for i, r in enumerate(rows) if i * DT >= 2.0]
    wz = [float(r["wz"]) for r in ss]; cwz = [float(r["cmd_wz"]) for r in ss]
    vx = [float(r["vx"]) for r in ss]; cvx = [float(r["cmd_vx"]) for r in ss]
    vy = [float(r["vy"]) for r in ss]; cvy = [float(r["cmd_vy"]) for r in ss]
    roll = [abs(float(r["roll"])) for r in rows]; pitch = [abs(float(r["pitch"])) for r in rows]
    tau = [max(abs(float(r[f"tau{j}"])) for j in range(12)) for r in rows]
    rms = lambda a: math.sqrt(sum(x * x for x in a) / len(a))
    ewz = rms([w - c for w, c in zip(wz, cwz)])
    evx = rms([v - c for v, c in zip(vx, cvx)])
    evy = rms([v - c for v, c in zip(vy, cvy)])
    fell = max(roll) > 0.5 or max(pitch) > 0.5
    return dict(errwz=ewz, peakwz=max(abs(w) for w in wz), errvx=evx, errvy=evy,
                mvx=sum(vx) / len(vx), rollmx=max(roll), pitchmx=max(pitch),
                taumx=max(tau), fell=fell)


print(f"=== SET {tag}: Qomega_z={qz:.0f} " +
      " ".join(f"{k}={v}" for k, v in overrides.items()) + " ===")
res = {}
worst_wz_peak = 0.0
any_fall = False
for name, (vx, vy, wz) in SUITE.items():
    out = os.path.join(outdir, f"{name}.csv")
    lite3_srbd.main(headless=True, steps=1000, metrics=out,
                    cmd=lambda k, vx=vx, vy=vy, wz=wz: (min(1, k / ramp_steps) * vx,
                                                         min(1, k / ramp_steps) * vy,
                                                         min(1, k / ramp_steps) * wz))
    m = metrics(out); res[name] = m
    worst_wz_peak = max(worst_wz_peak, m["peakwz"])
    any_fall = any_fall or m["fell"]
    print(f"  {name:<10} errwz={m['errwz']:.3f} peakwz={m['peakwz']:.3f} "
          f"errvx={m['errvx']:.3f} mvx={m['mvx']:.3f} errvy={m['errvy']:.3f} "
          f"roll={m['rollmx']:.2f} pitch={m['pitchmx']:.2f} tau={m['taumx']:.0f}"
          f"{'  <<FALL' if m['fell'] else ''}")
print(f"  -> VALID={not any_fall}  worst_peak_wz={worst_wz_peak:.3f}")
json.dump({"tag": tag, "qz": qz, "overrides": overrides, "res": res,
           "valid": not any_fall, "worst_peak_wz": worst_wz_peak},
          open(os.path.join(outdir, "summary.json"), "w"), indent=2)
