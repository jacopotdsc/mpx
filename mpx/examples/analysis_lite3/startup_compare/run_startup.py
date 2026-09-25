"""Confronto del transitorio di avvio Aliengo (SRBD nativo) vs Lite3 (SRBD portato).

Stesso comando e stesso profilo di onset per entrambi, reset al rispettivo home.
Onset = smoothing esponenziale analitico cmd(t)=target*(1-exp(-t/tau)), tau=0.39s,
che riproduce quello dell'env joystick (`command += 0.05*(target-command)` a 50Hz).

GPU occupata -> CPU forzata prima degli import. 1 env.

Uso:
    python run_startup.py <aliengo|lite3> <out.csv> [vx vy wz] [tau]
"""
import os, sys, math

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import jax
assert jax.default_backend() == "cpu"

robot = sys.argv[1]
out = sys.argv[2]
vx, vy, wz = (float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])) \
    if len(sys.argv) > 5 else (1.0, 0.4, 0.0)
tau = 0.39
extras = {}
for a in sys.argv[6:]:
    if "=" in a:
        k, v = a.split("="); extras[k] = float(v)
    else:
        tau = float(a)

DT = 1.0 / 200.0   # whole-body step (200 Hz per entrambi i config)


def cmd_fn(k):
    a = 1.0 - math.exp(-(k * DT) / tau)
    return (a * vx, a * vy, a * wz)


import jax.numpy as jnp
if robot == "aliengo":
    import mpx.config.config_srbd as cfg
    import srbd_quad as runner
elif robot == "lite3":
    import mpx.config.config_lite3 as cfg
    import lite3_srbd as runner
else:
    raise SystemExit("robot must be aliengo|lite3")

for k, v in extras.items():
    if k == "qz":
        cfg.Qomega = jnp.diag(jnp.array([1.0, 1.0, v / 10.0])) * 1e1
    else:
        setattr(cfg, k, v)
if extras:
    cfg.W = jax.scipy.linalg.block_diag(cfg.Qp, cfg.Qrot, cfg.Qdp, cfg.Qomega, cfg.Qgrf)
    print("[override]", extras)

if robot == "aliengo":
    runner.main(headless=True, steps=400, scene="flat", metrics=out, cmd=cmd_fn)
else:
    runner.main(headless=True, steps=400, metrics=out, cmd=cmd_fn)
