"""Prova CPU (1 env) dell'effetto del peso di yaw-rate dell'MPC sull'oscillazione di wz.

Obiettivo (consegna 2026-09-24): ridurre le oscillazioni di wz della baseline a
vx=1.0 senza peggiorare il tracking di vx o la stabilita'. Sovrascrive SOLO la
componente z di Qomega (peso sulla yaw-rate) e ricalcola W, PRIMA che
lite3_srbd.main costruisca il wrapper MPC. Un parametro alla volta.

La GPU e' occupata: forziamo JAX su CPU PRIMA di importare jax.
Il comando e' RAMPATO 0->target in `ramp_s` s e poi tenuto: il baseline a step
vx=1.0 cade nel transitorio (vedi REPORT_mpc_lite3.md), quindi per misurare
l'oscillazione a regime serve un profilo stabile. Stesso reset (keyframe home)
e stesso comando per baseline e varianti.

Uso:
    python diag_yaw.py <out.csv> <vx> <vy> <wz> <yaw_scale> [ramp_s] [key=val ...]
    yaw_scale=1.0 -> baseline (Qomega_z = 10). yaw_scale=5 -> Qomega_z = 50.
    key=val extra (float) su config, es. duty_factor=0.7 step_freq=1.4.
"""
import os
import sys

# --- forza CPU prima di qualunque import di jax ---
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# examples/ (per lite3_srbd) e repo root; mpx e' installato come pacchetto.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import jax
import jax.numpy as jnp

assert jax.default_backend() == "cpu", f"backend != cpu: {jax.default_backend()}"

import mpx.config.config_lite3 as config

out = sys.argv[1]
vx, vy, wz = float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
yaw_scale = float(sys.argv[5])
rest = sys.argv[6:]
ramp_s = 1.5
extras = []
for a in rest:
    if "=" in a:
        extras.append(a)
    else:
        ramp_s = float(a)

# override SOLO la yaw-rate (componente z di Qomega); wx,wy invariati.
config.Qomega = jnp.diag(jnp.array([1.0, 1.0, yaw_scale])) * 1e1
# override extra (duty_factor, step_freq, step_height, ...) come float su config.
for kv in extras:
    k, v = kv.split("=")
    setattr(config, k, float(v))
    print(f"[override] config.{k} = {getattr(config, k)}")
config.W = jax.scipy.linalg.block_diag(
    config.Qp, config.Qrot, config.Qdp, config.Qomega, config.Qgrf
)
print(f"[override] Qomega = diag(10, 10, {float(config.Qomega[2,2]):.0f})  (yaw_scale={yaw_scale})")

import lite3_srbd

DT = 1.0 / 200.0
ramp_steps = max(1, int(ramp_s / DT))


def cmd_fn(step):
    a = min(1.0, step / ramp_steps)
    return (a * vx, a * vy, a * wz)


lite3_srbd.main(headless=True, steps=1000, metrics=out, cmd=cmd_fn)
