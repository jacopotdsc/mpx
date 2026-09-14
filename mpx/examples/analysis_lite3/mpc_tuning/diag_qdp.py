"""Prova rapida: scala il peso di tracking velocita' Qdp (e ricalcola W)
prima di costruire l'MPC. Non modifica file. Uso:
    python diag_qdp.py <out.csv> <vx> <vy> <wz> <qdp_scale>
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import jax
import jax.numpy as jnp
import mpx.config.config_lite3 as config

out, vx, vy, wz, scale = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
config.Qdp = jnp.diag(jnp.array([1, 1, 1])) * 1e3 * scale
config.W = jax.scipy.linalg.block_diag(config.Qp, config.Qrot, config.Qdp, config.Qomega, config.Qgrf)
print(f"[override] Qdp scale = {scale} -> diag {float(config.Qdp[0,0]):.0f}")

import lite3_srbd
lite3_srbd.main(headless=True, steps=1000, metrics=out, cmd=(vx, vy, wz))
