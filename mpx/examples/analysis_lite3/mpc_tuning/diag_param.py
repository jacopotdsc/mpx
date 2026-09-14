"""Prova rapida di override di parametri di config su un comando a STEP.

Sovrascrive attributi di mpx.config.config_lite3 PRIMA che lite3_srbd.main
costruisca il wrapper MPC (che legge i parametri alla costruzione). Non
modifica file su disco: serve solo a valutare pochi valori plausibili prima
di scegliere la modifica definitiva.

Uso:
    python diag_param.py <out.csv> <vx> <vy> <wz> [key=val ...]
Esempi di key: step_freq, duty_factor, step_height, clearence_speed
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import mpx.config.config_lite3 as config

out, vx, vy, wz = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
for kv in sys.argv[5:]:
    k, v = kv.split("=")
    setattr(config, k, float(v))
    print(f"[override] config.{k} = {getattr(config, k)}")

import lite3_srbd
lite3_srbd.main(headless=True, steps=1000, metrics=out, cmd=(vx, vy, wz))
