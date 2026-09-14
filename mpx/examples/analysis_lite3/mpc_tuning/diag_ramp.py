"""Diagnostica non invasiva: esegue lite3_srbd.main con un comando a RAMPA.

Non modifica il controller: usa il parametro `cmd` callable gia' previsto da
lite3_srbd.main (step -> (vx,vy,wz)). Serve solo a distinguere se la caduta a
vx=1.0 e' un transitorio da step aggressivo o un'instabilita' di regime.

Uso:
    python diag_ramp.py <vx> <vy> <wz> <ramp_s> <out.csv>
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import lite3_srbd

vx, vy, wz, ramp_s, out = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
DT = 1.0 / 200.0
ramp_steps = max(1, int(ramp_s / DT))


def cmd_fn(step):
    a = min(1.0, step / ramp_steps)
    return (a * vx, a * vy, a * wz)


lite3_srbd.main(headless=True, steps=1000, metrics=out, cmd=cmd_fn)
