#!/usr/bin/env bash
# Esegue la suite di test SRBD MPC del Lite3 con lite3_srbd.py.
# Un processo per comando (stato iniziale fisso, 5 s = 1000 step @200Hz).
# Uso: bash run_suite.sh <out_dir>
set -u
OUT="${1:?uso: run_suite.sh <out_dir>}"
STEPS="${STEPS:-1000}"
cd "$(dirname "$0")/../.."   # -> mpx/mpx/examples
export PYTHONPATH="$HOME/Desktop/repo_lite3/mpx:$HOME/Desktop/repo_lite3/mujoco_playground"
mkdir -p "$OUT"

# nome  vx vy wz  (NIENTE vx=1.5: escluso per scelta, vedi §5)
run () {
  local name="$1" vx="$2" vy="$3" wz="$4"
  echo ">>> $name  cmd [$vx,$vy,$wz]  $(date '+%T')"
  conda run --no-capture-output -n mjpl python lite3_srbd.py \
     --headless --steps "$STEPS" --cmd "$vx" "$vy" "$wz" \
     --metrics "$OUT/${name}.csv" 2>"$OUT/${name}.err" | grep -vi "os.fork\|_fork_exec\|self.pid"
  echo "    exit=${PIPESTATUS[0]}  righe=$(wc -l < "$OUT/${name}.csv" 2>/dev/null)"
}

run vx1p0        1.0 0.0 0.0
run vy0p4        0.0 0.4 0.0
run vx1p0_vy0p4  1.0 0.4 0.0
run wz0p6        0.0 0.0 0.6
run vx1p0_wz0p6  1.0 0.0 0.6
run vy0p4_wz0p6  0.0 0.4 0.6
echo "=== SUITE COMPLETATA $(date '+%T') -> $OUT ==="
