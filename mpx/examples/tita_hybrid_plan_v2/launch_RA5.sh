#!/bin/bash
# RA5: fine-tune of RA4 on the 10 deg tilted plane (linear commands <= 1 m/s,
# same tracking terms). Starts after 10 consecutive minutes with GPU
# utilization < 15% (user rule).
EX=/home/jacopo/Desktop/repo_rl/tita_rl/test/mpx/mpx/examples
cd $EX
n=0
while [ $n -lt 10 ]; do
  s=0; for i in 1 2 3 4 5 6; do s=$((s + $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits))); sleep 10; done
  u=$((s / 6)); if [ $u -lt 15 ]; then n=$((n + 1)); else n=0; fi
  echo "$(date +%T) util=$u free_minutes=$n"
done
source /home/jacopo/miniconda3/etc/profile.d/conda.sh; conda activate mjpl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
echo "$(date +%T) launching RA5"
python -u train_srbd.py --name tita --algo ppo --seed 6 --timesteps 3000000 --num-evals 3 --load 20261019_065047 \
  --env-override '{"terrain_config.scene": "tilt10", "terrain_config.tilt_deg": 10.0, "randomize_reset": 1.0, "command_config.mode": "joystick", "command_config.a": [1.0, 0.0], "command_config.a_learned": [0.8, 0.0], "episode_length": 1000, "reward_config.scales.tracking_lin_vel": 1.0, "reward_config.scales.tracking_ang_vel": 2.0}' \
  > tita_hybrid_plan_v2/logs/RA5_tilt10_track_finetune_ppo3M.log 2>&1
echo "$(date +%T) RA5 done"
