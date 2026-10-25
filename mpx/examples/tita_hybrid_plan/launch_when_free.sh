#!/bin/bash
# Launch the HRC SAC run(s) when the GPU utilization stays below 15% for 10
# consecutive minutes (mean of 6 samples, 10 s apart, per minute), or at 23:00
# at the latest. Other users may run through the MPS server, whose clients are
# not listed by nvidia-smi, so only the utilization is used.
EX=/home/jacopo/Desktop/repo_rl/tita_rl/test/mpx/mpx/examples
LOGS=$EX/tita_hybrid_plan/logs
mkdir -p $LOGS
mean_util() {
  local sum=0
  for i in 1 2 3 4 5 6; do
    sum=$((sum + $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)))
    sleep 10
  done
  echo $((sum / 6))
}
free_count=0
while true; do
  now=$(date +%H%M)
  util=$(mean_util)
  if [ "$util" -lt 15 ]; then free_count=$((free_count + 1)); else free_count=0; fi
  echo "$(date +%T) mean_util=$util free_count=$free_count"
  if [ $free_count -ge 10 ] || [ "$now" -ge 2300 ]; then break; fi
done
echo "$(date +%T) launching"
source /home/jacopo/miniconda3/etc/profile.d/conda.sh
conda activate mjpl
cd $EX
export XLA_PYTHON_CLIENT_PREALLOCATE=false
stamp=$(date +%Y%m%d_%H%M)
# B5: paper Group 1 (slopes 0/5/10 deg), paper SAC, posture PD, tau_max [5,5,2].
nohup python -u train_srbd.py --name tita --algo sac --seed 0 \
  --env-override '{"terrain_config.scene": "slopes"}' > $LOGS/B5_slopes_paper_pd_tau5_${stamp}.log 2>&1 &
sleep 10
echo "$(date +%T) launched: $(pgrep -f 'train_srbd.py --name tita --algo sac' | tr '\n' ' ')"
