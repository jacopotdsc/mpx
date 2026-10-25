# Tita hybrid final: paired episode archive

This directory evaluates the four final policies from the two Tita hybrid
plans: RA3 and RS4 from `tita_hybrid_plan`, plus RA4 and RA5 from
`tita_hybrid_plan_v2`. Every policy is paired with MPC-only from the same
reset key, terrain and command.

The archive uses the command sets from `tita_hybrid_plan_v2/eval_robust.py`:
8 commands × 5 random spawns (40 episodes) per selected policy/terrain.
`seed=20261019` is the base for the stable list of reset keys and can be
changed with `--seed`. The old aggregate JSON files do not contain enough
episode data to recover their individual random spawns.

The targeted reruns follow the model selection in the prior results: RA4 on
rough and Perlin, RA5 on slope10. Flat already has 40 paired episodes for all
four checkpoints. Thus RA5 is not rerun on rough/Perlin, where it was not
selected; RA3 and RS4 remain available as historical aggregate references.
The `prior_results/` directory preserves the earlier result tables, aggregate
JSONs, and selected paired sample videos/CSVs/rollouts.

## Run

Use the `mjpl` Python environment from the examples directory or any working
directory:

```bash
/home/jacopo/miniconda3/envs/mjpl/bin/python \
  tita_rl/test/mpx/mpx/examples/tita_hybrid_final/build_archive.py --simulate
/home/jacopo/miniconda3/envs/mjpl/bin/python \
  tita_rl/test/mpx/mpx/examples/tita_hybrid_final/build_archive.py --render
```

To run both phases together, pass `--simulate --render`. A subset can be
selected with `--terrains flat,rough`, using `flat`, `rough`, `perlin`, and
`slope10`.

## Outputs

- `manifest.json`: policy/checkpoint, command, spawn key, results and file
  paths for every newly collected episode.
- `rollouts/<terrain>/*.npz`: MPC and residual qpos, done/alive flags and
  per-step velocity, orientation and heading drift measurements.
- `data_csv/<episode>.csv`: paired per-step command and tracking telemetry.
- `data_csv/episode_metrics.csv`: one row per policy/terrain/command/spawn,
  including fall outcome, tracking, body tilt, heading drift and review flags.
- `plots/<episode>.png`: paired velocity-tracking plots styled after
  `compare.py`.
- `videos/<terrain>__<policy>_candidates.mp4`: paired videos for recoveries
  (MPC falls, residual survives), residual episodes with review flags, and
  representative flat-terrain straight runs.
- `videos/video_index.csv`: episode list for the rendered videos.

The first pass keeps all episode data. Quality flags are prompts for review,
not filters: heading drift over 90°, mean velocity error over 0.5 m/s or
0.5 rad/s, or mean absolute roll/pitch over 0.2 rad. Videos include the
residual-surviving MPC falls even when one of those flags is present, so
unusual locomotion remains available for manual selection.

`cache/` stores each simulated command batch as soon as it completes. If a
run is interrupted, rerunning the same command resumes from those batches;
the same seed must be used. `resume_tonight.sh` resumes the selected rollouts
on GPU at 23:00 Europe/Rome, then renders the videos on CPU.
