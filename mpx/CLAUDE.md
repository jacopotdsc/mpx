# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A fork of [`iit-DLSLab/mpx`](https://github.com/iit-DLSLab/mpx) — a JAX/MJX library for
GPU-parallel Model Predictive Control and trajectory optimization on legged robots
(paper: arXiv 2506.07823). This fork's active work is **TITA** (a wheeled-legged robot)
reinforcement learning, where an MPC controller is wrapped as a differentiable, batched
environment and a PPO policy learns a residual on top of it.

## Repository layout gotcha

The git root is one level **up** from this package:

- Git root: `/home/jacopo/Desktop/repo_rl/tita_rl/test/mpx` (holds `pyproject.toml`,
  `README.md`, and large audit docs: `AUDIT_MPX_CONTROLLER.md`, `FINAL_REPORT.md`, etc.)
- Python package + all active development: the inner `mpx/` (this directory).

So `import mpx` resolves to this directory, and everything below is relative to here.

## Environment & commands

**The README's install instructions are stale.** It describes a `mpx_env` / Python 3.13 /
`jax[cuda12]` setup, but the environment actually used is different:

- Conda env **`mjpl`**, Python 3.11.15, `jax==0.10.2` (cuda13), `brax==0.14.2`,
  `mujoco_playground` (installed as a **local git checkout**, not a wheel).
- The real env spec is `examples/environment.yml` (untracked). Activate with `conda activate mjpl`.
- Editable install from the git root: `pip install -e .` (after
  `git submodule update --init --recursive` — `jax_ocp_solvers` is a submodule).

Because of the jax 0.10 / brax 0.14 mismatch, `examples/train_srbd.py` installs a
compatibility shim restoring `jax.device_put_replicated` — keep it when editing that file.

There is **no test suite and no linter configured.** The only checked-in check is
`diagnostics/smoke_test_a.py` (import/config sanity + short rollouts). Verify changes by
running the relevant example.

Run examples directly (first run JITs for >1 min; headless rendering auto-selects EGL when
`$DISPLAY` is unset):

```bash
python examples/mjx_quad.py          # whole-body MJX MPC (interactive, arrow keys to drive)
python examples/srbd_quad.py         # SRBD MPC
python examples/train_srbd.py --name tita          # PPO train TITA (residual policy)
python examples/train_srbd.py --eval --name tita [--headless]
python examples/compare.py --name tita --load      # MPC baseline vs residual vs e2e policy
```

`--name` shortcuts in `train_srbd.py`: `tita` → `TitaJoystickFlatTerrain`,
`titae2e` → `TitaJoystickE2EFlatTerrain`, `go1`, `aliengo`. Any other value is passed
straight to the MuJoCo Playground registry. Checkpoints land in
`examples/checkpoints/<env_name>/<timestamp>/` (a few TITA runs are force-tracked in git via
`.gitignore` exceptions).

## Architecture

**Solver core** — `jax_ocp_solvers/` (git submodule): GPU-parallel OCP solvers. Two backends,
selectable via `solver_mode = "primal_dual"` or `"fddp"` in a config: primal-dual iLQR and a
GPU-optimized FDDP. Uses temporal + state-space parallel scans so complexity is
polylogarithmic in horizon rather than linear.

**MPC wrappers** — `utils/`, each wrapping the solver for a different dynamics model:
- `mpc_wrapper.py` (`MPCWrapper`) — whole-body MJX dynamics.
- `mpc_wrapper_srbd.py` (`BatchedMPCControllerWrapper`) — Single Rigid Body Dynamics, **batched**
  over environments; this is the one used for RL. Key surface: `init_state`, `run`,
  `whole_body_run`, `reset`.
- `mpc_wrapper_dfcip.py` — a further variant (DFCIP).
- Supporting: `models.py`, `objectives.py`, `mpc_utils.py`, `offline_solver.py`,
  `rotation.py`, `sim.py`.

**Config-driven** — `config/config_<robot>.py`: each is a plain Python module of constants
(MuJoCo model path, joint/contact names, `dt`, horizon `N`, `mpc_frequency`, gait timers,
`q0`/`p0`, cost weights, `solver_mode`). Examples select one by importing it directly, e.g.
`import mpx.config.config_srbd as config`. To change dynamics/cost/gait, edit the config, not
the wrapper. Note `config_srbd.py` currently targets Aliengo geometry.

**Robot models** — `data/<robot>/` MuJoCo XMLs (`tita`, `aliengo`, `go2`, `unitree_h1`,
`unitree_g1`, `pal_talos`, `acrobot`).

**RL pipeline** (the TITA focus):
- `examples/rl_env_srbd.py` (`QuadrupedMPCEnv(brax PipelineEnv)`) — wraps the batched SRBD MPC
  as a brax env; the policy action is a **residual** added to the MPC command/output. Contains
  the reward/observation/command-sampling logic (`_compute_reward`, `_get_obs`, `_sample_command`).
- `examples/train_srbd.py` — the ~1800-line PPO training/eval entry point: loads a MuJoCo
  Playground env via `registry.load`, trains with brax PPO, snapshots source, writes
  checkpoints. TITA envs come from Playground (`TitaJoystick*`), not from the `config/` system.
- `examples/compare.py` — runs MPC baseline, residual policy, and (optional) end-to-end policy
  through fixed-command sequences; emits per-reward-term CSVs, plots, and videos. Reads the
  command layout from the env's `command_config.names` rather than hardcoding it.

**Analysis / diagnosis artifacts** (read for context on the TITA training investigation):
`TITA_PPO_TRAINING_DIAGNOSIS.md` (this dir) and `examples/analysis_training/` (experiment
plans, reward-vs-tracking studies, aggregated results).
