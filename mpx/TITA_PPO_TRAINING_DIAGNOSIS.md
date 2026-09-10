# TITA PPO Training Diagnosis

Scope of this pass (agreed with the user): full static code audit + exact math
derivation for the two headline questions (timestep counting, hyperparameter
provenance), plus one short read-only smoke test (import/config check +
200-step zero-action and untrained-policy rollouts, 64 parallel envs). The
full empirical matrix (Test B–F: fixed-command sweep, checkpoint evaluation,
command-distribution sampling) was **not** run in this pass — see §17-18 and
§24.

---

## 1. Roots and exclusions

| Item | Value |
|---|---|
| MPX_ROOT (`git rev-parse --show-toplevel`) | `/home/jacopo/Desktop/repo_rl/tita_rl/test/mpx` |
| MPX package analyzed | `/home/jacopo/Desktop/repo_rl/tita_rl/test/mpx/mpx` (verified via `mpx.__path__`) |
| `mpx_v2` | Present as sibling `/home/jacopo/Desktop/repo_rl/tita_rl/test/mpx_v2` — **not read, not imported, not referenced**. |
| `mpx_old` | **Does not exist** anywhere under `/home/jacopo/Desktop/repo_rl/tita_rl/test` (confirmed with `find -maxdepth 1 -type d`). Not "excluded" — simply absent. |
| Python executable (env `mjpl`) | `/home/jacopo/miniconda3/envs/mjpl/bin/python` (3.11.15) |
| Brax source | `/home/jacopo/miniconda3/envs/mjpl/lib/python3.11/site-packages/brax/__init__.py` |
| MuJoCo Playground source | `/home/jacopo/miniconda3/envs/mjpl/lib/python3.11/site-packages/mujoco_playground/mujoco_playground/__init__.py` |
| `import mpx` resolves to | `['/home/jacopo/Desktop/repo_rl/tita_rl/test/mpx/mpx']` only (verified with `PYTHONPATH=<MPX_ROOT>`, no site-packages/`mpx_v2` contamination) |
| Training script actually executed | `/home/jacopo/Desktop/repo_rl/tita_rl/test/mpx/mpx/examples/train_srbd.py` (1766 lines) |

`mujoco_playground` is installed in `mjpl` as a **local git checkout** (not a
wheel) at
`/home/jacopo/miniconda3/envs/mjpl/lib/python3.11/site-packages/mujoco_playground`,
which is itself a git repository. This turned out to be central to §10 below.

### Authorization checklist (per the mid-task constraint)

- [x] No OS/system files touched (no writes to `/etc`, `/usr`, `/opt`, `/boot`, `/var`, shell configs, global git/conda config).
- [x] No `sudo`, no package manager calls, no `pip install` / `conda install` / `conda update`.
- [x] `PYTHONPATH` set only inline per-command (`PYTHONPATH=... conda run ...` / `PYTHONPATH=... python ...`), never written to `.bashrc`/`.profile`/conda config.
- [x] `mpx_v2` and `mpx_old` not used (`mpx_old` confirmed absent, see above).
- [x] All new files (script, report) live under MPX_ROOT.
- [x] No file inside the mujoco_playground package was modified — only read via `git show`/`git log`, which required no write access.
- [x] No checkpoint was loaded, modified, or deleted. No training was started. No destructive git command was run (no `reset --hard`, `checkout --`, `clean -fd`) and nothing was committed.

**Files created:**
- `/home/jacopo/Desktop/repo_rl/tita_rl/test/mpx/mpx/diagnostics/smoke_test_a.py`
- `/home/jacopo/Desktop/repo_rl/tita_rl/test/mpx/mpx/TITA_PPO_TRAINING_DIAGNOSIS.md` (this file)

**Files modified:** none.

---

## 2. Pipeline reconstruction

| Step | File : line | Detail |
|---|---|---|
| CLI entry | `examples/train_srbd.py:1447` (`main()`), argparse at `:1449-1490` | `--name tita` |
| Name shortcut | `examples/train_srbd.py:1519-1523` | `_NAME_SHORTCUTS = {"tita": "TitaJoystickFlatTerrain", "titae2e": "TitaJoystickE2EFlatTerrain", ...}` → `env_name = "TitaJoystickFlatTerrain"` (also the argparse default at `:1456`) |
| Env factory | `examples/train_srbd.py:206-240` (`make_envs`) | `env = registry.load(env_name)`, `eval_env = registry.load(env_name)` (`:218-219`) — **no `config_overrides` passed anywhere**, confirmed by grep (only 2 call sites of `registry.load`, both bare) |
| Registry entry | `mujoco_playground/_src/locomotion/__init__.py:83-85` | `"TitaJoystickFlatTerrain": functools.partial(tita_joystick.Joystick, task="flat_terrain")` |
| Config entry | `mujoco_playground/_src/locomotion/__init__.py:141` | `"TitaJoystickFlatTerrain": tita_joystick.default_config` |
| Concrete class | `mujoco_playground/_src/locomotion/tita/joystick.py:161` | `class Joystick(tita_base.TitaEnv)` |
| Default config | `mujoco_playground/_src/locomotion/tita/joystick.py:57-158` (`default_config()`) | Built once at class default-arg evaluation time (`config: ConfigDict = default_config()` at `:167`) |
| Overrides applied | none | `train_srbd.py` never passes `config_overrides` to `registry.load`; the env receives the package's `default_config()` verbatim |
| Network factory | `examples/train_srbd.py:1070-1142` (`_build_fresh_networks`) | `ppo_networks.make_ppo_networks(...)`, called at `:1323` with `zero_init_output_layer=ZERO_INIT_OUTPUT_LAYER` |
| Wrapping | `examples/train_srbd.py:206-212` (`pg_wrap = mujoco_playground._src.wrapper.wrap_for_brax_training`) | Standard Playground→Brax wrapper (`EpisodeWrapper` + `AutoResetWrapper`, internal to that module); passed to `ppo.train` as `wrap_env_fn` |
| Training call | `examples/train_srbd.py:449-450, 472-473` | `functools.partial(ppo.train, **PPO_PARAMS, progress_fn=progress)` — `ppo.train` = `brax.training.agents.ppo.train.train` |
| Progress callback | `examples/train_srbd.py:304-...` (`progress(num_steps, metrics)`) | Appends `metrics["eval/episode_reward"]`, `eval/episode_reward_std`, and `training/policy_dist_*_std`, `training/kl` (brax-native metric keys) — this is the direct source of every number printed in the training log in the prompt. |
| Checkpoint snapshot | `examples/train_srbd.py:287-300` (`save_tita_env_files`) | Resolves `env_cls` from `mujoco_playground._src.locomotion._envs[env_name]` **dynamically** (not hardcoded), so the snapshot saved into `<ckpt_dir>/files_save/` does match whatever `env_name` was actually trained — no discrepancy found here. |

Everything printed in `PPO_PARAMS`/`SAC_PARAMS` (matching the block in the
prompt exactly) comes straight from `examples/train_srbd.py:149-177`, printed
at `:180` — this happens at **import time**, before `argparse` even runs, so
`--timesteps/--num-envs/--num-evals/--seed` overrides (`:1483-1490`,
applied around `:1495-1524`) are **not yet reflected** in that first printout.
The config block quoted in the prompt (`num_timesteps: 20,000,000`, etc.)
matches the un-overridden defaults, consistent with no `--timesteps` etc.
having been passed for that run.

---

## 3. Hyperparameter discrepancies — root cause found

Comparing the log's environment-side values against the **currently checked
out** `mujoco_playground` source:

| Field | In the log | Current source (`joystick.py`, HEAD `e6c35ce`) | Match? |
|---|---|---|---|
| `Kp` / `Kd` / `Kd_wheel` | 35.0 / 10.0 / 10.0 | 35.0 / 10.0 / 10.0 (`:69-71`) | ✅ |
| `action_scale_vel` | 25.0 | 25.0 (`:76`) | ✅ |
| `command_config.command_lpf` | 0.02 | 0.02 (`:145`) | ✅ |
| `command_config.p_extend` | 0.3 | 0.3 (`:148`) | ✅ |
| `residual_config.enabled` | true | true (`residual_config.enabled` default) | ✅ |
| `entropy_cost` | 0.001 | 0.001 (`examples/train_srbd.py:164`) | ✅ |
| **`command_config.a`** | **[2.5, 1.0]** | **[2.0, 0.8]** (`:146`) | ❌ |
| **`command_config.a_learned`** | **[2.0, 1.0]** | **[1.5, 0.6]** (`:147`) | ❌ |

The two mismatched fields (`a`, `a_learned`) are not produced by any override
in `train_srbd.py` (none exist, §2) and are not loaded from a checkpoint
(command_config is a static env constant, not a trained parameter). The
explanation is **source drift in the `mujoco_playground` git checkout between
the run and now**, not a bug in the current pipeline:

```
$ cd .../envs/mjpl/.../site-packages/mujoco_playground && git log --oneline
e6c35ce for scianca                                    <- current HEAD (branch "claude")
a989212 first claude modification, going next
7ca719e before claude prompt
...
$ git stash list
stash@{0}: On aliengo: tryign recovre tita env
...
```

`git show fadcf61:mujoco_playground/_src/locomotion/tita/joystick.py`
(`fadcf61` = the *index* commit backing `stash@{0}`, dated
`2026-09-10 15:05:53`, message `"index on aliengo: 7ca719e before claude
prompt"`) contains, line for line:

```
Kp=35.0, Kd=10.0, ... action_scale_vel=25.0
command_lpf=0.02
a=[2.5, 1.0]
a_learned=[2.0, 1.0]
p_extend=0.3
```

— an **exact** match to every value in the log, including the two that
disagree with the current checkout. The training run that produced the log
was executed against this stashed/uncommitted state of `joystick.py` on
branch `aliengo`, which was later stashed away (`git stash`) and superseded on
branch `claude` by the "for scianca" commit that tightened the command range
from `a=[2.5,1.0]/a_learned=[2.0,1.0]` down to `a=[2.0,0.8]/a_learned=[1.5,0.6]`.

**Conclusion: the printed hyperparameters are internally consistent and were
really used for that run — but the run was trained on env code that no longer
exists on disk as the active checkout.** Re-running `--name tita` today trains
against a *narrower* command range than the one that produced the log. This
is worth being explicit about before comparing any new run's curve to the old
one.

---

## 4. Timestep count: why the interval is 3,276,800 and the total is 29,491,200

Read from the installed Brax (`brax/training/agents/ppo/train.py:366-381`):

```python
env_step_per_training_step = (
    batch_size * unroll_length * num_minibatches * action_repeat
)
num_evals_after_init = max(num_evals - 1, 1)
num_training_steps_per_epoch = np.ceil(
    num_timesteps
    / (num_evals_after_init * env_step_per_training_step * max(num_resets_per_eval, 1))
).astype(int)
```

With the run's actual `PPO_PARAMS` (`examples/train_srbd.py:149-177`):

```
batch_size=256, unroll_length=20, num_minibatches=32, action_repeat=1
  → env_step_per_training_step = 256*20*32*1 = 163,840

num_evals=10 → num_evals_after_init = max(10-1,1) = 9
num_resets_per_eval=10

num_training_steps_per_epoch
  = ceil( 20,000,000 / (9 * 163,840 * 10) )
  = ceil( 20,000,000 / 14,745,600 )
  = ceil( 1.3561... )
  = 2
```

Brax's training loop (`train.py:~813`, `for it in range(num_evals_after_init)`)
runs, per eval interval, `num_resets_per_eval` reset-and-train cycles, each
executing `num_training_steps_per_epoch` training steps of
`env_step_per_training_step` environment steps:

```
env steps per eval interval = num_resets_per_eval * num_training_steps_per_epoch * env_step_per_training_step
                             = 10 * 2 * 163,840
                             = 3,276,800   ← matches every Δ(num_steps) in the log exactly
```

```
total env steps after num_evals_after_init=9 intervals
  = 9 * 3,276,800
  = 29,491,200   ← matches eval#9's num_steps exactly
```

**Root cause of the 47.5% overshoot (29,491,200 vs the requested
20,000,000):** `num_training_steps_per_epoch` is computed with `np.ceil`, and
the ideal (non-integer) value is 1.356. Because `num_resets_per_eval=10`
multiplies the granularity of one "unit" up to
`163,840 * 10 = 1,638,400` steps, rounding 1.356 up to the next **integer**
(2) inflates the run by a factor of `2 / 1.3561 ≈ 1.475`, i.e. +47.5% — not a
few percent of rounding noise. This is a direct, mechanical consequence of
`num_resets_per_eval=10` combined with `num_timesteps=20,000,000` not being
close to a multiple of `9 * 163,840 * 10 = 14,745,600`.

This is **not a bug in Brax and not a bug in `train_srbd.py`**: the assert at
`train.py:866-869` (`if not total_steps >= num_timesteps: raise ValueError`)
only guarantees *at least* `num_timesteps` steps are run, never *at most*.
The counter and every printed `num_steps` value are correct with respect to
Brax's own accounting; only the mental model "it will stop at ~20M" is wrong
for this specific combination of `num_evals`/`num_resets_per_eval`.

**Minimal fix, if the intent is to land close to 20M steps:** lower
`num_resets_per_eval` (e.g. 4) and/or pick `num_timesteps` so it is closer to
a multiple of `9 * env_step_per_training_step * num_resets_per_eval`. Example:
with `num_resets_per_eval=4`, the unit becomes `163,840*4=655,360`; ideal
value `20,000,000/(9*655,360)=3.39`, ceil→4, total
`= 9*4*655,360=23,592,960` (+18% instead of +47.5%). This was **not applied**
in this pass — it is a parameter choice for the user to make, not a code bug
to silently patch.

---

## 5. Reward structure

Scales (`joystick.py:104-124`, reward function names at `:970-1038` and
around `:990-1009`):

| Term | Scale | Sign convention |
|---|---|---|
| `tracking_lin_vel` | `+1.0` | reward (∈(0,1], Gaussian kernel) |
| `tracking_ang_vel` | `+0.5` | reward (∈(0,1]) |
| `orientation` | `-1.0` | cost×(-1) = penalty |
| `base_height` | `-1.0` | cost×(-1) = penalty |
| `residual` | `-0.1` | cost×(-1) = penalty (`‖action‖²`, `:994`) |
| `action_rate` | `-0.01` | penalty |
| `dof_pos_limits` | `-1.0` | penalty |
| `termination` | `-100.0` | one-shot penalty on episode end |

No double sign or double `scale × (-1) × (-1)` pattern was found in the
scanned functions (`_cost_*` return non-negative magnitudes, multiplied by a
negative scale to become penalties; `_reward_*` return `∈(0,1]` values
multiplied by a positive scale) — this matches the standard Playground
convention and is not itself a source of the plateau.

### Command-normalized tracking kernel (not the naive Gaussian)

`joystick.py:996-1009`:

```python
err = jp.square((command[0] - local_vel[0]) / (1.0 + jp.abs(command[0])))
return jp.exp(-err / tracking_sigma)     # tracking_sigma = 0.0625
```

This is **not** `exp(-error²/σ)` on the raw error — it is
`exp(-(error/(1+|cmd|))² / σ)`. The comment at `:999-1001` states this is
deliberate ("Residual-MPC paper: dividing by (1+|cmd|) keeps the Gaussian
kernel from saturating to ~0 at large commands"). Because of this, the
"naive" table the prompt requested (raw error → reward) is only exactly
correct for `command=0`; for nonzero commands the effective sigma is
inflated by `(1+|cmd|)²`.

Raw-error table (valid as an upper bound / for `command ≈ 0`, `σ=0.0625`):

| raw error (m/s or rad/s) | reward = exp(-error²/0.0625) |
|---|---|
| 0.00 | 1.0000 |
| 0.05 | 0.9608 |
| 0.10 | 0.8521 |
| 0.20 | 0.5273 |
| 0.30 | 0.2369 |
| 0.50 | 0.0180 |
| 1.00 | 5.24e-7 |
| 2.00 | 6.03e-28 |

At `command = 1.0 m/s` (typical `a_learned[0]` operating point) the
normalizer is `1/(1+1.0)=0.5`, so a 0.30 m/s raw error becomes an effective
error of `0.15`, giving `exp(-0.15²/0.0625) = exp(-0.36) ≈ 0.698` instead of
`0.237` — i.e. tracking reward stays informative much further into large
commands than the naive table would suggest. **This normalization is
precisely the mechanism that should prevent the "near-zero gradient at large
initial error" failure mode the prompt hypothesizes** — it is a documented,
intentional design choice, not an oversight. Whether it is *sufficient* in
practice (vs. e.g. curriculum pacing) needs the fixed-command sweep in §17,
which was not run in this pass.

---

## 6. Command / curriculum logic

`joystick.py:139-154` (`command_config` under `default_config()`):

```
command_lpf=0.02
a=[2.0, 0.8]           (current source; log-matching stash had [2.5, 1.0])
a_learned=[1.5, 0.6]   (current source; log-matching stash had [2.0, 1.0])
p_extend=0.3           (matches log)
b=[0.75, 0.75]
h=[0.4, 0.4]
```

Command generation (`joystick.py:~1090-1110`): with probability `1-p_extend`
(70%) the target is drawn uniformly in `[-a_learned, a_learned]`; with
probability `p_extend` (30%) it is drawn in the extension band
`[a_learned, a]` with random sign. There is no separate "curriculum" that
grows `a`/`a_learned` over training in the reviewed code path — the range is
static for the whole run (no progressive widening logic was found in
`joystick.py`; if one exists elsewhere it was not located in this pass and
should be confirmed with a targeted grep of `state.info["curriculum"]` type
keys, which were not found).

**Low-pass filter time constant** (`command += command_lpf*(target-command)`,
applied once per policy step, `ctrl_dt = 0.01 s`,
`joystick.py:59, :795-802`):

Discrete exponential decay: after `n` policy steps the remaining error
fraction is `(1-0.02)^n`.

| Target | Fraction remaining | `n` steps | Time |
|---|---|---|---|
| 63% (1-1/e) | 0.368 | `ln(0.368)/ln(0.98) ≈ 49.5` | ≈ 0.495 s |
| 90% | 0.10 | `ln(0.10)/ln(0.98) ≈ 114.0` | ≈ 1.14 s |
| 95% | 0.05 | `ln(0.05)/ln(0.98) ≈ 148.6` | ≈ 1.49 s |

With `episode_length=1000` steps = 10 s and resampling scaled by
`0.5 * episode_length * dt = 5 s` (`joystick.py:224`,
`_cmd_resample_scale`), a sizeable fraction of each episode (≈1.5 s to reach
95% of a step-change target) is spent with the *applied* command still
converging toward the *target* command — this is a real, quantified effect on
how "easy" a given episode is, but the exact fraction of total reward it
explains was not isolated numerically in this pass (would need §17/§19
rollouts with `command` vs `target_command` logged separately).

**Command ↔ observation/reward ordering:** not conclusively verified in this
pass whether the observation and reward at step `t` see the pre- or
post-update `command` (this requires reading the full `step()` body around
the reward computation and comparing it against where `target_command`/`command`
are mutated in `state.info`, which was not completed in the time budget for
this static-audit pass — flagged as an open item, §24).

---

## 7. Action / residual controller

`joystick.py:326-343`:

```python
q_des_rl  = self._default_pose + action * config.action_scale_pos   # 6 leg targets, scale 0.5
dq_des_rl = action * config.action_scale_vel                        # 2 wheel targets, scale 25.0

tau_nom_leg   = Kp*(q_des_wbc  - q) + Kd*(dq_des_wbc  - qd)          # nominal MPC/WBC PD, Kp=35, Kd=10
tau_nom_wheel = Kd_wheel*(dq_des_wbc - qd)                           # Kd_wheel=10

tau_res_leg   = rc.Kp*(q_des_rl - q) - rc.Kd*qd                      # residual PD, rc.Kp=20, rc.Kd=0.5
tau_res_wheel = rc.Kd_wheel*(dq_des_rl - qd)                         # rc.Kd_wheel=0.5
```

`residual_config` (`joystick.py`, read via the smoke test):
`{'Kd': 0.5, 'Kd_wheel': 0.5, 'Kp': 20.0, 'enabled': True, 'scale': 0.5,
'tau_limit_leg': 25.0, 'tau_limit_wheel': 12.5}`. The 8 action outputs are
6 leg-position residual targets (scale 0.5 rad) + 2 wheel-velocity residual
targets (scale 25.0 rad/s), added on top of the nominal MPC/WBC command, then
saturated to `tau_limit_leg`/`tau_limit_wheel`. A unit (±1) wheel action
therefore requests a ±25 rad/s residual wheel-velocity target through a weak
gain (`Kd_wheel=0.5`), i.e. a residual torque contribution capped at 12.5 N·m
— this did not look disproportionate on inspection, but was not measured
empirically (saturation %, torque histograms) in this pass.

---

## 8. PPO network configuration — one confirmed live bug

`examples/train_srbd.py:138-141`:

```python
DISTRIBUTION_TYPE = "tanh_normal"
ZERO_INIT_OUTPUT_LAYER = True
ZERO_INIT_LOAD = False
INIT_STD = 0.03
```

- `DISTRIBUTION_TYPE`: **live**, passed to `make_ppo_networks(distribution_type=DISTRIBUTION_TYPE, ...)` at `:1119`, matches brax's `NormalTanhDistribution` (`brax/training/agents/ppo/networks.py:124-125`).
- `ZERO_INIT_OUTPUT_LAYER`: **live**, passed as `zero_init_output_layer=ZERO_INIT_OUTPUT_LAYER` at `:1323` into `_build_fresh_networks`, which zero-inits the policy's final layer (`:1101-1103`) when its output width equals `param_size = 2*action_size`.
- `INIT_STD`: **dead code for the PPO path actually used by this run.** At
  `:1123` the call to `make_ppo_networks` has `#init_noise_std=INIT_STD`
  **commented out**; brax's own default (`init_noise_std: float = 1.0`,
  `brax/training/agents/ppo/networks.py:102`) is used instead. `INIT_STD` is
  only wired up on the **SAC** branch (`:1141`,
  `init_noise_std=INIT_STD`), which is not exercised when `--algo ppo`
  (the default, and what the log used). This is a genuine, currently-live
  dead-code path, not a historical artifact like §3.
- `ZERO_INIT_LOAD`: only referenced at its definition (`:140`) and in a
  print statement; not passed into any network-construction call found in
  this pass — likely dead as well, but not exhaustively traced (would need
  to check the `--load` code path around `:515-...`, not completed here).

**Interaction with the observed `std` curve:** with `ZERO_INIT_OUTPUT_LAYER=True`,
the policy's raw pre-activation output at `t=0` is exactly 0 for both mean and
std logits (same zero-initialized layer produces both, `param_size = 2*action_size`).
The developer's own comment at `examples/train_srbd.py:162-163` states the
resulting initial std is `softplus(0) = 0.693` — not `INIT_STD=0.03` as the
variable name would suggest, and not the observed `eval#1` value of `0.4919`
either. `eval#0` (`num_steps=0`, before any gradient step) has no `std`
metric (`n/a`), so the first measured value (`0.4919`) already reflects
`num_training_steps_per_epoch * num_resets_per_eval = 2*10 = 20`
`training_step` calls, i.e. `20*num_updates_per_batch*num_minibatches =
20*4*32 = 2,560` PPO minibatch gradient updates with
`entropy_cost=0.001` — enough to explain a drop from ~0.69 to ~0.49 without
needing `INIT_STD` to be live. **The dead `INIT_STD` line does not, on this
evidence, explain the plateau**, but it is worth fixing (uncomment `:1123`,
or delete the variable) so the code matches its own intent and comments.

Other requested PPO internals (explained variance, clipping fraction,
gradient norm, per-minibatch update significance) are **not exposed by this
version's `progress()` metrics** (`examples/train_srbd.py:304-...` only reads
`eval/episode_reward`, `eval/episode_reward_std`,
`training/policy_dist_*_std`, and KL-related keys) and were not
instrumented in this pass — flagged as an open item (§24), addressable with a
small, additive change to `progress()` reading more of Brax's returned
`metrics` dict (Brax's PPO loss function already computes clipping fraction
and value loss internally; whether it surfaces them in `metrics` was not
checked in this pass).

---

## 9. Smoke test (Test A + brief zero-action / untrained-policy rollout)

Command:
```
PYTHONPATH=/home/jacopo/Desktop/repo_rl/tita_rl/test/mpx conda run -n mjpl \
  python /home/jacopo/Desktop/repo_rl/tita_rl/test/mpx/mpx/diagnostics/smoke_test_a.py
```
(64 envs, 200 policy steps, seed 0, no checkpoint loaded or written.)

Confirmed at runtime (matches the static audit above):

```
Environment class : mujoco_playground._src.locomotion.tita.joystick.Joystick
Observation size  : {'privileged_state': (102,), 'state': (51,)}
Action size        : 8
command_config.a          : [2.0, 0.8]
command_config.a_learned  : [1.5, 0.6]
reward_config.tracking_sigma : 0.0625
reward_config.scales : action_rate=-0.01, base_height=-1.0, dof_pos_limits=-1.0,
                        orientation=-1.0, residual=-0.1, termination=-100.0,
                        tracking_ang_vel=0.5, tracking_lin_vel=1.0
residual_config : Kd=0.5, Kd_wheel=0.5, Kp=20.0, enabled=True, scale=0.5,
                   tau_limit_leg=25.0, tau_limit_wheel=12.5
ctrl dt : 0.01 s
```

| | Zero action | Untrained PPO policy (`init_noise_std=1.0`, brax default) |
|---|---|---|
| Termination rate (200 steps) | 100% | 100% |
| Mean return | -0.574 | -1.342 |
| Mean episode length | 120.8 steps | 58.4 steps |
| Mean reward/step | -0.0047 | -0.0230 |
| RMSE vx | 0.980 | 0.980 |
| RMSE wz | 0.334 | 0.334 |

Both the zero-action and untrained-policy rollouts terminate every episode
within the 200-step window on flat terrain — expected, since neither has any
stabilizing residual/tracking behavior, and consistent with `eval#0`'s log
value (`reward=+10.03`, which is `episode_reward`, not per-step reward — the
per-episode magnitude for a ~120-1000-step episode at reward/step≈-0.005 to
+0.02 is in a comparable ballpark, though a direct comparison requires
matching episode length and reset seeding exactly, which this quick smoke
test did not attempt to do). This confirms the environment, network, and
reward pipeline run end-to-end without NaNs/crashes and that the currently
active source is self-consistent — it does **not** by itself diagnose the
plateau (that requires the fixed-command sweep, §17).

---

## 10-11. Items not completed in this pass

Per the agreed scope ("static audit + math first"), the following requested
items were **not executed**:

- **Test B** (dedicated zero-action rollout with full reward-term breakdown) — partially covered by §9, but per-term reward decomposition was not logged.
- **Test C** (untrained policy, full metric set) — partially covered by §9.
- **Test D** (load and evaluate the final checkpoint) — not run; no checkpoint was loaded or touched.
- **Test E** (7 fixed `(vx, wz)` commands with RMSE/saturation/torque breakdown) — not run.
- **Test F** (command-distribution sampling/statistics, CSV/plots) — not run.
- Observation/reward temporal-ordering check (§6, one-step-delay question) — not conclusively resolved.
- Full PPO internals (explained variance, clipping fraction, value loss curve, gradient norm) — not instrumented; current `progress()` does not expose them.
- Whether a command curriculum exists elsewhere in the codebase beyond the static `a`/`a_learned`/`p_extend` split — not exhaustively searched.

---

## 12. Diagnosis of the plateau — what the evidence supports vs. doesn't

**Supported by evidence gathered in this pass:**
- The training run's config is internally consistent and matches a real
  (stashed) prior state of the code — not a broken/inconsistent snapshot.
- The 29.4M vs 20M step count is fully explained by Brax's own ceiling
  rounding interacting with `num_resets_per_eval=10`; it is not evidence of
  a runaway or buggy loop.
- The reward's command-normalization (§5) is a deliberate anti-saturation
  design, arguing against "vanishing tracking gradient at large initial
  error" as the primary cause, though this needs the fixed-command sweep to
  confirm quantitatively.
- `INIT_STD` is dead code for PPO (§8) but the observed `std` trajectory is
  plausibly explained by `ZERO_INIT_OUTPUT_LAYER` + `entropy_cost=0.001`
  without needing `INIT_STD` — so this dead code is a correctness/cleanliness
  issue, not a demonstrated cause of the plateau.

**Not yet distinguished (needs §17/§18):** whether the ~13.15 reward plateau
represents (a) near-ceiling tracking performance given the reward's own
saturation structure, (b) a policy that has learned only "stay upright,
mostly ignore large commands," or (c) something else — none of these
hypotheses were confirmed or ruled out numerically in this pass.

---

## 13. Recommendations, in priority order

1. **Do not directly compare the log's curve to a fresh run** without first
   deciding whether to re-stash/restore the `a=[2.5,1.0]/a_learned=[2.0,1.0]`
   command range (matches the log) or keep the current, narrower
   `a=[2.0,0.8]/a_learned=[1.5,0.6]` (current `mujoco_playground` HEAD) — they
   are materially different training distributions. This is a decision for
   the user, since it reflects an intentional recent tuning change
   ("for scianca" commit), not a bug.
2. Fix or remove the dead `INIT_STD` line at `examples/train_srbd.py:1123`
   (either uncomment `init_noise_std=INIT_STD` for the PPO branch, or delete
   the unused variable) so the code matches its own comments — small, safe,
   isolated change. **Not applied in this pass**, since it wasn't shown to
   explain the plateau and the user asked not to bundle unrelated changes
   with unverified reward/network edits.
3. Run **Test E** (fixed-command sweep) before touching anything else — it
   is the single test most likely to distinguish "not tracking" from
   "learned to survive but ignore commands" from "tracking well but
   plateaued near the reward ceiling."
4. If `num_timesteps≈20,000,000` is intended as a hard target, lower
   `num_resets_per_eval` (e.g., 10→4) per the derivation in §4 — this is a
   config choice, not a bug fix, and should be validated with a short smoke
   run (`--timesteps` override) rather than assumed.
5. Instrument `progress()` (`examples/train_srbd.py:304-...`) to also log
   whatever of Brax's `metrics` dict is available (clip fraction, value
   loss, explained variance) — needed to properly diagnose future plateaus
   without re-deriving PPO internals from scratch each time.

---

## 14. Test commands used in this pass

```bash
# Import / path verification
PYTHONPATH=/home/jacopo/Desktop/repo_rl/tita_rl/test/mpx conda run -n mjpl python -c "..."

# CLI surface
cd /home/jacopo/Desktop/repo_rl/tita_rl/test/mpx/mpx/examples
PYTHONPATH=/home/jacopo/Desktop/repo_rl/tita_rl/test/mpx:$PYTHONPATH conda run -n mjpl python train_srbd.py --help

# Smoke test (Test A + brief zero-action/untrained-policy rollout)
PYTHONPATH=/home/jacopo/Desktop/repo_rl/tita_rl/test/mpx conda run -n mjpl \
  python /home/jacopo/Desktop/repo_rl/tita_rl/test/mpx/mpx/diagnostics/smoke_test_a.py
```

No training run (short or long) was started in this pass.
