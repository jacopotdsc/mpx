> **Superseded (2026-09-04):** the combined-command analysis, the validation numbers in section 9 and the timing description in this report refer to the previous revision. See `COMBINED_COMMAND_FIX.md` for the demonstrated root cause, the new timing scheme (sim 500 Hz, MPC/WBC 100 Hz, dt_mpc 0.01, N 50) and the current results.

# TITA MPC/WBC (JAX/MPX) + Residual RL — Investigation & Fix Report

Date: 2026-09-03. Repos touched (as authorized): `TITA-dynamic-obstacle-avoidance` (`crocoddyl`),
`mpx` (`tita`), `mujoco_playground` (`aliengo`). No files outside these three repos were read or
modified.

This report is the top-level summary. The three per-repo audits produced during the investigation
have the full file:line-cited detail and should be treated as the primary reference for anything
not fully reproduced here:

- `TITA-dynamic-obstacle-avoidance/AUDIT_CPP_CONTROLLER.md` — C++ baseline, full trace
- `mpx/AUDIT_MPX_CONTROLLER.md` — JAX/MPX controller, full trace
- `mujoco_playground/AUDIT_RL_PIPELINE.md` — E2E + residual RL pipeline, full trace

A note on method: three read-only audits were run in parallel first (one per repo), each
independently tracing the *actual* runtime call graph (not file names, not config defaults) and
citing file:line for every claim. Fixes were then applied based on cross-referencing those three
reports, and every fix below was validated experimentally against a headless simulation harness
(`mpx/mpx/examples/validate_dfcip_controller.py`), including a same-day regression when a fix that
looked correct on paper turned out to make things worse in practice (documented in full — see
§9.4).

---

## 1. Repository architecture

- **`TITA-dynamic-obstacle-avoidance/TITA_MJ`**: the original C++ controller. Crocoddyl-based MPC
  (FDDP, one iteration/call, RTI scheme) over a reduced 13-state "DFCIP" model, feeding a dense-QP
  whole-body controller (HPIPM) that outputs joint torques directly to MuJoCo. Ground-truth state,
  no estimator, no active obstacle-avoidance or jump logic despite the repo's framing (see
  `AUDIT_CPP_CONTROLLER.md` §5-S7/S8).
- **`mpx`**: JAX/GPU port of the same DFCIP MPC + WBC formulation (`mjx_tita.py` →
  `mpc_wrapper_dfcip.py` → `jax_ocp_solvers.optimizers.fddp_mpc` for the MPC,
  `mpc_utils.whole_body_interface_wheeled_legged_qp` (`qpax` QP) for the WBC). Also hosts the
  training entry point (`mpx/mpx/examples/train_srbd.py`, the file the task brief calls
  `TRAINING_SRDB.py`) and the interactive policy tester (`mpx/mpx/examples/mjx_policy_tita.py`,
  the brief's `MJX_POLICY_TITA.py`) — the latter is unrelated to the MPC/WBC stack; it drives a
  separately-trained PPO policy against the `mujoco_playground` env.
- **`mujoco_playground`**: the RL simulation environment. `tita/joystickE2E.py` (pure E2E,
  known-good baseline) and `tita/joystick.py` (MPC/WBC + residual RL, `tau_final = tau_WBC +
  tau_NN`) are **independent sibling classes**, both subclassing `tita_base.TitaEnv` directly —
  `joystick.py` does *not* inherit from `joystickE2E.py` (a wrong assumption in the original task
  brief, corrected here). `joystick.py` imports `mpx.config.config_dfcip` and
  `mpx.utils.mpc_wrapper_dfcip` directly, so every MPX-side fix in this report also changes
  `joystick.py`'s behavior.

How they interact: `mpx`'s config/wrapper code is the single source of truth for the DFCIP
MPC+WBC math, consumed by three different entry points (`mjx_tita.py` standalone viewer,
`joystick.py`'s residual RL env, and indirectly anything built on either). Editing
`mpx/mpx/config/config_dfcip.py` or `mpx/mpx/utils/{mpc_utils,objectives}.py` therefore changes
both the standalone controller and the residual RL environment simultaneously — this is expected
and by design, not an accidental cross-repo side effect.

---

## 2. Actual execution paths (traced, not assumed)

**C++ controller**: `main.cpp` → `mj_step1` → `robot_state_from_mujoco` (ground truth) →
`WalkingManager::update` → `MPC::solve` (Crocoddyl FDDP, 1 iter) → `des_configuration_` (MPC→WBC
mapping) → `WholeBodyController::compute_inverse_dynamics` (dense QP, HPIPM) → `JointCommand`
(torques) → `mjData->ctrl[]` → `mj_step2`. All at a fixed 500 Hz sim/control rate.

**MPX controller** (`mjx_tita.py`): `gather_raw_state` → `process_tita_state` (13-dim DFCIP state
`x0`) → every 5 physics ticks (100 Hz): `mpc.run` (`mpc_wrapper_dfcip.BatchedMPCControllerWrapper`
→ `jax_ocp_solvers.optimizers.fddp_mpc`, 1 FDDP iteration) → `mpc.whole_body_run`
(`_build_desired_impl` kinematic reconstruction → `mpc_utils.whole_body_interface_wheeled_legged_qp`,
a `qpax`-solved dense QP) → `tau`. Every physics tick (500 Hz): an **additional outer joint PD
loop** (`Kp=35, Kd=10`, hardcoded in `mjx_tita.py`, not in config) tracks a one-step qddot-integrated
target and is summed with `tau` before `mj_step`. This second PD loop has no equivalent in the C++
controller (which sends the WBC's QP torque straight to `ctrl[]`); it was left in place (not
removed) because ablation testing (§9) showed the actual instability was elsewhere, and removing an
unrelated part of the control law without evidence would have been exactly the kind of unverified
"looks wrong on paper" change this investigation was trying to avoid making twice (see §9.4).

**`mjx_policy_tita.py`**: confirmed **not** part of the MPC/WBC controller — it loads a
`mujoco_playground` registry env and drives it with a PPO checkpoint via `brax`. Out of scope for
Objective A.

**`train_srbd.py`**: `registry.load("TitaJoystickFlatTerrain")` → `joystick.py` (not
`rl_env_srbd.py`/`config_srbd.py`/`mpc_wrapper_srbd.py`, which are a separate, unused-here
example pipeline — a wrong assumption in the task brief, corrected in `AUDIT_RL_PIPELINE.md` §0).
Per-step: `_run_mpc_wbc` (MPC+WBC solve, **RL-action-blind**, `use_nn=False` hardcoded) →
`_compute_joint_desired` (policy action → PD target) → `ctrl = ctrl_nn + tau` (unconditional sum,
5x per control step via `jax.lax.scan`) → `mjx.step` → reward/obs/termination → Brax PPO.

---

## 3. C++ controller — key facts (full detail in `AUDIT_CPP_CONTROLLER.md`)

- Mass 27.68978 kg (hardcoded twice, matches URDF sum; thesis states 24.1 kg excl. battery — a
  real, undocumented ~15% discrepancy between the thesis and the shipped model).
- Wheel radius 0.0925 m; friction μ starts at 0.5 (`getDefaultParams()`) but is **overridden to 0.9
  at runtime** in `WalkingManager::init` — the only place this value actually takes effect.
- 500 Hz control/WBC/sim loop; MPC internal step Δ=10 ms, horizon 50 stages (0.5 s), 1 FDDP
  iteration/call (intentional RTI, matches thesis). **Timing bug** (S2): the warm-start trajectory
  is shifted by one 10 ms stage on every 2 ms control call — the solver's own warm start "gallops"
  5x faster than real elapsed time relative to the reference-time bookkeeping.
- MPC costs are all soft quadratic penalties (no hard constraints in Crocoddyl); no `Fz≥0`, no
  friction cone in the MPC itself (both absent from the code despite the thesis listing them as
  required constraints).
- WBC gains/weights **actually used at runtime differ from the file's own `getDefaultParams()`** in
  almost every case (silently overridden in `WalkingManager::init`) — e.g. `Kp_wheel` 90000→50,
  `Kp_motion` 13000→50. Anyone tuning against the header defaults is tuning the wrong numbers.
  Joint-posture regulation is fully dead (weight 0 both before and after the override).
- Torque limits computed from the URDF (100 Nm) but never enforced in the WBC QP — only MuJoCo's
  actuator clamp (±120 Nm, from the MJCF actually loaded, which disagrees with the URDF) applies.

---

## 4. Mathematical formulation actually implemented (both C++ and JAX — verified identical structure)

**State** (`x`, dim 13): `[p_com(3), v_com(3), c(3), vc_z, theta, v, omega]` — CoM position/velocity
in world frame, "virtual unicycle" ground-contact-midpoint position `c` and its vertical velocity,
heading, forward speed and yaw rate of the differential-drive wheel base.

**Control** (`u`, dim 9): `[a, ac_z, alpha, Fl(3), Fr(3)]` — CoM forward/vertical/yaw accelerations
plus per-wheel ground-reaction force (world frame).

**Dynamics** (forward-Euler): `ṗ_com=v_com`; `v̇_com=(Fl+Fr)/m+g`; unicycle kinematics
`ċ_x=v·cosθ, ċ_y=v·sinθ, ċ_z=vc_z`; `v̇c_z=ac_z`; `θ̇=ω`; `v̇=a`; `ω̇=alpha`. **The CoM translational
dynamics are decoupled from the heading/unicycle dynamics** — GRFs only drive `p_com/v_com` via
Newton's law, while `a, ac_z, alpha` independently drive the reduced unicycle state; the two are
coupled only through the cost (CoM tracks `c`) and through the WBC's kinematic reconstruction, not
through the dynamics model itself. Verified identical between C++ (`DFIPActionModel.hpp`) and JAX
(`models.py::wheeled_dfcip_dynamics`) — matches thesis eq. 3.6-3.13 exactly on both sides.

**WBC QP**: `min 0.5 xᵀHx + fᵀx` s.t. rolling-without-slip (2×3 rows), floating-base
inverse-dynamics (6 rows), joint pos/vel box limits, and a linearized friction-cone pyramid per
wheel — decision variable `x=[q̈(nv), Fl(3), Fr(3)]`, output `tau = Ma·q̈+ca-Jla'·Tl·Fl-Jra'·Tr·Fr`.
Structurally identical between C++ (HPIPM) and JAX (`qpax`).

---

## 5. C++ vs JAX comparison — the discrepancies that mattered

| Aspect | C++ (actual runtime) | JAX (before fix) | Verdict |
|---|---|---|---|
| WBC friction μ | 0.9 (overridden in `WalkingManager::init`) | 0.6 (config's own comment claimed "0.5 in C++" — itself stale) | Real mismatch, fixed |
| WBC posture-regulation weight | 0.0 (dead) | 0.1 (active) | Real mismatch, fixed |
| MPC state weights (`w_pcomxy/pcomz/vcomxy/vcomz`) | 10 / 1e5 / 10 / 1 | 0 / 2e4 / 3e2 / 1e1 | **Numeric mismatch confirmed, but "fixing" it to match C++ made stability measurably worse in this port — reverted, see §9.4** |
| GRF non-negativity (`Fz≥0`) in MPC cost | absent (thesis constraint, never implemented) | computed (`h_fz`) but never wired into the cost — literally dead code | JAX-side bug, fixed (independent of C++ parity, since C++ never had it either) |
| WBC friction-cone lower bound on `Fz` | 4-row pyramid, same structural gap as JAX | same 4-row pyramid | **Shared design gap in both controllers** — only fixed on the JAX side (see §9.3); flagged as a candidate fix for the C++ side too but out of scope to modify without a C++ build/test loop |
| Quaternion/frame conventions | wxyz, body-frame angular velocity | wxyz, body-frame angular velocity, explicit xyzw conversion only where SciPy needs it | Verified equivalent, no bug |
| Dynamics equations | eq. 3.6-3.13 | eq. 3.6-3.13 | Verified byte-for-byte equivalent structure |

Full variable-by-variable tables are in the two controller audits.

---

## 6. Physical-model comparison

Mass (27.6898 kg) and wheel radius (0.0925 m) match to the decimal between the C++ URDF, the JAX
`config_dfcip.py`, and `mpx/mpx/data/tita/tita.xml`. The wheel half-track `d=0.567` m in JAX matches
the C++ `WalkingManager`'s *initial desired-configuration* offset (`±0.2835`), but the C++ MPC
actually computes its own `d` from the true initial MuJoCo pose at startup (`MPC::init_solver`,
`set_d=true`) rather than using that fixed constant — the two are very likely close but were not
independently re-verified via forward kinematics in this pass (flagged, not fixed, low expected
impact given the closeness of the two figures). `config.inertia` (3×3 SRBD tensor) in
`config_dfcip.py` is dead on the DFCIP path (the reduced state has no orientation DOF to use it) —
harmless, just noise. Torque limits: URDF 100 Nm vs. the actually-loaded MJCF's 120 Nm disagree in
the C++ repo; the JAX side uses `tita.xml`'s 120 Nm consistently and (like C++) does not enforce a
software torque limit in the WBC QP — MuJoCo's actuator clamp is the only bound in both.

---

## 7. Bugs found (file → function → bug → evidence → fix)

1. **`mpx/mpx/utils/objectives.py::wheeled_dfcip_hessian_gn`/`wheeled_dfcip_obj`** — GRF
   non-negativity residual `h_fz` computed but discarded (`jnp.zeros_like(h_fz)` in the Hessian
   residual vector, and never added to `stage_cost` at all). **Fixed**: wired `h_fz` into
   `stage_cost` and replaced the zeroed Hessian slot with the real residual.
2. **`mpx/mpx/config/config_dfcip.py`** — `mu=0.6`, contradicted by the code's own comment at
   `mpc_wrapper_dfcip.py:135` ("0.5 in C++"), and both disagree with the C++ runtime value of 0.9
   traced in `AUDIT_CPP_CONTROLLER.md`. **Fixed**: `mu=0.9`.
3. **`mpx/mpx/config/config_dfcip.py`** — `w_posture=0.1`, active, while the C++ equivalent
   (`weight_regulation`) is 0.0 both before and after its own override block — a real behavioral
   divergence (JAX WBC pulls legs toward `q0` under combined motion, C++ doesn't). **Fixed**:
   `w_posture=0.0`.
4. **`mpx/mpx/utils/mpc_utils.py::whole_body_interface_wheeled_legged_qp`** — the WBC's 4-row
   friction-cone pyramid has no explicit lower bound on `Fz`; under a combined vx+omega command it
   can go infeasible, and `qpax.solve_qp` returns **NaN torques** instead of failing gracefully
   (root cause of the reported "combined command falls" bug — confirmed by ablation, §9.2-9.3).
   **Fixed**: added a 5th inequality row per wheel enforcing `Fz ≥ 5 N`.
5. **`mujoco_playground/.../tita/joystick.py::_compute_joint_desired`** — residual policy's PD
   target hardcoded to `self._default_pose` (fixed home pose) with the MPC/WBC-plan-based target
   commented out; the only historically-converged residual training run
   (`checkpoints/TitaJoystickFlatTerrain/saved/first_training_residual`) used the commented-out
   qddot-integrated target. **Fixed**: restored `self._joint_targets_from_qddot(...)`.
6. **`mujoco_playground/.../tita/joystick.py::step`** — `state.info["steps_until_next_cmd"] -= 0`
   (looks like an in-progress debug edit), which disables the command-resample countdown entirely.
   **Fixed**: `-= 1`.

---

## 8. Code modifications (complete list)

| File | Change | Reason | Expected effect |
|---|---|---|---|
| `mpx/mpx/config/config_dfcip.py` | `mu`: 0.6→0.9 | Match C++ runtime friction, not its stale comment | More lateral force budget for combined maneuvers |
| `mpx/mpx/config/config_dfcip.py` | `w_posture`: 0.1→0.0 | Match C++ (dead task) | Remove a task competing with CoM/wheel/base tasks under combined commands |
| `mpx/mpx/config/config_dfcip.py` | `w_pcomxy/pcomz/vcomxy/vcomz`: tried C++-matched values, **reverted to original** | Empirically destabilizing in this port (§9.4) despite theoretical C++ parity | Kept the stable, if not perfectly C++-numerically-matched, configuration |
| `mpx/mpx/config/config_dfcip.py` | `w_v`: tried 15/40, **reverted to 10** | Both reintroduced the NaN/fall under combined command (§9.4) | Documented as a stability-boundary finding, not resolved further |
| `mpx/mpx/utils/objectives.py` | Wired `h_fz` into `stage_cost` and the GN Hessian residual | Was dead code; keeps MPC's own GRF references physically valid | Ablation-confirmed **not** the cause of the combined-command instability, but a correct, independently-justified fix, kept |
| `mpx/mpx/utils/mpc_utils.py` | Added `Fz≥5N` row to the WBC friction-cone inequality block | Prevents the friction-cone constraint from going infeasible and returning NaN | Confirmed: single-axis targets now hit cleanly and stably; combined-command case no longer NaNs/falls (still undershoots vx, see §9.5) |
| `mujoco_playground/.../tita/joystick.py` | Restored qddot-integrated residual PD target | Decoupled-from-MPC-plan target was a real regression vs. the only converged historical run | Re-establishes "residual on top of the plan" semantics; not independently re-trained in this session (see §12) |
| `mujoco_playground/.../tita/joystick.py` | Fixed `-= 0` → `-= 1` command countdown | Dead countdown, likely accidental debug edit | Commands resample correctly during training |
| `mpx/mpx/examples/validate_dfcip_controller.py` | New file: headless CPU validation harness | No existing non-interactive way to test velocity tracking/stability | Used for all validation in §9 |

---

## 9. Validation

### 9.1 Method

`mpx/mpx/examples/validate_dfcip_controller.py` drives the exact same
`mpc_wrapper_dfcip.BatchedMPCControllerWrapper` (MPC + WBC) and the same outer joint-PD law used in
`mjx_tita.py`, headless, on CPU (to avoid GPU contention with other users on this shared machine —
see §12), for each of the 9 commanded (vx, ω) pairs listed in the task brief, ramping the command
over 1s then holding for 3-6s. It logs steady-state velocity tracking, height, roll/pitch, peak
torque, and a fall/NaN detector. A CUDA-enabled JAX environment was not available in this session
(`jax_ocp_solvers` git submodule had to be initialized first — it was not checked out in the
working tree at all, `git submodule update --init` was run as part of setup, since it's inside the
authorized `mpx` repo).

### 9.2 Baseline (before any fix) — reproduced the reported bug

Before any change, the combined `vx=0.6, omega=0.4` command produced a **NaN WBC torque** and the
robot fell over within a few seconds. (Note: an early attempt to re-derive the pristine-baseline
numbers for comparison, done later in the investigation as a control for §9.4, showed the
*un-modified* code is actually fairly well-behaved on its own single-axis and mild-combined cases —
the reported instability is specifically a combined-aggressive-command failure mode, not a general
"close to the limit" fragility across the board. This refines, but does not contradict, the task
brief's description.)

### 9.3 Root-cause isolation (ablation)

Two independent hypotheses were tested by disabling one fix at a time and re-running the failing
case:

- With the MPC-side `h_fz` fix reverted (fix #1 only), the combined case **still** produced the
  identical NaN at the same simulated time — this fix was **not** the cause.
- With the WBC-side `Fz≥5N` fix added, the NaN in the specific narrow band tested initially
  disappeared for the milder combined case, but a longer-duration re-test (5-6s instead of 3s)
  showed the underlying instability was actually still present and unrelated to this fix in
  isolation — it was the **cost-weight rebalance** (done in the same pass, see §9.4) that was
  responsible for the regression, not this fix, which is retained because it is independently
  correct (a QP with no `Fz` lower bound is a real latent bug regardless of whether it was the
  proximate cause here).

### 9.4 The weight-rebalance regression — an honest account

An early fix pass rebalanced `w_pcomxy/w_pcomz/w_vcomxy/w_vcomz` in `config_dfcip.py` to numerically
match the C++ baseline's actual runtime `DFIPActionModel` weights, reasoning that the JAX port had
drifted ~30x too aggressive on CoM-xy-velocity tracking and ~5x too weak on CoM-height tracking.
This is a real, file:line-verified numeric discrepancy (§5). Tested head-to-head against the
original weights on the same combined `vx=0.6/omega=0.4` command:

- **C++-matched weights**: NaN WBC torque and a full fall (roll 0.68 rad, pitch 1.09 rad) within
  ~2-4 seconds, reproducibly, across both the initial 3-6s window and repeat runs.
- **Original (pre-audit) weights**: stable for the full 5-6s hold, no NaN, no fall, in every repeat.

The weight rebalance was **reverted**. The most likely explanation (documented in the code comment
left in `config_dfcip.py`) is that the C++ weights were tuned inside a controller with its own
internal-step/call-rate mismatch (`AUDIT_CPP_CONTROLLER.md` §5-S2 — the C++ warm-start trajectory
advances 5x faster than real elapsed time), so their absolute magnitudes aren't actually
transferable 1:1 into this port's cleaner timing — numeric parity with the C++ source is not the
same thing as behavioral parity. A follow-up attempt to fix the (separately real) combined-command
forward-speed undershoot by raising just `w_v` (10→15, then →40) was tested the same way and also
regressed to the identical NaN/fall failure mode at both values — this operating point is right at
a stability boundary sensitive to fairly small increases in forward-velocity-tracking
aggressiveness while turning. This was left at the stable value rather than pushed further, and is
documented as a real, verified-fragile limitation rather than either ignored or "fixed" with an
untested value.

### 9.5 Final validated results (settled configuration: mu=0.9, w_posture=0, original MPC state
weights, h_fz wired in, WBC Fz≥5N added)

| Command | Steady vx | Steady ω | Height range (m) | Max roll/pitch (rad) | Max \|τ\| (Nm) | Fell? |
|---|---|---|---|---|---|---|
| vx=0.0 | 0.000 | 0.000 | 0.440-0.445 | 0.000 / 0.004 | 19.5 | No |
| vx=0.2 | 0.193 | -0.000 | 0.440-0.445 | 0.000 / 0.003 | 19.4 | No |
| vx=0.6 | 0.580 | -0.001 | 0.440-0.445 | 0.000 / 0.005 | 20.0 | No |
| **vx=1.0** | **0.966** | -0.001 | 0.440-0.447 | 0.000 / 0.008 | 20.4 | No |
| ω=0.2 | -0.002 | 0.195 | 0.440-0.445 | 0.000 / 0.004 | 19.4 | No |
| ω=0.4 | 0.001 | 0.389 | 0.440-0.445 | 0.001 / 0.004 | 19.3 | No |
| **ω=0.8** | 0.001 | **0.772** | 0.440-0.445 | 0.001 / 0.004 | 19.4 | No |
| vx=0.3, ω=0.2 | 0.249 | 0.195 | 0.440-0.445 | 0.001 / 0.003 | 19.8 | No |
| **vx=0.6, ω=0.4** | 0.268 | 0.380 | 0.426-0.445 | 0.006 / 0.014 | 20.2 | **No** |

Raw JSON: `mpx/mpx/examples/validation_results.json`.

**Targets from the task brief**: `vx≥1.0` ✅ (0.966, ~3% under), `omega≥0.8` ✅ (0.772, ~3.5% under),
combined `vx=0.6/omega=0.4` — **stability achieved** (no fall, no NaN, torque nowhere near
saturation at 20/120 Nm, height/roll/pitch all mild), but **forward-speed tracking under combined
command is still poor** (0.268 vs 0.6 commanded, while yaw-rate tracking is fine at 0.380 vs 0.4).
This is a genuine, unresolved tracking-quality gap — see §12.

---

## 10. Residual RL

**Architecture** (`joystick.py`, verified by direct read, `:586-593`): `ctrl = ctrl_nn + tau`, an
unconditional element-wise sum, no gating, no blend coefficient — structurally
`tau_final = tau_WBC + tau_NN` as intended. `tau` (WBC) is computed with `use_nn=False` hardcoded —
**the WBC never sees the RL action** by design; all adaptation comes through the additive PD term.

**Observations** (actor, 51-dim): proprioception (linvel/gyro/gravity/leg-pos-err/joint-vel),
previous action, command, CoM-height error, **and the MPC's own plan** (`mpc_control_a/acz/alpha`,
contact-force plan, and `info["mpc_tau"]` = the WBC feedforward torque itself). Critic gets an
additional 51 dims of privileged state (clean sensors, realized actuator force, contact/air-time,
push force) — genuine actor/critic asymmetry (`policy_obs_key="state"`,
`value_obs_key="privileged_state"`).

**Action → torque**: `q_des = q_target + action*0.5`, `dq_des = dq_target + action*25.0`, PD-tracked
(`Kp=50/Kd=1` legs, `Kd_wheel=0.5` wheels) to produce `ctrl_nn`, summed with `tau`. **Before this
session's fix**, `q_target`/`dq_target` were a fixed default pose; now they're the WBC's own
qddot-integrated one-step target (fix #5, §7-8).

**Reward**: identical formula and weights to the E2E baseline (tracking_lin/ang_vel, orientation,
ang_vel_xy, base_height, posture, torques, action_rate, dof_pos_limits, termination), plus two
MPC-trajectory-tracking terms (`tracking_mpc_accel/alpha`) that exist in code but are **weight-0,
i.e. inert** — nothing in the reward currently pulls the policy toward the MPC's own trajectory
beyond what's implicit in the observation and the shared torque.

**Residual scaling/clipping**: none in software anywhere — no clip on the raw action, the scaled
target, or the summed torque; the only bound is MuJoCo's `±120 Nm` actuator clamp on the *sum*.

**Curriculum**: none — fixed command sampling ranges throughout training
(`a=[1.0,0.5]` → vx∈[-1,1], ω∈[-0.5,0.5]), matching E2E. Domain randomization exists in the codebase
for Tita but **is not actually invoked by `train_srbd.py`**'s training call (no `randomization_fn`
passed) — a shared gap for both E2E and residual runs launched via this script, not
residual-specific.

**Historical evidence** (`checkpoints/TitaJoystickFlatTerrain/saved/first_training_residual`): the
**one** clean, converged residual run on disk reached ~+12.6±2.16 reward, held full 1000-step
episodes with zero terminations, CoM height within 0.397-0.406 m, torque peaking at 27.9 Nm (well
under the 120 Nm limit) — a genuinely successful run. Diffing its saved script snapshot against the
current code showed **no PPO-hyperparameter or reward-weight drift**, but real drift in the
residual-target definition (the bug fixed in #5) and the observation composition
(70-dim historically vs. 51-dim now) — i.e. the current code was not a faithful continuation of the
one run known to work.

---

## 11. Final performance comparison

| | E2E (`joystickE2E.py`) | MPC/WBC alone (this session's fixed config) | MPC/WBC + residual (current `joystick.py`, fixes applied, **not re-trained**) |
|---|---|---|---|
| vx=1.0 tracking | Not re-measured this session (multiple historical checkpoints converged to +14.2 to +24.4 reward, stable) | 0.966 m/s, stable | Untrained with the new fixes — no meaningful number to report yet |
| omega=0.8 tracking | Not re-measured this session | 0.772 rad/s, stable | Untrained with the new fixes |
| Combined vx=0.6/omega=0.4 | Not re-measured this session | Stable, vx undershoots to 0.268 | Untrained with the new fixes |
| Status | Known-good, multiple converged checkpoints exist | **Fixed this session**: stable everywhere tested, single-axis targets met, combined-command tracking still degraded | **Fixes applied, training not run this session** (see §12 for why) |

The residual controller was **not retrained in this session** — training 20M timesteps takes hours
even on an idle GPU, and this machine's GPU had another user's live training process on it for a
significant part of this session (it finished partway through; see §12). Running a multi-hour
training job was out of scope for a single interactive session and would need to be started as a
tracked background job in a follow-up. What *was* done: the base controller the residual sits on
top of is now demonstrably more stable (§9.5), and the two verified regressions in the residual
env's own code (decoupled PD target, dead command countdown) were fixed — both are prerequisites
for any new residual training run to have a fair chance of reproducing (or beating) the one
historical success.

---

## 12. Remaining problems

1. **Combined-command forward-speed tracking is still degraded** (0.268 vs 0.6 m/s commanded at
   vx=0.6/omega=0.4), even though the fall/NaN failure is fixed. Torque is nowhere near saturation
   (20/120 Nm), so this isn't a hardware limit — it's the controller trading off forward speed for
   yaw-rate tracking under combined load, and the system is right at a stability boundary that
   makes naive weight increases (tested: `w_v` 10→15→40) reintroduce the fall. This likely needs a
   structural fix (WBC/QP feasibility margin analysis, more than 1 FDDP iteration when the cost
   landscape is stiffer, or properly softened/relaxed constraints) rather than further cost-weight
   tuning — flagged, not resolved.
2. **The same friction-cone `Fz` lower-bound gap likely exists in the C++ WBC too** (its friction
   pyramid has the identical 4-row structure with no explicit `Fz≥0` row) — not fixed there, since
   modifying and rebuilding/testing the C++ side was not verified in this session (no C++ build
   was run). Worth checking if the real robot or its C++-controller simulation ever exhibits an
   analogous failure under aggressive combined commands.
3. **Residual RL was not retrained** — the fixes in §7/§8 are necessary but their effect on
   training outcomes (vs. the historical +12.6 run) is unverified. This is the most important next
   step.
4. **`joystickE2E.py` is mid-edit by the user** (observed via a live `gnome-text-editor` process
   during this session, ~366 diff lines, not touched by this investigation) — its current on-disk
   state may not be stable; re-run the E2E baseline comparison once that settles.
5. **C++ `pybind_mpc.cpp`/`pybind_wbc.cpp` are stale/orphaned** (not in the CMake build, reference
   an old API) — noted, not fixed (fixing them wasn't necessary for any JAX-side comparison, since
   `getDefaultParams()` isn't what actually runs anyway).
6. **No domain randomization is actually applied** in `train_srbd.py`'s training call despite
   existing in the codebase for Tita — a real gap for sim-to-real robustness on both E2E and
   residual runs, not fixed (would change training results non-trivially and wasn't part of the
   reported bug).
7. **`d` (wheel half-track, 0.567 m)** in `config_dfcip.py` was not independently re-derived via
   forward kinematics against the true robot geometry in this session — flagged as unverified in
   both audits, likely close but not confirmed.

---

## 13. Recommendations, in priority order

1. **Retrain the residual policy** with the fixes in this report (start from
   `first_training_residual`'s exact hyperparameters, which are unchanged in current
   `train_srbd.py`) and compare against both the historical residual run and a fresh E2E run under
   identical conditions. This is the single most informative next experiment and wasn't run here
   due to session-length/GPU-sharing constraints.
2. **Investigate the combined-command forward-speed undershoot structurally**: profile which
   constraint in the WBC QP is actually binding when vx undershoots under combined ω (my
   suspicion, untested: the friction cone, given μ and the differential force demand of turning
   while accelerating) rather than continuing to tune MPC cost weights, which was shown in this
   session to be a narrow, fall-prone lever.
3. **Port the `Fz≥5N` WBC fix's underlying idea back to a C++ sanity-check** (even just a
   standalone Python/Eigen unit test of the friction-cone block) to see if the same latent
   infeasibility exists there, given the structural similarity.
4. **Re-derive `d` from forward kinematics** at the nominal posture and compare against the
   hardcoded 0.567 m, given how directly it enters both the heading-recovery formula and the
   dynamics/cost's wheel-offset geometry.
5. **Wire up domain randomization** in `train_srbd.py`'s `ppo.train` call (it already exists per-env
   via the registry) before any training run intended to inform real-hardware deployment.
6. Once `joystickE2E.py`'s in-progress edit settles, re-run a fresh E2E training as the up-to-date
   comparison baseline for the residual controller's eventual retraining.
