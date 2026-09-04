# TITA DFCIP MPC + WBC: simultaneous linear + angular velocity command fix

Date: 2026-09-04. Scope: `mpx` only (`mpx/mpx/config/config_dfcip.py`, `mpx/mpx/utils/{mpc_wrapper_dfcip,mpc_utils,timing}.py`,
`mpx/mpx/examples/{mjx_tita,tita,validate_dfcip_controller}.py`). `TITA-dynamics-obstacle`, `mujoco_playground`, PPO and the
residual policy were not modified. This report supersedes the combined-command sections of `FINAL_REPORT.md` and
`AUDIT_MPX_CONTROLLER.md` (both refer to the previous revision). Every number below was measured with
`mpx/mpx/examples/validate_dfcip_controller.py` (headless, CPU, x64), 1 s ramp + 8 s hold unless stated, steady-state = mean over
the last 2 s. Raw results: `mpx/mpx/examples/validation_results.json` (final) and the scratch logs quoted in the tables.

## 0. Environment finding that invalidated earlier runs

`mpx` is installed editable in the `mjpl` conda environment from `/home/jacopo/Desktop/repo_rl/tita_rl/test/mpx`, and
`mujoco_playground` from a git checkout inside `site-packages`. The example scripts used `sys.path.append(...)`, which loses
against the editable-install `.pth` entry, so **every script run imported the other checkout, not this repository**. Fixed in
`mjx_tita.py` and in the harness with `sys.path.insert(0, <repo>/mpx)`; the RL pipeline (`train_srbd.py -> registry ->
site-packages joystick.py -> installed mpx`) still uses the other copies until the editable installs are re-pointed
(`pip install -e /home/jacopo/Desktop/repo_jacopo/mpx`, not done: environment change, user decision). The site-packages
`joystick.py` also does **not** contain the two RL-env fixes of `FINAL_REPORT.md` (it still has `q_target = default_pose` and the
`-= 0` countdown), i.e. those fixes were applied to a copy that does not run.

## 1. Main cause (demonstrated)

**The MPC soft-constraint penalty `w_eq = 1e8` made the real-time-iteration FDDP unable to converge as soon as the reference
rotates, i.e. exactly when `v` and `omega` are commanded together.**

Mechanism, measured step by step (`--verbose-every`, scratch logs `base_*.npz`):

1. With `omega != 0` the reference CoM velocity `v_ref (cos theta_ref, sin theta_ref)` and the world-frame GRF plan
   (`F_lat = m v omega` rotating with `theta`) change at every node; the shifted warm start is no longer near-optimal.
2. The moment-balance residual `h_moment = (p_l - p_com) x F_l + (p_r - p_com) x F_r` is bilinear in `(theta, c, p_com)` and
   `(F_l, F_r)`; with weight 1e8 the Gauss-Newton curvature on `theta`/`c` reaches ~1e11-1e12 per stage against tracking weights of
   5-300: the quadratic model of the step is poor, the Goldstein line search rejects all 11 step sizes (`acc=N`, cost before ==
   cost after), the plan is only shifted.
3. The multiple-shooting defects grow (`|def|` 0.02 -> 0.2 -> 0.6 -> 2.0), the moment residual reaches 0.4-0.9 N m, the
   first-stage `(a, alpha, F_l, F_r)` handed to the WBC is physically inconsistent, the WBC task residuals explode
   (`|res_com|` 0.05 -> 1.0 m/s^2) and `qpax` returns NaN torques -> fall.

Evidence:

| configuration (old timing, dt_mpc 0.002, N 250) | (0.6, 0.4) | rejected steps | result |
|---|---|---|---|
| as committed, 1 FDDP iteration | fall at 1.8 s, NaN | 62 | reported "0.268 m/s stable" came from a 3 s run cut before the fall |
| 3 FDDP iterations per update (same weights) | vx 0.600, omega 0.398 | 19 | tracks: the problem is convergence, not feasibility |
| no outer joint PD | fall, NaN | 24 | outer PD is not the cause |

| mandated timing (dt_mpc 0.01, N 50, WBC 100 Hz) | (0.6, 0.4) | (1.0, 0.8) | (1.0, -0.8) | rejected | max moment residual |
|---|---|---|---|---|---|
| 1 iteration, w_eq 1e8 | 0.506 / 0.430 | fall + NaN | fall + NaN | 18-24 | 0.36-0.85 N m |
| 1 iteration, w_eq 1e7 | 0.599 / 0.415 | 0.999 / 0.860 | 0.999 / -0.832 | 1-3 | 0.2-0.35 N m |
| **1 iteration, w_eq 1e6** | 0.600 / 0.417 | 1.002 / 0.832 | 1.002 / -0.832 | **0** | **0.011 N m** |
| 1 iteration, w_eq 1e5 | 0.600 / 0.416 | 1.002 / 0.830 | 1.002 / -0.830 | 0 | 0.033 N m |
| 2 iterations, w_eq 1e8 | 0.603 / 0.396 | fall + NaN | fall + NaN | 8-33 | 0.2-0.8 N m |
| 2 iterations, w_eq 1e6 | 0.601 / 0.420 | 1.003 / 0.839 | 1.003 / -0.839 | 0 | 0.015 N m |

The larger penalty produces **larger** constraint violations because the optimiser cannot converge; 1e6 keeps the violation
physically negligible (0.01 N m = 0.04 mm lever arm at body weight) and is still five orders of magnitude above the tracking
weights. Doubling the iterations at 1e8 does not fix the fast turns, so the fix is the conditioning, not the iteration budget
(kept configurable: `mpc_iterations = 1`).

Physical check of the converged plan on (0.6, 0.4): planned lateral force 6.96 N vs the required `m v omega` = 6.52 N; planned
normal-load transfer `Fz_l - Fz_r` = -9.8 N vs the required `2 h m v omega / d` = 9.2 N (sign: outer wheel loaded).

## 2. Secondary causes, by importance

1. **WBC reference lookahead** (`_build_desired_impl`): `pos_ref = p + dt v`, `vel_ref = v + dt a` compared with the *current*
   state turns the task PD into `a_des = a_ref (1 + Kd dt) + Kp dt v`, a positive velocity feedback whose gain scales with `dt`.
   Inherited from the C++ (2 ms). Naively using `dt_wbc = 0.01` destabilises the yaw. Measured at (1.0, 0.8), w_eq 1e6:
   lookahead 0 -> omega 0.800 (base yaw rate), 0.002 -> 0.825, 0.01 -> 5.0 rad/s and QP NaN. Fixed: references evaluated at
   the current instant (`wbc_lookahead_dt = 0.0`), so the WBC is the pure feedforward inverse-dynamics stage of the RTI cascade;
   the only WBC feedback left is the base-orientation task (needed: with `w_base = 0` the robot falls).
2. **Stale state read** (`mjx_tita.py`, harness): after `mj_step` the derived quantities (`geom_xpos`, `subtree_com`,
   `mj_objectVelocity`) still describe the pre-integration state while `qpos/qvel` are integrated, so `x0` lags the WBC input by
   one simulation step (2 mm and 2 mm/s at 1 m/s). With Kp 50 / Kd 30 that is 0.1 m/s^2 of spurious task acceleration, the same
   order as the MPC feedforward (the old lookahead bias `+Kp dt v` and this `-Kp dt v` were partly cancelling each other).
   Fixed with `mj_step1` / read / set ctrl / `mj_step2`. Effect at (0.6, 0.4): omega 0.389 -> 0.397.
3. **Reference generator discretisation**: the CoM velocity of node k+1 used `theta` of node k, so the reference CoM path lagged
   the unicycle path by one node (`N dt^2 v omega` = 4 mm at (1.0, 0.8) with dt 0.01, 25x smaller with dt 0.002). Through the
   terminal stability constraint `p_com(N) = c(N)` (1e6) the MPC traded the cheap omega tracking (weight 5) for alignment: omega
   0.75 for 0.8. Fixed: `vx_next = v_next cos(theta_next)`, `theta_next = theta + omega dt`, `p_next = p + v cos(theta) dt`
   (exact trajectory of the MPC Euler model). Isolated MPC test on an ideal circle: converged plan within 0.5% of the command.
4. **Unanchored CoM lean in the WBC**: the hip-abduction joints (track width) are the kinematic redundancy of the stance and no
   task anchors them; under the lateral load they drift (19 mrad at (1.0, 0.8)) and the base rolls (15 mrad with `w_base = 0.01`),
   so the measured `p_com - c` is 3-5 mm inside the turn and the terminal constraint bends the base path: omega +5%. Fixed with
   `w_base = 1.0` (roll within 2 mrad) and the posture task restricted to the abduction joints (`posture_joint_ids = (0, 4)`,
   `w_posture = 0.1`); regulating all leg joints instead fights the CoM height (pitch -0.02 rad). Measured at (1.0, 0.8):
   0.843 (none) / 0.846 (abduction only) / 0.839 (w_base 1 only) / 0.825 (w_base 0.1 + abduction) / **0.807 (w_base 1 + abduction)**.
5. **Timing scheme** (mandated): `dt_mpc 0.002 / N 250` -> `0.01 / 50`; simulation, MPC and WBC rates made independent and
   checked; reference generated directly at the MPC discretisation; WBC joint-limit prediction step `dt_wbc = 0.01`;
   warm-start shift 1 node (was described as "simulation steps"). The coarser discretisation is also 2x cheaper (section 7).

Verified and **not** a cause: command layout (`[vx, vz, omega, height]`, `omega = cmd[2]`, consistent between
`KeyboardVelocityCommand.mpc_wheeled_input`, the RL env and the reference generator; only the docstring was wrong, fixed);
`config.d = 0.567` is the full wheel track (forward kinematics at the home keyframe: 0.5670 m) and is used consistently as
`d/2` offsets; wheel reference signs (inner wheel slower, centripetal `v omega` and `omega^2 d/2` terms correct in
`_build_desired_impl`); friction cone (margins >= 100 N in all converged runs, mu 0.9, `Fz >= 5 N` never active); joint limits
(min slack 0.27 rad, never active); `qpax` (11 iterations, converged, whenever the MPC plan was consistent); outer joint PD
(ablation: no effect on tracking, `dt_sim` correctly used); warm-start shift and regularisation (defects -> 0 once w_eq is
rescaled); `u_ref` with zero horizontal force on a curve (weight 1e-7, no measurable effect).

## 3. Why the commands worked separately but not together

Let `theta` be the heading, `F = F_l + F_r`. The soft constraints are `h_moment(x, u) = 0` (bilinear in state and GRF) and
`p_com(N) = c(N)`.

- `omega` alone (`v = 0`): no centripetal force is needed, `F_xy = 0`, `p_com = c`; `h_moment` is satisfied identically for
  any `theta`, so the 1e8 term contributes no curvature and the problem is quadratic in `(theta, omega, alpha)`: one GN step is
  the exact solution; the shifted warm start is optimal at every update.
- `v` alone: `theta` is constant, `F_fwd = m a` requires a pitch lean `l_x = h a / g` but the geometry is fixed in the world
  frame; after the ramp the solution is stationary and the shifted warm start is again optimal.
- `v` and `omega` together: `F_lat = m v omega` is nonzero and must rotate with `theta` at every node
  (`F_xy(k) = m v omega (-sin theta_k, cos theta_k)`), balanced by `Fz_l - Fz_r = -2 h F_lat / d`. Every error in `theta` or
  `v` changes the required GRF at all future nodes through `h_moment`, whose GN curvature is `w_eq |dR/dtheta off x F|^2 ~
  1e8 x 0.08 x 136^2 ~ 1.5e11`. The step computed from this model is rejected by the Goldstein test (the true cost of the
  bilinear term is not captured by the GN model far from the manifold), the single RTI iteration makes no progress, and since
  the yaw is kinematic (`alpha` is free, weight 1e-3) while forward speed needs `F_fwd` through the same constraint, the
  forward velocity is the quantity that is sacrificed first (0.27 m/s of 0.6). With `w_eq = 1e6` the curvature drops by 100x,
  the model is accurate enough for full steps, and both velocities converge.

## 4. Minimal structural correction

1. `w_eq: 1e8 -> 1e6` (conditioning of the soft equality constraints).
2. `wbc_lookahead_dt = 0.0` (WBC task references at the current instant).
3. Reference generator consistent with the MPC Euler model.
4. `mj_step1` / read / `mj_step2` in the drivers (state consistent with the WBC input).
5. `w_base 1e-2 -> 1.0`, posture task on the abduction joints only (`w_posture 0.1`, `posture_joint_ids (0, 4)`).

## 5. Timing scheme (all mandated values, all checked at import and at wrapper construction)

```python
simulation_frequency = 500   # dt_sim 0.002 s, must equal the XML timestep (checked, never silently changed)
whole_body_frequency = 100   # dt_wbc 0.01 s
mpc_frequency = 100          # MPC update period 0.01 s
dt_mpc = 0.01; N = 50; mpc_horizon_s = 0.5
mpc_period_sim_steps = 5; wbc_period_sim_steps = 5; mpc_shift_nodes = 1
```

`mpx/mpx/utils/timing.py::derive_timing` raises on: non-integer frequency ratios, `N dt_mpc != 0.5`, non-integer or `< 1`
warm-start shift; `check_model_timestep` raises if `model.opt.timestep != 1/simulation_frequency`. Between two controller
updates the WBC torque and joint accelerations are held for 5 simulation steps; the outer joint PD predicts one *simulation*
step ahead from the current joint state (`dt_sim`). The reference generator is called with `(N, dt_mpc)` (N+1 nodes, no
dense generation / decimation). `_st` (joint-limit prediction) is `dt_wbc`.

## 6. Files and functions modified

- `mpx/mpx/utils/timing.py` (new): `derive_timing`, `check_model_timestep`, `describe_timing`.
- `mpx/mpx/config/config_dfcip.py`: timing block; `mpc_iterations`; `wbc_lookahead_dt`; `w_eq`; `w_base`; `w_posture`,
  `posture_joint_ids`; derived timing + import-time checks.
- `mpx/mpx/utils/mpc_wrapper_dfcip.py`: `__init__` (timing, shift, lookahead, iterations loop, posture mask, diag functions),
  `run` (reference at MPC discretisation), `run_diag`, `whole_body_run_diag`, `_build_desired_impl` (lookahead dt).
- `mpx/mpx/utils/mpc_utils.py`: `reference_generator_dfcip_online` (docstring + consistent integration);
  `_whole_body_interface_wheeled_legged_qp_impl` (diagnostics, `posture_mask`), `whole_body_interface_wheeled_legged_qp`
  (unchanged public signature), `whole_body_interface_wheeled_legged_qp_diag`.
- `mpx/mpx/examples/mjx_tita.py`: import precedence, timing/checks, separate MPC and WBC update blocks, `mj_step1/mj_step2`,
  outer PD with explicit `dt_sim`, headless branch crash (`get_command`) fixed.
- `mpx/mpx/examples/tita.py`: `simulation_frequency` / `dt_sim` (legacy driver).
- `mpx/mpx/examples/validate_dfcip_controller.py`: rewritten harness (see docstring; `--set`, `--stale-state`, `--wbc-every`,
  `--no-outer-pd`, `--outer-pd-mode`, `--save-logs`, `--verbose-every`).

Full diff: `COMBINED_COMMAND_FIX.diff`.

## 7. Results before / after (1 s ramp, 8 s hold, same harness; "before" = previous control law reproduced through overrides)

| (vx, omega) | before: vx / omega | before: outcome | after: vx / omega (base yaw rate) | after: outcome |
|---|---|---|---|---|
| (1.0, 0.0) | 1.000 / 0.000 | ok | 1.000 / 0.000 | ok, 0 rejected steps |
| (0.0, 0.8) | fall at 4.0 s | NaN torque | 0.000 / 0.800 (0.800) | ok |
| (0.3, 0.2) | 0.296 / 0.200 | ok (13 rejected steps) | 0.300 / 0.201 (0.201) | ok |
| (0.6, 0.4) | fall at 2.1 s | NaN torque | 0.600 / 0.405 (0.404) | ok |
| (1.0, 0.8) | fall at 1.9 s | NaN torque | 1.000 / 0.807 (0.802) | ok |
| (1.0, -0.8) | fall at 2.4 s | NaN torque | 1.000 / -0.807 (-0.802) | ok |

After: height 0.440-0.445 m, |roll| <= 2 mrad, |pitch| <= 6 mrad, |tau| <= 20.5 N m, friction margin >= 100 N, QP converged
in 11 iterations at every update, WBC task residuals <= 0.15 m/s^2 (CoM) / 0.04 m/s^2 (wheels), no NaN. The single-axis cases
are unchanged or better (vx 1.000 vs 1.000; omega 0.800 vs a fall).

Cost per MPC update (reference + FDDP, x64): CPU single env 24.6 ms (old, N 250) -> 12.8 ms (new, N 50, 1 iteration);
vmapped 256 envs on the shared GPU 1434 ms -> 644 ms; CPU 32 envs 469 ms -> 174 ms. `mjx_tita.py` headless on CPU: MPC 2.9-3.3 ms,
WBC 1.2-1.6 ms per 10 ms update.

## 8. Not done / to be validated by the user

- Residual RL not retrained; the RL env (`joystick.py`, MJX) reads `geom_xpos`/sensors after `mjx.step` (same one-substep
  staleness as item 2.2) and integrates its residual PD target with `sim_dt` once per control step; both are outside the allowed
  scope and are flagged only.
- The editable installs (section 0) must be re-pointed for training to use this repository.
- The interactive viewer run of `mjx_tita.py` was not exercised (no display); the headless path was run for 3 s.
