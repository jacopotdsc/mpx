# MPX / TITA Controller Audit

Read-only technical audit. Working tree: `/home/mattia/Desktop/repo_jacopo/mpx`, branch `tita`.
All paths below are relative to the repo root `/home/mattia/Desktop/repo_jacopo/mpx` unless given absolute.

**Critical environment note (affects every claim in §1):** `mpx/jax_ocp_solvers` is declared as a
git submodule (`.gitmodules:1-3`, pinned commit `f397c9a87d9f90170a9b461821a9a973901fa49a`,
url `https://github.com/iit-DLSLab/jax_ocp_solvers.git`) but **the submodule is not checked out** in
this working tree (`git submodule status` prints a leading `-`, and
`mpx/mpx/jax_ocp_solvers/` is an empty directory). The actual optimizer source is therefore not
present on disk. To classify the solver call graph faithfully I fetched the pinned-commit source
of `jax_ocp_solvers/optimizers.py` from GitHub (read-only, not written into the repo) and cross-checked
it against how `mpc_wrapper_dfcip.py` calls it. Findings for `jax_ocp_solvers` are based on that fetched
source, which the local checkout does **not** contain — this must be fixed (`git submodule update --init`)
before anyone tries to run or re-derive gradients locally.

---

## 1. Execution-path map

### 1.1 `mjx_tita.py` — the actual MPC+WBC runtime entry point

Call chain (`mpx/mpx/examples/mjx_tita.py`):

1. `main()` (`mjx_tita.py:293`) loads `mujoco.MjModel` from `data/tita/scene_{scene}.xml`
   (`:296-298`), sets `model.opt.timestep = 1/config.whole_body_frequency` = `1/500 = 0.002 s`
   (`:300-301`).
2. `mpc = mpc_wrapper_dfcip.BatchedMPCControllerWrapper(config, n_env=1)` (`:328`) — `config` is
   `mpx.config.config_dfcip` (`:15`). This is the **only** wrapper class instantiated.
3. `solve_mpc = _build_solve_fn(mpc)` → `jax.jit(mpc.run)` (`:45-50, 339`).
4. `solve_wbc = _build_wbc(mpc)` → `jax.jit(mpc.whole_body_run)` (`:52-57, 341`).
5. Per-step loop `step_controller()` (`:397-536`), called every physics tick (500 Hz):
   - `gather_raw_state()` (`:233-248`) reads `MjData` (subtree CoM, wheel geom pos/rot/vel).
   - `process_tita_state()` (`:200-231`, `@jax.jit`) builds the 18-dim "tita_state" and the
     13-dim DFCIP state `x0` (see §2.1).
   - Every `period = sim_frequency/mpc_frequency = 500/100 = 5` physics steps (`:391, 438`):
     command is low-pass filtered (`command += 0.02*(target-command)`, `:439-441`, height command
     `command[3]` **not** filtered — set directly from keyboard target, `:444`), then
     `mpc_state, reference = solve_mpc(mpc_state, x0[None,:], command[None,:])` (`:447`) →
     `BatchedMPCControllerWrapper.run()` (`mpc_wrapper_dfcip.py:184-232`).
     Then `solve_wbc(mpc_state, x0, qpos, qvel, pl_world, pr_world, dpl_world, dpr_world)`
     (`:466-475`) → `BatchedMPCControllerWrapper.whole_body_run()`
     (`mpc_wrapper_dfcip.py:461-493`).
   - Joint-level PD tracking is added **on top of** the QP torque every physics tick (`:504-519`,
     see §2 and §6): `total_ctrl = p_ctrl + d_ctrl + tau[0]`, `data.ctrl = total_ctrl` (`:519-520`),
     `mujoco.mj_step(model, data)` (`:522`).

`BatchedMPCControllerWrapper.run()` (`mpc_wrapper_dfcip.py:184-232`):
- `x_ref, u_ref = self._ref_gen(x0, cmd)` → `jax.vmap(mpc_utils.reference_generator_dfcip_online)`
  (`:97, 153`), full reference downsampled by `ref_substeps` (`:198-199`, currently `= 1`, see §5).
- `X, U, D = self._solve(reference, parameter, W_tiled, x0, X0_shifted, U0_shifted)` (`:203-210`)
  → `jax.jit(jax.vmap(work))` where `work = partial(optimizers.fddp_mpc, self.cost, self.dynamics,
  self.hessian_approx, False)` (`:93, 152`). **This is a single FDDP (feasibility-driven DDP)
  iteration per call**, not an iterate-to-convergence solve (see §1.3) — real-time-iteration (RTI)
  style, relying on warm-starting across the 100 Hz replan calls.
- Trajectory is shifted by `s = self.shift = int(1/(dt_mpc*mpc_frequency)) = 5` stages for the next
  warm start (`:217-221`).

`BatchedMPCControllerWrapper.whole_body_run()` (`mpc_wrapper_dfcip.py:461-493`):
- `_build_desired_impl()` (`:234-459`) converts the MPC's first-stage control
  `(a, ac_z, alpha, fcl, fcr)` into desired CoM/wheel/base kinematic references via closed-form
  integration (this is a literal, commented port of the C++ `des_configuration_` block — see the
  inline C++ comments at `:259-262, 287-295, 309-457`).
- `self._whole_body_interface(qpos, qvel, desired)` (`:476-478`) →
  `jax.vmap(mpc_utils.whole_body_interface_wheeled_legged_qp)` (`:158`) — a per-step QP
  (`mpc_utils.py:968-1438`, solved with `qpax.solve_qp`, `:1423-1426`) that outputs
  `(tau, qddot, fl, fr)`.

### 1.2 `tita.py` — parallel/legacy standalone entry point, **not on the `mjx_tita.py` call graph**

`mpx/mpx/examples/tita.py` (1433 lines) imports the *same* backend as `mjx_tita.py`:
`mpx.config.config_dfcip` (`tita.py:25`), `mpx.utils.mpc_wrapper_dfcip` (`:67`),
`mpx.utils.mpc_utils` (`:68`). It defines its own `main(headless, steps, scene)`
(`tita.py:690`) with an almost identical `step_controller`/viewer loop and its own
`if __name__ == "__main__"` block (`:1380`). **Neither `mjx_tita.py` nor `train_srbd.py` imports
`tita.py`.** The only reference to it anywhere in the repo is a *lazy, function-local* import in
`plot_rollout_info.py:1695` (`from tita import TITA_PATH, dir_path`, inside
`main_replot_from_csv()`, a CSV-replot CLI utility never called from `mjx_tita.py`). Conclusion:
`tita.py` is an earlier/duplicate driver script exercising the identical DFCIP MPC + WBC solve
functions against the identical config — useful as a secondary code sample of the same formulation,
but it is **not invoked** when `mjx_tita.py` (or `mjx_policy_tita.py`) runs.

### 1.3 `mjx_policy_tita.py` — a *different* system entirely (RL policy tester, not the MPC)

Despite the name symmetry with `mjx_tita.py`, `mjx_policy_tita.py` does **not** call
`mpc_wrapper_dfcip`, `jax_ocp_solvers`, or any file under `mpx/utils/objectives.py` /
`mpc_utils.py`'s DFCIP path. Its own docstring says so explicitly (`mjx_policy_tita.py:1-29`):
it is "an interactive real-time frontend of *exactly* the same MJX environment used by
`train_srbd.py --eval`". Concretely:
- `env = registry.load(env_name, ...)` (`:401-404`) where `env_name` defaults to
  `"TitaJoystickFlatTerrain"` (`:731, 750-756`) and `registry` is
  `from mujoco_playground import registry` (`:56`) — an **external package** (not present under
  `mpx/`), i.e. a MuJoCo-Playground RL joystick environment with its own dynamics/observation/reward
  pipeline, unrelated to `mpx.utils.mpc_wrapper_dfcip`.
- Actions come from a PPO policy (`ppo_networks`/`brax`, `:193-203, 420-421`) loaded from a
  checkpoint (`load_params`, `:179-190`), not from `optimizers.fddp_mpc`.
- `env.step()` (a `brax`-style `State` transition) is called every tick (`:570, 609`); no MPC solve,
  no WBC QP, no `mpc_utils.whole_body_interface_wheeled_legged_qp` call anywhere in the file.

**Classification: `mjx_policy_tita.py` is out of scope for "the JAX MPC controller" — it exercises a
separately-trained RL policy against the `mujoco_playground` Tita joystick env, a different
codebase from `mpx/mpx/jax_ocp_solvers` + `mpc_wrapper_dfcip.py`.** It is documented here because it
was a named entry point to check, but none of the MPC/WBC formulation details in §2-§6 apply to it.

### 1.4 USED / CONDITIONALLY USED / UNUSED classification

**`mpx/jax_ocp_solvers` (fetched pinned-commit source, since not checked out locally):**

| Symbol | Status on `mjx_tita.py` path | Evidence |
|---|---|---|
| `fddp_mpc()` | **USED** — single FDDP/DDP step per MPC call | `mpc_wrapper_dfcip.py:93` (`work = partial(optimizers.fddp_mpc, self.cost, self.dynamics, self.hessian_approx, False)`), `:152` (`jax.jit(jax.vmap(work))`), called at `:203-210` |
| `compute_fddp_search_direction()` | **USED** (called inside `fddp_mpc`) | fetched `optimizers.py`, `fddp_mpc` body |
| `direct_dynamics_defect_helper`, `direct_cost_evaluator_helper` | **USED** (called inside `fddp_mpc`) | same |
| `parallel_goldstein_line_search()` | **USED** — fixed grid of `num_alpha=11` step sizes `2^0..2^-10`, Goldstein test, evaluated in parallel via `vmap`, best accepted index picked, **no `lax.while_loop`** (fully unrolled/parallel, GPU-friendly) | fetched `optimizers.py` |
| `tvlqr_gpu`, `rollout_gpu` (from `primal_tvlqr`) | **USED** — because `mpc_wrapper_dfcip.py:93` passes `limited_memory=False` as the 4th positional arg to `fddp_mpc`, which forces the `else` branch (`if limited_memory: tvlqr/rollout else: tvlqr_gpu/rollout_gpu`) inside `compute_fddp_search_direction` | `mpc_wrapper_dfcip.py:93`; fetched `optimizers.py` |
| `tvlqr`, `rollout` (limited-memory/non-GPU TVLQR) | **UNUSED** on this path (would only run if `limited_memory=True`) | same |
| `mpc()` (the KKT/dual-multiplier equality-constrained solver, uses `compute_search_direction`, `dual_lqr`, `parallel_filter_line_search`/`filter_line_search`) | **UNUSED** — `mpc_wrapper_dfcip.py` never imports or calls `optimizers.mpc`; it is used by `mpc_wrapper.py`/`mpc_wrapper_srbd.py` (other robots, not on the Tita path — see below) | grep confirms `mpc_wrapper_dfcip.py` only references `optimizers.fddp_mpc` |
| `compute_search_direction()`, `dual_lqr*`, `line_search`, `parallel_line_search`, `filter_line_search`, `parallel_filter_line_search`, `regularize`, `linearize_scan`, `linearize_obj_scan`, `merit_rho`, `slope` | **UNUSED** on the DFCIP/Tita path (dependencies of `mpc()`, not `fddp_mpc()`) | same |
| `direct_model_improvement()` | **UNUSED** — defined but not called by either `mpc()` or `fddp_mpc()` in the fetched source | fetched `optimizers.py` |
| `quadratize(cost)` (from `trajax.optimizers`, autodiff Hessian) | **CONDITIONALLY UNUSED** — only triggered `if hessian_approx is None`; the Tita path always supplies `hessian_approx = wheeled_dfcip_hessian_gn` (`mpc_wrapper_dfcip.py:89`), so the analytic Gauss-Newton branch is always taken and `quadratize` never runs for Tita | `optimizers.py` `compute_fddp_search_direction`, `mpc_wrapper_dfcip.py:89` |

**`mpc_wrapper_dfcip.py` internals:**

| Symbol | Status | Evidence |
|---|---|---|
| `mpc_objectives.wheeled_dfcip_obj` | **USED** — bound as `self.cost` (`:88`) | |
| `mpc_objectives.wheeled_dfcip_hessian_gn` | **USED** — bound as `self.hessian_approx` (`:89`), Gauss-Newton curvature, analytic Jacobians via `jax.jacobian` on a hand-built residual (not full autodiff Hessian of the cost) | `objectives.py:132-203` |
| `mpc_dyn_model.wheeled_dfcip_dynamics` | **USED** — bound as `self.dynamics` (`:90-91`) | `models.py:27-70` |
| `mpc_utils.reference_generator_dfcip_online` | **USED** (`:97`) | `mpc_utils.py:466-667` |
| `mpc_utils.whole_body_interface_wheeled_legged_qp` | **USED** (`:141-150, 158`) | `mpc_utils.py:968-1438` |
| `_build_desired_impl(..., use_nn=True, action_nn=...)` branch | **UNUSED** on the `mjx_tita.py` path — `mjx_tita.py` calls `solve_wbc(...)` with only the 7 positional args (`mjx_tita.py:379-388, 466-475`), so `action_nn=None, use_nn=False` (defaults, `mpc_wrapper_dfcip.py:243-244, 462`); the residual-learning/NN-correction code (`:270-284, 350-353`) is dead on this path | |

**`mpx/utils/objectives.py` — every `*_obj`/`*_hessian_gn` function other than the two above
(`quadruped_srbd_obj/hessian_gn`, `quadruped_wb_obj/hessian_gn`, `h1_wb_obj/hessian_gn`,
`h1_kinodynamic_obj`, `talos_wb_obj/hessian_gn`) is UNUSED on the Tita path** — they belong to the
quadruped/H1/Talos configs (`config_srbd.py`, `config_h1.py`, `config_talos.py`, …), which are wired
through `mpc_wrapper.py`/`mpc_wrapper_srbd.py`, used only by `mjx_quad.py`, `mjx_h1.py`,
`mjx_talos.py`, `srbd_quad.py`, `acrobot.py`, `multi_env.py`, `offline_task.py` — none of which sit
on the `mjx_tita.py`/`mjx_policy_tita.py` execution path (grep confirms `mpc_wrapper_dfcip` is
imported only by `mjx_tita.py` and `tita.py`).

---

## 2. MPC formulation as actually implemented (DFCIP, `mjx_tita.py` path)

### 2.1 State vector `x` (dim 13) — `models.py:29-35`, mirrored in `mpc_utils.py:477-484`

```
x = [ pcom_x, pcom_y, pcom_z,     # 0:3  CoM position, world frame, meters
      dpcom_x, dpcom_y, dpcom_z,  # 3:6  CoM linear velocity, world frame, m/s
      c_x, c_y, c_z,              # 6:9  "virtual unicycle" ground-projected contact-midpoint
                                   #      position, world frame, meters (c_z tracked to 0)
      vc_z,                       # 9    vertical velocity of the contact-midpoint, m/s
      theta,                      # 10   heading (yaw of the wheel axle), rad, world frame
      v,                          # 11   forward speed of the unicycle point, m/s (body-x)
      omega ]                     # 12   yaw rate, rad/s
```
`config_dfcip.py:57` declares `nx = 13` matching. Note: **no orientation quaternion and no leg/wheel
joint states are in the MPC state** — this is a reduced "DFCIP" (differential-drive floating-inverted-
pendulum-like) model, not a full-order or even SRBD-with-quaternion model. Full-order tracking
(quaternion, joints) only re-enters at the WBC QP layer (§2.6), driven off a first-order kinematic
reconstruction (`_build_desired_impl`) of what this reduced state implies.

`x0` is built at runtime by `process_tita_state()` (`mjx_tita.py:200-231`, `@jax.jit`) from raw
MuJoCo state: `pcom, vcom` = `subtree_com`/`subtree_linvel` of `base_link` (`gather_raw_state`,
`:233-248`); `pl_world, pr_world` = wheel-geom center + `get_rCP()` contact-point offset
(`mpc_utils.py:897-904`, ports the C++ `labrob::get_rCP`); `theta` recovered from
`atan2(-diff[0], diff[1])` with `diff = pl_world - pr_world` and unwrapped against the previous
`theta` (`mjx_tita.py:216-219`) — this is algebraically consistent with the wheel-offset
parametrization used in `wheeled_dfcip_obj`/`_build_desired_impl` (`vector_off = [0, d/2, 0]`,
`objectives.py:25`, `mpc_wrapper_dfcip.py:313`): `pl - pr = R(theta)@[0,d,0]` ⇒
`atan2(-Δx, Δy) = theta` exactly, verified symbolically — no sign bug found here.
`v, omega` are recovered from the two wheel-contact velocities rotated into the heading frame
(`mjx_tita.py:220-226`): `w = (dpr_body_x - dpl_body_x)/config.d`, `v = (dpr_body_x + dpl_body_x)/2`
— a rigid-bar/differential-drive kinematic relation using `config.d = 0.567 m` as the wheel track.

### 2.2 Control vector `u` (dim 9) — `models.py:36-40`, `objectives.py:19-20`

```
u = [ a, ac_z, alpha,             # 0:3  CoM forward accel, CoM vertical accel, yaw ang. accel
      Fl_x, Fl_y, Fl_z,           # 3:6  left  wheel ground-reaction force, world frame, N
      Fr_x, Fr_y, Fr_z ]          # 6:9  right wheel ground-reaction force, world frame, N
```
`config_dfcip.py:58` declares `nu = 9`, matches. `u_ref` default: zero accelerations, GRF split
`m*g/2` each wheel (`config_dfcip.py:66-67`, `mpc_utils.py:555-568`).

### 2.3 Dynamics (`models.py:27-70`, function `wheeled_dfcip_dynamics`)

Forward-Euler, `dt = config.dt_mpc = 0.002 s`:
```
ddpcom = (Fl + Fr)/m + [0,0,-|g|]                     # CoM Newton's law (point-mass, world frame)
dc_x = v cos(theta),  dc_y = v sin(theta),  dc_z = vc_z # unicycle kinematics of contact midpoint
dvc_z = ac_z
dtheta = omega
dv = a
domega = alpha

pcom_{k+1}  = pcom_k  + dt*dpcom_k
dpcom_{k+1} = dpcom_k + dt*ddpcom_k
c_{k+1}     = c_k + dt*[dc_x,dc_y,dc_z]
vc_z_{k+1}  = vc_z_k + dt*dvc_z
theta_{k+1} = theta_k + dt*omega_k
v_{k+1}     = v_k + dt*a_k
omega_{k+1} = omega_k + dt*alpha_k
```
Note the CoM translational dynamics (`ddpcom`) are **decoupled** from the unicycle/heading dynamics
(`c, theta, v, omega`): the GRF inputs `Fl,Fr` only drive `pcom/dpcom` via Newton's law, while
`a, ac_z, alpha` (control inputs, not derived from `Fl,Fr`) independently drive the reduced unicycle
state. The two halves are coupled only through the cost (which penalizes `pcom` tracking `c`,
`objectives.py:38,56,100-106,114`) and through the WBC's kinematic reconstruction — **not through
the dynamics model itself**. `mjx_model` is accepted as the first positional argument
(`models.py:27`) but is **never referenced inside the function body** — dead parameter, presumably
kept for a uniform `dynamics(mjx_model, ..., x, u, t, parameter)` call signature across robot
configs.

### 2.4 Horizon / solver iteration / warm-start

- `dt_mpc = 0.002 s` (`config_dfcip.py:21`), `N = 250` stages (`:22`) ⇒ **0.5 s prediction horizon**.
- MPC replanned at `mpc_frequency = 100 Hz` (`:24`), i.e. every `period = 5` physics steps at the
  500 Hz whole-body rate (`mjx_tita.py:391`); `shift = 5` stages per replan
  (`mpc_wrapper_dfcip.py:55`).
- **Exactly one FDDP iteration is performed per `run()` call** (`fddp_mpc`, fetched
  `optimizers.py`) — this is a real-time-iteration scheme, not iterate-to-tolerance. Convergence
  relies entirely on warm-starting (`X0_shifted, U0_shifted`, `mpc_wrapper_dfcip.py:217-230`) across
  the 100 Hz replan cadence, i.e. ~5 physics steps / 10 ms between "SQP iterations" of a nominally
  0.5 s-horizon problem. No iteration-count or convergence-tolerance parameter exists to increase
  inner iterations for this call site (`fddp_mpc`'s signature has no loop over itself).
- Line search: `parallel_goldstein_line_search`, 11 step sizes `α ∈ {2^0,...,2^-10}` evaluated in
  parallel with `vmap`, Goldstein acceptance test (`b1=0.1, b2=2.0`, fetched `optimizers.py`,
  defaults not overridden by `mpc_wrapper_dfcip.py:93,152`).
- Regularization: fixed `regularization=1e-3` added to `Q,R` block diagonals inside
  `compute_fddp_search_direction` (fetched `optimizers.py`) — **not adaptive** (no
  regularization-increase-on-failure logic visible in `fddp_mpc`/`compute_fddp_search_direction`).

### 2.5 Cost terms — `objectives.py:15-130` (`wheeled_dfcip_obj`), weights from `config_dfcip.py:135-143`

All terms are **quadratic penalties on hand-built residuals** (not autodiff of a generic scalar via
`quadratize`, since `hessian_approx` is always supplied — see §1.4); gradients used by the solver
(`q, r` in `compute_fddp_search_direction`) come from `jax.jacobian`/`linearize(cost)` (autodiff of
`wheeled_dfcip_obj` itself), while curvature (`Q,R,M`) comes from the **separate, hand-differentiated
Gauss-Newton** residual in `wheeled_dfcip_hessian_gn` (`objectives.py:132-203`) — i.e. **gradient is
autodiff of the true cost, Hessian is GN-approximate from a residual that must be kept consistent by
hand** (see §6 for a consistency check).

Reference source: `reference[t,:13]` / `reference[t,13:22]`, produced per-call by
`reference_generator_dfcip_online` (`mpc_utils.py:466-667`, downsampled by `ref_substeps`,
`mpc_wrapper_dfcip.py:198-199`).

Stage cost (`objectives.py:76-95`), all weights read from the 15×15 diagonal `W`
(`config_dfcip.py:135-143`; entries indexed `W[0,0]..W[14,14]`, `objectives.py:40-46`):

| Residual | Weight (config value) | W row |
|---|---|---|
| `pcom_xy - x_ref[0:2]` | `w_pcomxy = 0` | `W[0,0]` |
| `pcom_z - x_ref[2]` | `w_pcomz = 2e4` | `W[1,1]` |
| `vcom_xy - x_ref[3:5]` | `w_vcomxy = 3e2` | `W[2,2]` |
| `vcom_z - x_ref[5]` | `w_vcomz = 1e1` | `W[3,3]` |
| `c - x_ref[6:9]` | `w_c = 0` | `W[4,4]` |
| `vc_z - x_ref[9]` | `w_vcz = 0` | `W[5,5]` |
| `theta - x_ref[10]` | `w_theta = 0` | `W[6,6]` |
| `v - x_ref[11]` | `w_v = 1e1` | `W[7,7]` |
| `omega - x_ref[12]` | `w_omega = 5e0` | `W[8,8]` |
| `a - u_ref[0]` | `w_a = 1e-1` | `W[9,9]` |
| `ac_z - u_ref[1]` | `w_ac_z = 1e-1` | `W[10,10]` |
| `alpha - u_ref[2]` | `w_alpha = 1e-3` | `W[11,11]` |
| `Fl_xy - u_ref[3:5]`, `Fr_xy - u_ref[6:8]` | `w_fcxy = 1e-7` | `W[12,12]` |
| `Fl_z - u_ref[5]`, `Fr_z - u_ref[8]` | `w_fcz = 1e-4` | `W[13,13]` |
| soft-constraint `h_contact = vc_z` | `w_eq = 1e8` | `W[14,14]` |
| soft-constraint `h_moment = (pl-pcom)×Fl + (pr-pcom)×Fr` | `w_eq = 1e8` | `W[14,14]` |

**`w_pcomxy = w_c = w_vcz = w_theta = 0`** — CoM-xy tracking, contact-midpoint tracking, and heading
tracking are all **structurally present but numerically switched off** in the current tuning (weight
literally 0); only `v` and `omega` (velocity-level) tracking, plus `pcom_z`/`vcom` and force-effort
terms, are active. This is a tuning choice worth flagging to whoever compares against the C++
baseline's weight table, since a `w=0` term still costs compute (still differentiated) but has zero
effect.

`h_fz = min(Fl_z,0)^2 + min(Fr_z,0)^2` (GRF-pulls-the-ground penalty) is **computed at
`objectives.py:37` but never added to `stage_cost` or `term_cost`** — see §6 (dead/unwired cost
term).

`h_stability = pcom_xy - c_xy` is **only used in the terminal cost** (`objectives.py:114,127`),
weight `w_eq = 1e8`; it does not appear in the stage cost.

Terminal cost (`objectives.py:100-128`, selected via `jnp.where(t==N, term_cost, stage_cost)`,
`:130`): same state-tracking terms as stage cost (`pcom, vcom, c, vc_z, theta, v, omega`, all with
the same, mostly-zero weights) plus `h_contact` and `h_stability` soft constraints; **no control
terms** (there is no control at the terminal node).

### 2.6 Whole-Body Controller (QP) — `mpc_utils.py:968-1438`, invoked every physics tick (500 Hz)

This is a per-tick **instantaneous QP**, not a receding-horizon problem: `min 0.5 xᵀHx + fᵀx`
s.t. `A_eq x = b_eq`, `d_min ≤ C_ineq x ≤ d_max`, solved with `qpax.solve_qp(..., solver_tol=1e-3)`
(`:1423-1426`). Decision variable `x = [qddot (nv), Fl (3), Fr (3)]`.

- **Task-space PD-tracking cost** (`:1198-1235`): weighted least-squares on `qddot` tracking desired
  CoM/left-wheel/right-wheel/base-orientation accelerations built from PD laws on top of the MPC's
  `_build_desired_impl` references (`Kp_motion=50, Kd_motion=30, Kp_wheel=50, Kd_wheel=30,
  Kp_reg=100, Kd_reg=20`, `config_dfcip.py:75-80`; weights `w_qddot=1e-12, w_com=1, w_lwheel=1,
  w_rwheel=1, w_base=1e-2, w_posture=0.1`, `:81-87`). Jacobians (`J_com, J_left_wheel, J_right_wheel,
  J_base_link`) and their time-derivatives are obtained via `jax.jvp` through `mjx.jac`
  (`mpc_utils.py:1048-1066`, forward-mode AD wrapping MJX's own analytic Jacobian, an efficient way
  to get `J̇q̇` without a second explicit derivative).
- **Equality constraints** (`:1308-1386`): (a) rolling-without-slip at each wheel contact point
  (`A_roll_L/R`, 3 rows each, ported from the `labrob` C++ contact model comments inline), (b) full
  floating-base inverse-dynamics equation on the 6 unactuated DOF (`A_dyn`, `Mu@qddot - Jlu'Tl*Fl -
  Jru'Tr*Fr = -cu`). Total 12 equality rows (a "no_contact" 6-row block is present in the C++ port
  comments but explicitly zeroed/commented out at `:1351-1353,1379-1382,1385`, i.e. structurally
  dead here since both wheels are treated as always-in-contact).
- **Inequality constraints** (`:1238-1306`): joint position/velocity box limits from
  `mjx_model.jnt_range` (`:1246-1248`, **note:** `vel_limit = 100 rad/s` is a hardcoded placeholder,
  commented `# adatta al tuo URDF`, i.e. explicitly flagged by the author as not sourced from the
  model — see §4/§6) expressed as one-step-ahead linear constraints on `qddot`
  (`C_acc_ineq`, `:1254-1262`); and linearized friction cone per wheel (`|Fx|,|Fy| ≤ μ Fz`,
  `μ = config.mu = 0.6`, `config_dfcip.py:36`, note the WBC binds this as a **separate**
  `config.mu = 0.5` comment mismatch — see §6) as a 4-row pyramid per wheel in the wheel's local
  contact frame (`compute_contact_frame`, `mpc_utils.py:889-895`).
- **Output**: `qddot = x[:nv]`, `Fl,Fr = x[nv:nv+3], x[nv+3:nv+6]`; torque via inverse dynamics
  `tau = Ma@qddot + ca - Jla'@Tl@Fl - Jra'@Tr@Fr` (`:1436`), returned to `mjx_tita.py`.

`mjx_tita.py` then **adds a second, independent joint-space PD loop** on top of this WBC torque
(`:504-519`): desired joint pos/vel from forward-integrating `qddot` one physics step
(`dq_desired = qvel + qddot*dt`, `q_desired = qpos + qvel*dt + 0.5*qddot*dt²`, `:511-512`), tracked
with `Kp=35, Kd=10` (**hardcoded in `mjx_tita.py`, not in `config_dfcip.py`**, `:514-515`), P-term
zeroed for the two wheel joints (`:517`), summed with the WBC's QP torque (`:519`). This
double-loop structure (QP torque + outer joint PD on the QP's own predicted `qddot`) is unusual and
should be compared explicitly against the C++ baseline's actuator command path — it is not evident
from the WBC QP formulation alone that this second PD loop is intentional/needed vs. legacy
scaffolding.

---

## 3. Quaternion / state-convention check

- **MuJoCo/MJX qpos convention (wxyz), used correctly and explicitly converted:**
  `mpc_utils.py:1013` — `q_wxyz = qpos[3:7]` (comment explicitly labels it `wxyz`); quaternion
  kinematics `dq/dt` from body-frame angular velocity computed by hand at
  `mpc_utils.py:1014-1019` using the standard `q̇ = 0.5 * q ⊗ [0,ω]` (wxyz Hamilton convention,
  consistent with the sign pattern `dqw = 0.5*(-qx*wx - qy*wy - qz*wz)`, etc.).
  This `dqpos` tangent is then used as a `jax.jvp` seed for finite-`Jq̇` (Jacobian time-derivative)
  computation (`:1048-1066`), i.e. **the whole "Jacobian-dot via JVP" trick depends on the wxyz
  quaternion-kinematics formula being correct** — worth an explicit unit check against the C++
  baseline's `Jq̇` since a sign error here would silently bias every `a_*_drift` term (§2.6).
- **Explicit wxyz→xyzw conversion before calling `scipy`'s `Rotation.from_quat`** (which expects
  xyzw): `mpc_utils.py:1102-1112` —
  `current_base_quat = mjx_data.xquat[base_body_id]` (MuJoCo convention, wxyz) →
  `current_base_quat_xyzw = [q[1],q[2],q[3],q[0]]` → normalized → `Rotation.from_quat(...).as_matrix()`.
  This conversion is correct and necessary; it is the one place in the WBC where an implicit
  convention mismatch (MuJoCo wxyz vs. SciPy xyzw) would silently corrupt `current_base_link_pos`
  if omitted, and it is handled.
- **`mpx/utils/rotation.py`** (`quaternion_product`, `quaternion_to_rpy`,
  `rotation_matrix_to_quaternion`, `:4-124`) is **wxyz throughout** (`w1=q1[0]`, `:5`), consistent
  with the above, but this module is **not imported by `mpc_wrapper_dfcip.py`/`mpc_utils.py`'s
  DFCIP+WBC path** at all (grep found no reference) — it backs the SRBD/quadruped/H1/Talos configs
  (`objectives.py`'s `math.quat_sub` calls, which use `mujoco.mjx._src.math`, MuJoCo's own wxyz
  convention, not this file). Listed here for completeness since it was a named area to check.
- **The 13-dim DFCIP MPC state itself carries no quaternion at all** — orientation is represented
  purely as scalar yaw `theta` (§2.1); roll/pitch are not part of the reduced model and are only
  implicitly regulated by the WBC's `err_rotation(des['base_rot'], current_base_link_pos)`
  (`mpc_utils.py:1154-1155,854-868`, an `AngleAxis`-style full-3D orientation error, ported
  from `Eigen::AngleAxisd`), where `des['base_rot']` is a pure-yaw rotation matrix
  (`mpc_wrapper_dfcip.py:415-433`, only θ, no roll/pitch reference is ever generated) — i.e. roll/pitch
  regulation at the WBC level always targets zero roll/pitch, not a value derived from the MPC.
- **Angular-velocity frame**: `omega = qvel[3:6]` (`mpc_utils.py:1014`) is MuJoCo's free-joint
  angular velocity, which MuJoCo defines in the **body frame** for a free joint's rotational DOF —
  used directly (not rotated) in the quaternion-derivative formula (`:1015-1018`), consistent with
  that convention. `J_base_link_rot @ qvel` (`:1114`, `current_base_link_vel`) is likewise MJX's
  Jacobian output, whose rotational rows map generalized velocities to the same body-frame angular
  velocity by construction of `mjx.jac`.

---

## 4. Physical model parameters — cited against `data/tita/tita.xml`

| Parameter | Config value (`config_dfcip.py`) | XML source | Match? |
|---|---|---|---|
| Total mass | `mass = 27.6898` (`:59`) | Sum of all body `mass=` attributes in `tita.xml` (`base_link` 13.2 `:84`, `imu_link` 0.001 `:94`, `left/right_leg_1` 2.064×2 `:100,123`, `left/right_leg_2` 3.0984/3.0987 `:104,127`, `left/right_leg_3` 0.57244×2 `:108,131`, `left/right_leg_4` 1.5094×2 `:113,136`) = **27.68978 kg** | Matches to 4 decimals |
| Wheel radius | `wheel_radius = 0.0925` (`:74`) | `<geom name="left/right_leg_4_collision" type="sphere" size="0.0925 0.017" .../>` (`tita.xml:114,137`) | Matches exactly |
| Wheel/leg track separation `d` | `d = 0.567` m (`:60`), used as `vector_off=[0,d/2,0]` (half-track) in both dynamics/cost (`objectives.py:25`) and reference/state reconstruction (`mjx_tita.py:183-184`, `mpc_wrapper_dfcip.py:313`) | Not directly present as a single XML attribute — it is an *effective* lateral offset between the two wheel contact geoms after the full leg kinematic chain (`left/right_leg_1` hip offset `±0.0895` m, `tita.xml:98,121`, plus downstream links). **Not independently re-derived/verified from forward kinematics in this audit** — flag for cross-check against the C++ baseline's own `d` (or against `mj_kinematics` at the nominal `q0` posture) since it is a hand-set constant, not computed from the model | Unverified — flag |
| Gravity | `grav = 9.81` (`:25`) | `<option gravity="0 0 -9.81" .../>` (`tita.xml:2`) | Matches |
| Joint angle limits (leg joints 1-3) | Not used by the MPC (reduced state has no joint DOF); used by WBC via `mjx_model.jnt_range` at runtime (`mpc_utils.py:1246-1248`), not hardcoded in `config_dfcip.py` | `joint_1: [-0.785398, 0.785398]`, `joint_2: [-1.919862, 3.490659]`, `joint_3: [-2.670354, -0.698132]` rad (`tita.xml:8,12,16`) | Sourced live from the model at runtime — correct design |
| Wheel joint (joint_4) limit | n/a | `limited="false"` (`tita.xml:20`, effectively continuous rotation; `mpc_utils.py:1246` still includes it in `jnt_ids` with `mjx_model.jnt_range` giving `[-6.283185e4, 6.283185e4]`, i.e. a huge-but-finite box, `tita.xml:112,135`) | consistent, harmless due to huge range |
| Joint velocity limit used in WBC | `vel_limit = 100.0 rad/s` **hardcoded constant**, `mpc_utils.py:1251`, comment `# rad/s — adatta al tuo URDF` (author's own note: "adapt to your URDF") | No `<joint ... velocity=...>` or actuator `forcerange`/`velocity` limit tag exists in `tita.xml` to source this from — the value is **not derived from the MJCF at all** | **Placeholder, flagged by the author's own comment** — see §6 |
| Torque limit (per joint) | Not enforced anywhere in the MPC or WBC QP (no torque box constraint appears in `whole_body_interface_wheeled_legged_qp`) | `<motor ... ctrlrange="-120 120" forcerange="-120 120" gear="1" .../>` for all 4 joint classes (`tita.xml:9,13,17,21`) — **120 Nm per joint** | **WBC does not constrain `tau` against this limit** — see §6 |
| Friction coefficient (WBC QP) | `mu = 0.6` in `config_dfcip.py:36`, comment says `# Coefficient of friction`; passed into `whole_body_interface_wheeled_legged_qp` as `_mu = config.mu` (`mpc_wrapper_dfcip.py:135`) — **but the parameter docstring/comment at `mpc_wrapper_dfcip.py:135` says `# 0.5 in C++`**, i.e. the author's own comment records the C++ baseline value as 0.5, while the Python config actually ships `0.6` | `tita.xml` collision default `friction="1 0.01 0.01"` (`:26`, MuJoCo tangential/torsional/rolling friction, not directly the same as the WBC's Coulomb `μ`) | **Config value (0.6) disagrees with the value the code's own comment says the C++ baseline uses (0.5)** — flag, see §6 |
| Inertia (SRBD-style 3×3 about CoM) | `inertia = [[1.14468,-2.63e-5,-0.02409],[-2.63e-5,0.55351,-2.10e-5],[-0.02409,-2.10e-5,0.70152]]` (`config_dfcip.py:61-63`) | Not cross-checked against XML — would require running `mj_forward`/composite rigid-body inertia at the nominal posture; **`inertia` does not even appear to be consumed anywhere on the `mjx_tita.py` path** (grep found no use of `config.inertia` in `mjx_tita.py`, `mpc_wrapper_dfcip.py`, `mpc_utils.py`, `objectives.py`, `models.py` for the DFCIP formulation — the reduced DFCIP state has no orientation DOF to need it) | **Appears unused / dead config value on this path** — see §6 |
| Nominal joint posture `q0` | `[0, 0.5, -1.0, 0, 0, 0.5, -1.0, 0]` (`config_dfcip.py:43`) | Used as WBC posture reference (`mpc_wrapper_dfcip.py:438`) and MJX-model keyframe reset target (`mjx_tita.py:76`, `mj_resetDataKeyframe(..., 0)` — the XML's own keyframe 0, not directly `config.q0`; **not independently confirmed these two are the same posture** since `tita.xml`'s keyframe block was not inspected in this pass) | Partially verified |

---

## 5. Timing

- **MPC internal step** `dt_mpc = 0.002 s` (`config_dfcip.py:21`), horizon `N = 250` ⇒ **0.5 s
  horizon**, `T_TRAJECTORY = 60` (`:23`, appears to be a total-sim-time constant for `tita.py`'s
  `MAX_STEPS`, not used by `mjx_tita.py`).
- **MPC replan rate**: `mpc_frequency = 100 Hz` (`:24`) — `mpc_wrapper_dfcip.py:184-232` `run()` is
  called every `period = 5` physics steps in `mjx_tita.py` (`:391,438`), i.e. every 10 ms.
- **Whole-body / physics rate**: `whole_body_frequency = 500 Hz` (`config_dfcip.py:26`), used both
  as `model.opt.timestep` (`mjx_tita.py:301`) and as the WBC QP's `sample_time` (`_st =
  1/whole_body_frequency`, `mpc_wrapper_dfcip.py:127,144`, used in the QP's joint-limit
  discretization, `mpc_utils.py:1254-1273`) and as `dt` for `_build_desired_impl`'s kinematic
  integration (`mpc_wrapper_dfcip.py:348`).
- **Reference generator "dense" resampling**: `dt_ref = 1/whole_body_frequency = 0.002 s`
  (`config_dfcip.py:27`) ⇒ `ref_substeps = round(dt_mpc/dt_ref) = 1` (`mpc_wrapper_dfcip.py:95`),
  `N_dense = N*ref_substeps = 250` (`:96`). With the current config values `dt_mpc == dt_ref`
  exactly, so `reference_full[:, ::1, :]` (`:199`) is a **no-op downsample** — the
  "generate-dense-then-decimate" machinery is present but currently degenerate; it only does
  something if `dt_ref` is changed to be finer than `dt_mpc`.
- **Interpolation/resampling between MPC output and applied torque**: **none at the MPC-to-WBC
  boundary** — the WBC (`whole_body_interface_wheeled_legged_qp`) is called fresh every physics tick
  (500 Hz) using only the MPC's **first-stage** control `U[:,0,:]` (`mpc_wrapper_dfcip.py:212-215`,
  held constant / re-integrated via `_build_desired_impl`'s closed-form kinematics until the next MPC
  solve, `:340-457`) — i.e. zero-order-hold on the MPC's first control between the 100 Hz MPC solves,
  with the WBC re-deriving fresh position/velocity/acceleration references every 500 Hz tick by
  forward-integrating that held control (not by re-sampling a stored MPC trajectory row). The
  physics-to-command chain applies one more resampling layer: `mjx_tita.py:504-519` PD loop, itself
  re-run every 500 Hz tick using the WBC's own `qddot` output from the *same* tick (no additional
  delay/hold there).

---

## 6. Suspicious / fragile findings

1. **`jax_ocp_solvers` submodule not checked out** (`.gitmodules:1-3`, empty
   `mpx/mpx/jax_ocp_solvers/`). Anyone cloning this branch as-is cannot run `mjx_tita.py` /
   `tita.py` at all (`import mpx.jax_ocp_solvers.optimizers` will `ModuleNotFoundError`). This audit
   only had the solver source because it was fetched from GitHub at the pinned commit — the local
   repo state does not actually reflect what will execute. **Action: `git submodule update --init`.**

2. **`h_fz` GRF non-negativity soft constraint is computed but never applied to the cost.**
   `objectives.py:37`: `h_fz = jnp.minimum(fl[2],0.)**2 + jnp.minimum(fr[2],0.)**2` is computed, but
   neither `stage_cost` (`:76-95`) nor `term_cost` (`:116-128`) references `sc_fz`/`tc_fz` or `h_fz`
   anywhere — grep of the whole function confirms `h_fz` is a dead local. Consistently,
   `wheeled_dfcip_hessian_gn` zeros the corresponding residual slots
   (`jnp.zeros_like(h_fz)` at `objectives.py:177,190`), so the GN curvature for those 2 of 30 residual
   rows is exactly zero too — internally consistent dead code, but it means **the MPC itself applies
   no penalty preventing `Fl_z`/`Fr_z` from going negative** (pulling on the ground); the only actual
   enforcement of non-negative/friction-bounded normal force happens downstream in the WBC QP's
   friction-cone rows (§2.6), which the MPC's own force *reference* is not required to respect.

3. **Height command is silently ignored by the reference generator, and the HUD advertises a control
   that does not exist in `mjx_tita.py`.**
   `mpc_utils.py:502`: `z_com_ref = 0.4` is a **hardcoded local constant** inside
   `reference_generator_dfcip_online`; the function's `cmd` argument's 4th component
   (`command[3]`, the height set by `command_handle.mpc_wheeled_input(config.com_z_to_track)`,
   `mjx_tita.py:359-362,434-437,444`) is **never read** inside the reference generator (only
   `cmd[0]` and `cmd[2]` are used, `mpc_utils.py:522-523`). Meanwhile `mjx_tita.py`'s viewer HUD text
   says `"PgUp/PgDown: z_com_ref"` (`:608`), but `command_handle.key_callback`
   (`sim.py:106-127`) has **no PageUp/PageDown handling at all** (only `KEY_UP/DOWN/LEFT/RIGHT/
   SPACE/ENTER/BACKSPACE`) — unlike `mjx_policy_tita.py`, which *does* implement
   `KEY_PAGE_UP/KEY_PAGE_DOWN` height adjustment via its own `key_callback` wrapper
   (`mjx_policy_tita.py:465-476`). Net effect: in `mjx_tita.py` the commanded CoM height is currently
   dead in two independent ways (no UI to change it, and even if changed the reference generator
   ignores it) — it happens to be silently correct only because `config.com_z_to_track == 0.4 ==`
   the generator's hardcoded value. This will silently break the moment either constant is changed
   independently.

4. **Friction coefficient value disagrees with its own inline comment about the C++ baseline.**
   `config_dfcip.py:36`: `mu = 0.6`. `mpc_wrapper_dfcip.py:135`: `_mu = config.mu  # 0.5 in C++` — the
   comment explicitly records that the reference C++ controller uses `μ=0.5`, while the ported
   Python config ships `0.6`. This is exactly the kind of parameter drift the downstream C++
   comparison should catch; flagging verbatim.

5. **WBC joint velocity limit is a hardcoded placeholder, by the author's own admission.**
   `mpc_utils.py:1251`: `vel_limit = 100.0 * jnp.ones(nj)  # rad/s — adatta al tuo URDF` ("adapt to
   your URDF"). No `tita.xml` joint or actuator tag defines a velocity limit (checked `tita.xml`),
   so this generous fixed bound is unlikely to ever bind — effectively unconstrained joint speed in
   the WBC QP.

6. **No torque limit enforced anywhere in the MPC or WBC QP**, despite `tita.xml` defining
   `forcerange="-120 120"` / `ctrlrange="-120 120"` per joint (`tita.xml:9,13,17,21`). MuJoCo itself
   will clip `data.ctrl` at simulation time (`mj_step`'s actuator clamping), so this is not a runtime
   crash risk, but it means the WBC's QP can request torques (`tau`, `mpc_utils.py:1436`) that exceed
   what the real actuators (or the C++ baseline's own explicit torque-limit constraint, if it has
   one) could deliver, silently degrading tracking rather than being accounted for in the
   optimization.

7. **Large block of dead config in `config_dfcip.py`.** Grep confirms none of the following are
   referenced anywhere in `mjx_tita.py`, `tita.py`, or `mpc_wrapper_dfcip.py`:
   `Kp`/`Kd` (generic, `:69-70`), `Qp,Qrot,Qdp,Qomega,Qacc,Qgrf` (`:90-98`, an old block-diagonal `W`
   assembly, itself commented out at `:107`), `timer_t, duty_factor, step_freq, step_height,
   clearence_speed` (gait-timer params for a walking/trotting reference generator not used by
   DFCIP's own `reference_generator_dfcip_online`), `p_legs0, q0_init, use_terrain_estimator,
   n_joints, joints_name` (the last two are computed but not consumed by the DFCIP wrapper), and
   `grf_ref` (only `u_ref`, which embeds the same values, is actually used). These look like
   leftovers copy-pasted from a quadruped `config_srbd.py`-style template; harmless at runtime but
   noise for anyone diffing configs against the C++ baseline.

8. **`config.inertia` (3×3 SRBD inertia matrix, `config_dfcip.py:61-63`) appears unused on the
   `mjx_tita.py` path** — grep found no reference to `config.inertia` in
   `mjx_tita.py`/`mpc_wrapper_dfcip.py`/`mpc_utils.py`/`objectives.py`/`models.py`'s DFCIP code. The
   reduced 13-state DFCIP model has no orientation DOF and thus no use for a rotational inertia
   tensor in its own dynamics; the WBC computes its own mass matrix `M = mjx_data.qM` live from MJX
   instead (`mpc_utils.py:1009`). Flag for the C++ comparison in case the baseline's SRBD-style
   controller *does* use this matrix somewhere the Python port dropped.

9. **`d = 0.567` (half-track wheel offset) is a hand-set constant, not derived from the MJCF**
   (§4) — recommend the downstream fix task independently re-derive it via forward kinematics at the
   nominal posture (`q0`) and compare, since it directly enters both the dynamics-adjacent
   heading-recovery formula (`mjx_tita.py:183-184`) and the cost's wheel-offset geometry
   (`objectives.py:25`).

10. **Un-epsilon'd/clip-fragile spots worth a numerical-robustness pass** (none observed to
    actually divide by exactly zero in the nominal gait, but all lack a defensive epsilon):
    - `mpc_utils.py:883,893,902`: `a/jnp.linalg.norm(a)` in `compute_virtual_frame`,
      `compute_contact_frame`, `get_rCP` — `a` is `(I-nnᵀ)z₀` or `n×z₀`, which is exactly zero only
      if the wheel normal `n` is aligned with `z₀` (wheel lying flat, i.e. robot fully overturned);
      not reachable in normal operation but not guarded either (contrast with `err_rotation` at
      `mpc_utils.py:865`, which *does* guard its own norm with `jnp.where(axis_norm<1e-7,...)`).
    - `objectives.py:8-9` (`penalty()`'s `safe_log`) clips its argument to `[1e-10,1e6]` before
      `log` — defensive, but note `penalty()` itself is **unused on the DFCIP path** (only consumed
      by the quadruped/H1/Talos objective functions, §1.4), so this doesn't affect Tita.
    - `mjx_tita.py:184` (`get_dfip_current_state`, itself dead code superseded by
      `process_tita_state` — see finding 11) divides by `config.d` with no epsilon; harmless since
      `d=0.567` is a compile-time constant, not a runtime-computed quantity that could hit zero.

11. **Dead duplicate state-construction path left in `mjx_tita.py`.** `build_tita_state()`
    (`:103-145`) and `get_dfip_current_state()` (`:147-197`) implement the same logic as
    `process_tita_state()` (`:200-231`, the one actually called in the hot loop, `:426`) using plain
    NumPy/non-jitted MuJoCo calls instead of the jitted MJX-free-array version; they are still called
    once at startup for the initial `x0`/warm `solve_mpc`/`solve_wbc` calls (`:351-388`) but the
    per-step loop uses only `process_tita_state` (`:420-421` are commented out in favor of `:422-426`,
    literally showing the migration in-place: `#tita_state = build_tita_state(...)` /
    `#x0, theta_prev = get_dfip_current_state(...)` commented out, replaced by
    `gather_raw_state`+`process_tita_state`). Not a bug, but worth pruning since it doubles the
    surface area a C++-parity reviewer has to check for the "same" computation.

---

## Summary of file:line references used

- `mpx/.gitmodules:1-3`
- `mpx/mpx/examples/mjx_tita.py` (whole file; key lines cited above: 15,45-57,103-145,147-197,
  184,199-231,233-248,293-536,391,397-536,438,447,461-475,504-519,522,594-655,660-676)
- `mpx/mpx/examples/mjx_policy_tita.py` (1-29,56,193-203,401-609,731,750-756)
- `mpx/mpx/examples/tita.py` (1-90,690,1377-1433)
- `mpx/mpx/examples/plot_rollout_info.py:1694-1697`
- `mpx/mpx/utils/mpc_wrapper_dfcip.py` (whole file; key lines 37-182,88-97,127-158,184-232,
  234-459,461-493)
- `mpx/mpx/utils/mpc_utils.py` (466-667,837-1438,897-904,968-1438,1009-1066,1102-1112,1154-1157,
  1246-1273,1283-1301,1423-1436)
- `mpx/mpx/utils/objectives.py:7-13,15-130,132-203`
- `mpx/mpx/utils/models.py:27-70`
- `mpx/mpx/utils/rotation.py:4-124`
- `mpx/mpx/utils/sim.py:14-165`
- `mpx/mpx/config/config_dfcip.py` (whole file)
- `mpx/mpx/data/tita/tita.xml:1-166` (option, defaults, actuators, bodies/inertials, contacts)
- Fetched (not present locally) `jax_ocp_solvers/optimizers.py` at pinned commit
  `f397c9a87d9f90170a9b461821a9a973901fa49a` — used only to classify the solver call graph in §1.4;
  re-fetch/`git submodule update --init` before trusting any numeric behavior of `fddp_mpc` beyond
  what is quoted here.
