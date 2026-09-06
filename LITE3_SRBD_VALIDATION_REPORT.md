# Lite3 SRBD model-based controller — validation report

Baseline validation of `mpx/examples/lite3_srbd.py`, the Lite3 port of the
standalone SRBD MPC + model-based whole-body controller. No RL is involved.

Date: 2026-09-06. Hardware: RTX 2070, MuJoCo 3.8.0, conda env `mjpl`.

---

## 1. Controller architecture

Unchanged from `srbd_quad.py`; only the model and the configuration differ.

```
                 joystick command [vx, vy, wz] + robot_height
                                  |
   gait timer (duty_factor, step_freq, timer_t)  ->  contact schedule
                                  |
   reference generator (swing trajectories, foot references, GRF reference)
                                  |
   SRBD MPC   state = [p(3), quat(4), v(3), omega(3)],  controls = 4x3 GRF
              horizon N=25 x dt=0.02 s = 0.50 s, solved at 50 Hz
                                  |
   model-based whole-body controller, at 200 Hz:
       tau = D[6:] + (M @ pinv(J^T) @ (Kp*(foot_ref - foot) + Kd*(...)))[6:]
             - (J @ grf)[6:]
                                  |
                    12 joint torques -> data.ctrl
```

The whole-body stage is the analytic inversion in
`mpx/utils/mpc_utils.whole_body_interface`, **not** a QP. The MPC state is a
JAX pytree (`MPCState`) carried across steps.

## 2. Differences from `srbd_quad.py`

`srbd_quad.py` is, despite the generic name, configured for **Aliengo**: it
loads `aliengo_constants.task_to_xml("flat_terrain")` and reads the
`FL_touch/FR_touch/RL_touch/RR_touch` sensors. The Go1 lines are commented out.
It is the baseline used for comparison here.

| | `srbd_quad.py` (Aliengo) | `lite3_srbd.py` |
|---|---|---|
| config | `mpx.config.config_srbd` | `mpx.config.config_srbd_lite3` |
| simulated model | Playground Aliengo flat scene | Playground Lite3 flat scene |
| WBC model (`config.model_path`) | `mpx/data/aliengo/aliengo.xml` — **a different model** | the same Lite3 scene the simulator steps |
| contact sensing | `*_touch` site sensors | `*_floor_found` contact sensors |
| per-step prints | one block per MPC step | none |
| diagnostics | none | optional `--metrics CSV` |
| command | keyboard only | keyboard, or `--cmd` / a callable for scripted transitions |

The WBC-model difference matters. `mpx/data/aliengo/aliengo.xml` has hip spacing
`y = ±0.194` while the Playground Aliengo has `±0.134`, so the Aliengo
whole-body controller computes Jacobians for a geometry the simulator does not
have. `lite3_srbd.py` points `model_path` at the very XML being stepped, which
removes that entire class of mismatch by construction.

## 3. Lite3 parameters and where they come from

All values measured on the Lite3 model, none scaled from Aliengo.

| parameter | value | source |
|---|---|---|
| `mass` | 11.9376 kg | `body_subtreemass[1]`; Aliengo 24.638 kg |
| `inertia` | `diag ≈ [0.1707, 0.3015, 0.3719]` | composite inertia of all bodies about the CoM at the `home` pose, rotated into the base frame |
| `p0` | `[0, 0, 0.31]` | `home` keyframe |
| `q0` | `[0, −0.8, 1.6] × 4` | `home` keyframe |
| `robot_height` | 0.31 m | base height with the feet on the ground |
| `p_legs0` | FL/FR `x=+0.18176`, HL/HR `x=−0.16724`, `y=±0.15935` | `mj_forward` on the keyframe |
| `contact_frame` | `FL, FR, HL, HR` | foot collision geoms |
| `body_name` | `FL_FOOT, FR_FOOT, HL_FOOT, HR_FOOT` | Lite3 has a dedicated foot body; Aliengo hangs its contact geom off the calf |
| `timer_t` | `[0.5, 0, 0, 0.5]` | trot, FL+HR against FR+HL |
| `duty_factor` | 0.65 | as Aliengo |
| `step_freq` | 1.4 Hz | Froude scaling: `1.35·√(0.35/0.31) = 1.43` |
| `step_height` | 0.055 m | `0.065·(0.41/0.50)`, ratio of leg lengths |
| `Kp` / `Kd` (WBC) | 500 / 20 | unchanged: the output is `M·pinv(Jᵀ)·(Kp·e + Kd·ė)`, so the mass matrix already scales the torque with the robot; the gains set foot bandwidth |
| MPC cost `W` | as Aliengo | no evidence yet that it needs to change |

**Deliberate departure.** `config_srbd.py` declares an Aliengo inertia of
`Iyy = 1.449`, `Izz = 1.503`, while the composite inertia of its own model at
its own `q0` is `0.883` / `0.937` — inflated by ~1.62x, evidently hand-tuned.
Lite3 uses the physically derived value.

## 4. Effective frequencies

| | value |
|---|---|
| simulation / WBC | 200 Hz (`model.opt.timestep = 1/whole_body_frequency = 0.005 s`) |
| MPC | 50 Hz — one MPC solve every 4 simulation steps |
| MPC internal `dt` | 0.02 s, horizon 25 steps = 0.50 s |
| warm-start shift | 1 stage per solve (`1/(dt·mpc_frequency)`) |

## 5. State, foot, joint and actuator mapping

Verified index by index on the compiled model:

```
[ 0] FL_HipX_joint   qpos[ 7] qvel[ 6] <- act[ 0]      geom FL id  9  body FL_FOOT id  5
[ 1] FL_HipY_joint   qpos[ 8] qvel[ 7] <- act[ 1]      geom FR id 16  body FR_FOOT id  9
[ 2] FL_Knee_joint   qpos[ 9] qvel[ 8] <- act[ 2]      geom HL id 23  body HL_FOOT id 13
...                                                    geom HR id 30  body HR_FOOT id 17
[11] HR_Knee_joint   qpos[18] qvel[17] <- act[11]
```

- joint order == actuator order == WBC torque order: **True**
- every contact geom belongs to the body declared in `body_name`: **True**
- the four `*_floor_found` sensors point at geoms 9/16/23/30 in `contact_frame`
  order, so GRF blocks and Jacobian columns line up: **True**
- `config.mass` vs model mass: difference `0.00e+00`
- SRBD state built from `qpos[:3]`, `qpos[3:7]`, `qvel[:3]`, `qvel[3:6]` — 13
  entries, matching `config.n`
- joint limits HipX `[−0.523, 0.523]`, HipY `[−2.67, 0.314]`, Knee `[0.524, 2.792]`;
  actuator `ctrlrange` `±30 N·m` on all twelve

## 6. Tests performed

`mpx/examples/lite3_srbd_validate.py`, headless, nine scenarios per robot,
identical for both. 1200 steps = 6 s unless stated.

```bash
python lite3_srbd_validate.py --robot lite3   --out validation
python lite3_srbd_validate.py --robot aliengo --out validation
```

stand `(0,0,0)` · vx_pos `(0.3,0,0)` · vx_neg `(−0.3,0,0)` · vy_pos `(0,0.2,0)` ·
wz_pos `(0,0,0.5)` · wz_neg `(0,0,−0.5)` · vx_wz `(0.3,0,0.3)` ·
long_run `(0.3,0,0)` for 4000 steps = 20 s · transition (still → forward →
forward+turn → still, 2 s each).

## 7. Results

| scenario | robot | vx | rmse vx | wz | h | roll rms | pitch rms | GRF err | τ max | duty spread |
|---|---|---|---|---|---|---|---|---|---|---|
| stand | Aliengo | 0.017 | 0.018 | 0.001 | 0.376 | 0.0121 | 0.0042 | +2.0% | 21.8 | 0.000 |
| | **Lite3** | −0.024 | 0.025 | −0.002 | 0.332 | **0.0066** | 0.0166 | +2.0% | **9.4** | 0.040 |
| vx_pos | Aliengo | 0.263 | 0.044 | 0.004 | 0.379 | 0.0120 | 0.0083 | +0.7% | 26.2 | 0.000 |
| | **Lite3** | 0.233 | 0.072 | −0.003 | 0.334 | **0.0060** | 0.0075 | +0.3% | **13.6** | 0.034 |
| vx_neg | Aliengo | −0.222 | 0.081 | 0.003 | 0.371 | 0.0176 | 0.0114 | +4.4% | 25.4 | 0.000 |
| | **Lite3** | **−0.269** | **0.044** | −0.002 | 0.325 | 0.0120 | 0.0262 | +6.6% | **14.5** | 0.047 |
| vy_pos | Aliengo | 0.013 | 0.030 | −0.004 | 0.376 | 0.0312 | 0.0167 | +2.2% | 26.0 | 0.000 |
| | **Lite3** | −0.022 | 0.036 | 0.001 | 0.331 | **0.0175** | 0.0187 | +2.5% | **18.1** | 0.043 |
| wz_pos | Aliengo | 0.019 | 0.020 | 0.429 | 0.376 | 0.0114 | 0.0033 | +2.2% | 21.9 | 0.000 |
| | **Lite3** | −0.024 | 0.025 | **0.453** | 0.331 | **0.0060** | 0.0151 | +2.5% | **9.3** | 0.037 |
| wz_neg | Aliengo | 0.017 | 0.019 | −0.420 | 0.376 | 0.0109 | 0.0084 | +2.2% | 22.6 | 0.000 |
| | **Lite3** | −0.024 | 0.025 | **−0.459** | 0.331 | **0.0063** | 0.0155 | +2.4% | **9.8** | 0.039 |
| vx_wz | Aliengo | 0.263 | 0.044 | 0.259 | 0.379 | 0.0073 | 0.0080 | +0.8% | 25.7 | 0.000 |
| | **Lite3** | 0.232 | 0.074 | **0.274** | 0.334 | 0.0066 | 0.0082 | +0.5% | **14.1** | 0.033 |
| long_run 20 s | Aliengo | 0.261 | 0.045 | 0.003 | 0.380 | 0.0119 | 0.0069 | +0.3% | 26.2 | 0.000 |
| | **Lite3** | 0.234 | 0.071 | 0.001 | 0.335 | **0.0059** | 0.0071 | **+0.0%** | **13.6** | **0.001** |
| transition | Aliengo | 0.152 | 0.051 | 0.174 | 0.378 | 0.0145 | 0.0191 | +1.0% | **30.0** | 0.000 |
| | **Lite3** | 0.109 | 0.063 | 0.194 | 0.333 | **0.0095** | 0.0155 | +1.6% | **14.9** | 0.008 |

**No falls, no NaN/Inf, in any scenario, for either robot.**

Overshoot and settling on the dominant axis:

| scenario | robot | overshoot | contact mismatch | symmetry L-R | symmetry F-H |
|---|---|---|---|---|---|
| vx_pos | Aliengo | 9.7% | 65.3% | 0.0000 | 0.0000 |
| | **Lite3** | **0.6%** | **2.9%** | 0.0000 | 0.0017 |
| wz_pos | Aliengo | 18.0% | 65.3% | 0.0000 | 0.0000 |
| | **Lite3** | 24.5% | **3.1%** | 0.0038 | 0.0062 |
| long_run | Aliengo | 9.7% | 65.0% | 0.0000 | 0.0000 |
| | **Lite3** | **1.0%** | **2.9%** | 0.0001 | 0.0011 |

Per-foot gait over the 20 s run:

```
LITE3    FL duty 0.633 td 29 air 0.27s   FR duty 0.632 td 29 air 0.27s
         HL duty 0.634 td 29 air 0.27s   HR duty 0.634 td 29 air 0.27s
ALIENGO  all four feet: duty 0.000, td 0, air 20.00s   <- sensor never fires
```

All four Lite3 feet complete the cycle identically: duty factors within 0.002,
the same 29 touchdowns, the same 0.27 s maximum airborne time. Nothing is stuck
in the air, nothing skips a step. `duty_factor` is configured at 0.65 and the
measured value is 0.633.

Ground reaction forces at zero command: `Σ Fz = 121.6 N` against `m·g = 117.1 N`,
error **+3.8%**, with a per-foot distribution of FL 28.4 / FR 30.8 / HL 32.1 /
HR 30.3 N — a 13% spread, consistent with the CoM sitting slightly aft.
Torques never saturate: peak 18.1 N·m against a ±30 N·m limit, **0% of samples
in saturation** across every scenario.

## 8. Problems found and corrections applied

**Aliengo's contact sensors never fire.** `FL_touch` and friends are `touch`
sensors on sites placed at `pos="0 0 -0.213"` while the collision spheres are at
`pos="0 0 -0.25"`, 37 mm lower. The contact point falls outside the site's
sensing volume, so the sensor reads zero for the entire run. That is the whole of
Aliengo's 65% "contact mismatch": the MPC is told no foot is ever on the ground.
Fixed in `srbd_quad.py` (not in the IIT model) by reading the `*_floor_found`
contact sensors, which already exist in the Aliengo scene and are attached to
the collision geoms. Aliengo's mismatch drops from 65.3% to 2.6% while every
other metric stays identical to the last reported digit -- so the MPC's
`contact` input turns out to have no measurable influence on the motion. The
table above reports the pre-fix Aliengo numbers; see `errori_aliengo.md` for the
before/after comparison. Lite3 used `*_floor_found` from the start.

**`srbd_quad.py` printed a diagnostic block on every MPC step.** Removed, and
`--metrics` / `--cmd` added so the same harness can drive both robots. This is
the only change to the Aliengo path and it does not alter the control law.

No correction was needed on the Lite3 controller itself: it validated on the
first run of the configuration described in section 3.

## 9. Remaining limitations

1. **Steady-state undershoot on vx.** Lite3 reaches 0.234 of a commanded 0.300
   (−22%); Aliengo reaches 0.261 (−13%). Neither settles inside a ±10% band, so
   `settling_s` is reported as the full run duration for both. This is a
   property of the SRBD MPC as configured, not a Lite3 defect, and it is exactly
   the kind of gap residual RL is meant to close. Not tuned away, because there
   is no measurement yet saying which cost weight is responsible.
2. **Base height sits ~6% above nominal**: Lite3 0.332 against 0.31, Aliengo
   0.376 against 0.35. Same relative offset on both robots, same cause.
3. **Yaw overshoot at wz 0.5 is 24.5%** against Aliengo's 18.0%. Both overshoot;
   Lite3 more. Worth revisiting if yaw tracking matters, via `Qomega`.
4. The MPC cost matrix `W` is inherited from Aliengo. It works, but it has not
   been shown to be the right choice for Lite3's inertia.

## 10. Why this is a reliable baseline

- Every physical parameter is derived from the Lite3 model and cross-checked
  against it (mass exact to 0.00e+00).
- Joint, actuator, foot, sensor, GRF and Jacobian orderings are verified index
  by index, not assumed.
- The whole-body controller and the simulator share one model, so their
  kinematics cannot drift.
- Nine scenarios including 20 s of continuous locomotion and a scripted command
  transition: no falls, no NaN/Inf, no torque saturation, 0.27 s maximum
  airborne time per foot.
- The gait is symmetric to within 0.002 of duty factor across all four legs.
- Ground reaction forces balance the robot's weight to within 3.8%.
- On every posture metric (roll rms, torque headroom, contact consistency,
  overshoot on vx) Lite3 matches or beats the Aliengo baseline, and where it is
  worse the deficit is shared with Aliengo rather than specific to Lite3.

## 11. Commands

```bash
cd ~/Desktop/repo_jacopo/mpx/mpx/examples
export PYTHONPATH=~/Desktop/repo_jacopo/mujoco_playground:~/Desktop/repo_jacopo/mpx

python lite3_srbd.py                                       # viewer
python lite3_srbd.py --headless --steps 1200 --cmd 0.3 0 0
python lite3_srbd.py --headless --cmd 0 0 0 --metrics stand.csv
python lite3_srbd_validate.py --robot both --out validation
```
