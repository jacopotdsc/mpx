# Aliengo — issues found while porting the SRBD controller to Lite3

Found while validating `lite3_srbd.py` against the Aliengo standalone
`srbd_quad.py`. The Aliengo model is the IIT release and is treated as
authoritative: **nothing in the model has been modified**. One issue was small
enough to fix in the example script; the rest are documented only.

Date: 2026-09-06.

---

## 1. FIXED — the foot touch sensors never fire

**Severity: diagnostic only. Does not affect Aliengo's motion.**

`srbd_quad.py` read foot contact from the `FL_touch / FR_touch / RL_touch /
RR_touch` sensors. Measured on the compiled model at the `home` keyframe:

```
FL: site_z +0.0622 (radius 0.023)   geom_z +0.0392 (radius 0.0265)   offset +0.0230 m
FR: same        RL: same        RR: same
```

A MuJoCo `touch` sensor reports contact forces inside the **site's** volume. The
site sphere spans z ∈ [0.0392, 0.0852], while the foot's contact point is at the
bottom of the collision sphere, z ≈ 0.0127 at rest. The contact therefore never
falls inside the sensing volume and the sensor reads zero permanently.

Measured over a 20 s run before the fix:

```
all four feet: duty 0.000, touchdowns 0, longest airborne interval 20.00 s
contact mismatch (planned vs measured): 65.3%
```

The SRBD MPC was being told that no foot is ever on the ground, for the entire
run.

**Fix applied** (`mpx/examples/srbd_quad.py`, 2 lines): read the
`*_floor_found` contact sensors instead. They already exist in
`aliengo/xmls/sensor_feet.xml`, are attached to the foot collision geoms rather
than to a site, and report contact correctly.

```python
touch_adr = [model.sensor_adr[model.sensor(f"{n}_floor_found").id]
             for n in config.contact_frame]
```

**Result — and this is the important part:** the contact mismatch drops from
65.3% to 2.6%, while every other metric is unchanged to the last digit reported.
Nine scenarios, before → after:

| scenario | vx | rmse vx | height | GRF err | τ max | mismatch |
|---|---|---|---|---|---|---|
| stand | 0.017 → 0.017 | 0.018 → 0.018 | 0.376 → 0.376 | +2.0% → +2.0% | 21.8 → 21.8 | **65.3% → 2.6%** |
| vx_pos | 0.263 → 0.263 | 0.044 → 0.044 | 0.379 → 0.379 | +0.7% → +0.7% | 26.2 → 26.2 | **65.3% → 2.7%** |
| vx_neg | −0.222 → −0.222 | 0.081 → 0.081 | 0.371 → 0.371 | +4.4% → +4.4% | 25.4 → 25.4 | **65.3% → 2.0%** |
| wz_pos | 0.019 → 0.019 | 0.020 → 0.020 | 0.376 → 0.376 | +2.2% → +2.2% | 21.9 → 21.9 | **65.3% → 2.6%** |
| long_run 20 s | 0.261 → 0.261 | 0.045 → 0.045 | 0.380 → 0.380 | +0.3% → +0.3% | 26.2 → 26.2 | **65.0% → 2.6%** |

Aliengo walks identically either way. That tells us something worth knowing:
**the `contact` input to `mpc.run()` has no measurable influence on the
resulting motion** — the gait timer and the reference generator dominate. The
fix is therefore risk-free, but it also means the contact input is currently
close to inert in this controller.

The touch sensors themselves were left alone. Making them work would mean moving
the `FL`/`FR`/`RL`/`RR` sites down onto the collision spheres, and those same
sites feed `FL_pos`, `FL_global_linvel` and the `feet_clearance` / `feet_height`
rewards of every Aliengo RL environment. That is not a small change.

---

## 2. NOT FIXED — the WBC runs on a different model than the simulator

**Severity: potentially significant. Left alone deliberately.**

`config_srbd.py` sets `model_path = mpx/data/aliengo/aliengo.xml`, and
`BatchedMPCControllerWrapper` builds the whole-body controller's kinematics from
that file. `srbd_quad.py` meanwhile simulates the Playground Aliengo scene. They
are **not the same robot**:

| | `mpx/data/aliengo/aliengo.xml` | Playground Aliengo |
|---|---|---|
| mass | 24.638 kg | 24.638 kg |
| hip spacing (foot y at nominal pose) | ±0.194 m | ±0.134 m |
| foot x, front / hind | +0.271 / −0.209 | +0.211 / −0.269 |
| nominal `q0` | `[0.2, 0.8, −1.8]` | `[0, 0.9, −1.8]` |

So the Jacobians, foot positions and `p_legs0` used by the controller describe a
geometry 45% wider at the hips than the one being simulated, with the front/hind
asymmetry reversed. `config.p_legs0` matches the mpx model exactly
(`[0.27092872, 0.193, 0]` against a measured `[0.27093, 0.19378, 0.0434]`), so
the configuration is self-consistent — it is simply consistent with the wrong
model.

Not corrected: changing `model_path` to the Playground scene would silently
change the behaviour of the Aliengo controller and of the Aliengo residual-RL
environment, which is the opposite of a small fix. For **Lite3** the new
`config_srbd_lite3.py` points at the same XML the simulator steps, so the
problem does not arise there.

---

## 3. NOT FIXED — the declared SRBD inertia is hand-inflated

**Severity: a tuning choice, not a bug. Documented for reference.**

`config_srbd.py` declares

```
inertia = [[0.2311, -0.0015, -0.0214],
           [-0.0015,  1.4485,  0.0005],
           [-0.0214,  0.0005,  1.5032]]
```

Computing the composite inertia of `mpx/data/aliengo/aliengo.xml` at its own
`q0` — summing every body about the CoM and rotating into the base frame — gives

```
           [[0.2438, -0.0010, -0.0082],
            [-0.0010,  0.8830, -0.0005],
            [-0.0082, -0.0005,  0.9373]]
```

`Ixx` agrees (0.2311 vs 0.2438), but `Iyy` and `Izz` are inflated by **1.64x and
1.60x**. The trunk-only inertia (0.161 / 0.175) is far smaller still, so the
declared value is neither the composite nor the trunk. Inflating the rotational
inertia of an SRBD model is a known way to make the MPC more conservative in
pitch and yaw, so this looks deliberate rather than mistaken.

Not corrected: it is a tuning parameter of a controller that works, and changing
it would alter Aliengo's behaviour. Lite3 uses its physically derived composite
inertia instead, and validates.

---

## 4. FIXED — per-step debug printing in `srbd_quad.py`

**Severity: cosmetic.**

`srbd_quad.py` printed a five-line block (`foot_z`, two contact vectors, a match
flag, a separator) on every MPC step, i.e. 50 times per simulated second.
Removed. `--metrics <csv>` and `--cmd VX VY WZ` were added in its place so the
same headless harness can drive Aliengo and Lite3 with identical scenarios.
Neither change touches the control law.

---

## 5. Observations, no action taken

- **Steady-state undershoot on vx.** Commanded 0.300, measured 0.261 (−13%).
  Lite3 shows the same effect more strongly (−22%). Shared property of the SRBD
  MPC as configured.
- **Base height sits ~7% above nominal**: 0.376 against `robot_height = 0.35`.
  Lite3: 0.332 against 0.31, ~7% as well. Same behaviour on both robots.
- **Yaw overshoot 18%** at a commanded 0.5 rad/s.
- **Torque reaches the limit during command transitions**: peak 30.0 N·m in the
  still → forward → turn → still scenario. Within the ±35.278 N·m of the
  thigh/calf actuators, but the hip limit is ±44.4 and the peak is at the
  actuator's own ceiling for at least one joint, so there is no margin left
  there.
- `sim_utils.estimate_contacts` and a `foot_z < 0.035` threshold are both
  present but commented out in `srbd_quad.py`. Left as they were.

---

## Summary

| # | issue | severity | action |
|---|---|---|---|
| 1 | touch sensors never fire | diagnostic | **fixed in the script**, model untouched, behaviour unchanged |
| 2 | WBC model ≠ simulated model | potentially significant | documented, not touched |
| 3 | inflated SRBD inertia | tuning choice | documented, not touched |
| 4 | per-step debug prints | cosmetic | **removed** |
| 5 | tracking undershoot, height offset, yaw overshoot, torque peak at transitions | inherent | documented |

Files changed: `mpx/examples/srbd_quad.py` only. No Aliengo model, config or RL
environment was modified.
