# Residual learning — notes from the papers in `mpx/mpx/paper/`

Read with one question in mind: what does the literature actually prescribe for
adding an RL residual on top of an MPC + whole-body controller, at the level of
detail needed to write `joystick.py`. Not a literature review.

Every point below is tagged with its origin:

- **[P]** taken from a paper, with the reference
- **[A]** inherited from the existing Aliengo residual environment
- **[E]** my own engineering decision for Lite3

Primary source: **Jeon, Lee, Hong, Kim — "Residual MPC: Blending Reinforcement
Learning with GPU-Parallelized Model Predictive Control"** (MIT, arXiv
2510.12717, Oct 2025). It is the closest match to what we are building: a
torque-level residual policy blended with a whole-body MPC, both evaluated at
100 Hz, trained with PPO on thousands of parallel GPU environments.

Secondary: **"RL-Augmented MPC for Non-Gaited Legged and Hybrid Locomotion"** —
a *hierarchical* architecture (the policy sets the MPC's inputs) rather than a
residual one, so only its observation-design argument transfers.

---

## 1. Where the residual is added

**[P]** The MIT paper tests three blending strategies and measures them against
each other with the same rewards, the same zero-initialized network and
λ = 0.1:

| # | name | formula | result (Fig. 6) |
|---|---|---|---|
| 1 | joint action, joint blending | `τ = Kp(q_cmd + λa − q) + Kd(q̇_cmd − q̇) + τ_cmd` (22) | reward plateau ≈ 12.5 |
| 2 | joint action, torque blending | `τ_res = Kp(a + q̂ − q) − Kd q̇` (23), `τ = τ_MPC + λ τ_res` (24) | reward plateau ≈ 12.5 |
| 3 | **torque action, torque blending** | `τ = τ_MPC + λ a` (25) | **reward plateau ≈ 8** |

Their conclusion, quoted: *"While there is little difference between
joint-space action representations for the residual policy, a torque-space
action representation performs significantly worse."* The stated reason: *"The
torque space representation controls the joints at the acceleration level,
making it more difficult for the policy to smoothly explore actions with high
advantage. Additionally, the scale of torque-space actions are significantly
larger than joint-space actions."*

They ship **strategy 2**, choosing it over 1 only to avoid a failure mode:
*"in the event of a diverging MPC solution, the joint-joint strategy would output
actions relative to potentially infeasible q_MPC setpoints."*

**This contradicts the architecture requested for our implementation**
(`tau_final = tau_mpc + tau_rl`, i.e. strategy 3). The contradiction is
documented here rather than resolved silently — see section 8.

**[P]** Note what τ_MPC means in that paper (eq. 21):

```
τ_MPC = Kp(q_MPC − q) + Kd(q̇_MPC − q̇) + τ_RNEA
```

Their *nominal* controller already contains a PD term. It tracks the **MPC's own
predicted joint trajectory**, not a fixed default pose. That distinction matters:
a PD toward `q_MPC` is part of the model-based controller; a PD toward a fixed
`q_default` is not, it is a separate hand-designed stabiliser.

**[A]** The Aliengo environment currently computes
`Kp*(q_default + action*action_scale − q) + Kd*(−q̇)` and adds it to the WBC
torque. Structurally this is the paper's strategy 2 (eq. 23) with `q̂ = q_default`
and λ folded into `action_scale`. So the Aliengo code is not arbitrary — it
matches the strategy the paper ships. What it lacks is an explicit λ and the
knowledge that at `action = 0` the residual is **not** zero.

**[E]** Our mpx whole-body controller outputs torque only — it does not expose a
`q_MPC` / `q̇_MPC` setpoint — so strategy 1 is not directly available without
first inverting torque back into a joint setpoint. `srbd_quad.py` does contain a
`tau_to_qdes` helper that does exactly this, so it is reachable, but it adds a
mass-matrix solve per step.

## 2. Blending factor λ

**[P]** λ = 0.1, a scalar, applied to the policy output before adding it to the
MPC torque. Its stated purpose: *"we introduce a weighting parameter to control
the initial influence of the policy"* — because with strategy 2 a zero-initialized
network still emits non-zero torque.

**[E]** With strategy 3 (torque action) λ and the action scale are the same
parameter: `τ = τ_MPC + λ·a` with `a ∈ [−1, 1]` means λ *is* the residual torque
budget in N·m.

## 3. Policy initialization

**[P]** *"It is important that the residual policy has little effect on the
baseline controller when initialized. This is to ensure that the residual
controller is aligned with the control prior at the start of training, instead
of destabilizing the system with random outputs."* They zero-initialize the
network weights and biases (their ref [23]).

**[P]** *"If the action space of the policy output is the same as that of the
nominal policy, this can easily be achieved by initializing the policy network to
have weights and biases of zero (e.g., a position controlled manipulator, with
position-space actions for the residual policy)."*

**[A]** `train_srbd.py` already has a `zero_init_output_layer` path in
`_build_fresh_networks`, used by its `--zero` flag. It zero-initializes the
**output layer** only, which is the standard practical form.

## 4. Why residual at all — what to expect

**[P]** Measured against an end-to-end policy given identical rewards,
environments and initial conditions (Figs. 6–8):

- the residual converges **faster and to a higher asymptote** than end-to-end;
- end-to-end *"learns to rapidly oscillate the ankle joint to glide across the
  floor, exploiting the physics of the environment"*, while the residual
  *"learns to stay close to the MPC solutions, exhibiting clear stepping
  behavior at the same frequency as the MPC"*;
- the residual shows *"a smaller median and range"* in joint velocities,
  torques, mechanical power and ground reaction force;
- against the MPC baseline it widens the trackable velocity envelope by roughly
  **78% in vx, 12% in vy, 9% in ωz**;
- wall-clock cost: 3–4 hours against 30 minutes for end-to-end, ~8x slower per
  gradient update, because the MPC runs inside the RL loop.

**[P]** *"we find that the residual torques are generally antagonistic to the MPC
torques, particularly shortly before a planned touchdown contact"* — a useful
diagnostic signature to look for in our own logs.

## 5. Observations

**[P]** `o = [p, θ, q, ω, v, q̇, τ_MPC] ∈ R^54`, and the policy outputs leg
actions only, `a ∈ R^10`. The only MPC information given to the policy is
**τ_MPC**. Explicitly: *"While we originally allowed the network to observe the
output torques and predicted states, we found that this made sim-to-sim transfer
and sim-to-real transfer less reliable, as the distributions of these quantities
were sensitive to the IsaacGym simulator."*

**[P]** The hierarchical paper reaches the same conclusion from the other side:
*"Instead of observing the full MPC state — which is impractical — we construct
an observation subset that captures the minimal MPC information we deem
sufficient for task completion."*

**[E]** Takeaway for us: give the actor the proprioception it would have on the
real robot plus, at most, a compact MPC signal. Do **not** feed it foot
references, GRF or the full contact schedule. Privileged MPC quantities belong to
the critic if anywhere.

## 6. Reward

**[P]** Table I, verbatim, with weights:

| term | weight | function |
|---|---|---|
| linear velocity tracking | **10.0** | `exp(−‖(c_vxy − v_xy)/(1+\|c_vxy\|)‖² / σ)` |
| angular velocity tracking | **5.0** | `exp(−‖c_ω − ω_z‖² / σ)` |
| 1st order action rate | −1e−3 | `‖(a_t − a_{t−1})/Δt‖²` |
| 2nd order action rate | −1e−4 | `‖(a_t − 2a_{t−1} + a_{t−2})/Δt‖²` |
| torques | −1e−4 | `‖τ‖²` |
| orientation | 1.0 | `exp(−‖g_xy‖² / σ)` |
| height | 1.0 | `exp(−‖c_z − p_z‖² / σ)` |
| joint regularization | 1.0 | per-joint deviation term |
| self-collision | −1.0 | indicator |
| termination | **−100** | indicator |

Three things worth copying:

1. **[P]** *"The set of rewards given to the system is intentionally minimal. We
   avoid giving overly specific rewards such as foot guidance, air time, or
   contact-scheduling terms."* The MPC already schedules the gait; paying the
   policy for gait-shaped behaviour is redundant and invites the degenerate
   solutions we already hit in the Lite3 E2E environment.
2. **[P]** The linear-velocity error is **normalised by `1 + |command|`**, so
   tracking a large command is not penalised more harshly in absolute terms than
   tracking a small one.
3. **[P]** Both a **first and a second order action-rate penalty**. The second
   order term is exactly the discrete jerk metric that came out at 1.33–1.69 in
   the Lite3 E2E diagnosis, where nothing penalised it.

**[P]** Termination is weighted −100, two orders of magnitude above the tracking
terms. Our Lite3 E2E environment uses −1.0 and clips the total reward at zero,
which makes the termination penalty nearly invisible.

## 7. Training

**[P]** PPO with a clipped objective and GAE, 2048 parallel environments, MPC and
policy both at 100 Hz. *"Note that any RL algorithm could be used to optimize the
policy, but we choose PPO for its simplicity and effectiveness in continuous
control tasks."*

**[A]** `train_srbd.py` uses PPO with GAE at 1024 environments — the same family.
Our MPC runs at 50 Hz with the WBC at 200 Hz, i.e. the control rate is 50 Hz
rather than 100 Hz.

## 8. The open decision, stated plainly

The requested architecture is `tau_final = clip(tau_mpc + tau_rl, ±30)` with
`tau_rl = action · residual_torque_scale`, and the requirement that
`action = 0 ⟹ tau_final == tau_mpc` exactly. That is the paper's **strategy 3**,
the one it measures as clearly worst (reward plateau ≈ 8 against ≈ 12.5).

The two properties are in genuine tension:

| | exact baseline at `a = 0` | learning performance (Fig. 6) |
|---|---|---|
| strategy 1, joint–joint | **yes** | good |
| strategy 2, joint–torque | no (PD toward `q̂` remains) | good |
| strategy 3, torque–torque | **yes** | measurably worse |

Strategy 1 gives both, but it needs the nominal controller to expose a joint
setpoint, which our torque-only WBC does not — it would require running
`tau_to_qdes` every step.

**[E] Decision:** implement **strategy 3** as specified, because exact baseline
reproducibility was made an explicit requirement and because it is by far the
simplest thing to read and to verify. Keep the blending factor explicit and
separate from the action so that strategy 2 remains a small, local edit if
learning stalls. Record the paper's result in the code comment so the trade-off
is not lost, and make the comparison against the pure-MPC baseline part of the
evaluation from the start, since that is the only way to detect the failure mode
the paper is warning about.

## 9. What we take, and from where

| choice | origin |
|---|---|
| residual added at the torque level, after the WBC | [P] |
| explicit scalar blending factor, small at the start | [P] λ = 0.1 |
| zero-initialized output layer | [P], [A] already available in `train_srbd.py` |
| actor sees proprioception + command, no MPC internals | [P] |
| minimal reward, no gait-shaping terms | [P] |
| velocity error normalised by `1 + \|command\|` | [P] |
| first **and** second order action-rate penalties | [P] |
| large termination penalty relative to tracking | [P] |
| torque action space (strategy 3) | [E], against [P]'s measurement, for baseline reproducibility |
| PPO, GAE, parallel environments | [P], [A] |
| Lite3 torque limits, mass, gait parameters | measured, see `LITE3_SRBD_VALIDATION_REPORT.md` |
