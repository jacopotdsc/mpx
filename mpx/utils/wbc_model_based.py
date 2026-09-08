"""Model-based whole-body controller for the wheeled biped (no QP).

This is the Jacobian-inversion counterpart of the QP whole-body controller in
``mpx.utils.mpc_utils.whole_body_interface_wheeled_legged_qp``: same model, same
tasks, same task PD law, same inverse-dynamics torque map -- but the joint
accelerations come from a *closed-form linear solve* instead of an inequality
constrained QP (qpax).  It is the wheeled-biped analogue of
``mpc_utils.whole_body_interface``, the quadruped WBC used with the SRBD model,
where the MPC ground reaction forces are mapped to joint torques through the
contact Jacobian and the task tracking is closed with a Jacobian pseudo-inverse.

What is solved
--------------
Tasks (identical to the QP cost): CoM linear acceleration, left/right wheel
centre linear acceleration, base angular acceleration, joint posture, plus the
``w_qddot`` regularisation on the whole acceleration vector.  Stacking them as a
weighted least-squares problem in ``qddot``,

    min_qddot  1/2 || A qddot - b ||^2_W  + 1/2 w_qddot ||qddot||^2

with ``A = [J_com; J_lwheel; J_rwheel; J_base_rot; S]`` and the right-hand side
``b_i = a_i_total - a_i_drift``, the stationarity condition is a single linear
system

    H qddot = g,   H = w_qddot I + sum_i w_i J_i' J_i,   g = sum_i w_i J_i' b_i

i.e. exactly a damped, weighted pseudo-inverse of the stacked task Jacobian
(``qddot = H^-1 g``).  This is the same optimum the QP would return if none of
its constraints were active.

Wheel rolling constraint
------------------------
The QP additionally enforces, as a hard equality, that the contact point of each
wheel has zero acceleration in the rolling directions.  Dropping it entirely
would let the solution violate the non-holonomic rolling of the wheels, so by
default (``enforce_rolling=True``) it is kept and the least-squares problem is
solved subject to it through its KKT system

    [ H    Ac' ] [qddot ]   [ g  ]
    [ Ac   -e I] [lambda] = [ bc ]

which is still one linear solve of size ``nv + 6`` (14 + 6 for Tita), no
iterations, no active-set/interior-point loop.  ``enforce_rolling=False`` gives
the bare pseudo-inverse ``qddot = H^-1 g``.

Contact forces
--------------
``force_source``:

* ``"dynamics"`` (default): the forces are recovered from the six unactuated
  (floating base) rows of the dynamics given ``qddot``, i.e. the equality the QP
  enforces, ``Mu qddot + cu = Jlu' T_l fl + Jru' T_r fr``, solved in least
  squares.  Since the wheel wrench map is square (6 equations, 3+3 forces) this
  is again a plain linear solve.
* ``"mpc"``: the contact forces are taken straight from the MPC solution
  (``fcl``/``fcr``, appended to the ``desired`` vector by
  ``BatchedMPCControllerWrapper._build_desired_impl``) and fed forward, exactly
  like ``tau_mpc = -(J grf)`` in the quadruped SRBD controller.

  This is the literal transposition of the quadruped controller, but it does NOT
  work on Tita: the quadruped's point feet make the SRBD forces a good match for
  the full model, whereas here the DFCIP forces and the ``qddot`` chosen by the
  tasks do not satisfy the six floating-base rows together (``dyn_res_norm`` is
  ~1.3 N even standing still), so the torque does not produce the acceleration
  it was computed for.  Measured on validate_dfcip_controller.py: the robot
  falls at vx=1.0 and at vx=0.6/omega=0.4, while ``"dynamics"`` tracks every
  case.  Kept as an option for comparison, not as a default.

Either way the actuated torque is the same inverse-dynamics expression used by
the QP controller:

    tau = Ma qddot + ca - Jla' T_l fl - Jra' T_r fr

What is lost with respect to the QP
-----------------------------------
Inequalities cannot be represented by a linear solve: the friction cones, the
positive normal-force floor and the joint position/velocity limits are *not*
enforced -- they are only measured and reported in the diagnostics dict
(``fric_margin_l/r``, ``ineq_slack_min_joint``).  In exchange the controller is
a fixed-cost, differentiable, always-finite linear solve: no solver failure and
no NaN on infeasible commands (see the ``fz_min`` comment in the QP for the
failure mode this avoids), and it is roughly 5x cheaper (0.29 ms vs 1.59 ms per
call on CPU, batch of one).

Measured against the QP on the six commands of
``mpx/examples/validate_dfcip_controller.py`` (4 s hold, flat scene, defaults):
steady-state vx / omega tracking is identical to three decimals on every case
(e.g. vx=1.0/omega=0.8 -> 0.997 / 0.805 for both), no fall, no NaN, same peak
torque (20.8 vs 20.5 N m).  The one visible difference is the pitch transient
during the command ramp (max 0.16 rad vs 0.006 rad), where the QP's joint-limit
and friction rows damp the initial lunge that this controller does not bound.

Usage: ``config.wbc_type = "model_based"`` (see mpx/config/config_dfcip.py), or
``validate_dfcip_controller.py --wbc model_based``.
"""

import jax
import jax.numpy as jnp
from mujoco import mjx

from mpx.utils.mpc_utils import (
    _REF_JOINTS,
    compute_contact_frame,
    compute_virtual_frame,
    err_rotation,
    get_rCP,
    skew,
    unpack_reference,
)


def contact_force_offsets(nj):
    """Offsets of the (optional) MPC contact forces appended to ``desired``.

    The reference vector built by ``BatchedMPCControllerWrapper`` keeps the
    layout of ``mpc_utils.pack_reference`` (CoM / wheels / base / joints) and
    appends ``fcl`` (3) and ``fcr`` (3) at the end, so every existing offset is
    unchanged and the QP controller simply ignores the tail.
    """
    base = _REF_JOINTS + 3 * nj
    return base, base + 3


def desired_size(nj):
    """Length of the ``desired`` vector including the two contact forces."""
    return _REF_JOINTS + 3 * nj + 6


def unpack_contact_forces(desired, nj):
    """MPC contact forces from ``desired``; zeros if the tail is not there."""
    fl_off, fr_off = contact_force_offsets(nj)
    if desired.shape[0] < fr_off + 3:
        z = jnp.zeros(3, dtype=desired.dtype)
        return z, z
    return desired[fl_off:fl_off + 3], desired[fr_off:fr_off + 3]


def _whole_body_interface_wheeled_legged_model_based_impl(
    mjx_model,
    # ── static (via partial) ──────────────────────────────────────────────
    mass, grav, d,
    contact_id, body_id, base_body_id,
    wheel_radius, sample_time, n_contacts,
    # ── gains ─────────────────────────────────────────────────────────────
    Kp_motion,     Kd_motion,
    Kp_wheel,      Kd_wheel,
    Kp_regulation, Kd_regulation,
    w_posture,
    # ── task weights ──────────────────────────────────────────────────────
    w_qddot, w_com, w_lwheel, w_rwheel, w_base,
    mu_,
    # ── runtime ───────────────────────────────────────────────────────────
    qpos, qvel, desired,
    posture_mask=None,
    # ── model-based specific options ──────────────────────────────────────
    enforce_rolling=True,
    force_source="dynamics",
    enforce_base_dynamics=True,
    damping=1e-6,
):
    del mass, grav, d, n_contacts   # kept for signature parity with the QP WBC

    nv = mjx_model.nv
    nj = nv - 6

    des = unpack_reference(desired, nj)
    fl_mpc, fr_mpc = unpack_contact_forces(desired, nj)

    # ══════════════════════════════════════════════════════════════════════
    #  1. FORWARD KINEMATICS / JACOBIANS
    #     Mirrors section 1 of the QP controller: the Jacobian time
    #     derivatives come from a jvp of the Jacobian along dqpos/dt (MJX has
    #     no getFrameJacobianTimeVariation).
    # ══════════════════════════════════════════════════════════════════════
    mjx_data = mjx.make_data(mjx_model)
    mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel)
    mjx_data = mjx.fwd_position(mjx_model, mjx_data)
    mjx_data = mjx.fwd_velocity(mjx_model, mjx_data)

    M = mjx_data.qM
    c = mjx_data.qfrc_bias          # Coriolis + gravity

    # dqpos/dt (quaternion kinematics for the free joint)
    q_wxyz = qpos[3:7]
    omega = qvel[3:6]
    dqw = 0.5 * (-q_wxyz[1]*omega[0] - q_wxyz[2]*omega[1] - q_wxyz[3]*omega[2])
    dqx = 0.5 * ( q_wxyz[0]*omega[0] + q_wxyz[2]*omega[2] - q_wxyz[3]*omega[1])
    dqy = 0.5 * ( q_wxyz[0]*omega[1] - q_wxyz[1]*omega[2] + q_wxyz[3]*omega[0])
    dqz = 0.5 * ( q_wxyz[0]*omega[2] + q_wxyz[1]*omega[1] - q_wxyz[2]*omega[0])
    dqpos = jnp.concatenate([qvel[:3], jnp.array([dqw, dqx, dqy, dqz]), qvel[6:]])

    def _jac_geom(qpos_, geom_id, bid):
        d_ = mjx.make_data(mjx_model)
        d_ = d_.replace(qpos=qpos_, qvel=qvel)
        d_ = mjx.fwd_position(mjx_model, d_)
        Jp_, Jr_ = mjx.jac(mjx_model, d_, d_.geom_xpos[geom_id], bid)
        return Jp_.T, Jr_.T          # (3, nv), (3, nv)

    def _jac_body(qpos_, bid):
        d_ = mjx.make_data(mjx_model)
        d_ = d_.replace(qpos=qpos_, qvel=qvel)
        d_ = mjx.fwd_position(mjx_model, d_)
        Jp_, Jr_ = mjx.jac(mjx_model, d_, d_.xpos[bid], bid)
        return Jp_.T, Jr_.T

    def _jac_com(qpos_):
        d_ = mjx.make_data(mjx_model)
        d_ = d_.replace(qpos=qpos_, qvel=qvel)
        d_ = mjx.fwd_position(mjx_model, d_)
        total_mass = jnp.sum(mjx_model.body_mass)
        nb = mjx_model.body_mass.shape[0]

        def _acc(i, J):
            Jp_i, _ = mjx.jac(mjx_model, d_, d_.xipos[i], i)
            return J + mjx_model.body_mass[i] * Jp_i.T

        return jax.lax.fori_loop(1, nb, _acc, jnp.zeros((3, nv))) / total_mass

    (J_left_wheel_lin,  J_left_wheel_rot), \
    (J_left_wheel_dot_lin,  J_left_wheel_dot_rot) = jax.jvp(
        lambda qp: _jac_geom(qp, contact_id[0], body_id[0]), (qpos,), (dqpos,),
    )
    (J_right_wheel_lin, J_right_wheel_rot), \
    (J_right_wheel_dot_lin, J_right_wheel_dot_rot) = jax.jvp(
        lambda qp: _jac_geom(qp, contact_id[1], body_id[1]), (qpos,), (dqpos,),
    )
    (_, J_base_link_rot), (_, J_base_link_dot_rot) = jax.jvp(
        lambda qp: _jac_body(qp, base_body_id), (qpos,), (dqpos,),
    )
    J_com, J_com_dot = jax.jvp(_jac_com, (qpos,), (dqpos,))

    # ── current wheel frames / positions / velocities ─────────────────────
    r_wheel_center = mjx_data.geom_xpos[contact_id[1]]
    r_wheel_R      = mjx_data.xmat[body_id[1]].reshape(3, 3)
    l_wheel_center = mjx_data.geom_xpos[contact_id[0]]
    l_wheel_R      = mjx_data.xmat[body_id[0]].reshape(3, 3)

    right_rCP = get_rCP(r_wheel_R, wheel_radius)
    w_r = J_right_wheel_rot @ qvel
    left_rCP = get_rCP(l_wheel_R, wheel_radius)
    w_l = J_left_wheel_rot @ qvel

    a_com_drift              = J_com_dot             @ qvel
    a_lwheel_drift           = J_left_wheel_dot_lin  @ qvel
    a_rwheel_drift           = J_right_wheel_dot_lin @ qvel
    a_base_orientation_drift = J_base_link_dot_rot   @ qvel

    current_com_pos = mjx_data.subtree_com[0]
    current_com_vel = J_com @ qvel

    current_base_quat = mjx_data.xquat[base_body_id]
    current_base_quat_xyzw = jnp.array([
        current_base_quat[1], current_base_quat[2],
        current_base_quat[3], current_base_quat[0],
    ])
    current_base_quat_xyzw = current_base_quat_xyzw / (
        jnp.linalg.norm(current_base_quat_xyzw) + 1e-9
    )
    current_base_link_pos = jax.scipy.spatial.transform.Rotation.from_quat(
        current_base_quat_xyzw
    ).as_matrix()
    current_base_link_vel = J_base_link_rot @ qvel

    r_virtual_frame_R = compute_virtual_frame(r_wheel_R)
    l_virtual_frame_R = compute_virtual_frame(l_wheel_R)
    r_contact_frame   = compute_contact_frame(r_wheel_R)
    l_contact_frame   = compute_contact_frame(l_wheel_R)

    current_lwheel_pos = l_wheel_center
    current_lwheel_vel = J_left_wheel_lin @ qvel
    current_rwheel_pos = r_wheel_center
    current_rwheel_vel = J_right_wheel_lin @ qvel

    # ══════════════════════════════════════════════════════════════════════
    #  2. DESIRED TASK ACCELERATIONS (same PD law as the QP controller)
    # ══════════════════════════════════════════════════════════════════════
    err_com     = des['com_pos'] - current_com_pos
    err_com_vel = des['com_vel'] - current_com_vel

    err_lwheel     = des['lwheel_pos'] - current_lwheel_pos
    err_lwheel_vel = des['lwheel_vel'] - current_lwheel_vel
    err_rwheel     = des['rwheel_pos'] - current_rwheel_pos
    err_rwheel_vel = des['rwheel_vel'] - current_rwheel_vel

    err_base_orientation     = err_rotation(des['base_rot'], current_base_link_pos)
    err_base_orientation_vel = des['base_omega'] - current_base_link_vel

    q_jnt    = qpos[7:]
    qdot_jnt = qvel[6:]

    err_posture     = jnp.concatenate([jnp.zeros(6), des['qjnt']    - q_jnt])
    err_posture_vel = jnp.concatenate([jnp.zeros(6), des['qjntdot'] - qdot_jnt])

    if posture_mask is None:
        matrix_with_no_wheel = jnp.eye(nj).at[3, 3].set(0.0).at[7, 7].set(0.0)
    else:
        matrix_with_no_wheel = jnp.diag(jnp.asarray(posture_mask, dtype=qpos.dtype))
    S = jax.scipy.linalg.block_diag(jnp.zeros((6, 6)), matrix_with_no_wheel)

    desired_qddot = jnp.concatenate([jnp.zeros(6), des['qjntddot']])

    a_jnt_total    = desired_qddot + Kp_regulation * err_posture + Kd_regulation * err_posture_vel
    a_com_total    = des['com_acc']    + Kp_motion * err_com    + Kd_motion * err_com_vel
    a_lwheel_total = des['lwheel_acc'] + Kp_wheel  * err_lwheel + Kd_wheel  * err_lwheel_vel
    a_rwheel_total = des['rwheel_acc'] + Kp_wheel  * err_rwheel + Kd_wheel  * err_rwheel_vel
    a_base_orientation_total = (
        des['base_alpha']
        + Kp_motion * err_base_orientation
        + Kd_motion * err_base_orientation_vel
    )

    # ══════════════════════════════════════════════════════════════════════
    #  3. STACKED TASK LEAST SQUARES  ->  H qddot = g
    #     (H, g are the QP's H_acc, -f_acc: same optimum with no constraint
    #      active, so the two controllers agree task by task)
    # ══════════════════════════════════════════════════════════════════════
    H = (w_qddot + damping) * jnp.eye(nv)
    H = H + w_com    * (J_com.T             @ J_com)
    H = H + w_lwheel * (J_left_wheel_lin.T  @ J_left_wheel_lin)
    H = H + w_rwheel * (J_right_wheel_lin.T @ J_right_wheel_lin)
    H = H + w_base   * (J_base_link_rot.T   @ J_base_link_rot)
    H = H + w_posture * S

    g = w_com    * J_com.T             @ (a_com_total              - a_com_drift)
    g = g + w_lwheel * J_left_wheel_lin.T  @ (a_lwheel_total           - a_lwheel_drift)
    g = g + w_rwheel * J_right_wheel_lin.T @ (a_rwheel_total           - a_rwheel_drift)
    g = g + w_base   * J_base_link_rot.T   @ (a_base_orientation_total - a_base_orientation_drift)
    g = g + w_posture * (S @ a_jnt_total)

    # ══════════════════════════════════════════════════════════════════════
    #  4. WHEEL ROLLING CONSTRAINT (hard equality, same rows as the QP)
    #     a_contact_point = J_roll qddot - b_roll = 0
    # ══════════════════════════════════════════════════════════════════════
    I3 = jnp.eye(3)
    # spin components of the wheel angular velocity, removed from the rolling
    # centripetal term (nl / nr = wheel spin axis in the virtual frame)
    nl, nr = l_virtual_frame_R[:, 1], r_virtual_frame_R[:, 1]
    wl_virtual = (I3 - jnp.outer(nl, nl)) @ w_l
    wr_virtual = (I3 - jnp.outer(nr, nr)) @ w_r

    A_roll_L = J_left_wheel_lin - skew(left_rCP) @ J_left_wheel_rot
    b_roll_L = (
        -(J_left_wheel_dot_lin - skew(left_rCP) @ J_left_wheel_dot_rot) @ qvel
        - jnp.cross(w_l, jnp.cross(wl_virtual, left_rCP))
    )
    A_roll_R = J_right_wheel_lin - skew(right_rCP) @ J_right_wheel_rot
    b_roll_R = (
        -(J_right_wheel_dot_lin - skew(right_rCP) @ J_right_wheel_dot_rot) @ qvel
        - jnp.cross(w_r, jnp.cross(wr_virtual, right_rCP))
    )
    A_roll = jnp.vstack([A_roll_L, A_roll_R])       # (6, nv)
    b_roll = jnp.concatenate([b_roll_L, b_roll_R])  # (6,)

    # ══════════════════════════════════════════════════════════════════════
    #  5. CONTACT MAPS AND DYNAMICS BLOCKS
    # ══════════════════════════════════════════════════════════════════════
    left_rCP_local  = jnp.array([0.0, 0.0, -wheel_radius])
    right_rCP_local = jnp.array([0.0, 0.0, -wheel_radius])
    pcis_l = l_virtual_frame_R @ left_rCP_local
    pcis_r = r_virtual_frame_R @ right_rCP_local
    T_l = jnp.vstack([I3, skew(pcis_l)])    # (6, 3): contact force -> wheel wrench
    T_r = jnp.vstack([I3, skew(pcis_r)])

    J_left_wheel_  = jnp.vstack([J_left_wheel_lin,  J_left_wheel_rot])   # (6, nv)
    J_right_wheel_ = jnp.vstack([J_right_wheel_lin, J_right_wheel_rot])

    Mu, cu = M[:6, :], c[:6]        # unactuated (floating base) rows
    Ma, ca = M[6:, :], c[6:]        # actuated (joint) rows

    Jlu, Jla = J_left_wheel_[:, :6],  J_left_wheel_[:, 6:]
    Jru, Jra = J_right_wheel_[:, :6], J_right_wheel_[:, 6:]

    B = jnp.hstack([Jlu.T @ T_l, Jru.T @ T_r])      # (6, 6): forces -> base wrench

    # ══════════════════════════════════════════════════════════════════════
    #  6. SOLVE FOR qddot UNDER THE HARD EQUALITIES
    #
    #  Rolling of both wheels, and -- when the contact forces are dictated by
    #  the MPC -- the six floating-base rows with those forces substituted in:
    #      Mu qddot = B [fl_mpc; fr_mpc] - cu
    #  Constraining qddot this way is what makes the (qddot, f_mpc) pair
    #  dynamically consistent, so the torque produces the motion it was
    #  computed for.  Without it the MPC forces cannot be used (see the module
    #  docstring: the robot falls).
    # ══════════════════════════════════════════════════════════════════════
    if force_source not in ("dynamics", "mpc"):
        raise ValueError(f"unknown force_source {force_source!r} (use 'mpc' or 'dynamics')")

    rows, rhs_rows = [], []
    if enforce_rolling:
        rows.append(A_roll)
        rhs_rows.append(b_roll)
    if force_source == "mpc" and enforce_base_dynamics:
        rows.append(Mu)
        rhs_rows.append(B @ jnp.concatenate([fl_mpc, fr_mpc]) - cu)

    if rows:
        # KKT system of  min ||A qddot - b||_W  s.t.  A_c qddot = b_c.
        # The -eps block regularises the multipliers so the matrix stays
        # invertible when the constraint rows are (nearly) linearly dependent.
        A_c = jnp.vstack(rows)
        b_c = jnp.concatenate(rhs_rows)
        n_c = A_c.shape[0]
        KKT = jnp.block([
            [H,   A_c.T],
            [A_c, -1e-10 * jnp.eye(n_c)],
        ])
        sol = jnp.linalg.solve(KKT, jnp.concatenate([g, b_c]))
        qddot = sol[:nv]
    else:
        # Bare damped weighted pseudo-inverse of the stacked task Jacobian.
        qddot = jnp.linalg.solve(H, g)

    # ══════════════════════════════════════════════════════════════════════
    #  7. CONTACT FORCES
    # ══════════════════════════════════════════════════════════════════════
    if force_source == "dynamics":
        # Recover the forces the floating base needs for this qddot:
        #   Mu qddot + cu = B [fl; fr]   (the QP's dynamics equality).
        rhs_dyn = Mu @ qddot + cu
        f = jnp.linalg.solve(B.T @ B + 1e-8 * jnp.eye(B.shape[1]), B.T @ rhs_dyn)
        fl, fr = f[:3], f[3:]
    else:
        # The MPC ground reaction forces, fed straight through; qddot above was
        # constrained to be consistent with them.
        fl, fr = fl_mpc, fr_mpc

    # ══════════════════════════════════════════════════════════════════════
    #  8. INVERSE DYNAMICS -> TAU  (identical to the QP controller)
    # ══════════════════════════════════════════════════════════════════════
    tau = Ma @ qddot + ca - Jla.T @ T_l @ fl - Jra.T @ T_r @ fr

    # ══════════════════════════════════════════════════════════════════════
    #  9. DIAGNOSTICS — same keys as the QP controller so that the validation
    #     harness (validate_dfcip_controller.py) works unchanged.  The
    #     inequality quantities are *measured*, not enforced.
    # ══════════════════════════════════════════════════════════════════════
    fl_local = l_contact_frame.T @ fl
    fr_local = r_contact_frame.T @ fr
    fz_min = 5.0        # same reference floor as the QP, for a comparable margin

    # Joint position/velocity limits predicted one WBC period ahead (the QP's
    # inequality rows), reported as a slack: negative = limit violated.
    jnt_ids   = jnp.arange(1, mjx_model.njnt)
    q_jnt_min = mjx_model.jnt_range[jnt_ids, 0]
    q_jnt_max = mjx_model.jnt_range[jnt_ids, 1]
    vel_limit = 100.0 * jnp.ones(nj)
    qddot_jnt = qddot[6:]
    dv = sample_time * qddot_jnt
    dq = 0.5 * sample_time ** 2 * qddot_jnt
    slack_vel = jnp.concatenate([
        (vel_limit - qdot_jnt) - dv,
        dv - (-vel_limit - qdot_jnt),
    ])
    slack_pos = jnp.concatenate([
        (q_jnt_max - q_jnt - sample_time * qdot_jnt) - dq,
        dq - (q_jnt_min - q_jnt - sample_time * qdot_jnt),
    ])

    diag = dict(
        # a linear solve always "converges": kept for interface parity
        converged=jnp.array(True),
        iters=jnp.array(0),
        # task residuals: J qddot + Jdot qdot - a_total (zero = task realised)
        res_com=J_com @ qddot + a_com_drift - a_com_total,
        res_lwheel=J_left_wheel_lin @ qddot + a_lwheel_drift - a_lwheel_total,
        res_rwheel=J_right_wheel_lin @ qddot + a_rwheel_drift - a_rwheel_total,
        res_base=J_base_link_rot @ qddot + a_base_orientation_drift - a_base_orientation_total,
        # accelerations requested by the tasks (feedforward + PD)
        a_com_total=a_com_total,
        a_lwheel_total=a_lwheel_total,
        a_rwheel_total=a_rwheel_total,
        a_base_total=a_base_orientation_total,
        # PD error terms feeding those requests
        err_com=err_com,
        err_com_vel=err_com_vel,
        err_lwheel=err_lwheel,
        err_rwheel=err_rwheel,
        err_base=err_base_orientation,
        # constraint satisfaction (rolling is enforced, the base dynamics only
        # when force_source="dynamics", the inequalities never)
        eq_res_norm=jnp.linalg.norm(A_roll @ qddot - b_roll),
        roll_res_norm=jnp.linalg.norm(A_roll @ qddot - b_roll),
        dyn_res_norm=jnp.linalg.norm(Mu @ qddot + cu - B @ jnp.concatenate([fl, fr])),
        ineq_slack_min_joint=jnp.min(jnp.concatenate([slack_vel, slack_pos])),
        fric_margin_l=jnp.array([mu_ * fl_local[2] - jnp.abs(fl_local[0]),
                                 mu_ * fl_local[2] - jnp.abs(fl_local[1]),
                                 fl_local[2] - fz_min]),
        fric_margin_r=jnp.array([mu_ * fr_local[2] - jnp.abs(fr_local[0]),
                                 mu_ * fr_local[2] - jnp.abs(fr_local[1]),
                                 fr_local[2] - fz_min]),
        fl_local=fl_local,
        fr_local=fr_local,
    )
    return tau, qddot, fl, fr, diag


def whole_body_interface_wheeled_legged_model_based(*args, **kwargs):
    """Model-based whole-body controller. Returns ``(tau, qddot, fl, fr)``.

    Drop-in replacement for
    ``mpc_utils.whole_body_interface_wheeled_legged_qp``; the diagnostics are
    dropped here so the call signature matches the QP entry point.
    """
    tau, qddot, fl, fr, _ = _whole_body_interface_wheeled_legged_model_based_impl(
        *args, **kwargs
    )
    return tau, qddot, fl, fr


def whole_body_interface_wheeled_legged_model_based_diag(*args, **kwargs):
    """As above, also returning the diagnostics dict (see the implementation)."""
    return _whole_body_interface_wheeled_legged_model_based_impl(*args, **kwargs)
