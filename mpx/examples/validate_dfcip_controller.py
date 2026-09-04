"""Headless validation and diagnostic harness for the DFCIP MPC + WBC controller.

Reproduces exactly the control law of ``mjx_tita.py`` (MPC and WBC solved at
``mpc_frequency`` / ``whole_body_frequency``, MuJoCo integrated at
``simulation_frequency``, WBC torque and joint accelerations held between
controller updates, optional outer joint PD) without any viewer, video or
matplotlib dependency, and logs everything needed to locate where a commanded
(vx, omega) pair is lost along the chain

    command -> reference -> MPC (a, alpha, Fl, Fr) -> WBC references -> QP -> torque -> MuJoCo.

Usage examples::

    python validate_dfcip_controller.py                       # all cases, 1 s ramp + 6 s hold
    python validate_dfcip_controller.py --cases 3 --hold 8    # only (vx, omega) = (0.6, 0.4)
    python validate_dfcip_controller.py --set w_v=20          # ablation: override a config value
    python validate_dfcip_controller.py --no-outer-pd         # ablation: WBC torque only
    python validate_dfcip_controller.py --set mpc_iterations=3

Every ``--set key=value`` overrides one attribute of ``mpx.config.config_dfcip``
(the MPC weight matrix W is rebuilt from the individual weights).
"""
import argparse
import json
import os
import sys
import time
import types

import numpy as np

# JAX_PLATFORMS must be set before jax is imported.
if "--device" in sys.argv:
    _dev = sys.argv[sys.argv.index("--device") + 1]
    os.environ["JAX_PLATFORMS"] = "cuda" if _dev == "gpu" else _dev
else:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import mujoco

dir_path = os.path.dirname(os.path.realpath(__file__))
# Insert at the front so that THIS working tree wins over any editable/site-packages install of mpx.
sys.path.insert(0, os.path.abspath(os.path.join(dir_path, "..", "..")))

import mpx.config.config_dfcip as base_config
import mpx.utils.mpc_wrapper_dfcip as mpc_wrapper_dfcip
import mpx.utils.mpc_utils as mpc_utils
import mpx.utils.sim as sim_utils
import mpx.utils.timing as timing_utils

# (label, vx [m/s], omega [rad/s]) -- the mandatory validation set.
CASES = [
    ("vx=1.0,omega=0.0", 1.0, 0.0),
    ("vx=0.0,omega=0.8", 0.0, 0.8),
    ("vx=0.3,omega=0.2", 0.3, 0.2),
    ("vx=0.6,omega=0.4", 0.6, 0.4),
    ("vx=1.0,omega=0.8", 1.0, 0.8),
    ("vx=1.0,omega=-0.8", 1.0, -0.8),
]

W_KEYS = [
    "w_pcomxy", "w_pcomz", "w_vcomxy", "w_vcomz", "w_c", "w_vcz",
    "w_theta", "w_v", "w_omega", "w_a", "w_ac_z", "w_alpha",
    "w_fcxy", "w_fcz", "w_eq",
]

# Outer joint PD gains used by mjx_tita.py on top of the WBC torque.
OUTER_KP = 35.0
OUTER_KD = 10.0
COMMAND_SMOOTHING = 0.02


def make_config(overrides):
    """Copy the config module into a namespace and apply ``overrides``."""
    cfg = types.SimpleNamespace()
    for k in dir(base_config):
        if k.startswith("__"):
            continue
        setattr(cfg, k, getattr(base_config, k))
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            print(f"[config] NOTE: '{k}' is not a base config attribute (added as an override)")
        setattr(cfg, k, v)
    cfg.W = jnp.diag(jnp.array([float(getattr(cfg, k)) for k in W_KEYS]))
    return cfg


def derive_timing(cfg):
    """Sim / MPC / WBC timing from the config (shared checks in mpx.utils.timing)."""
    t = timing_utils.derive_timing(cfg)
    t["mpc_period_steps"] = t["mpc_period_sim_steps"]
    t["wbc_period_steps"] = t["wbc_period_sim_steps"]
    t["horizon"] = t["horizon_s"]
    return t


def parse_value(text):
    if text[:1] in "([{":
        import ast as _ast
        return _ast.literal_eval(text)
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    if text.lower() in ("true", "false"):
        return text.lower() == "true"
    return text


# ── state estimation (identical to mjx_tita.py) ─────────────────────────────
def gather_raw_state(model, data, base_body_id, contact_ids):
    mujoco.mj_subtreeVel(model, data)
    pcom = data.subtree_com[base_body_id].copy()
    vcom = data.subtree_linvel[base_body_id].copy()
    centers = data.geom_xpos[contact_ids].copy()
    Rs = data.geom_xmat[contact_ids].reshape(2, 3, 3).copy()
    radii = model.geom_size[contact_ids, 0].copy()
    feet_vel = np.zeros((2, 3))
    vel = np.zeros(6)
    for i, g in enumerate(contact_ids):
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_GEOM, int(g), vel, 0)
        feet_vel[i] = vel[3:6]
    return pcom, vcom, centers, Rs, radii, feet_vel


def make_process_tita_state(d_track):
    @jax.jit
    def process_tita_state(pcom, vcom, centers, Rs, radii, feet_vel, theta_prev):
        l_rcp = mpc_utils.get_rCP(Rs[0], radii[0])
        r_rcp = mpc_utils.get_rCP(Rs[1], radii[1])
        pl_world = centers[0] + l_rcp
        pr_world = centers[1] + r_rcp
        dpl_world, dpr_world = feet_vel[0], feet_vel[1]
        tita_state = jnp.concatenate([pcom, vcom, pl_world, pr_world, dpl_world, dpr_world])
        c_world = (pl_world + pr_world) / 2.0
        vc_world = (dpl_world + dpr_world) / 2.0
        diff = pl_world - pr_world
        theta_wrapped = jnp.arctan2(-diff[0], diff[1])
        a = (theta_wrapped - theta_prev + jnp.pi) % (2 * jnp.pi)
        a = jnp.where(a < 0, a + 2 * jnp.pi, a) - jnp.pi
        theta = theta_prev + a
        ct, st = jnp.cos(theta), jnp.sin(theta)
        R = jnp.array([[ct, -st, 0.], [st, ct, 0.], [0., 0., 1.]])
        dpl_b = R.T @ dpl_world
        dpr_b = R.T @ dpr_world
        w = (dpr_b[0] - dpl_b[0]) / d_track
        v = (dpr_b[0] + dpl_b[0]) / 2.0
        x0 = jnp.concatenate([pcom, vcom, c_world, jnp.array([vc_world[2]]),
                              jnp.array([theta]), jnp.array([v]), jnp.array([w])])
        return tita_state, x0, theta
    return process_tita_state


def reset_to_initial_state(model, data):
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def quat_to_rpy(q):
    w_, x_, y_, z_ = q
    roll = np.arctan2(2 * (w_ * x_ + y_ * z_), 1 - 2 * (x_ * x_ + y_ * y_))
    pitch = np.arcsin(np.clip(2 * (w_ * y_ - z_ * x_), -1.0, 1.0))
    yaw = np.arctan2(2 * (w_ * z_ + x_ * y_), 1 - 2 * (y_ * y_ + z_ * z_))
    return roll, pitch, yaw


class Context:
    def __init__(self, cfg, scene="flat"):
        self.cfg = cfg
        self.timing = derive_timing(cfg)
        self.model = mujoco.MjModel.from_xml_path(dir_path + f"/../data/tita/scene_{scene}.xml")
        # MuJoCo integrates at the simulation frequency, never at the WBC rate;
        # the XML timestep must agree with it.
        timing_utils.check_model_timestep(self.model.opt.timestep, self.timing)
        self.model.opt.timestep = self.timing["dt_sim"]
        self.base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, cfg.base_body_name)
        self.contact_ids = sim_utils.geom_ids(self.model, cfg.contact_frame)
        self.mpc = mpc_wrapper_dfcip.BatchedMPCControllerWrapper(cfg, n_env=1)
        self.process_tita_state = make_process_tita_state(float(cfg.d))
        mpc = self.mpc

        @jax.jit
        def solve_mpc(mpc_state, x0, command):
            return mpc.run_diag(mpc_state, x0, command)

        @jax.jit
        def solve_wbc(mpc_state, x0, qpos, qvel, pl_world, pr_world, dpl_world, dpr_world):
            return mpc.whole_body_run_diag(mpc_state, x0, qpos, qvel, pl_world, pr_world, dpl_world, dpr_world)

        self.solve_mpc = solve_mpc
        self.solve_wbc = solve_wbc


def _np(x):
    return np.asarray(jax.device_get(x))


def run_case(ctx, vx_cmd, wz_cmd, ramp_s, hold_s, label, outer_pd=True,
             steady_window_s=2.0, verbose_every=0, outer_pd_mode="legacy", fresh_state=True):
    cfg, T = ctx.cfg, ctx.timing
    model = ctx.model
    data = mujoco.MjData(model)
    reset_to_initial_state(model, data)

    n_steps = int(round((ramp_s + hold_s) * T["sim_f"]))
    dt_sim = T["dt_sim"]
    mpc_period = T["mpc_period_steps"]
    wbc_period = T["wbc_period_steps"]

    theta_prev = 0.0
    mpc_state = ctx.mpc.init_state()
    command = np.array([0.0, 0.0, 0.0, cfg.com_z_to_track], dtype=np.float64)
    tau = jnp.zeros((1, model.nu))
    qddot = jnp.zeros((1, model.nv))
    reference = None
    mdiag = None
    wdiag = None
    desired = None

    L = {k: [] for k in [
        "t", "target_vx", "target_wz", "cmd_vx", "cmd_wz",
        "ref_v0", "ref_w0", "ref_v1", "ref_w1", "ref_vcom1", "ref_theta1",
        "x0", "base_wz_body", "vcom_speed",
        "a", "ac_z", "alpha", "Fl", "Fr",
        "mpc_accepted", "mpc_cost_before", "mpc_cost_after", "mpc_defect", "mpc_defect0",
        "mpc_hmoment_max", "mpc_hmoment_0", "mpc_lean0", "mpc_lean_max", "mpc_ftot_body0", "mpc_stabN",
        "des_com_acc", "des_lw_acc", "des_rw_acc", "des_lw_vel", "des_rw_vel", "des_base_omega", "des_base_alpha",
        "wbc_a_com", "wbc_a_lw", "wbc_a_rw", "wbc_a_base",
        "wbc_res_com", "wbc_res_lw", "wbc_res_rw", "wbc_res_base",
        "wbc_err_com", "wbc_err_com_vel", "wbc_err_lw", "wbc_err_rw", "wbc_err_base",
        "wbc_converged", "wbc_iters", "wbc_eq_res", "wbc_joint_slack",
        "wbc_fric_l", "wbc_fric_r", "wbc_fl_local", "wbc_fr_local",
        "qddot", "tau_wbc", "tau_total", "q_joint", "qd_joint",
        "height", "roll", "pitch", "yaw", "com_acc_meas", "mpc_x1", "mpc_xN",
    ]}

    fell = False
    nan_hit = False
    nan_where = None
    vcom_prev = None
    total_ctrl = np.zeros(model.nu)

    for k in range(n_steps):
        t = k * dt_sim
        ramp = min(1.0, t / ramp_s) if ramp_s > 0 else 1.0
        target = np.array([vx_cmd * ramp, 0.0, wz_cmd * ramp, cfg.com_z_to_track])

        if fresh_state:
            # Forward pass (mj_step1) so that every derived quantity read below
            # (geom positions/velocities, subtree CoM) corresponds to the SAME
            # qpos/qvel that is handed to the WBC. After mj_step the derived
            # quantities are still those of the pre-integration state.
            mujoco.mj_step1(model, data)
        raw = gather_raw_state(model, data, ctx.base_body_id, ctx.contact_ids)
        tita_state, x0, theta_j = ctx.process_tita_state(*raw, theta_prev)
        theta_prev = float(theta_j)

        controller_tick = (k % mpc_period == 0)
        if controller_tick:
            command += COMMAND_SMOOTHING * (target - command)
            command[3] = target[3]
            mpc_state, reference, mdiag = ctx.solve_mpc(mpc_state, x0[None, :], jnp.asarray(command)[None, :])

        if k % wbc_period == 0:
            mpc_state, tau, qddot, fl, fr, desired, wdiag = ctx.solve_wbc(
                mpc_state, x0,
                jnp.asarray(data.qpos)[None, :], jnp.asarray(data.qvel)[None, :],
                tita_state[6:9][None, :], tita_state[9:12][None, :],
                tita_state[12:15][None, :], tita_state[15:18][None, :],
            )
            k_wbc = k
            q_wbc = data.qpos[7:].copy()
            qd_wbc = data.qvel[6:].copy()

        qpos_joint = data.qpos[7:].copy()
        qvel_joint = data.qvel[6:].copy()
        tau_np = _np(tau[0])
        if outer_pd:
            qddot_joint = _np(qddot[0, 6:])
            if outer_pd_mode == "plan":
                # Track the WBC plan integrated from the WBC-instant joint state over
                # the elapsed time inside the hold interval.
                tau_hold = (k - k_wbc + 1) * dt_sim
                dq_desired = qd_wbc + qddot_joint * tau_hold
                q_desired = q_wbc + qd_wbc * tau_hold + 0.5 * qddot_joint * tau_hold ** 2
            else:
                # mjx_tita.py outer joint PD: one-simulation-step (dt_sim) prediction
                # from the current joint state with the held qddot.
                dq_desired = qvel_joint + qddot_joint * dt_sim
                q_desired = qpos_joint + qvel_joint * dt_sim + 0.5 * qddot_joint * dt_sim ** 2
            p_ctrl = OUTER_KP * (q_desired - qpos_joint)
            d_ctrl = OUTER_KD * (dq_desired - qvel_joint)
            p_ctrl[[3, 7]] = 0.0
            total_ctrl = p_ctrl + d_ctrl + tau_np
        else:
            total_ctrl = tau_np.copy()

        if not np.all(np.isfinite(total_ctrl)):
            nan_hit = True
            nan_where = ("tau" if not np.all(np.isfinite(tau_np)) else "outer_pd", t)
            total_ctrl = np.nan_to_num(total_ctrl)

        data.ctrl = total_ctrl
        if fresh_state:
            mujoco.mj_step2(model, data)
        else:
            mujoco.mj_step(model, data)

        roll, pitch, yaw = quat_to_rpy(data.qpos[3:7])
        upvector_z = 1 - 2 * (data.qpos[4] ** 2 + data.qpos[5] ** 2)

        if controller_tick:
            x0n = _np(x0)
            xr = _np(reference[0])
            u_first = np.concatenate([np.atleast_1d(_np(mpc_state.sol.a[0])), np.atleast_1d(_np(mpc_state.sol.ac_z[0])),
                                      np.atleast_1d(_np(mpc_state.sol.alpha[0])), np.atleast_1d(_np(mpc_state.sol.grf[0]))])
            des = mpc_utils.unpack_reference(_np(desired[0]), model.nv - 6)
            vcom_now = x0n[3:6]
            com_acc = (vcom_now - vcom_prev) / (mpc_period * dt_sim) if vcom_prev is not None else np.zeros(3)
            vcom_prev = vcom_now.copy()
            L["t"].append(t)
            L["target_vx"].append(target[0]); L["target_wz"].append(target[2])
            L["cmd_vx"].append(command[0]); L["cmd_wz"].append(command[2])
            L["ref_v0"].append(xr[0, 11]); L["ref_w0"].append(xr[0, 12])
            L["ref_v1"].append(xr[1, 11]); L["ref_w1"].append(xr[1, 12])
            L["ref_vcom1"].append(xr[1, 3:5]); L["ref_theta1"].append(xr[1, 10])
            L["x0"].append(x0n)
            L["base_wz_body"].append(float(data.qvel[5]))
            L["vcom_speed"].append(float(np.linalg.norm(x0n[3:5])))
            L["a"].append(u_first[0]); L["ac_z"].append(u_first[1]); L["alpha"].append(u_first[2])
            L["Fl"].append(u_first[3:6]); L["Fr"].append(u_first[6:9])
            L["mpc_accepted"].append(bool(_np(mdiag["accepted"][0])))
            L["mpc_cost_before"].append(float(_np(mdiag["cost_before"][0])))
            L["mpc_cost_after"].append(float(_np(mdiag["cost_after"][0])))
            L["mpc_defect"].append(float(_np(mdiag["defect_norm"][0])))
            L["mpc_defect0"].append(float(_np(mdiag["defect0_norm"][0])))
            L["mpc_hmoment_max"].append(float(_np(mdiag["h_moment_max"][0])))
            L["mpc_hmoment_0"].append(_np(mdiag["h_moment_0"][0]))
            L["mpc_lean0"].append(_np(mdiag["lean_body_0"][0]))
            L["mpc_lean_max"].append(_np(mdiag["lean_body_max"][0]))
            L["mpc_ftot_body0"].append(_np(mdiag["f_tot_body_0"][0]))
            L["mpc_stabN"].append(_np(mdiag["stability_N"][0]))
            L["des_com_acc"].append(des["com_acc"]); L["des_lw_acc"].append(des["lwheel_acc"])
            L["des_rw_acc"].append(des["rwheel_acc"]); L["des_lw_vel"].append(des["lwheel_vel"])
            L["des_rw_vel"].append(des["rwheel_vel"]); L["des_base_omega"].append(des["base_omega"])
            L["des_base_alpha"].append(des["base_alpha"])
            g = lambda key: _np(wdiag[key][0])
            L["wbc_a_com"].append(g("a_com_total")); L["wbc_a_lw"].append(g("a_lwheel_total"))
            L["wbc_a_rw"].append(g("a_rwheel_total")); L["wbc_a_base"].append(g("a_base_total"))
            L["wbc_res_com"].append(g("res_com")); L["wbc_res_lw"].append(g("res_lwheel"))
            L["wbc_res_rw"].append(g("res_rwheel")); L["wbc_res_base"].append(g("res_base"))
            L["wbc_err_com"].append(g("err_com")); L["wbc_err_com_vel"].append(g("err_com_vel"))
            L["wbc_err_lw"].append(g("err_lwheel")); L["wbc_err_rw"].append(g("err_rwheel"))
            L["wbc_err_base"].append(g("err_base"))
            L["wbc_converged"].append(bool(g("converged"))); L["wbc_iters"].append(int(g("iters")))
            L["wbc_eq_res"].append(float(g("eq_res_norm"))); L["wbc_joint_slack"].append(float(g("ineq_slack_min_joint")))
            L["wbc_fric_l"].append(g("fric_margin_l")); L["wbc_fric_r"].append(g("fric_margin_r"))
            L["wbc_fl_local"].append(g("fl_local")); L["wbc_fr_local"].append(g("fr_local"))
            L["qddot"].append(_np(qddot[0])); L["tau_wbc"].append(tau_np.copy()); L["tau_total"].append(total_ctrl.copy())
            L["q_joint"].append(qpos_joint); L["qd_joint"].append(qvel_joint)
            L["height"].append(float(data.qpos[2])); L["roll"].append(roll); L["pitch"].append(pitch); L["yaw"].append(yaw)
            L["com_acc_meas"].append(com_acc)
            L["mpc_x1"].append(_np(mpc_state.X_prediction[0, 1])); L["mpc_xN"].append(_np(mpc_state.X_prediction[0, -1]))

            if verbose_every and (k // mpc_period) % verbose_every == 0:
                print(f"  t={t:5.2f} cmd=({command[0]:+.3f},{command[2]:+.3f}) v={x0n[11]:+.3f} w={x0n[12]:+.3f}"
                      f" |vcom|={np.linalg.norm(x0n[3:5]):.3f} a={u_first[0]:+.3f} al={u_first[2]:+.3f}"
                      f" Fl=({u_first[3]:+.1f},{u_first[4]:+.1f},{u_first[5]:.1f}) Fr=({u_first[6]:+.1f},{u_first[7]:+.1f},{u_first[8]:.1f})"
                      f" acc={'Y' if L['mpc_accepted'][-1] else 'N'} J={L['mpc_cost_after'][-1]:.3g}"
                      f" |def|={L['mpc_defect'][-1]:.2e} hM={L['mpc_hmoment_max'][-1]:.2e}"
                      f" lean0=({L['mpc_lean0'][-1][0]:+.3f},{L['mpc_lean0'][-1][1]:+.3f})"
                      f" qp={'ok' if L['wbc_converged'][-1] else 'FAIL'}/{L['wbc_iters'][-1]}"
                      f" fricL={np.min(L['wbc_fric_l'][-1]):+.1f} fricR={np.min(L['wbc_fric_r'][-1]):+.1f}"
                      f" |res_com|={np.linalg.norm(L['wbc_res_com'][-1]):.2e} q1=({qpos_joint[0]:+.2f},{qpos_joint[4]:+.2f})"
                      f" h={data.qpos[2]:.3f} rp=({roll:+.3f},{pitch:+.3f}) |tau|={np.max(np.abs(total_ctrl)):.1f}")

        if upvector_z < 0.3 or data.qpos[2] < 0.15 or not np.all(np.isfinite(data.qpos)):
            fell = True
            break

    for k2 in list(L.keys()):
        L[k2] = np.asarray(L[k2])

    t_arr = L["t"]
    steady = t_arr >= (ramp_s + hold_s - steady_window_s) if len(t_arr) else np.zeros(0, bool)
    x0s = L["x0"]

    def ss(arr):
        arr = np.asarray(arr)
        return (float(np.mean(arr[steady])), float(np.std(arr[steady]))) if steady.any() else (float("nan"), float("nan"))

    vx_ss = ss(x0s[:, 11]) if len(x0s) else (np.nan, np.nan)
    wz_ss = ss(x0s[:, 12]) if len(x0s) else (np.nan, np.nan)
    wz_body_ss = ss(L["base_wz_body"]) if len(x0s) else (np.nan, np.nan)
    vcom_ss = ss(L["vcom_speed"]) if len(x0s) else (np.nan, np.nan)
    summary = dict(
        label=label, vx_cmd=vx_cmd, wz_cmd=wz_cmd, ramp_s=ramp_s, hold_s=hold_s,
        fell=fell, fell_at_t=float(k * dt_sim) if fell else None,
        nan=nan_hit, nan_where=nan_where,
        steady_vx_mean=vx_ss[0], steady_vx_std=vx_ss[1],
        steady_wz_mean=wz_ss[0], steady_wz_std=wz_ss[1],
        steady_wz_body_mean=wz_body_ss[0], steady_vcom_speed_mean=vcom_ss[0],
        vx_tracking_ratio=(vx_ss[0] / vx_cmd) if vx_cmd else None,
        wz_tracking_ratio=(wz_ss[0] / wz_cmd) if wz_cmd else None,
        height_min=float(np.min(L["height"])) if len(x0s) else None,
        height_max=float(np.max(L["height"])) if len(x0s) else None,
        roll_max_abs=float(np.max(np.abs(L["roll"]))) if len(x0s) else None,
        pitch_max_abs=float(np.max(np.abs(L["pitch"]))) if len(x0s) else None,
        tau_max=float(np.max(np.abs(L["tau_total"]))) if len(x0s) else None,
        mpc_rejected_steps=int(np.sum(~L["mpc_accepted"])) if len(x0s) else None,
        mpc_hmoment_max=float(np.max(L["mpc_hmoment_max"])) if len(x0s) else None,
        wbc_failed_steps=int(np.sum(~L["wbc_converged"])) if len(x0s) else None,
        wbc_iters_max=int(np.max(L["wbc_iters"])) if len(x0s) else None,
        fric_margin_min=float(min(np.min(L["wbc_fric_l"]), np.min(L["wbc_fric_r"]))) if len(x0s) else None,
        joint_slack_min=float(np.min(L["wbc_joint_slack"])) if len(x0s) else None,
        q1_abs_max=float(np.max(np.abs(L["q_joint"][:, [0, 4]]))) if len(x0s) else None,
        res_com_max=float(np.max(np.linalg.norm(L["wbc_res_com"], axis=1))) if len(x0s) else None,
        res_wheel_max=float(max(np.max(np.linalg.norm(L["wbc_res_lw"], axis=1)),
                                np.max(np.linalg.norm(L["wbc_res_rw"], axis=1)))) if len(x0s) else None,
    )
    return summary, L


def print_summary(s):
    def f(v, fmt):
        return ("   nan " if v is None or (isinstance(v, float) and np.isnan(v)) else fmt.format(v))
    print(f"[{s['label']:>18}] fell={str(s['fell']):5} nan={str(s['nan']):5}"
          f" vx={f(s['steady_vx_mean'], '{:+.3f}')}/{s['vx_cmd']:+.1f}"
          f" wz={f(s['steady_wz_mean'], '{:+.3f}')}/{s['wz_cmd']:+.1f}"
          f" (base wz {f(s['steady_wz_body_mean'], '{:+.3f}')}, |vcom| {f(s['steady_vcom_speed_mean'], '{:.3f}')})"
          f" h=[{f(s['height_min'], '{:.3f}')},{f(s['height_max'], '{:.3f}')}]"
          f" rp_max=({f(s['roll_max_abs'], '{:.3f}')},{f(s['pitch_max_abs'], '{:.3f}')})"
          f" |tau|max={f(s['tau_max'], '{:.1f}')}"
          f" mpc_rej={s['mpc_rejected_steps']} hM={f(s['mpc_hmoment_max'], '{:.1e}')}"
          f" qp_fail={s['wbc_failed_steps']} fric_min={f(s['fric_margin_min'], '{:+.1f}')}"
          f" q1max={f(s['q1_abs_max'], '{:.2f}')} wall={s.get('wall_s', 0):.1f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default="all", help="comma-separated case indices (0-based) or 'all'")
    ap.add_argument("--ramp", type=float, default=1.0, help="command ramp duration [s]")
    ap.add_argument("--hold", type=float, default=8.0, help="hold duration after the ramp [s]")
    ap.add_argument("--steady-window", type=float, default=2.0, help="window at the end used for steady-state stats [s]")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config attribute")
    ap.add_argument("--no-outer-pd", action="store_true", help="apply the WBC torque only (no outer joint PD)")
    ap.add_argument("--outer-pd-mode", default="legacy", choices=["legacy", "plan"], help="outer PD target: legacy (one sim step ahead of the current state) or plan (WBC plan integrated over the hold interval)")
    ap.add_argument("--stale-state", action="store_true", help="legacy read: gather the state after mj_step without a forward pass (derived quantities one simulation step old)")
    ap.add_argument("--wbc-every", type=int, default=None, help="override the WBC period in simulation steps (legacy mjx_tita cadence = MPC period)")
    ap.add_argument("--scene", default="flat")
    ap.add_argument("--out", default=os.path.join(dir_path, "validation_results.json"))
    ap.add_argument("--save-logs", default=None, help="save full time series to this .npz file")
    ap.add_argument("--verbose-every", type=int, default=0, help="print one diagnostic line every N MPC steps")
    ap.add_argument("--device", default="cpu", help="cpu or gpu (must be given as '--device X')")
    ap.add_argument("--tag", default="", help="free-text tag stored in the results")
    args = ap.parse_args()

    overrides = {}
    for item in args.set:
        k, v = item.split("=", 1)
        overrides[k.strip()] = parse_value(v.strip())
    cfg = make_config(overrides)
    ctx = Context(cfg, scene=args.scene)
    if args.wbc_every is not None:
        ctx.timing["wbc_period_steps"] = int(args.wbc_every)
    T = ctx.timing
    print(f"[timing] sim {T['sim_f']} Hz (dt {T['dt_sim']:.4f}) | MPC {T['mpc_f']} Hz every {T['mpc_period_steps']} sim steps"
          f" | WBC {T['wbc_f']} Hz every {T['wbc_period_steps']} sim steps (dt_wbc {T['dt_wbc']:.4f})"
          f" | dt_mpc {T['dt_mpc']} N {T['N']} horizon {T['horizon']:.3f} s | shift {ctx.mpc.shift} node(s)"
          f" | fddp iterations {ctx.mpc.mpc_iterations} | WBC lookahead {ctx.mpc.wbc_lookahead_dt} s | outer PD {'off' if args.no_outer_pd else args.outer_pd_mode} | state read {'stale (after mj_step)' if args.stale_state else 'fresh (mj_step1/mj_step2)'}")
    if overrides:
        print(f"[config] overrides: {overrides}")

    idx = list(range(len(CASES))) if args.cases == "all" else [int(i) for i in args.cases.split(",")]
    results = []
    logs = {}
    for i in idx:
        label, vx_cmd, wz_cmd = CASES[i]
        t0 = time.time()
        summary, L = run_case(ctx, vx_cmd, wz_cmd, args.ramp, args.hold, label,
                              outer_pd=not args.no_outer_pd, steady_window_s=args.steady_window,
                              verbose_every=args.verbose_every, outer_pd_mode=args.outer_pd_mode,
                              fresh_state=not args.stale_state)
        summary["wall_s"] = time.time() - t0
        summary["overrides"] = overrides
        summary["outer_pd"] = not args.no_outer_pd
        summary["outer_pd_mode"] = args.outer_pd_mode
        summary["fresh_state"] = not args.stale_state
        summary["tag"] = args.tag
        results.append(summary)
        logs[label] = L
        print_summary(summary)

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2, default=lambda o: None if isinstance(o, float) and np.isnan(o) else str(o))
    print(f"Saved results to {args.out}")
    if args.save_logs:
        flat = {}
        for label, L in logs.items():
            for k, v in L.items():
                flat[f"{label}/{k}"] = v
        np.savez_compressed(args.save_logs, **flat)
        print(f"Saved logs to {args.save_logs}")


if __name__ == "__main__":
    main()
