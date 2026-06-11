import argparse
import csv
import shutil
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" 

import jax
print(jax.config.jax_enable_x64)
jax.config.update("jax_enable_x64", True)
print(jax.config.jax_enable_x64)
CACHE_DIR = os.path.expanduser("~/.jax_cache")
print(CACHE_DIR)
#jax.config.update("jax_compilation_cache_dir", CACHE_DIR)
#jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
#jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


import sys
import time
from datetime import datetime
from timeit import default_timer as timer

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, "..")))
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")


TITA_PATH = os.path.join(dir_path, "plots","tita_outputs")
os.makedirs(TITA_PATH, exist_ok=True)

MAX_STEPS = 10000
DEFAULT_VIDEO_SLOWDOWN_FACTOR = 1.0
LLC_ROLLOUT_CSV = os.path.join(TITA_PATH, "rollout_info_llc.csv")
TORQUE_LIST = []
FC_LIST = []
MPC_INPUT_LIST = []
MPC_OUTPUT_LIST = []
X_DES_LIST = []
U_DES_LIST = []
WBC_DESIRED_LIST = []
WBC_CURRENT_LIST = []

import jax.numpy as jnp
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np

import mpx.config.config_dfcip as config
import mpx.utils.mpc_wrapper_dfcip as mpc_wrapper_dfcip
import mpx.utils.mpc_utils as mpc_utils
import mpx.utils.sim as sim_utils
from plot_rollout_info import (
    _DEFAULT_DIR,
    plot_llc,
    plot_mpc_prediction_state,
    plot_mpc_prediction_control,
    plot_torques_and_contacts,
    plot_mpc_state_and_output,
    plot_wbc_desired,
)


def _snapshot_config_to_txt(output_dir: str = TITA_PATH) -> None:
    """Copy the active MPC config file to txt so run weights are traceable."""
    config_src = getattr(config, "__file__", None)
    if not config_src:
        print("[config_snapshot] Could not resolve config file path.")
        return

    config_src = os.path.abspath(config_src)
    dst_path = os.path.join(output_dir, "config_dfcip.txt")
    try:
        os.makedirs(output_dir, exist_ok=True)
        shutil.copy2(config_src, dst_path)
        print(f"[config_snapshot] Saved config snapshot: {dst_path}")
    except Exception as e:
        print(f"[config_snapshot] Failed to save config snapshot: {e}")

def _append_llc_row(
    csv_path: str,
    step: int,
    tau_cmd,
    kp: float,
    kd: float,
    action_scale: float = 1.0,
) -> None:
    tau = np.asarray(tau_cmd).reshape(-1)
    row = {
        "step": int(step),
        "low_level_controller/kp": float(kp),
        "low_level_controller/kd": float(kd),
        "low_level_controller/action_scale": float(action_scale),
    }
    for i in range(tau.size):
        row[f"low_level_controller/tau_ff_{i}"] = float(tau[i])
        row[f"low_level_controller/tau_p_{i}"] = 0.0
        row[f"low_level_controller/tau_d_{i}"] = 0.0
        row[f"low_level_controller/action_{i}"] = float(tau[i])

    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def _reset_to_initial_state(model, data):
    try:
        mujoco.mj_resetDataKeyframe(model, data, 0)
        data.qvel[:] = 0.0
    except Exception as e:
        print(f"Failed to reset to initial state: {e}")
    mujoco.mj_forward(model, data)

def _base_touches_floor(model, data, base_body_name: str = "base_link", floor_geom_name: str = "floor") -> bool:
    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body_name)
    floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, floor_geom_name)

    if base_body_id < 0 or floor_geom_id < 0:
        return False

    for contact_idx in range(data.ncon):
        contact = data.contact[contact_idx]
        geom1 = contact.geom1
        geom2 = contact.geom2
        body1 = model.geom_bodyid[geom1]
        body2 = model.geom_bodyid[geom2]

        if geom1 == floor_geom_id and body2 == base_body_id:
            return True
        if geom2 == floor_geom_id and body1 == base_body_id:
            return True

    return False

def build_tita_state(model, data, base_body_name, contact_ids) -> jnp.ndarray:
    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body_name)
    pcom = pcom = data.subtree_com[base_body_id] #jnp.asarray(data.qpos[0:3])
    vcom = vcom = data.subtree_linvel[base_body_id] #jnp.asarray(data.qvel[0:3])

    centers = np.asarray(sim_utils.geom_positions(data, contact_ids, flatten=False))  # (2, 3)

    def contact_offset_world(geom_id: int) -> np.ndarray:
        wheel_R = np.asarray(data.geom_xmat[geom_id]).reshape(3, 3)
        wheel_radius = float(data.model.geom_size[geom_id, 0])
        return np.asarray(mpc_utils.get_rCP(jnp.asarray(wheel_R), wheel_radius), dtype=np.float64)

    l_geom = int(contact_ids[0])
    r_geom = int(contact_ids[1])
    l_rcp = contact_offset_world(l_geom)
    r_rcp = contact_offset_world(r_geom)

    pl_world = jnp.asarray(centers[0] + l_rcp)
    pr_world = jnp.asarray(centers[1] + r_rcp)
    
    # velocità piedi nel world frame
    def geom_contact_velocity(data, geom_id: int, rcp_world: np.ndarray) -> np.ndarray:
        """Velocità del contact point: v_cp = v_center + omega x rcp."""
        vel = np.zeros(6)  # [rot(3), lin(3)]
        mujoco.mj_objectVelocity(
            data.model, data,
            mujoco.mjtObj.mjOBJ_GEOM,
            geom_id,
            vel,
            flg_local=0  # 0 = world frame
        )
        omega_world = vel[0:3]
        v_center_world = vel[3:6]
        return v_center_world + np.cross(omega_world, rcp_world)

    dpl_world = jnp.asarray(geom_contact_velocity(data, l_geom, l_rcp))
    dpr_world = jnp.asarray(geom_contact_velocity(data, r_geom, r_rcp))

    tita_state = jnp.hstack([
        pcom, vcom, pl_world, pr_world, dpl_world, dpr_world
    ])
    return tita_state

def get_dfip_current_state(self, tita_state: jax.Array, theta_prev: float) -> jax.Array:
        
    def unwrapNear(theta_wrapped: float, theta_prev: float) -> float:
        # wrapping to pi
        a = theta_wrapped - theta_prev # input

        a = (a + jnp.pi) % (2 * jnp.pi)
        a = jnp.where(a < 0, a + 2*jnp.pi, a)
        a = a - jnp.pi

        return theta_prev + a

    pcom        = tita_state[0:3]
    vcom        = tita_state[3:6]
    pl_world    = tita_state[6:9]
    pr_world    = tita_state[9:12]
    dpl_world   = tita_state[12:15]
    dpr_world   = tita_state[15:18]

    c_world     = (pl_world + pr_world) / 2.0
    vc_world    = (dpl_world + dpr_world) / 2.0

    # extract theta
    diff = pl_world - pr_world
    theta_wrapped = jnp.atan2( -diff[0], diff[1])
    theta = unwrapNear(theta_wrapped, theta_prev)

    R = jnp.array([
        [ jnp.cos(theta), -jnp.sin(theta), 0.],
        [ jnp.sin(theta),  jnp.cos(theta), 0.],
        [ 0.,              0.,             1.]
    ])

    dpl_body = R.T @ dpl_world
    dpr_body = R.T @ dpr_world

    d = config.d
    w = (dpr_body[0] - dpl_body[0]) / d # 0.1
    v = (dpr_body[0] + dpl_body[0]) / 2.0

    x0 = jnp.concatenate([
        pcom,
        vcom,
        c_world,
        jnp.array([vc_world[2]]),
        jnp.array([theta]),
        jnp.array([v]),
        jnp.array([w]),
    ])

    return x0, theta

def build_wbc_current_vector(model, data, contact_ids, mpc_utils, nj: int) -> np.ndarray:
    """
    Build a vector with the same layout as WBC desired.
    Useful for plotting desired vs current using the same _REF_* indices.
    """

    # stesso size usato dal wrapper:
    # _REF_JOINTS + q, dq, ddq
    current = np.zeros(mpc_utils._REF_JOINTS + 3 * nj, dtype=np.float64)

    # ── COM current ─────────────────────────────────────────────
    # Nota: nel WBC mjx viene usato subtree_com[0], cioè CoM globale del modello.
    com_pos = np.asarray(data.subtree_com[0]).copy()

    # MuJoCo non dà direttamente data.subtree_linvel[0] sempre in modo affidabile
    # dopo mj_forward? Qui usiamo cvel/com subtree se disponibile.
    # Se vuoi essere più coerente con il tuo x0, puoi sostituire con data.subtree_linvel[base_id].
    try:
        com_vel = np.asarray(data.subtree_linvel[0]).copy()
    except Exception:
        com_vel = np.zeros(3)

    current[mpc_utils._REF_COM_POS:mpc_utils._REF_COM_POS + 3] = com_pos
    current[mpc_utils._REF_COM_VEL:mpc_utils._REF_COM_VEL + 3] = com_vel
    current[mpc_utils._REF_COM_ACC:mpc_utils._REF_COM_ACC + 3] = 0.0

    # ── Wheel center current ────────────────────────────────────
    l_geom = int(contact_ids[0])
    r_geom = int(contact_ids[1])

    l_pos = np.asarray(data.geom_xpos[l_geom]).copy()
    r_pos = np.asarray(data.geom_xpos[r_geom]).copy()

    def geom_center_velocity(data, geom_id: int) -> np.ndarray:
        vel = np.zeros(6)  # [angular, linear]
        mujoco.mj_objectVelocity(
            data.model,
            data,
            mujoco.mjtObj.mjOBJ_GEOM,
            geom_id,
            vel,
            flg_local=0,
        )
        return vel[3:6].copy()

    l_vel = geom_center_velocity(data, l_geom)
    r_vel = geom_center_velocity(data, r_geom)

    current[mpc_utils._REF_LW_POS:mpc_utils._REF_LW_POS + 3] = l_pos
    current[mpc_utils._REF_RW_POS:mpc_utils._REF_RW_POS + 3] = r_pos

    current[mpc_utils._REF_LW_VEL:mpc_utils._REF_LW_VEL + 3] = l_vel
    current[mpc_utils._REF_RW_VEL:mpc_utils._REF_RW_VEL + 3] = r_vel

    current[mpc_utils._REF_LW_ACC:mpc_utils._REF_LW_ACC + 3] = 0.0
    current[mpc_utils._REF_RW_ACC:mpc_utils._REF_RW_ACC + 3] = 0.0

    # ── Base rotation / angular velocity current ────────────────
    base_body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        config.base_body_name,
    )

    base_R = np.asarray(data.xmat[base_body_id]).reshape(3, 3).copy()
    current[mpc_utils._REF_BASE_ROT:mpc_utils._REF_BASE_ROT + 9] = base_R.reshape(-1)

    # body velocity: [angular, linear]
    base_vel6 = np.zeros(6)
    mujoco.mj_objectVelocity(
        model,
        data,
        mujoco.mjtObj.mjOBJ_BODY,
        base_body_id,
        base_vel6,
        flg_local=0,
    )
    base_omega = base_vel6[0:3].copy()

    current[mpc_utils._REF_BASE_OMG:mpc_utils._REF_BASE_OMG + 3] = base_omega
    current[mpc_utils._REF_BASE_ALP:mpc_utils._REF_BASE_ALP + 3] = 0.0

    # ── Joints current ─────────────────────────────────────────
    qj = np.asarray(data.qpos[7:7 + nj]).copy()
    dqj = np.asarray(data.qvel[6:6 + nj]).copy()

    current[mpc_utils._REF_JOINTS:mpc_utils._REF_JOINTS + nj] = qj
    current[mpc_utils._REF_JOINTS + nj:mpc_utils._REF_JOINTS + 2 * nj] = dqj
    current[mpc_utils._REF_JOINTS + 2 * nj:mpc_utils._REF_JOINTS + 3 * nj] = 0.0

    return current

def main(headless=False, steps=500, scene="flat"):
    if os.path.exists(LLC_ROLLOUT_CSV):
        os.remove(LLC_ROLLOUT_CSV)

    model = mujoco.MjModel.from_xml_path(
        dir_path + f"/../data/tita/tita_world.xml"
    )
    nj = model.nv - 6
    data = mujoco.MjData(model)
    sim_frequency = float(config.whole_body_frequency)
    video_fps = max(1, int(round(sim_frequency / max(DEFAULT_VIDEO_SLOWDOWN_FACTOR, 1e-6))))
    counter = 0

    # ── MuJoCo video recording ───────────────────────────────────────────
    _sim_frames: list = []
    _renderer = mujoco.Renderer(model, height=480, width=640)
    _sim_cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(_sim_cam)
    _sim_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    _sim_cam.distance = 8.0
    _sim_cam.elevation = -15.0
    _sim_cam.azimuth = 60.0
    # ────────────────────────────────────────────────────────────────────
    model.opt.timestep = 1.0 / sim_frequency
    
    contact_ids = sim_utils.geom_ids(model, config.contact_frame)
    command_handle = sim_utils.KeyboardVelocityCommand(vx=0.0, vy=0.0, wz=0.0)
    # TODO: use mpc_wrapper_dfcip
    mpc = mpc_wrapper_dfcip.BatchedMPCControllerWrapper(config, n_env=1)

    _reset_to_initial_state(model, data)
    mujoco.mj_forward(model, data)
    data.qvel[:] = 0.0          # ← forza velocità a zero
    mujoco.mj_forward(model, data)  # ← ri-propaga con vel=0
    np.set_printoptions(precision=20, suppress=True)
    print("------------")
    total = 0.0
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or f"body_{i}"
        m = model.body_mass[i]
        total += m
        if m > 0.0001:
            print(f"  id: {i:2d},  {name:25s}  {m:.4f} kg")
    print(f"  {'TOTAL':25s}  {total:.4f} kg")
    M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, M, data.qM)
    print("Inertia matrix:\n", M[3:6, 3:6])
    print("------------")
    
    theta_prev = 0.0
    foot = jnp.asarray(sim_utils.geom_positions(data, contact_ids))
    tita_state = build_tita_state(model, data, base_body_name="base_link", contact_ids=contact_ids)
    x0, theta_prev = get_dfip_current_state(mpc, tita_state, theta_prev=theta_prev)
    print(f"pcom:    {x0[0:3]}")
    print(f"vcom:    {x0[3:6]}")
    print(f"c_world: {x0[6:9]}")
    print(f"vcz:     {x0[9]}")
    print(f"theta:   {x0[10]}")
    print(f"v:       {x0[11]}")
    print(f"omega:   {x0[12]}")
    print(f"robot floating base: {config.robot_height}")
    print(f"robot com: {data.subtree_com[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'base_link')]}")
    print(f"x_ref step0: {mpc._x_reference[0, :13]}")
    print(f"u_ref step0: {mpc._u_reference[0, :9]}")
    
    mpc_state = mpc.init_state()

    # ── print MPC config parameters ──────────────────────────────────────
    _mpc_params = [
        "dt", "dt_mpc", "N", "mpc_frequency", "grav", "whole_body_frequency",
        "duty_factor", "step_freq", "step_height", "robot_height", "clearence_speed",
        "mu", "mass", "d",
        "w_pcomxy", "w_pcomz", "w_vcomxy", "w_vcomz", "w_c", "w_vcz",
        "w_theta", "w_v", "w_omega",
        "w_a", "w_ac_z", "w_alpha", "w_fcxy", "w_fcz", "w_eq",
    ]
    print("\n── MPC parameters ──")
    for _p in _mpc_params:
        print(f"  {_p:20s} = {getattr(config, _p)}")
    print("────────────────────\n")

    print("[timing] mpc.run  ... ", end="", flush=True)
    _t0 = timer()
    mpc_state, reference = mpc.run(mpc_state, x0[None, :], counter, config.N)
    jax.block_until_ready(mpc_state.U0_shifted)
    _dt = timer() - _t0
    print(f"{int(_dt // 60)}m {_dt % 60:.1f}s")

    # ── save MPC prediction plots at timestep 0 ──────────────────────────
    X0_np = x0
    U0_np = np.concatenate([np.asarray(mpc_state.sol.a)[:, None], np.asarray(mpc_state.sol.ac_z)[:, None], np.asarray(mpc_state.sol.alpha)[:, None], np.asarray(mpc_state.sol.grf)], axis=1).copy()
    XN_np = np.asarray(mpc_state.X0_shifted[0][-1]).copy()  
    UN_np = np.asarray(mpc_state.U0_shifted[0][-1]).copy()
    np.set_printoptions(precision=4, suppress=True, linewidth=200)

    _x_names = ["pcom_x", "pcom_y", "pcom_z", "dpcom_x", "dpcom_y", "dpcom_z",
                "c_x",    "c_y",    "c_z",    "vcz",     "θ",       "v",       "ω"]
    _u_names = ["a", "acz", "α", "Fl_x", "Fl_y", "Fl_z", "Fr_x", "Fr_y", "Fr_z"]

    _PREFIX = "  step  0: "   # 11 chars — aligns header with data rows

    def _fmt_row(names, values, w=10):
        header = "".join(f"{n:>{w}}" for n in names)
        values = np.asarray(values).reshape(-1)
        row = "".join(f"{v:>10.4f}" for v in values)
        return header, row

    print("\n── MPC state  x0, XN ──")
    hdr, r0 = _fmt_row(_x_names, X0_np)
    _,   rN = _fmt_row(_x_names, XN_np)
    print(" " * len(_PREFIX) + hdr)
    print(f"  step  0: {r0}")
    print(f"  step  N: {rN}")

    print("\n── MPC control  u0, UN ──")
    hdr, r0 = _fmt_row(_u_names, U0_np)
    _,   rN = _fmt_row(_u_names, UN_np)
    print(" " * len(_PREFIX) + hdr)
    print(f"  step  0: {r0}")
    print(f"  step  N: {rN}")
    print()

    #plot_mpc_prediction_state(
    #    mpc_state.X0_shifted[0],
    #    timestep=0,
    #    filename="state_mpc_t000.png",
    #    out_dir=os.path.join(TITA_PATH, "mpc_prediction"),
    #)
    #plot_mpc_prediction_control(
    #    mpc_state.U0_shifted[0],
    #    timestep=0,
    #    filename="control_mpc_t000.png",
    #    out_dir=os.path.join(TITA_PATH, "mpc_prediction"),
    #)

    # ── run WBC and print results ─────────────────────────────────────────
    
    qpos_np = jnp.asarray(data.qpos)[None, :]   # (1, nq)
    qvel_np = jnp.asarray(data.qvel)[None, :]   # (1, nv)

    # ── print WBC-related config parameters ──────────────────────────────
    _wbc_params = [
        "base_body_name", "wheel_radius",
        "Kp_motion", "Kd_motion", "Kp_wheel", "Kd_wheel", "Kp_reg", "Kd_reg",
        "w_qddot", "w_com", "w_lwheel", "w_rwheel", "w_base", "w_friction", "w_joint_vel",
    ]
    print("\n── WBC parameters ──")
    for _p in _wbc_params:
        print(f"  {_p:20s} = {getattr(config, _p)}")
    print("────────────────────\n")

    print(f"[timing] wbc.run  start={datetime.now().strftime('%H:%M:%S')} ... ", end="", flush=True)
    _t0 = timer()
    tita_state_wbc = build_tita_state(model, data, base_body_name="base_link", contact_ids=contact_ids)
    pl_world_wbc = tita_state_wbc[6:9][None, :]
    pr_world_wbc = tita_state_wbc[9:12][None, :]
    dpl_world_wbc = tita_state_wbc[12:15][None, :]
    dpr_world_wbc = tita_state_wbc[15:18][None, :]
    mpc_state, tau_cmd, qddot, fl, fr, desired = mpc.whole_body_run(
        mpc_state,
        x0,
        qpos_np,
        qvel_np,
        counter,
        pl_world_wbc,
        pr_world_wbc,
        dpl_world_wbc,
        dpr_world_wbc,
    )
    jax.block_until_ready(tau_cmd)
    _dt = timer() - _t0
    print(f"{int(_dt // 60)}m {_dt % 60:.1f}s")

    #_append_llc_row(
    #    LLC_ROLLOUT_CSV,
    #    step=0,
    #    tau_cmd=tau_cmd[0],
    #    kp=getattr(config, "Kp_motion", 0.0),
    #    kd=getattr(config, "Kd_motion", 0.0),
    #)
    #plot_llc(
    #    LLC_ROLLOUT_CSV,
    #    out_path=os.path.join(TITA_PATH, "llc","llc_t000.png"),
    #)

    _tau_names = [f"τ{i}" for i in range(tau_cmd.shape[-1])]
    _q_names   = [f"q̈{i}" for i in range(qddot.shape[-1])]

    def _fmt_vec(names, values, w=9):
        hdr = "".join(f"{n:>{w}}" for n in names)
        row = "".join(f"{v:>{w}.4f}" for v in values)
        return hdr, row

    _tau_names = [f"τ{i}" for i in range(tau_cmd.shape[-1])]
    _q_names   = [f"q̈{i}" for i in range(qddot.shape[-1])]
    _f_names   = ["fx", "fy", "fz"]

    _tau_np  = np.asarray(tau_cmd[0])
    _q_np    = np.asarray(qddot[0])
    _fl_np   = np.asarray(fl[0])
    _fr_np   = np.asarray(fr[0])

    def _fmt_vec(names, values, w=9):
        hdr = "".join(f"{n:>{w}}" for n in names)
        row = "".join(f"{v:>{w}.4f}" for v in values)
        return hdr, row

    print("\n── WBC output ──")
    hdr, row = _fmt_vec(_tau_names, _tau_np)
    print("  tau  hdr: " + hdr)
    print("  tau  val: " + row)
    hdr, row = _fmt_vec(_q_names, _q_np)
    print("  qddot hdr:" + hdr)
    print("  qddot val:" + row)
    hdr, row = _fmt_vec(_f_names, _fl_np)
    print("  Fl   hdr: " + hdr)
    print("  Fl   val: " + row)
    hdr, row = _fmt_vec(_f_names, _fr_np)
    print("  Fr   hdr: " + hdr)
    print("  Fr   val: " + row)
    print()

    _tau_np  = np.asarray(tau_cmd[-1])
    _q_np    = np.asarray(qddot[-1])
    _fl_np   = np.asarray(fl[-1])
    _fr_np   = np.asarray(fr[-1])
    print("\n── WBC output ──")
    hdr, row = _fmt_vec(_tau_names, _tau_np)
    print("  tau  hdr: " + hdr)
    print("  tau  val: " + row)
    hdr, row = _fmt_vec(_q_names, _q_np)
    print("  qddot hdr:" + hdr)
    print("  qddot val:" + row)
    hdr, row = _fmt_vec(_f_names, _fl_np)
    print("  Fl   hdr: " + hdr)
    print("  Fl   val: " + row)
    hdr, row = _fmt_vec(_f_names, _fr_np)
    print("  Fr   hdr: " + hdr)
    print("  Fr   val: " + row)
    print()
    
    # ─────────────────────────────────────────────────────────────────────


    period = int(sim_frequency / config.mpc_frequency)
    print(f"sim_frequency: {sim_frequency} Hz, mpc_frequency: {config.mpc_frequency} Hz")
    #print(f"Controller period: {period} steps at {sim_frequency} Hz simulation frequency.")
    counter = 0
    theta_prev = 0.0
    mpc_state = mpc.init_state()

    def do_print(*args, **kwargs):
        debug_print = False
        if debug_print:
            print(*args, **kwargs)

    def step_controller(mpc_state, theta_prev=theta_prev):
        nonlocal counter

        do_print(f"\n=== step {counter} ===")
        
        qpos = data.qpos.copy()
        qvel = data.qvel.copy()

        if counter % period == 0:

            contact_ids = sim_utils.geom_ids(model, config.contact_frame)
            tita_state = build_tita_state(model, data, base_body_name="base_link", contact_ids=contact_ids)
            x0, theta_prev = get_dfip_current_state(mpc, tita_state, theta_prev=theta_prev)

            mpc_state, reference = mpc.run(mpc_state, x0[None, :], counter, config.N)

            def _f(v, d=4):
                s = f"{float(v):.{d}f}"
                return s if float(v) < 0 else f" {s}"
            
            do_print(
                f"[x0] "
                f" pc=({_f(x0[0])},{_f(x0[1])},{_f(x0[2])}) "
                f" vc=({_f(x0[3])},{_f(x0[4])},{_f(x0[5])}) "
                f"\n      cw=({_f(x0[6])},{_f(x0[7])},{_f(x0[8])}) "
                f" vcz={_f(x0[9])}\n      th={_f(x0[10],4)} v={_f(x0[11],4)} w={_f(x0[12],4)}"
            )
            do_print(
                f"[sol] "
                f"a={_f(mpc_state.sol.a[0])} "
                f"acz={_f(mpc_state.sol.ac_z[0])} "
                f"alpha={_f(mpc_state.sol.alpha[0], 4)}\n "
                f"      fl=({_f(mpc_state.sol.grf[0,0])},{_f(mpc_state.sol.grf[0,1])},{_f(mpc_state.sol.grf[0,2])})\n "
                f"      fr=({_f(mpc_state.sol.grf[0,3])},{_f(mpc_state.sol.grf[0,4])},{_f(mpc_state.sol.grf[0,5])})"
           
            )

            x_ref_step = reference[:, 0, :config.nx]
            u_ref_step = reference[:, 0, config.nx:]
            do_print(
                f"[ref0] "
                f"pc=({_f(x_ref_step[0,0])},{_f(x_ref_step[0,1])},{_f(x_ref_step[0,2])}) "
                f"vc=({_f(x_ref_step[0,3])},{_f(x_ref_step[0,4])},{_f(x_ref_step[0,5])}) "
                f"\n       cw=({_f(x_ref_step[0,6])},{_f(x_ref_step[0,7])},{_f(x_ref_step[0,8])}) "
                f"vcz={_f(x_ref_step[0,9])} "
                f"th={_f(x_ref_step[0,10],4)} v={_f(x_ref_step[0,11],4)} w={_f(x_ref_step[0,12],4)}\n"
                f"       a={_f(u_ref_step[0,0])} "
                f"acz={_f(u_ref_step[0,1])} "
                f"alpha={_f(u_ref_step[0,2],4)}\n "
                f"      fl=({_f(u_ref_step[0,3])},{_f(u_ref_step[0,4])},{_f(u_ref_step[0,5])})\n "
                f"      fr=({_f(u_ref_step[0,6])},{_f(u_ref_step[0,7])},{_f(u_ref_step[0,8])})"
            )

            x_ref_N = reference[:, -1, :config.nx]
            u_ref_N = reference[:, -1, config.nx:]
            do_print(
                f"[refN] "
                f"pc=({_f(x_ref_N[0,0])},{_f(x_ref_N[0,1])},{_f(x_ref_N[0,2])}) "
                f"vc=({_f(x_ref_N[0,3])},{_f(x_ref_N[0,4])},{_f(x_ref_N[0,5])}) "
                f"\n      cw=({_f(x_ref_N[0,6])},{_f(x_ref_N[0,7])},{_f(x_ref_N[0,8])}) "
                f"vcz={_f(x_ref_N[0,9])} "
                f"th={_f(x_ref_N[0,10],4)} v={_f(x_ref_N[0,11],4)} w={_f(x_ref_N[0,12],4)}\n"
                f"       a={_f(u_ref_N[0,0])} "
                f"acz={_f(u_ref_N[0,1])} "
                f"alpha={_f(u_ref_N[0,2],4)}\n "
                f"       fl=({_f(u_ref_N[0,3])},{_f(u_ref_N[0,4])},{_f(u_ref_N[0,5])})\n "
                f"       fr=({_f(u_ref_N[0,6])},{_f(u_ref_N[0,7])},{_f(u_ref_N[0,8])})"
            )

            X_DES_LIST.append( x_ref_step[0].copy() )
            U_DES_LIST.append( u_ref_step[0].copy() )
            mpc_output = np.array([
                float(mpc_state.sol.a[0]),
                float(mpc_state.sol.ac_z[0]),
                float(mpc_state.sol.alpha[0]),
                *np.asarray(mpc_state.sol.grf[0])   # 6 elementi
            ])

            MPC_INPUT_LIST.append(np.asarray(x0).copy())
            MPC_OUTPUT_LIST.append(mpc_output)
            
            pl_world_wbc = tita_state[6:9][None, :]
            pr_world_wbc = tita_state[9:12][None, :]
            dpl_world_wbc = tita_state[12:15][None, :]
            dpr_world_wbc = tita_state[15:18][None, :]

            mpc_state, tau_cmd, qddot, fl, fr, desired = mpc.whole_body_run(
                mpc_state,
                x0,
                jnp.asarray(qpos)[None, :],
                jnp.asarray(qvel)[None, :],
                counter,
                pl_world_wbc,
                pr_world_wbc,
                dpl_world_wbc,
                dpr_world_wbc,
            )

            d = desired[0] if desired.ndim == 2 else desired
            _nj = model.nv - 6
            
            do_print(
                f"[desired] "
                f"com_p=({_f(d[mpc_utils._REF_COM_POS + 0])},{_f(d[mpc_utils._REF_COM_POS + 1])},{_f(d[mpc_utils._REF_COM_POS + 2])}) "
                f"com_v=({_f(d[mpc_utils._REF_COM_VEL + 0])},{_f(d[mpc_utils._REF_COM_VEL + 1])},{_f(d[mpc_utils._REF_COM_VEL + 2])}) "
                f"\n          com_a=({_f(d[mpc_utils._REF_COM_ACC + 0])},{_f(d[mpc_utils._REF_COM_ACC + 1])},{_f(d[mpc_utils._REF_COM_ACC + 2])})"
                f"\n          lw_p=({_f(d[mpc_utils._REF_LW_POS + 0])},{_f(d[mpc_utils._REF_LW_POS + 1])},{_f(d[mpc_utils._REF_LW_POS + 2])}) "
                f"rw_p=({_f(d[mpc_utils._REF_RW_POS + 0])},{_f(d[mpc_utils._REF_RW_POS + 1])},{_f(d[mpc_utils._REF_RW_POS + 2])})"
                f"\n          lw_v=({_f(d[mpc_utils._REF_LW_VEL + 0])},{_f(d[mpc_utils._REF_LW_VEL + 1])},{_f(d[mpc_utils._REF_LW_VEL + 2])}) "
                f"rw_v=({_f(d[mpc_utils._REF_RW_VEL + 0])},{_f(d[mpc_utils._REF_RW_VEL + 1])},{_f(d[mpc_utils._REF_RW_VEL + 2])})"
                f"\n          lw_a=({_f(d[mpc_utils._REF_LW_ACC + 0])},{_f(d[mpc_utils._REF_LW_ACC + 1])},{_f(d[mpc_utils._REF_LW_ACC + 2])}) "
                f"rw_a=({_f(d[mpc_utils._REF_RW_ACC + 0])},{_f(d[mpc_utils._REF_RW_ACC + 1])},{_f(d[mpc_utils._REF_RW_ACC + 2])})"
                f"\n          omg=({_f(d[mpc_utils._REF_BASE_OMG + 0])},{_f(d[mpc_utils._REF_BASE_OMG + 1])},{_f(d[mpc_utils._REF_BASE_OMG + 2])}) "
                f"alp=({_f(d[mpc_utils._REF_BASE_ALP + 0])},{_f(d[mpc_utils._REF_BASE_ALP + 1])},{_f(d[mpc_utils._REF_BASE_ALP + 2])})"
                f"\n          qj0={_f(d[mpc_utils._REF_JOINTS + 0],4)} "
                f"dqj0={_f(d[mpc_utils._REF_JOINTS + _nj + 0],4)} "
                f"ddqj0={_f(d[mpc_utils._REF_JOINTS + 2*_nj + 0],4)}"
            )

            tau_to_apply = tau_cmd[0].copy()

            c = build_wbc_current_vector(
                model=model,
                data=data,
                contact_ids=contact_ids,
                mpc_utils=mpc_utils,
                nj=_nj,
            )
            
            TORQUE_LIST.append(tau_to_apply)
            FC_LIST.append( [fl[0].copy(), fr[0].copy()] )
            WBC_DESIRED_LIST.append(d.copy())
            WBC_CURRENT_LIST.append(c.copy())

        touch_floor = _base_touches_floor(model, data, base_body_name=config.base_body_name)

        do_print(
            f"[WBC]\n"
            f"  tau_cmd: {tau_to_apply}\n"
            f"  contact_forces_left (fl): {fl[0]}\n"
            f"  contact_forces_right (fr): {fr[0]}"
        )

        #_append_llc_row(
        #    LLC_ROLLOUT_CSV,
        #    step=counter + 1,
        #    tau_cmd=tau_to_apply,
        #    kp=getattr(config, "Kp_motion", 0.0),
        #    kd=getattr(config, "Kd_motion", 0.0),
        #)

        if False and (counter % 50 == 0 and counter < 505) or touch_floor:
            X0_np = np.asarray(mpc_state.X0_shifted[0])   # (N+1, 13)
            U0_np = np.asarray(mpc_state.U0_shifted[0])  

            plot_mpc_prediction_state(
                X0_np,
                timestep=counter+1,
                filename=f"state_mpc_t{counter+1:03d}.png",
                out_dir=os.path.join(TITA_PATH, "mpc_prediction"),
            )
            plot_mpc_prediction_control(
                U0_np,
                timestep=counter+1,
                filename=f"control_mpc_t{counter+1:03d}.png",
                out_dir=os.path.join(TITA_PATH, "mpc_prediction"),
            )
            plot_llc(
                LLC_ROLLOUT_CSV,
                out_path=os.path.join(TITA_PATH, f"llc.png"),
            )


        q_target = np.array([0.0, 0.5, -1.0, 0.0,]*2)
        pd_tau = 70*(q_target - data.qpos[7:15]) - 0.5*data.qvel[6:14]

        data.ctrl = np.asarray(tau_cmd[0])
        #data.ctrl = pd_tau
        #data.ctrl = np.zeros(model.nu)
        #print("-----------")
        #print(f"Base position: {data.qpos[0:3]}")
        #print(f"Joint positions: {list(map(lambda x: round(float(x), 2), data.qpos[7:]))}")
        mujoco.mj_step(model, data)
        _renderer.update_scene(data, camera=_sim_cam)
        _sim_frames.append(_renderer.render().copy())
        counter += 1
        return mpc_state, theta_prev, touch_floor

    def _save_sim_video() -> None:
        import imageio
        if not _sim_frames:
            return
        video_dir = os.path.join(TITA_PATH)
        os.makedirs(video_dir, exist_ok=True)
        video_path = os.path.join(video_dir, "simulation_video.mp4")
        try:
            imageio.mimwrite(video_path, _sim_frames, fps=video_fps, macro_block_size=1)
            print(
                f"[sim_video] saved ({len(_sim_frames)} frames, {video_fps} fps, slowdown x{DEFAULT_VIDEO_SLOWDOWN_FACTOR:.2f}): {video_path}"
            )
        except Exception as e:
            print(f"[sim_video] failed to save video: {e}")
        finally:
            _renderer.close()

    if headless:
        for _ in range(steps):
            mpc_state, theta_prev, touch_floor = step_controller(mpc_state, theta_prev=theta_prev)
            if touch_floor:
                print(f"Base touched the floor at step {counter}. Ending simulation.")
                break
        _save_sim_video()
        return

    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=command_handle.key_callback,
    ) as viewer:
        viewer.cam.distance *= 5.5
        viewer.sync()
        while viewer.is_running():
            overlay_text = command_handle.consume_overlay_text()
            tic = timer()
            overlay_text = command_handle.consume_overlay_text()

            timestep_text = (
                "Info",
                f"step: {counter}\nsim time: {data.time:.3f} s",
            )

            texts = []
            if overlay_text is not None:
                texts.extend(overlay_text)

            texts.append((None, None, timestep_text[0], timestep_text[1]))

            viewer.set_texts(texts)
            mpc_state, theta_prev, touch_floor = step_controller(mpc_state, theta_prev=theta_prev)

            toc = timer()
            if toc - tic < model.opt.timestep:
                time.sleep(model.opt.timestep - (toc - tic))

            if touch_floor:
                print(f"Base touched the floor at step {counter}. Ending simulation.")
                break

            if counter >= MAX_STEPS:
                print(f"Reached max steps ({MAX_STEPS}). Exiting.")
                break
            viewer.sync()

    _save_sim_video()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--scene", type=str, default="flat")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    if os.environ.get("DISPLAY") is None or os.environ.get("DISPLAY") == "":
        print("[WARN] No DISPLAY detected: forcing --headless mode (no viewer)")
        args.headless = True

    def do_plots():

        model = mujoco.MjModel.from_xml_path(
            dir_path + f"/../data/tita/tita_world.xml"
        )
        nj = model.nv - 6
        plot_torques_and_contacts(
            torques=TORQUE_LIST,
            contact_forces=FC_LIST,
            out_dir=TITA_PATH,
            filename_torques="wbc_torques.png",
            filename_contacts="wbc_contacts.png",
        )
        plot_mpc_state_and_output(
            x0_list=MPC_INPUT_LIST,
            u0_list=MPC_OUTPUT_LIST,
            x_ref_list=X_DES_LIST,
            u_ref_list=U_DES_LIST,
            out_dir=TITA_PATH,
            filename_state="mpc_input.png",
            filename_u0="mpc_output.png",
        )

        plot_wbc_desired(
            desired_list=WBC_DESIRED_LIST,
            current_list=None, #WBC_CURRENT_LIST,
            mpc_utils=mpc_utils,
            nj=model.nv - 6,
            out_dir=TITA_PATH,
            filename_com_base="wbc_desired_com_base.png",
            filename_wheels="wbc_desired_wheels.png",
            filename_joints="wbc_desired_joints.png",
        )

    _snapshot_config_to_txt(TITA_PATH)
    try:
        main(
            headless=args.headless,
            steps=args.steps,
            scene=args.scene,
        )
    except Exception as e:
        do_plots()
    do_plots()
