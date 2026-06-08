import argparse
import csv
import shutil
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" 

import jax

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

MAX_STEPS = 1000
DEFAULT_VIDEO_SLOWDOWN_FACTOR = 4.0
LLC_ROLLOUT_CSV = os.path.join(TITA_PATH, "rollout_info_llc.csv")
TORQUE_LIST = []
FC_LIST = []
MPC_INPUT_LIST = []
MPC_OUTPUT_LIST = []

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
    plot_llc,
    plot_mpc_prediction_state,
    plot_mpc_prediction_control,
    plot_torques_and_contacts,
    plot_mpc_state_and_output,
    render_mpc_prediction_video,
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

def build_tita_state(data, contact_ids) -> jnp.ndarray:
    pcom = jnp.asarray(data.qpos[0:3])
    vcom = jnp.asarray(data.qvel[0:3])

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


def _quat_wxyz_to_rpy(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to roll-pitch-yaw [rad]."""
    w, x, y, z = [float(v) for v in quat_wxyz]

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw], dtype=np.float64)

def _quat_wxyz_to_rotmat(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to a 3x3 rotation matrix."""
    w, x, y, z = [float(v) for v in quat_wxyz]

    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),     2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w),     1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w),     2.0 * (y * z + x * w),     1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)

def main(headless=False, steps=500, scene="flat"):
    if os.path.exists(LLC_ROLLOUT_CSV):
        os.remove(LLC_ROLLOUT_CSV)

    model = mujoco.MjModel.from_xml_path(
        dir_path + f"/../data/tita/tita_world.xml"
    )
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
    _sim_cam.distance = 5.0
    _sim_cam.elevation = -20.0
    # ────────────────────────────────────────────────────────────────────
    model.opt.timestep = 1.0 / sim_frequency
    
    contact_ids = sim_utils.geom_ids(model, config.contact_frame)
    command_handle = sim_utils.KeyboardVelocityCommand(vx=0.0, vy=0.0, wz=0.0)
    # TODO: use mpc_wrapper_dfcip
    mpc = mpc_wrapper_dfcip.BatchedMPCControllerWrapper(config, n_env=1)

    _reset_to_initial_state(model, data)
    mujoco.mj_forward(model, data)
    np.set_printoptions(precision=20, suppress=True)
    print("------------")
    total = 0.0
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or f"body_{i}"
        m = model.body_mass[i]
        total += m
        if m > 0.0001:
            print(f"  {name:25s}  {m:.4f} kg")
    print(f"  {'TOTAL':25s}  {total:.4f} kg")
    M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, M, data.qM)
    print("Inertia matrix:\n", M[3:6, 3:6])
    print("------------")
    
    theta_prev = 0.0
    foot = jnp.asarray(sim_utils.geom_positions(data, contact_ids))
    tita_state = build_tita_state(data, contact_ids)
    x0, theta_prev = get_dfip_current_state(mpc, tita_state, theta_prev=theta_prev)
    print(f"pcom:    {x0[0:3]}")
    print(f"vcom:    {x0[3:6]}")
    print(f"c_world: {x0[6:9]}")
    print(f"vcz:     {x0[9]}")
    print(f"theta:   {x0[10]}")
    print(f"v:       {x0[11]}")
    print(f"omega:   {x0[12]}")
    print(f"robot_height ref: {config.robot_height}")
    print(f"x_ref step0: {mpc._x_reference[0, :13]}")
    print(f"u_ref step0: {mpc._u_reference[0, :9]}")
    
    command = jnp.asarray(command_handle.mpc_input(config.robot_height))
    contact = jnp.asarray(sim_utils.estimate_contacts(data, contact_ids))

    mpc_state = mpc.init_state()

    # ── print MPC config parameters ──────────────────────────────────────
    _mpc_params = [
        "dt", "N", "mpc_frequency", "grav", "whole_body_frequency",
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
    mpc_state = mpc.run(mpc_state, x0[None, :], counter, config.N)
    jax.block_until_ready(mpc_state.U0)
    _dt = timer() - _t0
    print(f"{int(_dt // 60)}m {_dt % 60:.1f}s")

    # ── save MPC prediction plots at timestep 0 ──────────────────────────
    X0_np = np.asarray(mpc_state.X0[0])   # (N+1, 13)
    U0_np = np.asarray(mpc_state.U0[0])   # (N,   9)

    np.set_printoptions(precision=4, suppress=True, linewidth=200)

    _x_names = ["pcom_x", "pcom_y", "pcom_z", "dpcom_x", "dpcom_y", "dpcom_z",
                "c_x",    "c_y",    "c_z",    "vcz",     "θ",       "v",       "ω"]
    _u_names = ["a", "acz", "α", "Fl_x", "Fl_y", "Fl_z", "Fr_x", "Fr_y", "Fr_z"]

    _PREFIX = "  step  0: "   # 11 chars — aligns header with data rows

    def _fmt_row(names, values, w=10):
        header = "".join(f"{n:>{w}}" for n in names)
        row    = "".join(f"{v:>{w}.4f}" for v in values)
        return header, row

    print("\n── MPC state  X0  (N+1 × 13) ──")
    hdr, r0 = _fmt_row(_x_names, X0_np[0])
    _,   rN = _fmt_row(_x_names, X0_np[-1])
    print(" " * len(_PREFIX) + hdr)
    print(f"  step  0: {r0}")
    print(f"  step  N: {rN}")

    print("\n── MPC control  U0  (N × 9) ──")
    hdr, r0 = _fmt_row(_u_names, U0_np[0])
    _,   rN = _fmt_row(_u_names, U0_np[-1])
    print(" " * len(_PREFIX) + hdr)
    print(f"  step  0: {r0}")
    print(f"  step  N: {rN}")
    print()

    plot_mpc_prediction_state(
        X0_np,
        timestep=0,
        filename="state_mpc_t000.png",
        out_dir=os.path.join(TITA_PATH, "mpc_prediction"),
    )
    plot_mpc_prediction_control(
        U0_np,
        timestep=0,
        filename="control_mpc_t000.png",
        out_dir=os.path.join(TITA_PATH, "mpc_prediction"),
    )

    # ── run WBC and print results ─────────────────────────────────────────
    
    qpos_np = jnp.asarray(data.qpos)[None, :]   # (1, nq)
    qvel_np = jnp.asarray(data.qvel)[None, :]   # (1, nv)

    # ── print WBC-related config parameters ──────────────────────────────
    _wbc_params = [
        "base_body_name", "wheel_radius",
        "Kp_motion", "Kd_motion", "Kp_wheel", "Kd_wheel", "Kp_reg", "Kd_reg",
        "w_qddot", "w_com", "w_lwheel", "w_rwheel", "w_base", "w_eq_roll", "w_eq_dyn", "w_friction",
    ]
    print("\n── WBC parameters ──")
    for _p in _wbc_params:
        print(f"  {_p:20s} = {getattr(config, _p)}")
    print("────────────────────\n")

    print(f"[timing] wbc.run  start={datetime.now().strftime('%H:%M:%S')} ... ", end="", flush=True)
    _t0 = timer()
    tita_state_wbc = build_tita_state(data, contact_ids)
    pl_world_wbc = tita_state_wbc[6:9][None, :]
    pr_world_wbc = tita_state_wbc[9:12][None, :]
    dpl_world_wbc = tita_state_wbc[12:15][None, :]
    dpr_world_wbc = tita_state_wbc[15:18][None, :]
    mpc_state, tau_cmd, qddot, fl, fr = mpc.whole_body_run(
        mpc_state,
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

    _append_llc_row(
        LLC_ROLLOUT_CSV,
        step=0,
        tau_cmd=tau_cmd[0],
        kp=getattr(config, "Kp_motion", 0.0),
        kd=getattr(config, "Kd_motion", 0.0),
    )
    plot_llc(
        LLC_ROLLOUT_CSV,
        out_path=os.path.join(TITA_PATH, "llc","llc_t000.png"),
    )

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

    def step_controller(mpc_state, theta_prev=theta_prev):
        nonlocal counter

        print(f"\n=== step {counter} ===")
        
        qpos = data.qpos.copy()
        qvel = data.qvel.copy()
        tita_state = build_tita_state(data, contact_ids)

        if counter % period == 0:
            foot = jnp.asarray(sim_utils.geom_positions(data, contact_ids))
            command = jnp.asarray(command_handle.mpc_input(config.robot_height))
            contact = jnp.asarray(sim_utils.estimate_contacts(data, contact_ids))
            x0, theta_prev = get_dfip_current_state(mpc, tita_state, theta_prev=theta_prev)
            
            #print(f"pcom:    {x0[0:3]}")
            #print(f"vcom:    {x0[3:6]}")
            #print(f"c_world: {x0[6:9]}")
            #print(f"vcz:     {x0[9]}")
            #print(f"theta:   {x0[10]}")
            #print(f"v:       {x0[11]}")
            #print(f"omega:   {x0[12]}")

            start = timer()
            mpc_state = mpc.run(mpc_state, x0[None, :], counter, config.N)
            stop = timer()
            #print(f"MPC time: {1e3 * (stop - start):.2f} ms")
            print(f"x0 input MPC:  v={float(x0[11]):.4f}  theta={float(x0[10]):.4f}  pcom_z={float(x0[2]):.4f}")
            print(f"MPC output U0: a={float(mpc_state.U0[0,0,0]):.4f}  alpha={float(mpc_state.U0[0,0,2]):.4f}  fl_z={float(mpc_state.U0[0,0,5]):.4f}  fr_z={float(mpc_state.U0[0,0,8]):.4f}")
            print(f"MPC state X0:  v={float(mpc_state.X0[0,0,11]):.4f}  pcom_z={float(mpc_state.X0[0,0,2]):.4f}")
            
            print(f"MPC N output U0: a={float(mpc_state.U0[0,-1,0]):.4f}  alpha={float(mpc_state.U0[0,-1,2]):.4f}  fl_z={float(mpc_state.U0[0,-1,5]):.4f}  fr_z={float(mpc_state.U0[0,-1,8]):.4f}")
            print(f"MPC N state X0:  v={float(mpc_state.X0[0,-1,11]):.4f}  pcom_z={float(mpc_state.X0[0,-1,2]):.4f}")


            if jnp.isnan(mpc_state.U0).any(): 
                MPC_OUTPUT_LIST.append(np.full_like(mpc_state.U0[0, 0], np.nan))
            else:
                MPC_OUTPUT_LIST.append(np.asarray(mpc_state.U0[0, 0]).copy())
            if jnp.isnan(x0).any():
                MPC_INPUT_LIST.append(np.full_like(x0, np.nan))
            else:
                MPC_INPUT_LIST.append(np.asarray(x0).copy())

        pl_world_wbc = tita_state[6:9][None, :]
        pr_world_wbc = tita_state[9:12][None, :]
        dpl_world_wbc = tita_state[12:15][None, :]
        dpr_world_wbc = tita_state[15:18][None, :]

        mpc_state, tau_cmd, qddot, fl, fr = mpc.whole_body_run(
            mpc_state,
            jnp.asarray(qpos)[None, :],
            jnp.asarray(qvel)[None, :],
            counter,
            pl_world_wbc,
            pr_world_wbc,
            dpl_world_wbc,
            dpr_world_wbc,
        )

        touch_floor = _base_touches_floor(model, data, base_body_name=config.base_body_name)

        print(
            #f"timestep: {counter}\n"
            #f"  base_rot_desired [3x3]:\n{np.eye(3)}\n"
            #f"  base_rot_current [3x3]:\n{_quat_wxyz_to_rotmat(np.asarray(qpos[3:7]))}\n"
            f"  tau_cmd: {tau_cmd[0]}\n"
            f"  contact_forces_left (fl): {fl[0]}\n"
            f"  contact_forces_right (fr): {fr[0]}"
        )

        _append_llc_row(
            LLC_ROLLOUT_CSV,
            step=counter + 1,
            tau_cmd=tau_cmd[0],
            kp=getattr(config, "Kp_motion", 0.0),
            kd=getattr(config, "Kd_motion", 0.0),
        )

        if (counter % 50 == 0 and counter < 505) or touch_floor:
            X0_np = np.asarray(mpc_state.X0[0])   # (N+1, 13)
            U0_np = np.asarray(mpc_state.U0[0])  

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

        if np.isnan(tau_cmd[0]).any():
            TORQUE_LIST.append(np.full_like(tau_cmd[0], np.nan))
        else:
            TORQUE_LIST.append(tau_cmd[0].copy())
        
        if np.isnan(fl[0]).any() or np.isnan(fr[0]).any():
            FC_LIST.append( [np.full_like(fl[0], np.nan), np.full_like(fr[0], np.nan)] )
        else:
            FC_LIST.append( [fl[0].copy(), fr[0].copy()] )
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
            if overlay_text is not None:
                viewer.set_texts((None, None, *overlay_text))
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
            out_dir=TITA_PATH,
            filename_state="mpc_input.png",
            filename_u0="mpc_output.png",
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
