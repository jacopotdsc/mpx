import argparse
import csv
from pyexpat import model
from pyexpat import model
import shutil
import os
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" 

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
import jax
import mpx.config.config_dfcip as config

import jaxlib

print("JAX version:", jax.__version__)
print("JAXLIB version:", jaxlib.__version__)
print("JAX devices:", jax.devices())
print("JAX default backend:", jax.default_backend())
print("JAX x64 enabled:", jax.config.read("jax_enable_x64"))
print("XLA_FLAGS:", os.environ.get("XLA_FLAGS"))
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("JAX_PLATFORM_NAME:", os.environ.get("JAX_PLATFORM_NAME"))

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, "..")))
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")


TITA_PATH = os.path.join(dir_path, "plots","tita_outputs")
os.makedirs(TITA_PATH, exist_ok=True)

MAX_STEPS = round(int( (2 + config.T_TRAJECTORY ) / config.dt_ref) )
DEFAULT_VIDEO_SLOWDOWN_FACTOR = 4.0
LLC_ROLLOUT_CSV = os.path.join(TITA_PATH, "rollout_info_llc.csv")
TORQUE_LIST = []
FC_LIST = []
MPC_INPUT_LIST = []
MPC_OUTPUT_LIST = []
X_DES_LIST = []
U_DES_LIST = []
WBC_DESIRED_LIST = []
WBC_CURRENT_LIST = []
MPC_PREDICTION_FRAMES = {}
CMD = [0.5, 0.0, 0.0] 

import jax.numpy as jnp
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np

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

def update_com_tracking_camera(cam, model, data, base_body_id, alpha=0.10):
    """
    Camera free che segue il CoM del robot.
    alpha basso = tracking smooth.
    alpha = 1.0 = tracking istantaneo.
    """
    mujoco.mj_subtreeVel(model, data)

    com = np.asarray(data.subtree_com[base_body_id]).copy()

    # opzionale: blocca un po' la quota del lookat per evitare tremolio verticale
    # com[2] = 0.40

    cam.lookat[:] = (1.0 - alpha) * cam.lookat + alpha * com
    
def store_mpc_prediction_frame(
    storage: dict,
    timeframe: int,
    X0_np,
    x_ref,
    U0_np,
    u_ref,
    cmd,
):
    """
    storage[timeframe] = {
        "X0_np": ...,
        "x_ref": ...,
        "U0_np": ...,
        "u_ref": ...,
        "cmd": ...
    }
    """

    storage[int(timeframe)] = {
        "X0_np": np.asarray(X0_np).copy(),
        "x_ref": np.asarray(x_ref).copy(),
        "U0_np": np.asarray(U0_np).copy(),
        "u_ref": np.asarray(u_ref).copy(),
        "cmd": np.asarray(cmd).copy(),
    }

def plot_mpc_prediction_frames(
    prediction_frames: dict,
    out_dir: str,
):
    """
    Generate MPC state/control prediction plots for every saved timeframe.

    Expected structure:

    prediction_frames[timeframe] = {
        "X0_np": (N+1, nx),
        "x_ref": (N+1, nx),
        "U0_np": (N, nu) or (N+1, nu),
        "u_ref": (N+1, nu),
        "cmd": command array
    }
    """

    os.makedirs(out_dir, exist_ok=True)

    if len(prediction_frames) == 0:
        print("[plot_mpc_prediction_frames] no frames to plot.")
        return

    for timeframe in sorted(prediction_frames.keys()):
        item = prediction_frames[timeframe]

        X0_np = item["X0_np"]
        x_ref = item["x_ref"]
        U0_np = item["U0_np"]
        u_ref = item["u_ref"]
        cmd   = item["cmd"]

        plot_mpc_prediction_state(
            X0_np,
            x_ref=x_ref,
            cmd=cmd,
            timestep=timeframe,
            filename=f"state_mpc_t{timeframe:06d}.png",
            out_dir=os.path.join(out_dir, "state"),
        )

        plot_mpc_prediction_control(
            U0_np,
            u_ref=u_ref,
            cmd=cmd,
            timestep=timeframe,
            filename=f"control_mpc_t{timeframe:06d}.png",
            out_dir=os.path.join(out_dir, "control"),
        )

    print(f"[plot_mpc_prediction_frames] saved {len(prediction_frames)} frames in {out_dir}")

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

def debug_dfcip_ref0_cost_terms(reference, W, wheel_offset, N=None, prefix="[ref0 cost]"):
    """
    Debug dei termini del costo DFCIP valutati su:
        x = x_ref[0]
        u = u_ref[0]

    reference shape:
        (N+1, nx+nu) oppure (B, N+1, nx+nu)

    W shape:
        (15, 15)
    """

    ref = np.asarray(reference)

    if ref.ndim == 3:
        ref = ref[0]

    W_np = np.asarray(W)

    x_ref = ref[:, :13]
    u_ref = ref[:, 13:22]

    x = x_ref[0]
    u = u_ref[0]

    pcom = x[0:3]
    vcom = x[3:6]
    c = x[6:9]

    vc_z = x[9]
    theta = x[10]
    v = x[11]
    omega = x[12]

    a = u[0]
    ac_z = u[1]
    alpha = u[2]

    fl = u[3:6]
    fr = u[6:9]

    xr = x_ref[0]
    ur = u_ref[0]

    vector_off = np.array([0.0, wheel_offset / 2.0, 0.0])

    R = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta),  np.cos(theta), 0.0],
        [0.0,            0.0,           1.0],
    ])

    pl = c + R @ vector_off
    pr = c - R @ vector_off

    h_contact = vc_z
    h_moment = np.cross(pl - pcom, fl) + np.cross(pr - pcom, fr)
    h_stability = pcom[:2] - c[:2]

    h_fz_raw = np.array([
        min(fl[2], 0.0),
        min(fr[2], 0.0),
    ])

    h_fz_cost_style = min(fl[2], 0.0) ** 2 + min(fr[2], 0.0) ** 2

    w_pcomxy = W_np[0, 0]
    w_pcomz = W_np[1, 1]
    w_vcomxy = W_np[2, 2]
    w_vcomz = W_np[3, 3]
    w_c = W_np[4, 4]
    w_vcz = W_np[5, 5]
    w_theta = W_np[6, 6]
    w_v = W_np[7, 7]
    w_w = W_np[8, 8]

    w_a = W_np[9, 9]
    w_ac_z = W_np[10, 10]
    w_alpha = W_np[11, 11]

    w_fcxy = W_np[12, 12]
    w_fcz = W_np[13, 13]
    w_eq = W_np[14, 14]

    terms = {
        "pcom_xy": 0.5 * w_pcomxy * np.sum((pcom[:2] - xr[0:2]) ** 2),
        "pcom_z":  0.5 * w_pcomz  *        (pcom[2]  - xr[2])   ** 2,

        "vcom_xy": 0.5 * w_vcomxy * np.sum((vcom[:2] - xr[3:5]) ** 2),
        "vcom_z":  0.5 * w_vcomz  *        (vcom[2]  - xr[5])   ** 2,

        "c_xyz":   0.5 * w_c      * np.sum((c        - xr[6:9]) ** 2),
        "vcz":     0.5 * w_vcz    *        (vc_z     - xr[9])   ** 2,
        "theta":   0.5 * w_theta  *        (theta    - xr[10])  ** 2,
        "v":       0.5 * w_v      *        (v        - xr[11])  ** 2,
        "omega":   0.5 * w_w      *        (omega    - xr[12])  ** 2,

        "a":       0.5 * w_a      *        (a        - ur[0])   ** 2,
        "ac_z":    0.5 * w_ac_z   *        (ac_z     - ur[1])   ** 2,
        "alpha":   0.5 * w_alpha  *        (alpha    - ur[2])   ** 2,

        "fl_xy":   0.5 * w_fcxy   * np.sum((fl[:2]   - ur[3:5]) ** 2),
        "fl_z":    0.5 * w_fcz    *        (fl[2]    - ur[5])   ** 2,
        "fr_xy":   0.5 * w_fcxy   * np.sum((fr[:2]   - ur[6:8]) ** 2),
        "fr_z":    0.5 * w_fcz    *        (fr[2]    - ur[8])   ** 2,

        "h_contact":   0.5 * w_eq * h_contact ** 2,
        "h_moment":    0.5 * w_eq * np.dot(h_moment, h_moment),
        "h_stability": 0.5 * w_eq * np.dot(h_stability, h_stability),
        "h_fz_raw_unused": 0.5 * w_eq * h_fz_cost_style,
    }

    running_total = (
        terms["pcom_xy"]
        + terms["pcom_z"]
        + terms["vcom_xy"]
        + terms["vcom_z"]
        + terms["c_xyz"]
        + terms["vcz"]
        + terms["theta"]
        + terms["v"]
        + terms["omega"]
        + terms["a"]
        + terms["ac_z"]
        + terms["alpha"]
        + terms["fl_xy"]
        + terms["fl_z"]
        + terms["fr_xy"]
        + terms["fr_z"]
        + terms["h_contact"]
        + terms["h_moment"]
    )

    terminal_total = (
        terms["pcom_xy"]
        + terms["pcom_z"]
        + terms["vcom_xy"]
        + terms["vcom_z"]
        + terms["c_xyz"]
        + terms["vcz"]
        + terms["theta"]
        + terms["v"]
        + terms["omega"]
        + terms["h_contact"]
        + terms["h_stability"]
    )

    def fmt_vec(v):
        return np.array2string(np.asarray(v), precision=6, suppress_small=True)

    print("\n" + "═" * 80)
    print(f"{prefix}")
    print("═" * 80)

    print("\n── x_ref[0] ──")
    print(f"pcom      = {fmt_vec(pcom)}")
    print(f"vcom      = {fmt_vec(vcom)}")
    print(f"c         = {fmt_vec(c)}")
    print(f"vc_z      = {vc_z:.8f}")
    print(f"theta     = {theta:.8f}")
    print(f"v         = {v:.8f}")
    print(f"omega     = {omega:.8f}")

    print("\n── u_ref[0] ──")
    print(f"a         = {a:.8f}")
    print(f"ac_z      = {ac_z:.8f}")
    print(f"alpha     = {alpha:.8f}")
    print(f"fl        = {fmt_vec(fl)}")
    print(f"fr        = {fmt_vec(fr)}")
    print(f"fl + fr   = {fmt_vec(fl + fr)}")

    print("\n── contact points from x_ref[0] ──")
    print(f"pl        = {fmt_vec(pl)}")
    print(f"pr        = {fmt_vec(pr)}")
    print(f"pcom-c    = {fmt_vec(pcom - c)}")
    print(f"pcom_xy-c_xy = {fmt_vec(h_stability)}")

    print("\n── constraints / residuals at x_ref[0], u_ref[0] ──")
    print(f"h_contact       = {h_contact:.10f}")
    print(f"h_moment        = {fmt_vec(h_moment)}")
    print(f"||h_moment||    = {np.linalg.norm(h_moment):.10f}")
    print(f"h_stability     = {fmt_vec(h_stability)}")
    print(f"||h_stability|| = {np.linalg.norm(h_stability):.10f}")
    print(f"h_fz_raw        = {fmt_vec(h_fz_raw)}")
    print(f"h_fz_cost_style = {h_fz_cost_style:.10f}")

    print("\n── weights ──")
    print(f"w_pcomxy = {w_pcomxy:.3e}")
    print(f"w_pcomz  = {w_pcomz:.3e}")
    print(f"w_vcomxy = {w_vcomxy:.3e}")
    print(f"w_vcomz  = {w_vcomz:.3e}")
    print(f"w_c      = {w_c:.3e}")
    print(f"w_vcz    = {w_vcz:.3e}")
    print(f"w_theta  = {w_theta:.3e}")
    print(f"w_v      = {w_v:.3e}")
    print(f"w_omega  = {w_w:.3e}")
    print(f"w_a      = {w_a:.3e}")
    print(f"w_ac_z   = {w_ac_z:.3e}")
    print(f"w_alpha  = {w_alpha:.3e}")
    print(f"w_fcxy   = {w_fcxy:.3e}")
    print(f"w_fcz    = {w_fcz:.3e}")
    print(f"w_eq     = {w_eq:.3e}")

    print("\n── individual cost terms at x_ref[0], u_ref[0] ──")
    for k, val in terms.items():
        print(f"{k:18s}: {val:.10e}")

    print("\n── totals ──")
    print(f"running_total  = {running_total:.10e}")
    print(f"terminal_total = {terminal_total:.10e}")

    print("═" * 80 + "\n")

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
    mujoco.mj_subtreeVel(model, data)
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

    # body da usare per il CoM
    _sim_base_body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        config.base_body_name,
    )

    # inizializza lookat sul CoM
    mujoco.mj_subtreeVel(model, data)
    _sim_cam.lookat[:] = np.asarray(data.subtree_com[_sim_base_body_id])
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
    
    mpc_state = mpc.init_state()

    # ── print MPC config parameters ──────────────────────────────────────
    _mpc_params = [
        "dt_mpc", "dt_ref", "N", "mpc_frequency", "grav", "whole_body_frequency",
        "duty_factor", "step_freq", "step_height", "robot_height", "clearence_speed",
        "mu", "mass",
        "w_pcomxy", "w_pcomz", "w_vcomxy", "w_vcomz", "w_c", "w_vcz",
        "w_theta", "w_v", "w_omega",
        "w_a", "w_ac_z", "w_alpha", "w_fcxy", "w_fcz", "w_eq",
    ]
    print("\n── MPC parameters ──")
    for _p in _mpc_params:
        print(f"  {_p:20s} = {getattr(config, _p)}")
    print("────────────────────\n")

    print("[timing] mpc.run  ... ", end="", flush=True)
    cmd = command_handle.mpc_wheeled_input(config.com_z_to_track) 
    cmd = jnp.asarray(CMD)[None, :]
    _t0 = timer()
    mpc_state, reference = mpc.run(mpc_state, x0[None, :], cmd, counter, config.N)
    jax.block_until_ready(mpc_state.U0_shifted)
    _dt = timer() - _t0
    print(f"{int(_dt // 60)}m {_dt % 60:.1f}s")

    # ── save MPC prediction plots at timestep 0 ──────────────────────────
    X0_np = x0
    U0_np = np.concatenate([np.asarray(mpc_state.sol.a)[:, None], np.asarray(mpc_state.sol.ac_z)[:, None], np.asarray(mpc_state.sol.alpha)[:, None], np.asarray(mpc_state.sol.grf)], axis=1).copy()
    XN_np = np.asarray(mpc_state.X_prediction[0][-1]).copy()  
    UN_np = np.asarray(mpc_state.U_prediction[0][-1]).copy()
    np.set_printoptions(precision=4, suppress=True, linewidth=200)

    _x_names = ["pcom_x", "pcom_y", "pcom_z", "dpcom_x", "dpcom_y", "dpcom_z",
                "c_x",    "c_y",    "c_z",    "vcz",     "θ",       "v",       "ω"]
    _u_names = ["a", "acz", "α", "Fl_x", "Fl_y", "Fl_z", "Fr_x", "Fr_y", "Fr_z"]

    _PREFIX = "  step  0: "   # 11 chars — aligns header with data rows

    def clean_plots_prediction(path: str):
        n_removed = 0
        for name in os.listdir(path):
            item_path = os.path.join(path, name)

            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.remove(item_path)
                n_removed += 1
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
                n_removed += 1
        
        print(f"[clean] cleared directory: {path}  |  removed {n_removed} items")

    clean_plots_prediction(os.path.join(TITA_PATH, "mpc_prediction"))

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

    plot_mpc_prediction_state(
        mpc_state.X_prediction[0],
        x_ref = reference[0][:, :config.nx],
        cmd=cmd,
        timestep=0,
        filename="dummy_state_mpc_t000.png",
        out_dir=os.path.join(TITA_PATH, "mpc_prediction"),
    )
    plot_mpc_prediction_control(
        mpc_state.U_prediction[0],
        u_ref = reference[0][:, config.nx:],
        timestep=0,
        filename="dummy_control_mpc_t000.png",
        out_dir=os.path.join(TITA_PATH, "mpc_prediction"),
    )

    # ── run WBC and print results ─────────────────────────────────────────
    
    qpos_np = jnp.asarray(data.qpos)[None, :]   # (1, nq)
    qvel_np = jnp.asarray(data.qvel)[None, :]   # (1, nv)

    # ── print WBC-related config parameters ──────────────────────────────
    _wbc_params = [
        "base_body_name", "wheel_radius",
        "Kp_motion", "Kd_motion", "Kp_wheel", "Kd_wheel", "Kp_reg", "Kd_reg",
        "w_qddot", "w_com", "w_lwheel", "w_rwheel", "w_base",
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
        pl_world_wbc,
        pr_world_wbc,
        dpl_world_wbc,
        dpr_world_wbc
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
    _snapshot_config_to_txt(TITA_PATH)
    counter = 0
    theta_prev = 0.0
    mpc_state = mpc.init_state()

    def do_print(*args, **kwargs):
        debug_print = True
        if debug_print:
            print(*args, **kwargs)

    def step_controller(mpc_state, reference, cmd, theta_prev=theta_prev):
        nonlocal counter

        print(f"\n=== step {counter} ===")
        print(F"command: {cmd[0,0]:.2f} m/s forward, {cmd[0,1]:.2f} m/s lateral, {cmd[0,2]:.2f} rad/s angular")
        
        qpos = data.qpos.copy()
        qvel = data.qvel.copy()

        if True:
            contact_ids = sim_utils.geom_ids(model, config.contact_frame)
            tita_state = build_tita_state(model, data, base_body_name="base_link", contact_ids=contact_ids)
            x0, theta_prev = get_dfip_current_state(mpc, tita_state, theta_prev=theta_prev)

            if counter % 1 == 0:
                print(f"calling mpc.run at step {counter} with x0={x0}")
                start_time_mpc = timer()
                mpc_state, reference = mpc.run(mpc_state, x0[None, :], cmd, counter, config.N)
                end_time_mpc = timer()
                mpc_duration = end_time_mpc - start_time_mpc
                print(f"[timing] mpc.run {mpc_duration * 1000:.2f} ms")

            pl_world_wbc = tita_state[6:9][None, :]
            pr_world_wbc = tita_state[9:12][None, :]
            dpl_world_wbc = tita_state[12:15][None, :]
            dpr_world_wbc = tita_state[15:18][None, :]

            start_time_wbc = timer()
            mpc_state, tau_cmd, qddot, fl, fr, desired = mpc.whole_body_run(
                mpc_state,
                x0,
                jnp.asarray(qpos)[None, :],
                jnp.asarray(qvel)[None, :],
                pl_world_wbc,
                pr_world_wbc,
                dpl_world_wbc,
                dpr_world_wbc,
            )
            
            end_time_wbc = timer()
            wbc_duration = end_time_wbc - start_time_wbc
            print(f"[timing] wbc.run {wbc_duration * 1000:.2f} ms")
            
            tau_to_apply = tau_cmd[0].copy()

            #debug_dfcip_ref0_cost_terms(
            #    reference=reference,
            #    W=config.W,
            #    wheel_offset=config.d,
            #    N=config.N,
            #    prefix=f"[ref0 cost] step {counter}",
            #)
            
            def _f(v, d=4):
                s = f"{float(v):.{d}f}"
                return s if float(v) < 0 else f" {s}"
            
            do_print(
                f"[tita_state] "
                f"pcom=({_f(tita_state[0])},{_f(tita_state[1])},{_f(tita_state[2])}) "
                f"vcom=({_f(tita_state[3])},{_f(tita_state[4])},{_f(tita_state[5])}) "
                f"\n             "
                f"pl=({_f(tita_state[6])},{_f(tita_state[7])},{_f(tita_state[8])}) "
                f"pr=({_f(tita_state[9])},{_f(tita_state[10])},{_f(tita_state[11])}) "
                f"\n             "
                f"dpl=({_f(tita_state[12])},{_f(tita_state[13])},{_f(tita_state[14])}) "
                f"dpr=({_f(tita_state[15])},{_f(tita_state[16])},{_f(tita_state[17])})"
            )
            do_print(
                f"[x0] "
                f" pc=({_f(x0[0])},{_f(x0[1])},{_f(x0[2])}) "
                f" vc=({_f(x0[3])},{_f(x0[4])},{_f(x0[5])}) "
                f"\n      cw=({_f(x0[6])},{_f(x0[7])},{_f(x0[8])}) "
                f" vcz={_f(x0[9])}\n      th={_f(x0[10],4)} v={_f(x0[11],4)} w={_f(x0[12],4)}"
            )
            do_print(
                f"[pred_X0] "
                f" pc=({_f(mpc_state.X_prediction[0,0,0])},{_f(mpc_state.X_prediction[0,0,1])},{_f(mpc_state.X_prediction[0,0,2])}) "
                f" vc=({_f(mpc_state.X_prediction[0,0,3])},{_f(mpc_state.X_prediction[0,0,4])},{_f(mpc_state.X_prediction[0,0,5])}) "
                f"\n      cw=({_f(mpc_state.X_prediction[0,0,6])},{_f(mpc_state.X_prediction[0,0,7])},{_f(mpc_state.X_prediction[0,0,8])}) "
                f" vcz={_f(mpc_state.X_prediction[0,0,9])}\n      th={_f(mpc_state.X_prediction[0,0,10],4)} v={_f(mpc_state.X_prediction[0,0,11],4)} w={_f(mpc_state.X_prediction[0,0,12],4)}"
            )
            do_print(
                f"[pred_U0] "
                f" a, acz, alpha=({_f(mpc_state.U_prediction[0,0,0])},{_f(mpc_state.U_prediction[0,0,1])},{_f(mpc_state.U_prediction[0,0,2])}) "
                f"\n      fl=({_f(mpc_state.U_prediction[0,0,3])},{_f(mpc_state.U_prediction[0,0,4])},{_f(mpc_state.U_prediction[0,0,5])}) "
                f"\n      fr=({_f(mpc_state.U_prediction[0,0,6])},{_f(mpc_state.U_prediction[0,0,7])},{_f(mpc_state.U_prediction[0,0,8])}) "
            )
            do_print(
                f"[pred_XN] "
                f" pc=({_f(mpc_state.X_prediction[0, -1,0])},{_f(mpc_state.X_prediction[0, -1,1])},{_f(mpc_state.X_prediction[0, -1,2])}) "
                f" vc=({_f(mpc_state.X_prediction[0, -1,3])},{_f(mpc_state.X_prediction[0, -1,4])},{_f(mpc_state.X_prediction[0, -1,5])}) "
                f"\n      cw=({_f(mpc_state.X_prediction[0, -1,6])},{_f(mpc_state.X_prediction[0, -1,7])},{_f(mpc_state.X_prediction[0, -1,8])}) "
                f" vcz={_f(mpc_state.X_prediction[0, -1,9])}\n      th={_f(mpc_state.X_prediction[0, -1,10],4)} v={_f(mpc_state.X_prediction[0, -1,11],4)} w={_f(mpc_state.X_prediction[0, -1,12],4)}"
            )
            do_print(
                f"[pred_U0] "
                f" a, acz, alpha=({_f(mpc_state.U_prediction[0,-1,0])},{_f(mpc_state.U_prediction[0,-1,1])},{_f(mpc_state.U_prediction[0,-1,2])}) "
                f"\n      fl=({_f(mpc_state.U_prediction[0,-1,3])},{_f(mpc_state.U_prediction[0,-1,4])},{_f(mpc_state.U_prediction[0,-1,5])}) "
                f"\n      fr=({_f(mpc_state.U_prediction[0,-1,6])},{_f(mpc_state.U_prediction[0,-1,7])},{_f(mpc_state.U_prediction[0,-1,8])}) "
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



            c = build_wbc_current_vector(
                model=model,
                data=data,
                contact_ids=contact_ids,
                mpc_utils=mpc_utils,
                nj=model.nv - 6,
            )
            
            TORQUE_LIST.append(tau_to_apply)
            FC_LIST.append( [fl[0].copy(), fr[0].copy()] )
            WBC_DESIRED_LIST.append(d.copy())
            WBC_CURRENT_LIST.append(c.copy())

            #store_mpc_prediction_frame(
            #    MPC_PREDICTION_FRAMES,
            #    timeframe=counter,
            #    X0_np=mpc_state.X0_shifted[0],
            #    x_ref=reference[0][:, :config.nx],
            #    U0_np=mpc_state.U0_shifted[0],
            #    u_ref=reference[0][:, config.nx:],
            #    cmd=cmd,
            #)
            

        touch_floor = _base_touches_floor(model, data, base_body_name=config.base_body_name)
        
        if (counter % 100 == 0) or touch_floor:
            print(f"[step {counter}] touch_floor={touch_floor}, saving MPC prediction plots...")
            X0_np = np.asarray(mpc_state.X_prediction[0])   # (N+1, 13)
            U0_np = np.asarray(mpc_state.U_prediction[0])  

            plot_mpc_prediction_state(
                X0_np,
                x_ref=reference[0][:, :config.nx],
                cmd=cmd,
                timestep=counter,
                filename=f"state_mpc_t{counter:03d}.png",
                out_dir=os.path.join(TITA_PATH, "mpc_prediction"),
            )
            plot_mpc_prediction_control(
                U0_np,
                u_ref=reference[0][:, config.nx:],
                cmd=cmd,
                timestep=counter,
                filename=f"control_mpc_t{counter:03d}.png",
                out_dir=os.path.join(TITA_PATH, "mpc_prediction"),
            )

            plot_mpc_state_and_output(
                x0_list=MPC_INPUT_LIST,
                u0_list=MPC_OUTPUT_LIST,
                cmd=CMD,
                x_ref_list=None,
                u_ref_list=None,
                out_dir=TITA_PATH,
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
        update_com_tracking_camera(
            _sim_cam,
            model,
            data,
            _sim_base_body_id,
            alpha=0.10,
        )
        _renderer.update_scene(data, camera=_sim_cam)
        if counter % 2 == 0:  # record every 2nd frame to reduce video size
            _sim_frames.append(_renderer.render().copy())
        counter += 1
        return mpc_state, reference, theta_prev, touch_floor
    
    def _save_sim_video() -> None:
        print("Saving simulation video... ", end="\n", flush=True)
        try:
            import imageio
        except ImportError:
            print("imageio not installed, skipping video saving.")
            return
        if not _sim_frames:
            print("No frames captured, skipping video saving.")
            return
        print(f"Captured {len(_sim_frames)} frames at {video_fps} fps.")
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
            cmd = command_handle.get_command()
            mpc_state, reference, theta_prev, touch_floor = step_controller(mpc_state, reference, cmd, theta_prev=theta_prev)
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
        start_time = timer()
        while viewer.is_running():
            overlay_text = command_handle.consume_overlay_text()
            tic = timer()

            texts = []

            if overlay_text is not None:
                texts.append((None, None, *overlay_text))

            texts.append((
                None,
                None,
                "Info",
                f"step: {counter}/{MAX_STEPS}\nsim time: {data.time:.3f} s"
            ))

            viewer.set_texts(texts)

            cmd = command_handle.mpc_wheeled_input(config.com_z_to_track)  # dummy command for now
            cmd = jnp.asarray(CMD)[None, :]
            mpc_state, reference, theta_prev, touch_floor = step_controller(mpc_state, reference, cmd, theta_prev=theta_prev)

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

    end_time = timer()
    step_duration = end_time - start_time
    print(f"[timing] step {counter}/{MAX_STEPS}: {step_duration // 60:.0f}m {step_duration % 60:.1f}s")
    
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
        
        plot_torques_and_contacts(
            torques=TORQUE_LIST,
            contact_forces=FC_LIST,
            out_dir=TITA_PATH,
        )
        plot_mpc_state_and_output(
            x0_list=MPC_INPUT_LIST,
            u0_list=MPC_OUTPUT_LIST,
            cmd=CMD,
            x_ref_list=None,
            u_ref_list=None,
            out_dir=TITA_PATH,
        )

        plot_wbc_desired(
            desired_list=WBC_DESIRED_LIST,
            current_list=WBC_CURRENT_LIST,
            mpc_utils=mpc_utils,
            nj=model.nv - 6,
            out_dir=TITA_PATH,
        )

        if len(MPC_PREDICTION_FRAMES.keys()) > 0:
            plot_mpc_prediction_frames(
                MPC_PREDICTION_FRAMES,
                out_dir=os.path.join(TITA_PATH, "mpc_prediction"),
            )

    try:

        start_sim = timer()
        main(
            headless=args.headless,
            steps=args.steps,
            scene=args.scene,
        )
    except Exception as e:
        do_plots()
    do_plots()
