
import argparse
import os
import sys
import time
from timeit import default_timer as timer
import jax

jax.config.update("jax_enable_x64", True)

dir_path = os.path.dirname(os.path.realpath(__file__))
# Insert at the front so that THIS working tree wins over any editable/site-packages
# install of mpx (an editable install of another checkout would otherwise shadow it).
sys.path.insert(0, os.path.abspath(os.path.join(dir_path, "..", "..")))
TITA_PATH = os.path.join(dir_path, "plots","tita_validation")

import mpx.config.config_dfcip as config

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")

import jax.numpy as jnp
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np

import mpx.utils.mpc_wrapper_dfcip as mpc_wrapper_dfcip
import mpx.utils.mpc_utils as mpc_utils
import mpx.utils.sim as sim_utils
from mpx.utils.timing import derive_timing, check_model_timestep, describe_timing
from plot_validation import (
    SimLogger,
    plot_velocity_tracking,
    plot_velocity_error,
    plot_com_and_forces,
    plot_all,
)
from plot_rollout_info import (
    _save_sim_video
) 

jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1)

def _build_solve_fn(mpc):
    @jax.jit
    def solve_mpc(x0, command, counter):
        return mpc.run(x0, command, counter)

    return solve_mpc

def _build_wbc(mpc):
    @jax.jit
    def solve_wbc(mpc_state, x0, qpos, qvel, pl_world, pr_world, dpl_world, dpr_world):
        return mpc.whole_body_run(mpc_state, x0, qpos, qvel, pl_world, pr_world, dpl_world, dpr_world)

    return solve_wbc

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
        return v_center_world #+ np.cross(omega_world, rcp_world)

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

@jax.jit
def process_tita_state(pcom, vcom, centers, Rs, radii, feet_vel, theta_prev):
    # get_rCP dentro il jit, per entrambe le ruote
    l_rcp = mpc_utils.get_rCP(Rs[0], radii[0])
    r_rcp = mpc_utils.get_rCP(Rs[1], radii[1])

    pl_world = centers[0] + l_rcp
    pr_world = centers[1] + r_rcp
    dpl_world, dpr_world = feet_vel[0], feet_vel[1]

    tita_state = jnp.concatenate([pcom, vcom, pl_world, pr_world,
                                  dpl_world, dpr_world])

    # --- get_dfip_current_state, identica ma dentro il jit ---
    c_world  = (pl_world + pr_world) / 2.0
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
    w = (dpr_b[0] - dpl_b[0]) / config.d
    v = (dpr_b[0] + dpl_b[0]) / 2.0

    x0 = jnp.concatenate([pcom, vcom, c_world,
                          jnp.array([vc_world[2]]), jnp.array([theta]),
                          jnp.array([v]), jnp.array([w])])
    return tita_state, x0, theta

def gather_raw_state(model, data, base_body_id, contact_ids):
    mujoco.mj_subtreeVel(model, data)
    pcom = data.subtree_com[base_body_id].copy()
    vcom = data.subtree_linvel[base_body_id].copy()

    centers = data.geom_xpos[contact_ids].copy()                    # (2,3)
    Rs = data.geom_xmat[contact_ids].reshape(2, 3, 3).copy()        # (2,3,3)
    radii = model.geom_size[contact_ids, 0].copy()                  # (2,)

    feet_vel = np.zeros((2, 3))
    vel = np.zeros(6)
    for i, g in enumerate(contact_ids):
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_GEOM,
                                 int(g), vel, 0)
        feet_vel[i] = vel[3:6]
    return pcom, vcom, centers, Rs, radii, feet_vel

def build_hud(command_vec, target_vec, actual_vec, help_text):
    """Build the HUD with actual, filtered command and keyboard target."""
    labels = ("vx", "omega", "height")

    title_lines = [
        "",
        *labels,
    ]

    value_lines = [
        " Actual   Command   Target",
    ]

    for actual, command, target in zip(
        actual_vec,
        command_vec,
        target_vec,
    ):
        value_lines.append(
            f"{float(actual):+7.2f}  "
            f"{float(command):+7.2f}  "
            f"{float(target):+7.2f}"
        )

    entries = []

    if help_text is not None:
        entries.append((
            mujoco.mjtFont.mjFONT_NORMAL,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            help_text[0],
            help_text[1],
        ))

    entries.append((
        mujoco.mjtFont.mjFONT_NORMAL,
        mujoco.mjtGridPos.mjGRID_TOPRIGHT,
        "\n".join(title_lines),
        "\n".join(value_lines),
    ))

    return entries

def main(headless=False, steps=500, scene="flat"):

    sim_logger = SimLogger()
    model = mujoco.MjModel.from_xml_path(
        dir_path + f"/../data/tita/scene_{scene}.xml"
    )
    data = mujoco.MjData(model)
    # MuJoCo integrates at simulation_frequency (500 Hz); MPC and WBC are updated
    # at mpc_frequency / whole_body_frequency (100 Hz). The XML timestep must
    # already agree with 1/simulation_frequency (checked, never silently changed).
    timing = derive_timing(config)
    check_model_timestep(model.opt.timestep, timing)
    sim_frequency = float(timing["sim_f"])
    model.opt.timestep = timing["dt_sim"]
    print("[timing] " + describe_timing(timing, int(getattr(config, "mpc_iterations", 1))))

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
    
    contact_ids = sim_utils.geom_ids(model, config.contact_frame)
    mpc = mpc_wrapper_dfcip.BatchedMPCControllerWrapper(config, n_env=1)
    command_handle = sim_utils.KeyboardVelocityCommand(
        vx=0.0, 
        vy=0.0, 
        wz=0.0,
        forward_step=0.1,
        yaw_step=0.2,
        forward_limits=(-10.0, 10.0),
        yaw_limits=(-1.5, 1.5),
    )
    
    solve_mpc = _build_solve_fn(mpc)
    reset_mpc = jax.jit(mpc.reset)
    solve_wbc = _build_wbc(mpc)

    _reset_to_initial_state(model, data)
    mujoco.mj_forward(model, data)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)  
    np.set_printoptions(precision=20, suppress=True)
   
    counter = 0
    theta_prev = 0.0
    tita_state = build_tita_state(model, data, base_body_name="base_link", contact_ids=contact_ids)
    x0, theta_prev = get_dfip_current_state(mpc, tita_state, theta_prev=theta_prev)
    mpc_state = mpc.init_state()
    hud_x0 = x0

    COMMAND_SMOOTHING = 0.02

    # Raw keyboard target: [vx, vz, omega, height]
    target_command = np.asarray(
        command_handle.mpc_wheeled_input(config.com_z_to_track),
        dtype=np.float64,
    )

    # The filtered command starts with zero planar velocity, but with the correct
    # initial height. Starting height from zero would give the MPC an invalid
    # vertical command during startup.
    command = np.array([0.0, 0.0, 0.0, config.com_z_to_track,], dtype=np.float64)
    warm_command = jnp.asarray(command)
    mpc_state, reference = solve_mpc(mpc_state, x0[None, :], warm_command[None, :])
    
    qpos_np = jnp.asarray(data.qpos)[None, :]
    qvel_np = jnp.asarray(data.qvel)[None, :]

    tita_state_wbc = build_tita_state(model, data, base_body_name="base_link", contact_ids=contact_ids)
    pl_world_wbc = tita_state_wbc[6:9][None, :]
    pr_world_wbc = tita_state_wbc[9:12][None, :]
    dpl_world_wbc = tita_state_wbc[12:15][None, :]
    dpr_world_wbc = tita_state_wbc[15:18][None, :]
    mpc_state, tau, qddot, fl, fr, desired = solve_wbc(
        mpc_state,
        x0,
        qpos_np,
        qvel_np,
        pl_world_wbc,
        pr_world_wbc,
        dpl_world_wbc,
        dpr_world_wbc
    )
    tau.block_until_ready()

    period = timing["mpc_period_sim_steps"]        # simulation steps between MPC updates (5)
    wbc_period = timing["wbc_period_sim_steps"]    # simulation steps between WBC updates (5)
    counter = 0
    theta_prev = 0.0
    qddot = np.zeros_like(model.nv)
    mpc_state = mpc.init_state()

    def step_controller(mpc_state, tau, qddot, reference, theta_prev=theta_prev):
        nonlocal counter, command, target_command, hud_x0

        print(f"\n=== step {counter} ===")
        
        init_start = timer()
        # Forward pass (mj_step1) so that every derived quantity read below
        # (geom positions/velocities, subtree CoM) corresponds to the SAME
        # qpos/qvel handed to the WBC. After mj_step the derived quantities still
        # belong to the pre-integration state (one simulation step old), which
        # created spurious ~2 mm / 2 mm/s errors in the WBC task PD terms.
        mujoco.mj_step1(model, data)
        qpos = data.qpos.copy()
        qvel = data.qvel.copy()

        if True: #counter % period == 0:
            base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, config.base_body_name)
            data.xfrc_applied[base_body_id] = 0.0  # reset every step (force does NOT auto-clear)

            force_start = 50
            if force_start <= counter < force_start + 100:                       # 500 frames window
                force_world = np.array([0.0, 0.0, 0.0])    # 100 N forward push (world frame), at the CoM
                data.xfrc_applied[base_body_id, 0:3] = force_world
            
            contact_ids = sim_utils.geom_ids(model, config.contact_frame)

            init_stop = timer()

            raw_start = timer()
            #tita_state = build_tita_state(model, data, base_body_name="base_link", contact_ids=contact_ids)
            #x0, theta_prev = get_dfip_current_state(mpc, tita_state, theta_prev=theta_prev)
            raw = gather_raw_state(model, data, base_body_id, contact_ids)
            raw_stop = timer()
            print(f"[timing] gather_raw_state {1e3 * (raw_stop - raw_start):.2f} ms")
            state_start = timer()
            tita_state, x0, theta_prev = process_tita_state(*raw, theta_prev)
            state_stop = timer()
            hud_x0 = x0
            print(f"[timing] state {1e3 * (state_stop - state_start):.2f} ms")
            #print(F"command: {command[0]:.2f} m/s forward, {command[1]:.2f} m/s lateral, {command[2]:.2f} rad/s angular")

            print(f"[timing] init time: {1e3 * (init_stop - init_start):.2f} ms")
            print(f"---- {counter}%{period} = {counter % period} ----")
            target_command = np.asarray(
                command_handle.mpc_wheeled_input(config.com_z_to_track),
                dtype=np.float64,
            )
            if counter % period == 0:
                command += COMMAND_SMOOTHING * (
                    target_command - command
                )

                # In mjx_policy_tita the height target is not smoothed.
                command[3] = target_command[3]

                mpc_start = timer()
                mpc_state, reference = solve_mpc(mpc_state, x0[None, :], command[None, :])
                jax.block_until_ready((
                    reference,
                    mpc_state.X_prediction,
                    mpc_state.U_prediction,
                    mpc_state.sol.a,
                    mpc_state.sol.ac_z,
                    mpc_state.sol.alpha,
                    mpc_state.sol.grf,
                ))
                mpc_stop = timer()
                print(f"[timing] MPC time: {1e3 * (mpc_stop - mpc_start):.2f} ms")

            if counter % wbc_period == 0:
                # WBC update: uses the latest MPC first-stage control and the
                # current measured state. tau and qddot are then held for
                # wbc_period simulation steps.
                pl_world_wbc = tita_state[6:9][None, :]
                pr_world_wbc = tita_state[9:12][None, :]
                dpl_world_wbc = tita_state[12:15][None, :]
                dpr_world_wbc = tita_state[15:18][None, :]

                wbc_start = timer()
                mpc_state, tau, qddot, fl, fr, desired = solve_wbc(
                    mpc_state,
                    x0,
                    jnp.asarray(qpos)[None, :],
                    jnp.asarray(qvel)[None, :],
                    pl_world_wbc,
                    pr_world_wbc,
                    dpl_world_wbc,
                    dpr_world_wbc,
                )
                jax.block_until_ready((
                    tau,
                    qddot,
                    fl,
                    fr,
                    desired,
                ))
                wbc_stop = timer()
                print(f"[timing] wbc.run {1e3 * (wbc_stop - wbc_start):.2f} ms")
            logger_start = timer()
            sim_logger.append(
                t=counter, 
                model=model,
                data=data,
                tita_state=tita_state,
                x0=x0,
                cmd=command,
                ext_force=data.xfrc_applied[base_body_id, 0:3],
                fl=None,
                fr=None,
            )
            logger_stop = timer()
            print(f"[timing] logger.append {1e3 * (logger_stop - logger_start):.2f} ms")
            #print(f"timestep: {counter}, force: {data.xfrc_applied[base_body_id, 0:3]}")

        extra_start = timer()
        touch_floor = _base_touches_floor(model, data, base_body_name=config.base_body_name)

        # Outer joint PD on top of the held WBC torque, evaluated at every
        # simulation step: it predicts the joint motion ONE SIMULATION STEP ahead
        # (dt_sim = 0.002 s, not dt_wbc) from the current joint state using the
        # held WBC joint accelerations.
        dt_sim = model.opt.timestep
        qpos_joint  = qpos[7:]
        qvel_joint  = qvel[6:]
        qddot_joint = np.asarray(qddot[0, 6:])   # JAX -> numpy once per step

        dq_desired = qvel_joint + qddot_joint * dt_sim
        q_desired  = qpos_joint + qvel_joint * dt_sim + 0.5 * qddot_joint * dt_sim**2

        p_ctrl = 35.0 * (q_desired - qpos_joint)
        d_ctrl = 10.0 * (dq_desired - qvel_joint)

        p_ctrl[[3, 7]] = 0.0        # wheels: no position term

        total_ctrl = p_ctrl + d_ctrl + np.asarray(tau[0])
        data.ctrl = total_ctrl

        mujoco.mj_step2(model, data)   # integrate (mj_step1 was called at the top of this step)
        update_com_tracking_camera(
            _sim_cam,
            model,
            data,
            _sim_base_body_id,
            alpha=0.10,
        )
        _renderer.update_scene(data, camera=_sim_cam)
        if counter % 2 == 0:  
            _sim_frames.append(_renderer.render().copy())
        counter += 1
        extra_stop = timer()
        print(f"[timing] extra time: {1e3 * (extra_stop - extra_start):.2f} ms")
        return mpc_state, tau, qddot, reference, theta_prev, touch_floor
    
    def finalize_outputs():
        print("\n[finalize] Saving outputs...")

        try:
            # One frame every 2 simulation steps = 250 frames per simulated second at 500 Hz.
            video_fps = int(sim_frequency / 2)

            _save_sim_video(
                video_dir=TITA_PATH,
                video_fps=video_fps,
                frames=_sim_frames,
                slowdown_factor=1.0,
                name_video="simulation_video.mp4"
            )
            _save_sim_video(
                video_dir=TITA_PATH,
                video_fps=video_fps,
                frames=_sim_frames,
                slowdown_factor=4.0,
                name_video="simulation_video_slow4.mp4"
            )
            _save_sim_video(
                video_dir=TITA_PATH,
                video_fps=video_fps,
                frames=_sim_frames,
                slowdown_factor=15.0,
                name_video="simulation_video_slow15.mp4"
            )
        except Exception as e:
            print(f"[finalize] failed to save video: {e}")

        try:
            plot_all(
                sim_logger,
                save_path=os.path.join(TITA_PATH, "plots"),
                show=False,
            )
        except Exception as e:
            print(f"[finalize] failed to save plots: {e}")

        try:
            _renderer.close()
        except Exception:
            pass

    try:
        if headless:
            for _ in range(steps):
                mpc_state, tau, qddot, reference, theta_prev, touch_floor = step_controller(mpc_state, tau, qddot, reference, theta_prev=theta_prev)
                if touch_floor:
                    print(f"Base touched the floor at step {counter}. Ending simulation.")
                    break
            return

        with mujoco.viewer.launch_passive(
            model,
            data,
            key_callback=command_handle.key_callback,
        ) as viewer:
            viewer.cam.distance *= 5.5
            viewer.sync()
            while viewer.is_running():
                tic = timer()
                overlay = command_handle.consume_overlay_text()
                if overlay is not None:
                    help_cache = overlay

                help_text = (
                    f"{help_cache[0]} | PgUp/PgDown: z_com_ref",
                    help_cache[1],
                )

                actual_vec = np.array([
                    float(hud_x0[11]),  # Actual vx
                    float(hud_x0[12]),  # Actual omega
                    float(hud_x0[2]),   # Actual CoM height
                ])

                command_vec = np.array([
                    float(command[0]),  # Filtered vx
                    float(command[2]),  # Filtered omega
                    float(command[3]),  # Commanded height
                ])

                target_vec = np.array([
                    float(target_command[0]),  # Keyboard target vx
                    float(target_command[2]),  # Keyboard target omega
                    float(target_command[3]),  # Keyboard target height
                ])

                viewer.set_texts(build_hud(
                    command_vec=command_vec,
                    target_vec=target_vec,
                    actual_vec=actual_vec,
                    help_text=help_text,
                ))

                start_step = timer()
                mpc_state, tau, qddot, reference, theta_prev, touch_floor = step_controller(mpc_state, tau, qddot, reference, theta_prev=theta_prev)
                end_step = timer()
                step_time = end_step - start_step
                print(f"Step time: {1e3 * step_time:.2f} ms")

                toc = timer()
                if toc - tic < model.opt.timestep:
                    time.sleep(model.opt.timestep - (toc - tic))

                if touch_floor:
                    print(f"Base touched the floor at step {counter}. Ending simulation.")
                    break

                viewer.sync()
    except KeyboardInterrupt:
        print("\n[interrupt] Ctrl+C received. Finalizing outputs...")
    finally:
        finalize_outputs()
                
       # _save_sim_video(video_dir=TITA_PATH, video_fps=int(sim_frequency), frames=_sim_frames, slowdown_factor=1.0)
       # plot_all(sim_logger, save_path=os.path.join(TITA_PATH, "plots"), show=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--scene", type=str, default="flat")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    if os.environ.get("DISPLAY") is None or os.environ.get("DISPLAY") == "":
        print("[WARN] No DISPLAY detected: forcing --headless mode (no viewer)")
        args.headless = True

    main(
        headless=args.headless,
        steps=args.steps,
        scene=args.scene,
    )
    
