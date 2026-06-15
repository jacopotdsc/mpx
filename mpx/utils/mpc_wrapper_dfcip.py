import os
import jax

import jax.numpy as jnp
from functools import partial
from flax import struct     
import mpx.utils.mpc_utils as mpc_utils
import mpx.utils.models as mpc_dyn_model
import mpx.utils.objectives as mpc_objectives
import mujoco 
from mujoco import mjx
import mpx.jax_ocp_solvers.optimizers as optimizers
from jax import dlpack as jax_dlpack
from timeit import default_timer as timer
import time

@struct.dataclass
class ControlSol:
    a: jax.Array
    ac_z: jax.Array
    alpha: jax.Array
    grf: jax.Array

@struct.dataclass
class MPCState:
    sol          : ControlSol
    X0_shifted           : jax.Array
    U0_shifted           : jax.Array
    V0_shifted           : jax.Array
    X_prediction         : jax.Array
    U_prediction         : jax.Array
    # WBC warm-start
    #X0_wbc       : jax.Array
    #U0_wbc       : jax.Array
    #V0_wbc       : jax.Array

class BatchedMPCControllerWrapper:
    def __init__(self, config, n_env):
        """
        Initializes the MPC controller wrapper.
        
        Args:
            config: Configuration object containing MPC and gait parameters.
            mpc_frequency: Frequency (Hz) at which MPC updates occur.
        """
        #jax.config.update("jax_compilation_cache_dir", "./jax_cache")
        #jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        #jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

        self.n_env = n_env
        model = mujoco.MjModel.from_xml_path(config.model_path)
        mjx_model = mjx.put_model(model)
        self.config = config
        self.mpc_frequency = config.mpc_frequency
        self.shift = 1# int(1 / (config.dt_mpc * config.mpc_frequency))
        print(f"MPC update every {self.shift} simulation steps (mpc_frequency={self.mpc_frequency} Hz, dt={config.dt_mpc} s)")
        
        # Timer and liftoff states for the reference generator.
        self.q0 = config.q0.copy()          # Initial joint configuration
        
        pcom_init = config.p0.copy()
        dpcom_init = jnp.zeros(3)
        c_init = pcom_init.copy().at[2].set(0.0)
        vcz_init = jnp.zeros(1)
        theta = jnp.arctan2(c_init[1], c_init[0])
        v = jnp.zeros(1)
        omega = jnp.zeros(1)
        self.initial_state = jnp.concatenate([pcom_init, dpcom_init, c_init, vcz_init, jnp.array([theta]), v, omega])

        # Get contact and body IDs from configuration
        contact_id   = [mjx.name2id(mjx_model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in config.contact_frame]
        body_id      = [mjx.name2id(mjx_model, mujoco.mjtObj.mjOBJ_BODY,  n) for n in config.body_name]
        base_body_id = mjx.name2id(mjx_model, mujoco.mjtObj.mjOBJ_BODY, config.base_body_name)
        self._mjx_model = mjx_model
        self._model     = model
        self._contact_id   = contact_id
        self._body_id      = body_id
        self._base_body_id = base_body_id
        # Trajectory warm-start variables (used between MPC calls)
        U0 = jnp.tile(config.u_ref, (config.N, 1))
        X0 = jnp.tile(self.initial_state, (config.N + 1, 1))
        V0 = jnp.zeros((config.N + 1, config.nx))
        
        self.batch_U0 = jnp.tile(U0, (n_env, 1, 1))
        self.batch_X0 = jnp.tile(X0, (n_env, 1, 1))
        self.batch_V0 = jnp.tile(V0, (n_env, 1, 1))
        
        # Define cost, hessian approximation, and dynamics functions for MPC.
        # TODO: write cost function
        cost = partial(mpc_objectives.wheeled_dfcip_obj, config.d, config.N)
        # TODO: can be omitted
        hessian_approx = partial(mpc_objectives.wheeled_dfcip_hessian_gn, config.d, config.N)
        
        # TODO: write dynamics function - OK
        dynamics = partial(mpc_dyn_model.wheeled_dfcip_dynamics,
            mjx_model, config.mass, config.grav, config.dt_mpc)

        work = partial(optimizers.mpc, cost, dynamics, hessian_approx, False)
        
        # TODO: copy reference generator from colab - OK
        reference_generator = partial(mpc_utils.reference_generator_dfcip_offline,
            pcom=(0.0, 0.0, 0.4), nx=config.nx, nu=config.nu, 
            t_sec=config.T_TRAJECTORY, dt=config.dt_ref, m=config.mass, grav=config.grav)

        #reference_generator = partial(mpc_utils.reference_generator_dfcip_online, config.N, config.dt, config.mass, config.grav)

        # Whole-body controller: static args frozen via partial, runtime args
        # (X0_prev, U0_prev, V0_prev, qpos, qvel, desired) passed at call time.
        nq = mjx_model.nq
        nv = mjx_model.nv
        n_joints_           = nv - 6
        n_contacts_         = 1 #config.n_contact                          # 1
        n_wbc_variables_    = 6 + n_joints_ + 2 * 3 * n_contacts_
        wbc_U0_init = jnp.zeros((n_wbc_variables_,))     # era nv+3, ora nv+6
        wbc_X0_init = jnp.zeros((nq + nv,))
        wbc_V0_init = jnp.zeros((nq + nv,))
        self._wbc_U0_init = wbc_U0_init
        self._wbc_X0_init = wbc_X0_init
        self._wbc_V0_init = wbc_V0_init

        # Store nj and desired size for building default desired in whole_body_run.
        nj = nv - 6
        self._nj = nj
        self._n_joints_    = n_joints_
        self._desired_size = mpc_utils._REF_JOINTS + 3 * n_joints_
        # Use a lambda so the warm-start args (X0_prev, U0_prev, V0_prev) are
        # correctly placed at positions 8-10, while all static config args are
        # closed over.  A bare partial would silently fill the warm-start slots
        # with the gain matrices (wrong shapes → nd=3 crash in optimizers.mpc).
        _mjx   = mjx_model
        _cid   = contact_id
        _bid   = body_id
        _bbid  = base_body_id
        _wr    = config.wheel_radius
        _st    = 1.0 / config.whole_body_frequency
        _nc    = 1 #config.n_contact
        _Kpm   = config.Kp_motion;  _Kdm  = config.Kd_motion
        _Kpw   = config.Kp_wheel;   _Kdw  = config.Kd_wheel
        _Kpr   = config.Kp_reg;     _Kdr  = config.Kd_reg
        _wq    = config.w_qddot;    _wc   = config.w_com
        _wl    = config.w_lwheel;   _wrr  = config.w_rwheel;  _wb = config.w_base
        _mu    = config.mu               # 0.5 in C++

        _Kpr = config.Kp_reg;  _Kdr = config.Kd_reg


        def whole_body_control(qpos, qvel, desired):
            return mpc_utils.whole_body_interface_wheeled_legged_qp(
                _mjx, config.mass, config.grav, config.d,
                _cid, _bid, _bbid,
                _wr, _st, _nc,
                _Kpm, _Kdm, _Kpw, _Kdw, _Kpr, _Kdr,   # ← ora 6 gains
                _wq, _wc, _wl, _wrr, _wb,
                _mu,
                qpos, qvel, desired
            )

        self._solve = jax.jit(jax.vmap(work))
        #self._ref_gen = jax.jit(jax.vmap(reference_generator))
        self._ref_gen = jax.jit(reference_generator)
        self._x_reference, self._u_reference = self._ref_gen(vel_lin=0.5, vel_ang=0.0, vel_z=0.0)
        self._build_desired_jit = jax.jit(self._build_desired_impl)
        self._whole_body_interface = jax.jit(jax.vmap(whole_body_control))

        U0 = jnp.tile(config.u_ref, (config.N, 1))
        X0 = jnp.tile(self.initial_state, (config.N + 1, 1))
        V0 = jnp.zeros((config.N + 1, config.nx))

        self._U0_init = jnp.tile(U0, (n_env, 1, 1))
        self._X0_init = jnp.tile(X0, (n_env, 1, 1))
        self._V0_init = jnp.tile(V0, (n_env, 1, 1))
        
    def init_state(self) -> MPCState:
        n, cfg = self.n_env, self.config
        a = jnp.tile(cfg.u_ref[0], (n, 1))
        ac_z = jnp.tile(cfg.u_ref[1], (n, 1))
        alpha = jnp.tile(cfg.u_ref[2], (n, 1))
        grf = jnp.tile(cfg.u_ref[3:], (n, 1))
        
        return MPCState(
            sol= ControlSol(a=a, ac_z=ac_z, alpha=alpha, grf=grf),
            X0_shifted=self._X0_init, 
            U0_shifted=self._U0_init,
            V0_shifted=self._V0_init,
            X_prediction=self._X0_init,
            U_prediction=self._U0_init
        )

    def run(self, state: MPCState, x0, time_frame, horizon):
        """
        Runs one MPC update using the current state, input, and foot positions.
        
        Args:
            x0: Current system state vector.
        
        Returns:
            A tuple (X, U, V) representing the computed state trajectory, control sequence,
            and auxiliary variable trajectory.
        """
        # Generate reference trajectory and additional MPC parameters.
        
        x_T = self._x_reference.shape[0]
        u_T = self._u_reference.shape[0]

        x_idx = time_frame + jnp.arange(horizon + 1)
        u_idx = time_frame + jnp.arange(horizon)

        x_idx = jnp.clip(x_idx, 0, x_T - 1)
        u_idx = jnp.clip(u_idx, 0, u_T - 1)

        x_slice = self._x_reference[x_idx, :]   # (horizon+1, nx)
        u_slice = self._u_reference[u_idx, :]   # (horizon,   nu)
        u_slice_pad = jnp.concatenate([u_slice, u_slice[-1:, :]], axis=0)
        ref_slice = jnp.concatenate([x_slice, u_slice_pad], axis=1)  # (horizon+1, nx + nu)
        reference = jnp.tile(ref_slice[None, :, :], (self.n_env, 1, 1))

        #reference = self._ref_gen(vel_lin=0.0, vel_ang=0.0, vel_z=0.0)

        parameter = None

        X, U, V = self._solve(
            reference,
            parameter,
            jnp.tile(self.config.W, (self.n_env, 1, 1)),
            x0,
            state.X0_shifted,
            state.U0_shifted,
            state.V0_shifted
            )
        
        new_a = U[:,0,0]
        new_ac_z = U[:,0,1]
        new_alpha = U[:,0,2]
        new_grf = U[:,0,3:]
        
        # Warm-start for the next call: shift trajectories forward.
        s = self.shift
        new_X0  = jnp.concatenate([X[:, s:, :], jnp.tile(X[:, -1:, :], (1, s, 1))], axis=1)
        new_U0  = jnp.concatenate([U[:, s:, :], jnp.tile(U[:, -1:, :], (1, s, 1))], axis=1)
        new_V0  = jnp.concatenate([V[:, s:, :], jnp.tile(V[:, -1:, :], (1, s, 1))], axis=1)
        
        new_state = MPCState(
            sol=ControlSol(a=new_a, ac_z=new_ac_z, alpha=new_alpha, grf=new_grf),
            X0_shifted=new_X0, 
            U0_shifted=new_U0, 
            V0_shifted=new_V0,
            X_prediction=X,
            U_prediction=U,
        )

        return new_state, reference

    def _build_desired_impl(
        self,
        x0,
        qpos,
        state,
        pl_world,
        pr_world,
        dpl_world,
        dpr_world
    ):
        B = qpos.shape[0]
        desired = jnp.zeros((B, self._desired_size))
        nv = self._mjx_model.nv
        
        x_mpc = x0[None, :]
        u_mpc = jnp.concatenate([
            state.sol.a[:, None],
            state.sol.ac_z[:, None],
            state.sol.alpha[:, None],
            state.sol.grf
        ], axis=1)

        # ── inputs ───────────────────────────────────────────────────────
        # double a     = u_prediction(0);
        # double ac_z  = u_prediction(1);
        # double alpha = u_prediction(2);
        # Eigen::Vector3d fcl = u_prediction.segment<3>(3);
        # Eigen::Vector3d fcr = u_prediction.segment<3>(6);
        a     = u_mpc[:, 0]    # (B,)
        ac_z  = u_mpc[:, 1]    # (B,)
        alpha = u_mpc[:, 2]    # (B,)
        fcl   = u_mpc[:, 3:6]  # (B, 3)
        fcr   = u_mpc[:, 6:9]  # (B, 3)

        # ── current state ────────────────────────────────────────────────
        # Eigen::Vector3d pcom_curr = x_IN.segment<3>(0);
        # Eigen::Vector3d vcom_curr = x_IN.segment<3>(3);
        # Eigen::Vector3d pl_curr   = x_IN.segment<3>(6);
        # Eigen::Vector3d pr_curr   = x_IN.segment<3>(9);
        # Eigen::Vector3d dpl_curr  = x_IN.segment<3>(12);
        # Eigen::Vector3d dpr_curr  = x_IN.segment<3>(15);
        # double theta_curr = x0(10);
        # double v_curr     = x0(11);
        # double w_curr     = x0(12);
        pcom_curr  = x_mpc[:, 0:3]
        vcom_curr  = x_mpc[:, 3:6]
        pl_curr    = pl_world
        pr_curr    = pr_world
        dpl_curr   = dpl_world
        dpr_curr   = dpr_world
        theta_curr = x_mpc[:, 10]
        v_curr     = x_mpc[:, 11]
        w_curr     = x_mpc[:, 12]

        cos_t = jnp.cos(theta_curr)  # (B,)
        sin_t = jnp.sin(theta_curr)  # (B,)

        # ── Eigen::Vector3d g_vec = Eigen::Vector3d(0, 0, -grav) ─────────
        g_vec = jnp.array([0.0, 0.0, -self.config.grav])  # (3,)

        # ── Eigen::Vector3d vector_off = Eigen::Vector3d(0.0, d/2, 0.0) ──
        vector_off = jnp.array([0.0, self.config.d / 2.0, 0.0])  # (3,)

        # ── dR_curr ───────────────────────────────────────────────────────
        # dR_curr << -sin(theta_curr), -cos(theta_curr), 0,
        #             cos(theta_curr), -sin(theta_curr),  0,
        #             0,               0,                 0;
        dR_curr = jnp.stack([
            jnp.stack([-sin_t, -cos_t, jnp.zeros((B,))], axis=1),  # row 0
            jnp.stack([ cos_t, -sin_t, jnp.zeros((B,))], axis=1),  # row 1
            jnp.zeros((B, 3)),                                       # row 2
        ], axis=1)  # (B, 3, 3)

        # ── ddR_curr ──────────────────────────────────────────────────────
        # ddR_curr << -cos(theta_curr),  sin(theta_curr), 0,
        #             -sin(theta_curr), -cos(theta_curr),  0,
        #              0,                0,                0;
        ddR_curr = jnp.stack([
            jnp.stack([-cos_t,  sin_t, jnp.zeros((B,))], axis=1),  # row 0
            jnp.stack([-sin_t, -cos_t, jnp.zeros((B,))], axis=1),  # row 1
            jnp.zeros((B, 3)),                                       # row 2
        ], axis=1)  # (B, 3, 3)

        # ── ddc ───────────────────────────────────────────────────────────
        # ddc(0) = a * cos(theta_curr) - v_curr * sin(theta_curr) * w_curr;
        # ddc(1) = a * sin(theta_curr) + v_curr * cos(theta_curr) * w_curr;
        # ddc(2) = ac_z;
        ddc = jnp.stack([
            a * cos_t - v_curr * sin_t * w_curr,
            a * sin_t + v_curr * cos_t * w_curr,
            ac_z,
        ], axis=1)  # (B, 3)

        # ── acc_com_ = 1/m * (fcl + fcr) + g_vec ─────────────────────────
        # ── vel_com_ = vcom_curr + dt_ * acc_com_ ────────────────────────
        # ── pos_com_ = pcom_curr + dt_ * vcom_curr ───────────────────────
        dt = 1.0 / self.config.whole_body_frequency
        acc_com_ = (fcl + fcr) / self.config.mass + g_vec[None, :]
        vel_com_ = vcom_curr + dt * acc_com_
        pos_com_ = pcom_curr + dt * vcom_curr

        # ── acc_pl_ = ddc + (ddR_curr * w_curr^2 + dR_curr * alpha) * vector_off
        # ── vel_pl_ = dpl_curr + dt_ * acc_pl_
        # ── pos_pl_ = pl_curr + dt_ * dpl_curr
        acc_pl_ = ddc + (
            jnp.einsum('bij,j->bi', ddR_curr, vector_off) * w_curr[:, None] * w_curr[:, None]
            + jnp.einsum('bij,j->bi', dR_curr, vector_off) * alpha[:, None]
        )
        vel_pl_ = dpl_curr + dt * acc_pl_
        pos_pl_ = pl_curr + dt * dpl_curr

        # ── acc_pr_ = ddc - (ddR_curr * w_curr^2 + dR_curr * alpha) * vector_off
        # ── vel_pr_ = dpr_curr + dt_ * acc_pr_
        # ── pos_pr_ = pr_curr + dt_ * dpr_curr
        acc_pr_ = ddc - (
            jnp.einsum('bij,j->bi', ddR_curr, vector_off) * w_curr[:, None] * w_curr[:, None]
            + jnp.einsum('bij,j->bi', dR_curr, vector_off) * alpha[:, None]
        )
        vel_pr_ = dpr_curr + dt * acc_pr_
        pos_pr_ = pr_curr + dt * dpr_curr

        # ── alpha_ = alpha ────────────────────────────────────────────────
        # ── omega_ = w_curr + dt_ * alpha_ ───────────────────────────────
        # ── theta_ = theta_curr + dt_ * w_curr ───────────────────────────
        alpha_ = alpha
        omega_ = w_curr + dt * alpha_
        theta_ = theta_curr + dt * w_curr

        # ── contact_force_left_  = fcl ────────────────────────────────────
        # ── contact_force_right_ = fcr ────────────────────────────────────
        contact_force_left_  = fcl
        contact_force_right_ = fcr

        # ════════════════════════════════════════════════════════════════════
        # Assemble desired — specchio di:
        #   des_configuration_.com.pos = sol.com.pos;
        #   des_configuration_.com.vel = sol.com.vel;
        #   des_configuration_.com.acc = sol.com.acc;
        com_pos_ref = pos_com_
        com_vel_ref = vel_com_
        com_acc_ref = acc_com_

        #   des_configuration_.lwheel.pos.p.segment<2>(0) = sol.pl.pos.segment<2>(0);
        #   des_configuration_.lwheel.pos.p(2) = sol.pl.pos(2) + wheel_radius_;
        #   des_configuration_.lwheel.vel.segment<3>(0) = sol.pl.vel.segment<3>(0);
        #   des_configuration_.lwheel.acc.segment<3>(0) = sol.pl.acc.segment<3>(0);
        lwheel_pos_ref = pos_pl_.at[:, 2].set(pos_pl_[:, 2] + self.config.wheel_radius)
        lwheel_vel_ref = vel_pl_
        lwheel_acc_ref = acc_pl_

        #   des_configuration_.rwheel.pos.p.segment<2>(0) = sol.pr.pos.segment<2>(0);
        #   des_configuration_.rwheel.pos.p(2) = sol.pl.pos(2) + wheel_radius_;   // nota: usa sol.pl.pos(2) anche per destra!
        #   des_configuration_.rwheel.vel.segment<3>(0) = sol.pr.vel.segment<3>(0);
        #   des_configuration_.rwheel.acc.segment<3>(0) = sol.pr.acc.segment<3>(0);
        rwheel_pos_ref = pos_pr_.at[:, 2].set(pos_pl_[:, 2] + self.config.wheel_radius)
        rwheel_vel_ref = vel_pr_
        rwheel_acc_ref = acc_pr_

        #   Eigen::Matrix3d R_theta << cos(sol.theta), -sin(sol.theta), 0,
        #                               sin(sol.theta),  cos(sol.theta), 0,
        #                               0,               0,              1;
        #   des_configuration_.base_link.pos = R_theta;
        #   des_configuration_.base_link.vel = Eigen::Vector3d(0, 0, sol.omega);
        #   des_configuration_.base_link.acc = Eigen::Vector3d(0, 0, sol.alpha);
        #
        # base_link.pos è una matrice di rotazione — nel WBC Python è un quaternione
        # R_theta è solo yaw → quaternione [cos(theta/2), 0, 0, sin(theta/2)]
        cos_theta_ = jnp.cos(theta_)
        sin_theta_ = jnp.sin(theta_)
        R_theta = jnp.stack([
            jnp.stack([ cos_theta_, -sin_theta_, jnp.zeros((B,))], axis=1),  # row 0
            jnp.stack([ sin_theta_,  cos_theta_, jnp.zeros((B,))], axis=1),  # row 1
            jnp.stack([ jnp.zeros((B,)), jnp.zeros((B,)), jnp.ones((B,)) ], axis=1),  # row 2
        ], axis=1)  # (B, 3, 3)

        # flatten row-major per il desired vector: (B, 9)
        base_rot_ref = R_theta.reshape(B, 9)
        base_omega_ref = jnp.zeros((B, 3)).at[:, 2].set(omega_)
        base_alpha_ref = jnp.zeros((B, 3)).at[:, 2].set(alpha_)

        # Joints — postura nominale
        qjnt_ref     = jnp.tile(self.config.q0, (B, 1))
        qjntdot_ref  = jnp.zeros((B, self._nj))
        qjntddot_ref = jnp.zeros((B, self._nj))

        # ── fill desired vector ───────────────────────────────────────────
        desired = desired.at[:, mpc_utils._REF_COM_POS:mpc_utils._REF_COM_POS + 3].set(com_pos_ref)
        desired = desired.at[:, mpc_utils._REF_COM_VEL:mpc_utils._REF_COM_VEL + 3].set(com_vel_ref)
        desired = desired.at[:, mpc_utils._REF_COM_ACC:mpc_utils._REF_COM_ACC + 3].set(com_acc_ref)
        desired = desired.at[:, mpc_utils._REF_LW_POS:mpc_utils._REF_LW_POS + 3].set(lwheel_pos_ref)
        desired = desired.at[:, mpc_utils._REF_RW_POS:mpc_utils._REF_RW_POS + 3].set(rwheel_pos_ref)
        desired = desired.at[:, mpc_utils._REF_LW_VEL:mpc_utils._REF_LW_VEL + 3].set(lwheel_vel_ref)
        desired = desired.at[:, mpc_utils._REF_RW_VEL:mpc_utils._REF_RW_VEL + 3].set(rwheel_vel_ref)
        desired = desired.at[:, mpc_utils._REF_LW_ACC:mpc_utils._REF_LW_ACC + 3].set(lwheel_acc_ref)
        desired = desired.at[:, mpc_utils._REF_RW_ACC:mpc_utils._REF_RW_ACC + 3].set(rwheel_acc_ref)
        desired = desired.at[:, mpc_utils._REF_BASE_ROT:mpc_utils._REF_BASE_ROT + 9].set(base_rot_ref)
        desired = desired.at[:, mpc_utils._REF_BASE_OMG:mpc_utils._REF_BASE_OMG + 3].set(base_omega_ref)
        desired = desired.at[:, mpc_utils._REF_BASE_ALP:mpc_utils._REF_BASE_ALP + 3].set(base_alpha_ref)
        desired = desired.at[:, mpc_utils._REF_JOINTS:mpc_utils._REF_JOINTS + self._nj].set(qjnt_ref)
        desired = desired.at[:, mpc_utils._REF_JOINTS + self._nj:mpc_utils._REF_JOINTS + 2*self._nj].set(qjntdot_ref)
        desired = desired.at[:, mpc_utils._REF_JOINTS + 2*self._nj:mpc_utils._REF_JOINTS + 3*self._nj].set(qjntddot_ref)
        
        return desired
    
    def whole_body_run(self, state: MPCState, x0, qpos, qvel,
                   pl_world, pr_world, dpl_world, dpr_world):

        desired = desired = self._build_desired_jit(
            x0,
            qpos,
            state,
            pl_world,
            pr_world,
            dpl_world,
            dpr_world,
        )

        tau_cmd, qddot, fl, fr = self._whole_body_interface(
            qpos, qvel, desired
        )

        return state, tau_cmd, qddot, fl, fr, desired

    def reset(self):
        """
        Resets the MPC controller state."
        """
        print("MPC Controller Reset")
        return self.init_state()