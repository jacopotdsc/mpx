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


@struct.dataclass
class MPCState:
    """Tutto lo stato mutabile dell'MPC in un pytree JAX (compatibile con jax.vmap)."""
    contact_time : jax.Array
    liftoff      : jax.Array
    foot_ref     : jax.Array
    foot_ref_dot : jax.Array
    grf          : jax.Array
    contact      : jax.Array
    X0           : jax.Array
    U0           : jax.Array
    V0           : jax.Array


class BatchedMPCControllerWrapper:
    def __init__(self, config, n_env):
        jax.config.update("jax_compilation_cache_dir", "./jax_cache")
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

        self.n_env = n_env
        self.config = config
        self.mpc_frequency = config.mpc_frequency
        self.shift = int(1 / (config.dt * config.mpc_frequency))

        model = mujoco.MjModel.from_xml_path(config.model_path)
        mjx_model = mjx.put_model(model)

        contact_id = [mjx.name2id(mjx_model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in config.contact_frame]
        body_id    = [mjx.name2id(mjx_model, mujoco.mjtObj.mjOBJ_BODY,  n) for n in config.body_name]

        self.initial_state = jnp.concatenate([config.p0, config.quat0, jnp.zeros(6)])

        cost      = partial(mpc_objectives.quadruped_srbd_obj, config.n_contact, config.N)
        hess      = partial(mpc_objectives.quadruped_srbd_hessian_gn, config.n_contact)
        dynamics  = partial(mpc_dyn_model.quadruped_srbd_dynamics, config.mass, config.inertia, jnp.linalg.inv(config.inertia), config.dt)
        work      = partial(optimizers.mpc, cost, dynamics, hess, False)
        ref_gen   = partial(mpc_utils.reference_generator_srbd, config.use_terrain_estimator, config.N, config.dt, config.n_contact, mass=config.mass, clearence_speed=config.clearence_speed, duty_factor=config.duty_factor, step_freq=config.step_freq, step_height=config.step_height, foot0=config.p_legs0)
        wbc       = partial(mpc_utils.whole_body_interface, model, mjx_model, contact_id, body_id, config.whole_body_frequency, config.Kp, config.Kd)

        self._solve      = jax.jit(jax.vmap(work))
        self._ref_gen    = jax.jit(jax.vmap(ref_gen))
        self._timer_run  = jax.jit(jax.vmap(mpc_utils.timer_run, in_axes=(None, None, 0, None)))
        self._whole_body = jax.jit(jax.vmap(wbc))

        U0 = jnp.tile(config.u_ref, (config.N, 1))
        X0 = jnp.tile(self.initial_state, (config.N + 1, 1))
        V0 = jnp.zeros((config.N + 1, config.n))

        self._U0_init = jnp.tile(U0, (n_env, 1, 1))
        self._X0_init = jnp.tile(X0, (n_env, 1, 1))
        self._V0_init = jnp.tile(V0, (n_env, 1, 1))

    def init_state(self) -> MPCState:
        n, cfg = self.n_env, self.config
        ct = jnp.tile(cfg.timer_t.reshape(1, -1), (n, 1))
        z3 = jnp.zeros((n, 3*cfg.n_contact))
        zc = jnp.ones((n, cfg.n_contact))
        return MPCState(contact_time=ct, liftoff=z3, foot_ref=z3, foot_ref_dot=z3, grf=z3, contact=zc, X0=self._X0_init, U0=self._U0_init, V0=self._V0_init)

    def run(self, state: MPCState, x0, input, foot_pos, contact) -> MPCState:
        cfg = self.config

        new_contact, new_ct = self._timer_run(cfg.duty_factor, cfg.step_freq, state.contact_time, 1.0 / cfg.mpc_frequency)

        reference, parameter, new_liftoff, new_frd = self._ref_gen(t_timer=new_ct, x=x0, foot=foot_pos, input=input, contact=contact, liftoff=state.liftoff)

        new_foot_ref = parameter[:, 0, 4:]
        new_frd      = new_frd[:, 0, :]

        # _solve is vmapped over envs, so W must always have a batch dimension.
        W = jnp.tile(cfg.W, (self.n_env, 1, 1))
        X, U, V = self._solve(reference, parameter, W, x0, state.X0, state.U0, state.V0)

        s = self.shift
        new_X0  = jnp.concatenate([X[:, s:, :], jnp.tile(X[:, -1:, :], (1, s, 1))], axis=1)
        new_U0  = jnp.concatenate([U[:, s:, :], jnp.tile(U[:, -1:, :], (1, s, 1))], axis=1)
        new_V0  = jnp.concatenate([V[:, s:, :], jnp.tile(V[:, -1:, :], (1, s, 1))], axis=1)
        new_grf = U[:, 0, :]

        return MPCState(contact_time=new_ct, liftoff=new_liftoff, foot_ref=new_foot_ref, foot_ref_dot=new_frd, grf=new_grf, contact=new_contact, X0=new_X0, U0=new_U0, V0=new_V0)

    def whole_body_run(self, state: MPCState, qpos, qvel):
        return self._whole_body(qpos, qvel, state.grf, state.foot_ref, state.foot_ref_dot, state.contact)

    def reset(self) -> MPCState:
        print("MPC Controller Reset")
        return self.init_state()