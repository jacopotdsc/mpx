"""
rl_env_srbd.py  —  QuadrupedMPCEnv come Brax PipelineEnv

Regola fondamentale: reset() e step() NON devono contenere
np.asarray() o qualsiasi operazione NumPy su array tracciati da JAX,
perché brax wrappa tutto con jax.vmap / jax.jit.

Tutte le operazioni nel path reset/step usano jnp puro.
"""

from __future__ import annotations
import os, sys

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, "..")))
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf

import mpx.config.config_srbd as config
from mpx.utils.mpc_wrapper_srbd import BatchedMPCControllerWrapper


class QuadrupedMPCEnv(PipelineEnv):

    def __init__(
        self,
        episode_steps:    int   = 500,
        sim_frequency:    float = 200.0,
        cmd_limits              = (0.5, 0.2, 0.5),
        delta_limit:      float = 10.0,
        height_threshold: float = 0.15,
        sigma:            float = 0.25,
        **kwargs,
    ):
        model_path = os.path.abspath(
            os.path.join(dir_path, "..", "data", "aliengo", "scene_flat.xml")
        )
        sys = mjcf.load(model_path)
        sys = sys.replace(opt=sys.opt.replace(timestep=1.0 / sim_frequency))
        super().__init__(sys, backend="mjx", n_frames=1, **kwargs)

        self.episode_steps    = episode_steps
        self.cmd_limits       = jnp.array(cmd_limits)
        self.delta_limit      = delta_limit
        self.height_threshold = height_threshold
        self.sigma            = sigma
        self.n_joints         = config.n_joints
        self.mpc_period       = max(1, int(sim_frequency / config.mpc_frequency))
        self.mpc              = BatchedMPCControllerWrapper(config, 1)

        mj_model  = mujoco.MjModel.from_xml_path(model_path)
        # Use geom ids for contact_frame, matching srbd_quad.py (data.geom_xpos)
        self._foot_geom_ids = [
            mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in config.contact_frame
        ]
        print(f"[QuadrupedMPCEnv] foot geom ids: {self._foot_geom_ids}")

        self._qpos0 = jnp.concatenate([config.p0, config.quat0, config.q0])
        self._qvel0 = jnp.zeros(6 + self.n_joints)

        print(f"[QuadrupedMPCEnv] obs={self.observation_size}  "
              f"act={self.action_size}  mpc_period={self.mpc_period}")

    @property
    def observation_size(self) -> int:
        # base state + command + mpc input (7D) + mpc output torque + previous policy action
        return 13 + 2 * self.n_joints + 3 + 7 + self.n_joints + self.n_joints

    @property
    def action_size(self) -> int:
        return self.n_joints  # delta torque, one per joint

    def reset(self, rng: jax.Array) -> State:
        rng, cmd_rng = jax.random.split(rng)

        pipeline_state = self.pipeline_init(self._qpos0, self._qvel0)
        command        = jnp.zeros(3) #self._sample_command(cmd_rng)
        mpc_input      = self._build_mpc_input(command)
        prev_action    = jnp.zeros((self.n_joints,), dtype=jnp.float32)
        reward         = self._compute_reward(pipeline_state, command, delta=prev_action)
        mpc_state      = self.mpc.init_state()
        ctrl, mpc_state = self._compute_ctrl(pipeline_state, command, mpc_state)
        obs            = self._get_obs(pipeline_state, command, mpc_input, ctrl, prev_action)

        return State(
            pipeline_state = pipeline_state,
            obs            = obs,
            reward         = jnp.float32(0.0),
            done           = jnp.float32(0.0),
            metrics        = {"reward": reward},
            info           = {
                "command":    command,
                "mpc_input":  mpc_input,
                "mpc_ctrl":   ctrl,
                "ctrl":       ctrl,
                "prev_action": prev_action,
                "step_count": jnp.int32(0),
                "mpc_state":  mpc_state,
            },
        )

    def step(self, state: State, action: jax.Array) -> State:
        delta      = action*self.delta_limit #jnp.clip(action, -self.delta_limit, self.delta_limit)
        command    = state.info["command"]
        mpc_input  = state.info["mpc_input"]
        step_count = state.info["step_count"]

        mpc_ctrl, mpc_state = jax.lax.cond(
            step_count % self.mpc_period == 0,
            lambda: self._compute_ctrl(state.pipeline_state, command, state.info["mpc_state"]),
            lambda: (state.info["mpc_ctrl"], state.info["mpc_state"]),
        )
        ctrl = mpc_ctrl + delta

        pipeline_state = self.pipeline_step(state.pipeline_state, ctrl)
        obs            = self._get_obs(pipeline_state, command, mpc_input, mpc_ctrl, action)
        reward         = self._compute_reward(pipeline_state, command, delta)
        done           = self._is_done(pipeline_state, step_count + 1)

        return state.replace(
            pipeline_state = pipeline_state,
            obs            = obs,
            reward         = reward,
            done           = done,
            metrics        = {"reward": reward},
            info           = {
                **state.info,
                "mpc_ctrl": mpc_ctrl,
                "ctrl": ctrl,
                "prev_action": action,
                "step_count": step_count + 1,
                "mpc_state": mpc_state,
            },
        )

    def _build_mpc_input(self, command: jax.Array) -> jax.Array:
        command = jnp.nan_to_num(command, nan=0.0, posinf=0.0, neginf=0.0)
        return jnp.array([command[0], command[1], 0.0, 0.0, 0.0, command[2], config.robot_height])

    def _compute_ctrl(self, pipeline_state: base.State, command: jax.Array, mpc_state):
        qpos = jnp.nan_to_num(pipeline_state.q, nan=0.0, posinf=0.0, neginf=0.0)
        qvel = jnp.nan_to_num(pipeline_state.qd, nan=0.0, posinf=0.0, neginf=0.0)
 
        # pipeline_state in Brax mjx backend is mjx.Data → has geom_xpos shape (ngeom, 3)
        foot_world = jnp.stack([pipeline_state.geom_xpos[gid] for gid in self._foot_geom_ids], axis=0)
        foot_pos = foot_world.reshape(1, -1)
        foot_pos  = jnp.nan_to_num(foot_pos, nan=0.0, posinf=0.0, neginf=0.0)
        x0        = jnp.concatenate([qpos[:3], qpos[3:7], qvel[:3], qvel[3:6]])[None]
        mpc_input = self._build_mpc_input(command)[None, :]
        # Contact estimate from foot height, closer to what the MuJoCo loop does.
        contact = (foot_world[:, 2] < 0.035).astype(jnp.float32)[None, :]
 
        mpc_state = self.mpc.run(mpc_state, x0, mpc_input, foot_pos, contact)
        tau, _    = self.mpc.whole_body_run(mpc_state, qpos[None], qvel[None])

        tau = jnp.nan_to_num(tau[0], nan=0.0, posinf=0.0, neginf=0.0)
        tau = jnp.clip(tau, -60.0, 60.0)
        return tau, mpc_state

    def _get_obs(
        self,
        ps: base.State,
        command: jax.Array,
        mpc_input: jax.Array,
        mpc_ctrl: jax.Array,
        prev_action: jax.Array,
    ) -> jax.Array:
        obs = jnp.concatenate(
            [
                ps.q[:3],
                ps.q[3:7],
                ps.qd[:3],
                ps.qd[3:6],
                ps.q[7:],
                ps.qd[6:],
                command,
                mpc_input,
                mpc_ctrl,
                prev_action,
            ]
        )
        return jnp.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    def _compute_reward(self, ps, command, delta):
        r_lin = jnp.exp(-jnp.sum((ps.qd[:2] - command[:2])**2) / self.sigma)
        r_ang = jnp.exp(-(ps.qd[5] - command[2])**2 / self.sigma)
        r_alive  = jnp.float32(1.0)
        #r_delta  = -0.01 * jnp.sum(delta**2)
        r_delta = -0.1 * jnp.sum((delta / self.delta_limit)**2)
        return jnp.nan_to_num(r_lin + r_ang + r_alive + r_delta)

    def _is_done(self, ps: base.State, step_count: jax.Array) -> jax.Array:
        fallen  = ps.q[2] < self.height_threshold
        timeout = step_count >= self.episode_steps
        invalid = ~jnp.isfinite(jnp.sum(ps.q)) | ~jnp.isfinite(jnp.sum(ps.qd))
        return (fallen | timeout | invalid).astype(jnp.float32)

    def _sample_command(self, rng: jax.Array) -> jax.Array:
        return jax.random.uniform(rng, (3,), minval=-self.cmd_limits, maxval=self.cmd_limits)