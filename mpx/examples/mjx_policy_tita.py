"""
Interactive TITA policy test.

The simulation loop is intentionally kept in the same style as mjx_tita:
    MuJoCo MjModel/MjData
    -> keyboard command
    -> controller update
    -> data.ctrl
    -> mujoco.mj_step()
    -> viewer / camera / video

There is no eval_env.step(), no vmap rollout and no MPC/WBC.

From train_srbd.py this file only reuses:
  1. --name / MuJoCo Playground environment selection
  2. --load checkpoint resolution and PPO policy reconstruction
"""

import argparse
import os
import pickle
import re
import sys
import time
from timeit import default_timer as timer

import jax

jax.config.update("jax_enable_x64", True)

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, "..")))
TITA_PATH = os.path.join(dir_path, "plots", "tita_validation")

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")

import jax.numpy as jnp
import mujoco
import mujoco.viewer
from mujoco import mjx
import numpy as np

from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from mujoco_playground import registry
from mujoco_playground._src.locomotion.tita import tita_constants as consts

import mpx.utils.sim as sim_utils
from plot_rollout_info import _save_sim_video


jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1)


# Must match train_srbd.py / the checkpoint.
POLICY_HIDDEN_LAYER_SIZES = (512, 256, 128)
DISTRIBUTION_TYPE = "tanh_normal"

_RUN_DIR_RE = re.compile(r"^\d{8}_\d{6}$")


# -----------------------------------------------------------------------------
# Checkpoint / policy loading.
# Copied conservatively from train_srbd.py.
# -----------------------------------------------------------------------------

def load_params(ckpt_dir: str, suffix: str = "best"):
    pkl_path = None
    for s in (suffix, "final"):
        p = os.path.join(ckpt_dir, f"params_{s}.pkl")
        if os.path.exists(p):
            pkl_path = p
            print(f"  Loading checkpoint: {pkl_path}")
            break

    if pkl_path is None:
        return None

    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def _list_run_dirs(env_base_dir: str) -> list[str]:
    if not os.path.isdir(env_base_dir):
        return []

    return sorted(
        d
        for d in os.listdir(env_base_dir)
        if _RUN_DIR_RE.match(d)
        and os.path.isdir(os.path.join(env_base_dir, d))
    )

def _list_dir_names(path: str) -> list[str]:
    if not os.path.isdir(path):
        return []

    return sorted(
        d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))
    )

def _resolve_load(env_base_dir: str, load_arg: str):
    """Same --load semantics used by train_srbd.py.

    load_arg may be a timestamp/prefix run name (matched under env_base_dir),
    a name of a run saved under env_base_dir/saved/, or an explicit relative
    path such as 'saved/joystick_first_train'.
    """
    run_dirs = _list_run_dirs(env_base_dir)

    is_suffix = load_arg in ("best", "final", "crash")
    suffix = load_arg if is_suffix else "best"

    if not is_suffix:
        direct_dir = os.path.join(env_base_dir, load_arg)
        saved_dir = os.path.join(env_base_dir, "saved", load_arg)

        if os.path.isdir(direct_dir):
            run_dir = direct_dir
        elif os.path.isdir(saved_dir):
            run_dir = saved_dir
        else:
            matches = [d for d in run_dirs if d == load_arg] or [
                d for d in run_dirs if d.startswith(load_arg)
            ]
            if not matches:
                print(
                    f"  [INFO] Available runs under '{env_base_dir}': "
                    f"{_list_dir_names(env_base_dir)}"
                )
                print(
                    f"  [INFO] Available saved runs under "
                    f"'{os.path.join(env_base_dir, 'saved')}': "
                    f"{_list_dir_names(os.path.join(env_base_dir, 'saved'))}"
                )
                raise FileNotFoundError(
                    f"No run matching '{load_arg}' found under '{env_base_dir}'."
                )
            run_dir = os.path.join(env_base_dir, matches[-1])
    elif run_dirs:
        run_dir = os.path.join(env_base_dir, run_dirs[-1])
    else:
        run_dir = env_base_dir

    if not os.path.isdir(env_base_dir):
        raise FileNotFoundError(
            f"No checkpoints for this environment: '{env_base_dir}' does not exist."
        )

    if not any(
        os.path.isfile(os.path.join(run_dir, f"params_{s}.pkl"))
        for s in (suffix, "final")
    ):
        raise FileNotFoundError(
            f"Checkpoint not found: {os.path.join(run_dir, f'params_{suffix}.pkl')}"
        )

    print(f"  [INFO] Checkpoint directory: {run_dir}")
    return run_dir, suffix

def load_params(ckpt_dir: str, suffix: str = "best"):
    """Prefer the requested checkpoint; fall back to final, as in training."""
    for s in (suffix, "final"):
        pkl_path = os.path.join(ckpt_dir, f"params_{s}.pkl")
        if os.path.exists(pkl_path):
            print(f"  [INFO] Loading checkpoint: {pkl_path}")
            with open(pkl_path, "rb") as f:
                return pickle.load(f)

    raise FileNotFoundError(
        f"No params_{suffix}.pkl or params_final.pkl found in '{ckpt_dir}'."
    )


def build_policy(env, params):
    # Same active PPO network construction used in train_srbd.py.
    networks = ppo_networks.make_ppo_networks(
        observation_size=env.observation_size,
        action_size=env.action_size,
        policy_hidden_layer_sizes=POLICY_HIDDEN_LAYER_SIZES,
        preprocess_observations_fn=running_statistics.normalize,
        distribution_type=DISTRIBUTION_TYPE,
    )

    inference_fn = ppo_networks.make_inference_fn(networks)
    return inference_fn(params, deterministic=True)


# -----------------------------------------------------------------------------
# mjx_tita-style MuJoCo helpers.
# -----------------------------------------------------------------------------

def update_com_tracking_camera(cam, model, data, base_body_id, alpha=0.10):
    mujoco.mj_subtreeVel(model, data)
    com = np.asarray(data.subtree_com[base_body_id]).copy()
    cam.lookat[:] = (1.0 - alpha) * cam.lookat + alpha * com


def _reset_to_initial_state(model, data):
    try:
        mujoco.mj_resetDataKeyframe(model, data, 0)
        data.qvel[:] = 0.0
    except Exception as exc:
        print(f"Failed to reset to initial state: {exc}")

    mujoco.mj_forward(model, data)


def _base_touches_floor(
    model,
    data,
    base_body_name: str = "base_link",
    floor_geom_name: str = "floor",
) -> bool:
    base_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, base_body_name
    )
    floor_geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, floor_geom_name
    )

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


def _build_env_get_obs_fn(env):
    """Use the selected environment's own observation function.

    The MuJoCo simulation remains CPU/native (mujoco.mj_step).  At policy
    update time the current MjData is converted to mjx.Data and passed to
    env._get_obs(), so observation content/order/size stays defined only by
    the environment.
    """
    @jax.jit
    def get_obs(mjx_data, info, previous_action):
        # _get_obs updates info["rng"] while adding observation noise.
        # Copy the dict so this mutation is local to this function and return
        # the updated info explicitly.
        info = dict(info)
        obs = env._get_obs(mjx_data, info, previous_action)
        return obs, info

    return get_obs


def _keyboard_to_policy_command(command_handle, cmd_dim: int) -> np.ndarray:
    """
    Reuse the exact public KeyboardVelocityCommand interface used by mjx_tita.

    mpc_wheeled_input() starts with [vx, vy, wz].  The additional MPC-specific
    value, if any, is ignored here.
    """
    keyboard_cmd = np.asarray(
        command_handle.mpc_wheeled_input(0.0),
        dtype=np.float32,
    ).reshape(-1)

    if keyboard_cmd.size < 3:
        raise ValueError(
            "KeyboardVelocityCommand.mpc_wheeled_input() returned fewer "
            "than 3 values."
        )

    if cmd_dim == 2:
        # TITA two-command variant: [vx, wz].
        return np.array(
            [keyboard_cmd[0], keyboard_cmd[2]],
            dtype=np.float32,
        )

    if cmd_dim == 3:
        return keyboard_cmd[:3].astype(np.float32)

    raise ValueError(
        f"Unsupported command dimension: {cmd_dim}. "
        "Expected 2 ([vx, wz]) or 3 ([vx, vy, wz])."
    )


def apply_policy_action(env, data, action):
    """
    Same action -> actuator target mapping used by TitaJoystickE2EFlatTerrain.

    Legs   : position target = default_pose + action * action_scale_pos
    Wheels : velocity target = action * action_scale_vel
    """
    action = np.asarray(action, dtype=np.float64)

    leg_ids = np.asarray(env._leg_ids, dtype=np.int32)
    wheel_ids = np.asarray(env._wheel_ids, dtype=np.int32)
    default_pose = np.asarray(env._default_pose)

    ctrl = np.zeros(env.action_size, dtype=np.float64)

    ctrl[leg_ids] = (
        default_pose[leg_ids]
        + action[leg_ids] * float(env._config.action_scale_pos)
    )
    ctrl[wheel_ids] = (
        action[wheel_ids] * float(env._config.action_scale_vel)
    )

    data.ctrl[:] = ctrl


def main(
    env_name: str,
    ckpt_root: str,
    load_arg: str,
    headless: bool = False,
    steps: int = 500,
):
    print("=" * 60)
    print(f"  Interactive policy test — {env_name}")
    print("=" * 60)

    # --name selection: same MuJoCo Playground environment choice as training.
    env = registry.load(env_name)

    # Use the exact MuJoCo model belonging to the selected training env, but
    # simulate it with ordinary MuJoCo exactly like mjx_tita.
    model = env.mj_model
    data = mujoco.MjData(model)

    sim_frequency = 1.0 / float(model.opt.timestep)
    policy_period = max(
        1,
        int(round(float(env._config.ctrl_dt) / float(model.opt.timestep))),
    )

    print(f"  sim dt       : {model.opt.timestep}")
    print(f"  policy dt    : {env._config.ctrl_dt}")
    print(f"  policy period: {policy_period} sim steps")

    # --load: same checkpoint layout/resolution as train_srbd.py.
    env_base_dir = os.path.join(ckpt_root, env_name)
    run_dir, load_suffix = _resolve_load(env_base_dir, load_arg)
    params = load_params(run_dir, suffix=load_suffix)
    if params is None:
        raise FileNotFoundError(
            f"No checkpoint parameters found in '{run_dir}'."
        )

    policy_fn = build_policy(env, params)
    jit_policy = jax.jit(policy_fn)
    env_get_obs = _build_env_get_obs_fn(env)

    # Use reset only to obtain the environment's own info structure.
    # Its MJX simulation state is discarded: the actual simulation below
    # remains the native MuJoCo MjData/mujoco.mj_step loop.
    rng = jax.random.PRNGKey(42)
    rng, info_rng = jax.random.split(rng)
    info_state = env.reset(info_rng)
    policy_info = dict(info_state.info)

    # Manual joystick starts from zero.  Command dimensionality comes directly
    # from the selected environment instead of being hard-coded.
    policy_info["command"] = jnp.zeros_like(policy_info["command"])
    cmd_dim = int(policy_info["command"].shape[-1])

    command_handle = sim_utils.KeyboardVelocityCommand(
        vx=0.0,
        vy=0.0,
        wz=0.0,
        forward_step=0.1,
        yaw_step=0.2,
        forward_limits=(-10.0, 10.0),
        yaw_limits=(-1.5, 1.5),
    )

    _reset_to_initial_state(model, data)

    # Same neutral actuator initialization used by the E2E environment reset.
    leg_ids = np.asarray(env._leg_ids, dtype=np.int32)
    data.ctrl[:] = 0.0
    data.ctrl[leg_ids] = np.asarray(data.qpos[7:])[leg_ids]
    mujoco.mj_forward(model, data)

    np.set_printoptions(precision=4, suppress=True)

    # ------------------------------------------------------------------
    # MuJoCo video recording: same structure as mjx_tita.
    # ------------------------------------------------------------------
    _sim_frames: list = []
    _renderer = mujoco.Renderer(model, height=480, width=640)

    _sim_cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(_sim_cam)
    _sim_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    _sim_cam.distance = 8.0
    _sim_cam.elevation = -15.0
    _sim_cam.azimuth = 60.0

    base_body_name = consts.ROOT_BODY
    floor_geom_name = consts.FLOOR_GEOM

    _sim_base_body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        base_body_name,
    )

    mujoco.mj_subtreeVel(model, data)
    _sim_cam.lookat[:] = np.asarray(
        data.subtree_com[_sim_base_body_id]
    )

    previous_action = np.zeros(env.action_size, dtype=np.float32)
    current_action = previous_action.copy()
    current_command = np.zeros(cmd_dim, dtype=np.float32)

    counter = 0

    def step_controller():
        nonlocal counter
        nonlocal rng
        nonlocal policy_info
        nonlocal previous_action
        nonlocal current_action
        nonlocal current_command

        print(f"\n=== step {counter} ===")

        init_start = timer()

        base_body_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            base_body_name,
        )

        # Same external-force block kept from mjx_tita.
        data.xfrc_applied[base_body_id] = 0.0

        force_start = 50
        if force_start <= counter < force_start + 100:
            force_world = np.array([0.0, 0.0, 0.0])
            data.xfrc_applied[base_body_id, 0:3] = force_world

        init_stop = timer()
        print(
            f"[timing] init time: "
            f"{1e3 * (init_stop - init_start):.2f} ms"
        )

        # Policy runs at ctrl_dt.  Between policy updates data.ctrl is held,
        # while ordinary MuJoCo continues stepping at sim_dt.
        if counter % policy_period == 0:
            current_command = _keyboard_to_policy_command(
                command_handle,
                cmd_dim,
            )

            obs_start = timer()

            # Keep the command inside the same info dictionary used by the
            # environment's own _get_obs().
            policy_info = {
                **policy_info,
                "command": jnp.asarray(current_command, dtype=jnp.float32),
            }

            # Convert only the current MuJoCo data for observation extraction.
            # Physics still runs through mujoco.mj_step(), not env.step().
            mjx_data = mjx.put_data(model, data)
            obs, policy_info = env_get_obs(
                mjx_data,
                policy_info,
                jnp.asarray(previous_action, dtype=jnp.float32),
            )
            jax.block_until_ready(obs)
            obs_stop = timer()

            rng, act_rng = jax.random.split(rng)

            policy_start = timer()
            action, _ = jit_policy(obs, act_rng)
            action.block_until_ready()
            policy_stop = timer()

            current_action = np.asarray(
                jax.device_get(action),
                dtype=np.float32,
            )

            apply_policy_action(
                env=env,
                data=data,
                action=current_action,
            )

            previous_action = current_action.copy()

            print(
                f"[command] {current_command}"
            )
            print(
                f"[action]  {current_action}"
            )
            print(
                f"[timing] observation: "
                f"{1e3 * (obs_stop - obs_start):.2f} ms"
            )
            print(
                f"[timing] policy: "
                f"{1e3 * (policy_stop - policy_start):.2f} ms"
            )

        extra_start = timer()

        touch_floor = _base_touches_floor(
            model,
            data,
            base_body_name=base_body_name,
            floor_geom_name=floor_geom_name,
        )

        # This is the same simulation primitive used by mjx_tita.
        mujoco.mj_step(model, data)

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
        print(
            f"[timing] extra time: "
            f"{1e3 * (extra_stop - extra_start):.2f} ms"
        )

        return touch_floor

    def finalize_outputs():
        print("\n[finalize] Saving outputs...")

        try:
            video_fps = int(sim_frequency / 2)

            _save_sim_video(
                video_dir=TITA_PATH,
                video_fps=video_fps,
                frames=_sim_frames,
                slowdown_factor=1.0,
                name_video="policy_simulation_video.mp4",
            )
            _save_sim_video(
                video_dir=TITA_PATH,
                video_fps=video_fps,
                frames=_sim_frames,
                slowdown_factor=4.0,
                name_video="policy_simulation_video_slow4.mp4",
            )
            _save_sim_video(
                video_dir=TITA_PATH,
                video_fps=video_fps,
                frames=_sim_frames,
                slowdown_factor=15.0,
                name_video="policy_simulation_video_slow15.mp4",
            )
        except Exception as exc:
            print(f"[finalize] failed to save video: {exc}")

        try:
            _renderer.close()
        except Exception:
            pass

    try:
        if headless:
            for _ in range(steps):
                touch_floor = step_controller()

                if touch_floor:
                    print(
                        f"Base touched the floor at step {counter}. "
                        "Ending simulation."
                    )
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
                overlay_text = command_handle.consume_overlay_text()
                tic = timer()

                if overlay_text is not None:
                    viewer.set_texts((None, None, *overlay_text))

                start_step = timer()
                touch_floor = step_controller()
                end_step = timer()

                print(
                    f"Step time: "
                    f"{1e3 * (end_step - start_step):.2f} ms"
                )

                toc = timer()
                if toc - tic < model.opt.timestep:
                    time.sleep(model.opt.timestep - (toc - tic))

                if touch_floor:
                    print(
                        f"Base touched the floor at step {counter}. "
                        "Ending simulation."
                    )
                    break

                viewer.sync()

    except KeyboardInterrupt:
        print("\n[interrupt] Ctrl+C received. Finalizing outputs...")
    finally:
        finalize_outputs()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--name",
        type=str,
        default="TitaJoystickE2EFlatTerrain",
        help="Environment name, same convention used by train_srbd.py.",
    )
    parser.add_argument(
        "--load",
        nargs="?",
        const="best",
        default="best",
        metavar="RUN_OR_SUFFIX",
        help=(
            "Bare --load loads latest best; a timestamp/prefix selects a run; "
            "'best', 'final' or 'crash' selects the suffix."
        ),
    )
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default="checkpoints",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--headless",
        action="store_true",
    )

    args = parser.parse_args()

    # Same shortcuts used by train_srbd.py.
    _NAME_SHORTCUTS = {
        "go1": "Go1JoystickFlatTerrain",
        "aliengo": "AliengoJoystickE2EFlatTerrain",
        "tita": "TitaJoystickFlatTerrain",
        "titae2e": "TitaJoystickE2EFlatTerrain",
    }
    env_name = _NAME_SHORTCUTS.get(
        args.name.lower(),
        args.name,
    )

    if (
        os.environ.get("DISPLAY") is None
        or os.environ.get("DISPLAY") == ""
    ):
        print(
            "[WARN] No DISPLAY detected: "
            "forcing --headless mode (no viewer)"
        )
        args.headless = True

    main(
        env_name=env_name,
        ckpt_root=args.ckpt_dir,
        load_arg=args.load,
        headless=args.headless,
        steps=args.steps,
    )