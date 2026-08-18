"""
Interactive TITA policy test.

This is NOT a second implementation of the environment. It is an interactive
real-time frontend of *exactly* the same MJX environment used by
`train_srbd.py --eval`:

    state = env.reset(rng)                      # single source of truth
    while viewer.is_running():
        keyboard_cmd = <keyboard>               # keyboard sets ONLY the command
        state = set_command(state, keyboard_cmd)
        action, _ = policy(take_env0(state.obs))# deterministic inference
        state = env.step(state, action)         # all physics/obs/termination here
        state = set_command(state, keyboard_cmd)# defeat the env's random resampler
        sync_viewer(state)                      # native MjData = visualization mirror

The reset/step semantics, observation construction, actuator mapping, substeps,
termination and info buffers all live inside env.reset()/env.step(). Nothing of
that is re-implemented here.

Reset uses the same vmap(batch=1) + PRNGKey(42) path as `train_srbd.py --eval`
(single-instance MJX can trip cuSolver on GPU), so at equal seed/command the
initial state and the rollout match the eval path bit-for-bit up to the command
the keyboard injects.

From train_srbd.py this file reuses:
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
import numpy as np

from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from mujoco_playground import registry

import mpx.utils.sim as sim_utils
from plot_rollout_info import _save_sim_video


jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1)


# Must match train_srbd.py / the checkpoint.
POLICY_HIDDEN_LAYER_SIZES = (512, 256, 128)
DISTRIBUTION_TYPE = "tanh_normal"

# Single-instance MJX can crash cuSolver on GPU, so train_srbd.py --eval runs a
# batch and views env 0. We mirror that exactly for numerical parity.
EVAL_BATCH = 1

# Interactive test only: run effectively forever. The env has no time-limit
# termination inside step() (only physical fall/base-contact via state.done),
# so this just uncaps the tester loop. Applied via a config override on
# registry.load so joystickE2E.default_config().episode_length is untouched.
EPISODE_LENGTH = 10_000_000

# Interactive test only: keep steps_until_next_cmd this high (and re-assert it
# every step) so the env's automatic command resampler (sample_command) NEVER
# fires. The keyboard becomes the sole authority over target_command; the env's
# own smoothing (command += 0.02*(target_command - command)) still runs.
NO_RESAMPLE_STEPS = 10_000_000

_RUN_DIR_RE = re.compile(r"^\d{8}_\d{6}$")


# -----------------------------------------------------------------------------
# Checkpoint / policy loading. Copied conservatively from train_srbd.py.
# -----------------------------------------------------------------------------

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
    """Same --load semantics used by train_srbd.py."""
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
                    f"\n  [INFO] Available runs under '{env_base_dir}': "
                    f"\n{_list_dir_names(env_base_dir)}"
                )
                print(
                    f"\n  [INFO] Available saved runs under "
                    f"'{os.path.join(env_base_dir, 'saved')}': "
                    f"\n{_list_dir_names(os.path.join(env_base_dir, 'saved'))}\n"
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
    """Identical PPO network construction to train_srbd.py's eval path."""
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
# Tester-specific helpers (NOT copies of the environment).
# -----------------------------------------------------------------------------

def keyboard_target_command(command_handle, cmd_dim: int) -> np.ndarray:
    """Read the keyboard and map it onto the env's command layout.

    KeyboardVelocityCommand.mpc_wheeled_input() starts with [vx, vy, wz].
    TITA's command is [forward_vel, yaw_rate] = [vx, wz] (cmd_dim == 2);
    a quadruped uses [vx, vy, wz] (cmd_dim == 3).
    """
    kbd = np.asarray(
        command_handle.mpc_wheeled_input(0.0), dtype=np.float32
    ).reshape(-1)
    if kbd.size < 3:
        raise ValueError(
            "KeyboardVelocityCommand.mpc_wheeled_input() returned < 3 values."
        )
    if cmd_dim == 2:
        return np.array([kbd[0], kbd[2]], dtype=np.float32)  # [vx, wz]
    if cmd_dim == 3:
        return kbd[:3].astype(np.float32)                    # [vx, vy, wz]
    raise ValueError(
        f"Unsupported command dimension: {cmd_dim} (expected 2 or 3)."
    )


def set_target_command(state, keyboard_cmd: np.ndarray, cmd_dim: int):
    """Make the keyboard the SOLE authority over target_command, and disable
    the env's automatic resampler — without ever touching info['command'].

    The deployed env, inside step(), does:
        target_command = where(steps_until_next_cmd <= 0,
                                sample_command(...), target_command)
        command = command + 0.02 * (target_command - command)

    So we only need to (1) write target_command = keyboard and (2) keep
    steps_until_next_cmd huge so the `<= 0` branch never triggers. The env's
    own smoothing then drives command toward the keyboard target on its own.

    Crucially we do NOT write info['command'] here: letting the env's 0.02
    filter compute it is the whole point (the previous version overwrote
    command every step, which killed the smoothing and bypassed the env law).
    """
    tc = jnp.broadcast_to(
        jnp.asarray(keyboard_cmd, dtype=state.info["target_command"].dtype),
        (EVAL_BATCH, cmd_dim),
    )
    big = jnp.full_like(
        state.info["steps_until_next_cmd"], NO_RESAMPLE_STEPS
    )
    return state.replace(info={
        **state.info,
        "target_command": tc,
        "steps_until_next_cmd": big,
    })


def sync_viewer_data(state, viewer_model, viewer_data):
    """Native MjData = pure visualization mirror of the MJX state (env 0)."""
    viewer_data.qpos[:] = np.asarray(state.data.qpos[0])
    viewer_data.qvel[:] = np.asarray(state.data.qvel[0])
    mujoco.mj_forward(viewer_model, viewer_data)


def take_env0(obs):
    """Select env 0 from a batched observation.

    The env returns obs as a dict ({'state': ..., 'privileged_state': ...}),
    so we must index per key, exactly like train_srbd.py's eval helper. Falls
    back to plain array indexing if obs is a flat array.
    """
    if isinstance(obs, dict):
        return {k: v[0] for k, v in obs.items()}
    return obs[0]


def build_hud(command_vec, target_vec, help_text):
    """One set_texts() sequence, rebuilt every frame: keyboard help (TOPLEFT)
    plus a single Command + Target-command block (TOPRIGHT).

    command_vec  <- state.info['command']         (env-smoothed, moving)
    target_vec   <- state.info['target_command']  (held at the keyboard value)

    Both live in ONE overlay entry so a later set_texts for the keyboard help
    cannot wipe them: everything goes out together in a single call.
    """
    n = command_vec.shape[-1]
    labels = {2: ("vx", "wz"), 3: ("vx", "vy", "wz")}.get(
        n, tuple(f"c{i}" for i in range(n))
    )

    # Left column = labels/headers, right column = values, line-aligned.
    title_lines = ["Command"]
    value_lines = [""]
    for lab, v in zip(labels, command_vec):
        title_lines.append(f"  {lab}")
        value_lines.append(f"{float(v):+.2f}")
    title_lines += ["", "Target command"]
    value_lines += ["", ""]
    for lab, v in zip(labels, target_vec):
        title_lines.append(f"  {lab}")
        value_lines.append(f"{float(v):+.2f}")

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


# -----------------------------------------------------------------------------
# Main.
# -----------------------------------------------------------------------------

def main(
    env_name: str,
    ckpt_root: str,
    load_arg: str,
    headless: bool = False,
    steps: int = EPISODE_LENGTH,
    debug_cmd: bool = False,
):
    print("=" * 60)
    print(f"  Interactive policy test — {env_name}")
    print("=" * 60)

    # --name: same MuJoCo Playground environment as training/eval, but with a
    # very large episode_length so the interactive test never ends on a time
    # limit. Physical termination (state.done) still works. This override is
    # local to this tester; training/eval defaults are untouched.
    env = registry.load(
        env_name,
        config_overrides={"episode_length": EPISODE_LENGTH},
    )

    # --load: same checkpoint layout/resolution as train_srbd.py.
    env_base_dir = os.path.join(ckpt_root, env_name)
    run_dir, load_suffix = _resolve_load(env_base_dir, load_arg)
    params = load_params(run_dir, suffix=load_suffix)

    policy_fn = build_policy(env, params)
    jit_infer = jax.jit(policy_fn)

    # Same vmapped reset/step used by train_srbd.py --eval.
    batched_reset = jax.jit(jax.vmap(env.reset))
    batched_step = jax.jit(jax.vmap(env.step))

    action_size = env.action_size
    env_dt = float(env.dt)

    print(f"  sim dt    : {env.sim_dt}")
    print(f"  ctrl dt   : {env_dt}")
    print(f"  n_substeps: {env.n_substeps} (handled inside env.step)")

    # One-off obs recompute so the FIRST action reflects the keyboard command
    # instead of the env's random reset command. Uses the env's own _get_obs
    # with a zero previous action (exactly what reset() feeds it).
    _zeros_act = jnp.zeros((action_size,), dtype=jnp.float32)

    @jax.jit
    @jax.vmap
    def batched_reset_obs(data, info):
        return env._get_obs(data, info, _zeros_act)

    # Reset — identical RNG to train_srbd.py --eval.
    rng = jax.random.PRNGKey(42)
    rng, *reset_rngs = jax.random.split(rng, EVAL_BATCH + 1)
    state = batched_reset(jnp.stack(reset_rngs))

    cmd_dim = int(state.info["command"].shape[-1])

    command_handle = sim_utils.KeyboardVelocityCommand(
        vx=0.0,
        vy=0.0,
        wz=0.0,
        forward_step=0.1,
        yaw_step=0.2,
        forward_limits=(-10.0, 10.0),
        yaw_limits=(-1.5, 1.5),
    )

    # Reset init: start command at 0, set target_command from the keyboard,
    # and disable the resampler. From here on the env's 0.02 smoothing moves
    # command toward target on its own. Then rebuild obs so step 0 is clean.
    keyboard_cmd = keyboard_target_command(command_handle, cmd_dim)
    state = state.replace(info={
        **state.info,
        "command": jnp.zeros_like(state.info["command"]),
    })
    state = set_target_command(state, keyboard_cmd, cmd_dim)
    state = state.replace(obs=batched_reset_obs(state.data, state.info))

    # Native MjModel/MjData used ONLY as a visualization mirror.
    viewer_model = env.mj_model
    viewer_data = mujoco.MjData(viewer_model)
    sync_viewer_data(state, viewer_model, viewer_data)

    np.set_printoptions(precision=4, suppress=True)

    # ------------------------------------------------------------------
    # Video recording (renders the mirror MjData).
    # ------------------------------------------------------------------
    _sim_frames: list = []
    _renderer = mujoco.Renderer(viewer_model, height=480, width=640)
    _sim_cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(_sim_cam)
    _sim_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    _sim_cam.distance = 8.0
    _sim_cam.elevation = -15.0
    _sim_cam.azimuth = 60.0

    _base_body_id = env._torso_body_id
    mujoco.mj_subtreeVel(viewer_model, viewer_data)
    _sim_cam.lookat[:] = np.asarray(viewer_data.subtree_com[_base_body_id])

    def _record_frame():
        mujoco.mj_subtreeVel(viewer_model, viewer_data)
        com = np.asarray(viewer_data.subtree_com[_base_body_id]).copy()
        _sim_cam.lookat[:] = 0.9 * _sim_cam.lookat + 0.1 * com
        _renderer.update_scene(viewer_data, camera=_sim_cam)
        _sim_frames.append(_renderer.render().copy())

    def _step_once(debug: bool = False):
        """policy(state.obs) -> env.step. Keyboard drives ONLY target_command;
        the env's 0.02 smoothing drives command. Resampler kept disabled."""
        nonlocal state, rng, keyboard_cmd

        keyboard_cmd = keyboard_target_command(command_handle, cmd_dim)

        # Before step: pin target_command to the keyboard and keep the
        # resampler off. command is left untouched (env smooths it in step()).
        state = set_target_command(state, keyboard_cmd, cmd_dim)

        cmd_before = np.asarray(state.info["command"][0])
        tc_before = np.asarray(state.info["target_command"][0])
        steps_before = int(np.asarray(state.info["steps_until_next_cmd"]).reshape(-1)[0])

        obs0 = take_env0(state.obs)
        rng, act_rng = jax.random.split(rng)
        action0, _ = jit_infer(obs0, act_rng)
        action = jnp.broadcast_to(action0, (EVAL_BATCH, action_size))

        state = env_step(state, action)

        if debug:
            cmd_after = np.asarray(state.info["command"][0])
            tc_after = np.asarray(state.info["target_command"][0])
            steps_after = int(np.asarray(state.info["steps_until_next_cmd"]).reshape(-1)[0])
            print(
                f"[cmd] kbd={keyboard_cmd} | "
                f"target {tc_before}->{tc_after} | "
                f"command {cmd_before}->{cmd_after} | "
                f"steps_until_next_cmd {steps_before}->{steps_after}"
            )

        return bool(state.done[0])

    # bind batched_step to a local name for readability
    env_step = batched_step

    def finalize_outputs():
        print("\n[finalize] Saving outputs...")
        try:
            for slow, name in (
                (1.0, "policy_simulation_video.mp4"),
                (4.0, "policy_simulation_video_slow4.mp4"),
                (15.0, "policy_simulation_video_slow15.mp4"),
            ):
                _save_sim_video(
                    video_dir=TITA_PATH,
                    video_fps=int(round(1.0 / env_dt)),
                    frames=_sim_frames,
                    slowdown_factor=slow,
                    name_video=name,
                )
        except Exception as exc:
            print(f"[finalize] failed to save video: {exc}")
        try:
            _renderer.close()
        except Exception:
            pass

    try:
        if headless:
            for i in range(steps):
                done = _step_once(debug=debug_cmd)
                sync_viewer_data(state, viewer_model, viewer_data)
                if i % 2 == 0:
                    _record_frame()
                print(
                    f"step {i:4d} | cmd {np.asarray(state.info['command'][0])} "
                    f"| done {done}"
                )
                if done:
                    print(f"Episode ended (state.done) at step {i}.")
                    break
            return

        help_cache = None
        with mujoco.viewer.launch_passive(
            viewer_model,
            viewer_data,
            key_callback=command_handle.key_callback,
        ) as viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.distance = 8.0
            viewer.cam.elevation = -15.0
            viewer.cam.azimuth = 60.0
            viewer.sync()

            i = 0
            while viewer.is_running():
                tic = timer()

                ov = command_handle.consume_overlay_text()
                if ov is not None:
                    help_cache = ov

                done = _step_once(debug=debug_cmd)
                sync_viewer_data(state, viewer_model, viewer_data)
                if i % 2 == 0:
                    _record_frame()

                viewer.set_texts(build_hud(
                    command_vec=np.asarray(state.info["command"][0]),
                    target_vec=np.asarray(state.info["target_command"][0]),
                    help_text=help_cache,
                ))

                com = np.asarray(viewer_data.subtree_com[_base_body_id]).copy()
                viewer.cam.lookat[:] = 0.9 * viewer.cam.lookat + 0.1 * com
                viewer.sync()

                if done:
                    print(f"Episode ended (state.done) at step {i}.")
                    break

                i += 1
                elapsed = timer() - tic
                if elapsed < env_dt:
                    time.sleep(env_dt - elapsed)

    except KeyboardInterrupt:
        print("\n[interrupt] Ctrl+C received. Finalizing outputs...")
    finally:
        finalize_outputs()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="TitaJoystickE2EFlatTerrain")
    parser.add_argument(
        "--load", nargs="?", const="best", default="best", metavar="RUN_OR_SUFFIX"
    )
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints")
    parser.add_argument("--steps", type=int, default=EPISODE_LENGTH)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--debug-cmd",
        action="store_true",
        help="Print keyboard target / target_command / command / "
             "steps_until_next_cmd before and after each env.step().",
    )
    args = parser.parse_args()

    _NAME_SHORTCUTS = {
        "go1": "Go1JoystickFlatTerrain",
        "aliengo": "AliengoJoystickE2EFlatTerrain",
        "tita": "TitaJoystickFlatTerrain",
        "titae2e": "TitaJoystickE2EFlatTerrain",
    }
    env_name = _NAME_SHORTCUTS.get(args.name.lower(), args.name)

    if not os.environ.get("DISPLAY"):
        print("[WARN] No DISPLAY detected: forcing --headless mode (no viewer)")
        args.headless = True

    main(
        env_name=env_name,
        ckpt_root=args.ckpt_dir,
        load_arg=args.load,
        headless=args.headless,
        steps=args.steps,
        debug_cmd=args.debug_cmd,
    )