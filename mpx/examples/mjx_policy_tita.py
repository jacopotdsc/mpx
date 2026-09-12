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
import functools
import os
import pickle
import re
import sys
import time
from timeit import default_timer as timer

# This tester shares one GPU between JAX and TWO OpenGL contexts (the offscreen
# mujoco.Renderer used for the video and the passive viewer window). JAX's
# default preallocation grabs 75% of VRAM (6.3 GB of 8 GB on this box), and
# XLA allocates compiled-executable *constants* straight from the driver rather
# than from that pool -- so once MuJoCo's GL contexts have eaten what is left,
# even a 192 KB constant fails with
#   XlaRuntimeError: INTERNAL: Failed to allocate N bytes for new constant
# Halving the pool leaves room for GL + the constants. Must be set before the
# JAX backend is created (i.e. before the first device use), hence up here.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")

import jax

jax.config.update("jax_enable_x64", True)

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, "..")))
CKPT_DIR = "."

import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np
from etils import epath

from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from mujoco_playground import registry
from mujoco_playground._src import locomotion

import mpx.utils.sim as sim_utils
from plot_rollout_info import _save_sim_video
from plot_validation import (
    SimLogger,
    plot_velocity_tracking,
    plot_velocity_error,
    plot_com_and_forces,
    plot_all,
)


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
HEIGHT_TARGET_INIT = 0.4
HEIGHT_TARGET_STEP = 0.1

KEY_PAGE_UP = 266
KEY_PAGE_DOWN = 267

# Lateral (vy) keys, used only on envs whose command is [vx, vy, wz]
# (quadrupeds such as Lite3).
#
# NOT letters: MuJoCo's own viewer UI reserves every key A-Z for its
# visualization/rendering flags (mjVISSTRING/mjRNDSTRING -- 'A' is Auto Connect,
# 'D' is Static Body), so a letter binding is swallowed before reaching this
# callback. Home/End and PageUp/PageDown are free, and PageUp/PageDown are known
# to arrive here because TITA already drives its height target with them.
KEY_HOME = 268
KEY_END = 269
LATERAL_STEP = 0.1
_RUN_DIR_RE = re.compile(r"^\d{8}_\d{6}$")

# --scene: which terrain XML to load. The environment class stays the one
# --name selects (the joystick env with all its rewards/obs/commands); only the
# scene XML handed to its constructor changes. Every robot ships the same four
# scenes under <robot>/xmls/, so these aliases work for TITA and the quadrupeds
# alike. Anything else passed to --scene is treated as a file name inside that
# same xmls/ directory, or as an explicit path to an XML.
DEFAULT_SCENE = "flat"
SCENE_ALIASES = {
    "flat": "scene_flat.xml",
    "rough": "scene_rough.xml",
    "stairs": "scene_stairs.xml",
    "perlin": "scene_perlin.xml",
}


# -----------------------------------------------------------------------------
# Scene selection (--scene).
# -----------------------------------------------------------------------------

def _env_consts_module(env_name: str):
    """The *_constants module of the env class registered under env_name.

    Every locomotion env in the registry builds its model as
    `xml_path=consts.task_to_xml(task)`, and imports that constants module as
    `consts` in its own module -- so this is what we have to look at (and, in
    load_env_with_scene, temporarily redirect) to change the scene.
    """
    try:
        ctor = locomotion._envs[env_name]  # pylint: disable=protected-access
    except KeyError as exc:
        raise ValueError(
            f"Env '{env_name}' is not a locomotion env, so --scene cannot "
            f"select its terrain XML."
        ) from exc

    cls = ctor.func if isinstance(ctor, functools.partial) else ctor
    module = sys.modules[cls.__module__]
    consts = getattr(module, "consts", None)
    if consts is None or not hasattr(consts, "ROOT_PATH"):
        raise ValueError(
            f"Env '{env_name}' ({cls.__module__}) exposes no 'consts' module "
            f"with a ROOT_PATH; --scene is not supported for it."
        )
    return consts


def _resolve_scene_xml(consts, scene: str) -> str:
    """Map --scene onto an actual XML file. Returns a posix path."""
    xml_dir = consts.ROOT_PATH / "xmls"

    # An explicit path wins over the aliases.
    if os.sep in scene or scene.startswith("."):
        path = os.path.abspath(os.path.expanduser(scene))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Scene XML not found: {path}")
        return path

    file_name = SCENE_ALIASES.get(scene.lower(), scene)
    if not file_name.endswith(".xml"):
        file_name += ".xml"

    path = xml_dir / file_name
    if not path.exists():
        available = sorted(
            p.name for p in xml_dir.iterdir() if p.name.startswith("scene_")
        )
        raise FileNotFoundError(
            f"Scene XML not found: {path.as_posix()}\n"
            f"  aliases : {sorted(SCENE_ALIASES)}\n"
            f"  in {xml_dir.as_posix()}: {available}"
        )
    return path.as_posix()


def _env_has_config_key(env_name: str, key: str) -> bool:
    """Whether this env's default config declares `key`.

    registry.load rejects an override for a key the config does not have, and
    only the residual envs declare enable_residual -- so ask before overriding.
    """
    try:
        return key in registry.get_default_config(env_name)
    except Exception:
        return False


def load_env_with_scene(env_name: str, scene: str, config_overrides: dict):
    """registry.load(env_name), but with the terrain XML forced to --scene.

    The env is built exactly as registry.load builds it -- same class, same
    task, same config -- so rewards, observations, command layout and the
    checkpoint's obs/action sizes are untouched. Only the XML string its
    constructor reads is redirected, by temporarily swapping the constants
    module's task_to_xml (which is the single place every locomotion env looks
    up its scene). The original is restored right after construction, so
    nothing else in the process sees the patch.
    """
    consts = _env_consts_module(env_name)
    xml_path = _resolve_scene_xml(consts, scene)

    original_task_to_xml = consts.task_to_xml
    consts.task_to_xml = lambda task_name: epath.Path(xml_path)
    try:
        env = registry.load(env_name, config_overrides=config_overrides)
    except NotImplementedError as exc:
        # MJX ships no collision kernel for some geom pairs. TITA's wheels are
        # cylinders, and (cylinder, box) / (hfield, cylinder) are among the
        # missing ones -- so every non-flat TITA scene fails here, exactly as
        # registry.load("TitaJoystickRoughTerrain") does. Nothing to do with
        # --scene itself; the quadrupeds (capsule/sphere feet) load all four.
        raise NotImplementedError(
            f"{exc}\n"
            f"  Scene '{scene}' ({xml_path}) cannot run in MJX with this "
            f"robot: the pair above has no MJX collision kernel."
        ) from exc
    finally:
        consts.task_to_xml = original_task_to_xml

    print(f"  [scene] --scene {scene} -> {xml_path}")
    return env


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

    Read vx/vy/wz straight off the handle. Do NOT route this through
    mpc_wheeled_input(), which returns the wheeled-MPC layout
    [vx, vz, wz, com_z]: its second entry is the *vertical* velocity, so a
    quadruped would silently receive vz where it expects vy.

    TITA's command is [forward_vel, yaw_rate] = [vx, wz] (cmd_dim == 2) and has
    no lateral velocity. A quadruped (Lite3, Go1, Aliengo) uses
    [vx, vy, wz] (cmd_dim == 3).
    """
    vx = float(command_handle.vx)
    vy = float(command_handle.vy)
    wz = float(command_handle.wz)

    if cmd_dim == 2:
        return np.array([vx, wz], dtype=np.float32)
    if cmd_dim == 3:
        return np.array([vx, vy, wz], dtype=np.float32)
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

def set_base_height_target(state, height_target: float, height_min: float, height_max: float):
    """Set the interactive CoM-height target.

    Envs without a base-height command (any quadruped joystick env: Lite3, Go1,
    Aliengo) have no "base_height_target" in info, so this is a no-op there.
    """
    if "base_height_target" not in state.info:
        return state

    target = jnp.full_like(
        state.info["base_height_target"],
        jnp.clip(height_target, height_min, height_max),
    )

    return state.replace(info={
        **state.info,
        "base_height_target": target,
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


def build_hud(
    command_vec,
    target_vec,
    actual_vec,
    actual_height,
    target_height,
    command_handle,
    lateral_step=None,
    lateral_limits=None,
):
    """actual_height/target_height may be None on envs without a height command.
    lateral_step/lateral_limits are set only when the env commands vy."""
    n = command_vec.shape[-1]

    labels = {
        2: ("vx", "wz"),
        3: ("vx", "vy", "wz"),
    }.get(n, tuple(f"c{i}" for i in range(n)))

    # ----------------------------------------------------------
    # LEFT: keyboard controls
    # ----------------------------------------------------------
    
    fwd_step = command_handle.forward_step
    yaw_step = command_handle.yaw_step

    fwd_min, fwd_max = command_handle.forward_limits
    yaw_min, yaw_max = command_handle.yaw_limits

    has_lateral = lateral_step is not None and lateral_limits is not None
    has_height_row = actual_height is not None and target_height is not None

    key_rows = ["Up/Down"]
    cmd_rows = [
        f"forward     {fwd_step:.2f}      {fwd_min:+.1f} / {fwd_max:+.1f}"
    ]
    if has_lateral:
        lat_min, lat_max = lateral_limits
        key_rows.append("Home/End or PgUp/PgDn")
        cmd_rows.append(
            f"lateral     {lateral_step:.2f}      {lat_min:+.1f} / {lat_max:+.1f}"
        )
    key_rows.append("Left/Right")
    cmd_rows.append(
        f"yaw         {yaw_step:.2f}      {yaw_min:+.1f} / {yaw_max:+.1f}"
    )
    key_rows.append("Space")
    cmd_rows.append("stop        --        --")
    if has_height_row:
        key_rows.append("PageUp/Down")
        cmd_rows.append(f"height      {HEIGHT_TARGET_STEP:.2f}      --")

    help_left = "\n".join(["Key", *key_rows])
    help_right = "\n".join(["Command     Step      Min / Max", *cmd_rows])
    # ----------------------------------------------------------
    # RIGHT: tracking values
    # ----------------------------------------------------------
    # The height row only exists on envs with a base-height command (TITA).
    has_height = has_height_row

    title_lines = [
        "",
        *labels,
    ]
    if has_height:
        title_lines.append("height")

    value_lines = [
        " Actual   Command   Target",
    ]

    for actual, command, target in zip(
        actual_vec, command_vec, target_vec
    ):
        value_lines.append(
            f"{float(actual):+7.2f}  "
            f"{float(command):+7.2f}  "
            f"{float(target):+7.2f}"
        )

    if has_height:
        value_lines.append(
            f"{float(actual_height):+7.2f}  "
            f"{float(target_height):+7.2f}  "
            f"{float(target_height):+7.2f}"
        )

    return [
        (
            mujoco.mjtFont.mjFONT_NORMAL,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            help_left,
            help_right,
        ),
        (
            mujoco.mjtFont.mjFONT_NORMAL,
            mujoco.mjtGridPos.mjGRID_TOPRIGHT,
            "\n".join(title_lines),
            "\n".join(value_lines),
        ),
    ]
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
    zero: bool = False,
    scene: str = DEFAULT_SCENE,
    baseline: bool = False,
):
    global CKPT_DIR

    # Both flags drive the env with a zero action; the difference is the
    # network. --zero loads it and then ignores it, --baseline never builds it.
    zero_action_only = zero or baseline

    print("=" * 60)
    print(f"  Interactive policy test — {env_name} (scene: {scene})")
    if baseline:
        print("  [NET-INIT] --baseline flag: MPC + WBC only, no policy network loaded.")
    elif zero:
        print("  [NET-INIT] --zero flag: forcing zero actions (ignoring policy network).")
    print("=" * 60)

    # --name: same MuJoCo Playground environment as training/eval, but with a
    # very large episode_length so the interactive test never ends on a time
    # limit. Physical termination (state.done) still works. This override is
    # local to this tester; training/eval defaults are untouched.
    #
    # --scene only swaps the terrain XML the very same env class loads; with
    # the default (flat) this is exactly registry.load(env_name).
    config_overrides = {"episode_length": EPISODE_LENGTH}
    if baseline and _env_has_config_key(env_name, "enable_residual"):
        # The residual branch is a PD around the nominal stance, so it keeps
        # applying torque even at action = 0. Turning it off at the env is what
        # actually leaves the MPC + whole-body controller alone.
        config_overrides["enable_residual"] = False
        print("  [NET-INIT] env config: enable_residual=False (tau_rl x 0.0).")
    elif baseline:
        print(
            f"  [WARN] {env_name} has no 'enable_residual' config key: the "
            f"action is zeroed, but whatever its low-level controller does at "
            f"action=0 still runs."
        )
    env = load_env_with_scene(env_name, scene, config_overrides=config_overrides)
    sim_logger = SimLogger()

    env_base_dir = os.path.join(ckpt_root, env_name)
    if baseline:
        # Nothing to load: no network, so no checkpoint is needed and the run
        # works on a tree that has none. Outputs go directly under the env
        # directory rather than inside some checkpoint's run folder, which the
        # baseline has nothing to do with.
        run_dir, load_suffix = env_base_dir, None
    else:
        # --load: same checkpoint layout/resolution as train_srbd.py.
        run_dir, load_suffix = _resolve_load(env_base_dir, load_arg)

    CKPT_DIR = run_dir
    run_tag = time.strftime("run_%Y%m%d_%H%M%S") + ("_baseline" if baseline else "")
    joystick_eval_dir = os.path.join(CKPT_DIR, "joystick_evaluation", run_tag)
    os.makedirs(joystick_eval_dir, exist_ok=True)

    print(f"  [INFO] Joystick evaluation directory: {joystick_eval_dir}")

    if baseline:
        jit_infer = None
    else:
        params = load_params(CKPT_DIR, suffix=load_suffix)
        jit_infer = jax.jit(build_policy(env, params))

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

    @jax.jit
    @jax.vmap
    def batched_get_obs(data, info, action):
        return env._get_obs(data, info, action)

    _zero_actions = jnp.zeros(
        (EVAL_BATCH, action_size),
        dtype=jnp.float32,
    )
    # Reset — identical RNG to train_srbd.py --eval.
    rng = jax.random.PRNGKey(42)
    rng, *reset_rngs = jax.random.split(rng, EVAL_BATCH + 1)
    state = batched_reset(jnp.stack(reset_rngs))

    height_target = HEIGHT_TARGET_INIT
    cmd_dim = int(state.info["command"].shape[-1])

    # Env capabilities. TITA carries a base-height command and the DFCIP/WBC
    # logging buffers; the quadruped joystick envs (Lite3, Go1, Aliengo) carry
    # neither, so every block that needs them is guarded on these flags.
    has_height_cmd = "base_height_target" in state.info
    has_tita_logging = "tita_state" in state.info

    # Lite3/quadrupeds command [vx, vy, wz]; TITA commands [vx, wz] and has no
    # lateral velocity at all. Bind vy only where the env actually has it, and
    # take its range from the env's own command_config so the keyboard cannot
    # ask for a vy the policy was never trained on.
    _cmd_amp = np.asarray(env._config.command_config.a, dtype=np.float32)
    has_lateral_cmd = cmd_dim == 3
    lateral_limits = (
        (-float(_cmd_amp[1]), float(_cmd_amp[1])) if has_lateral_cmd else None
    )
    _h = getattr(env._config.command_config, "h", (0.0, 0.0))
    height_min, height_max = float(_h[0]), float(_h[1])
    print(
        f"  [env] cmd_dim={cmd_dim} | base-height command: "
        f"{'yes' if has_height_cmd else 'no'} | DFCIP logging: "
        f"{'yes' if has_tita_logging else 'no'}"
    )

    command_handle = sim_utils.KeyboardVelocityCommand(
        vx=0.0,
        vy=0.0,
        wz=0.0,
        forward_step=0.1,
        yaw_step=0.2,
        forward_limits=(-10.0, 10.0),
        yaw_limits=(-1.5, 1.5),
    )

    def key_callback(keycode):
        nonlocal height_target

        # Height target (TITA only): PageUp/PageDown keep their meaning there.
        if has_height_cmd and keycode == KEY_PAGE_UP:
            height_target += HEIGHT_TARGET_STEP
            return

        if has_height_cmd and keycode == KEY_PAGE_DOWN:
            height_target -= HEIGHT_TARGET_STEP
            return

        # Lateral velocity (quadrupeds only). Handled here and not in sim.py,
        # which the TITA examples share: KeyboardVelocityCommand binds only vx
        # and wz, and its _clip() does not cover vy. On these envs PageUp/Down
        # are free (no height command), so they double as lateral keys.
        if has_lateral_cmd:
            if keycode in (KEY_HOME, KEY_PAGE_UP):
                step = LATERAL_STEP
            elif keycode in (KEY_END, KEY_PAGE_DOWN):
                step = -LATERAL_STEP
            else:
                step = None

            if step is not None:
                command_handle.vy = float(
                    np.clip(command_handle.vy + step, *lateral_limits)
                )
                return

        command_handle.key_callback(keycode)

    # Reset init: start command at 0, set target_command from the keyboard,
    # and disable the resampler. From here on the env's 0.02 smoothing moves
    # command toward target on its own. Then rebuild obs so step 0 is clean.
    keyboard_cmd = keyboard_target_command(command_handle, cmd_dim)
    state = state.replace(info={
        **state.info,
        "command": jnp.zeros_like(state.info["command"]),
    })
    state = set_target_command(state, keyboard_cmd, cmd_dim)
    state = set_base_height_target(
        state, height_target, height_min=height_min, height_max=height_max
    )

    state = state.replace(
        obs=batched_get_obs(
            state.data,
            state.info,
            _zero_actions,
        )
    )
    # Compile the policy and the env step HERE, while the GPU still has free
    # memory outside JAX's pool: both executables allocate their constants
    # (policy weights, MJX model arrays) directly from the driver, and the
    # OpenGL contexts created just below (offscreen renderer + viewer window)
    # take that same memory. Compiling first also removes the multi-second
    # hitch the first interactive step would otherwise show.
    #
    # A throwaway key is used on purpose: `rng` must stay on the exact stream
    # train_srbd.py --eval uses, so the warm-up may not consume from it.
    if jit_infer is not None and not zero:
        jax.block_until_ready(
            jit_infer(take_env0(state.obs), jax.random.PRNGKey(0))
        )
    jax.block_until_ready(batched_step(state, _zero_actions))

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
        state = set_base_height_target(
            state, height_target, height_min=height_min, height_max=height_max
        )

        state = state.replace(
            obs=batched_get_obs(
                state.data,
                state.info,
                state.info["last_act"],
            )
        )

        cmd_before = np.asarray(state.info["command"][0])
        tc_before = np.asarray(state.info["target_command"][0])
        steps_before = int(np.asarray(state.info["steps_until_next_cmd"]).reshape(-1)[0])

        if zero_action_only:
            # --zero ignores the policy network (matching train_srbd.py --zero);
            # --baseline never loaded one. Either way the env is driven with a
            # zero action every step.
            action = _zero_actions
        else:
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

        info = state.info

        # SimLogger consumes TITA's DFCIP/WBC buffers. Quadruped joystick envs
        # (Lite3, Go1, Aliengo) do not publish them, so skip the logging there;
        # the viewer, the video and the rollout itself are unaffected.
        if has_tita_logging:
            command = np.asarray(info["command"][0])

            # TITA command is [vx, wz]; a quadruped's is [vx, vy, wz].
            omega = command[1] if cmd_dim == 2 else command[2]

            logger_cmd = np.array([
                command[0],                                      # vx
                0.0,                                             # vz
                omega,                                           # omega
                float(np.asarray(info["base_height_target"][0])), # height
            ])

            sim_logger.append(
                t=int(np.asarray(info["step"][0])),
                model=viewer_model,
                data=viewer_data,
                tita_state=np.asarray(info["tita_state"][0]),
                x0=np.asarray(info["dfcip_state"][0]),
                cmd=logger_cmd,
                ext_force=np.asarray(info["robot"]["ext_force"][0]),
                fl=None,
                fr=None,
            )

        return bool(state.done[0])

    # bind batched_step to a local name for readability
    env_step = batched_step

    def finalize_outputs():
        print("\n[finalize] Saving outputs...")

        try:
            _renderer.close()
        except Exception:
            pass
        if has_tita_logging:
            try:
                plot_all(
                    sim_logger,
                    save_path=joystick_eval_dir,
                    show=False
                )
            except Exception as e:
                print(f"[finalize] error on generating all plots: {e}")
        else:
            print(
                "[finalize] DFCIP plots skipped: this env publishes no "
                "tita_state/dfcip_state (video and rollout are unaffected)."
            )
        try:
            for slow, name in (
                (1.0, "policy_simulation_video.mp4"),
                (4.0, "policy_simulation_video_slow4.mp4"),
                (15.0, "policy_simulation_video_slow15.mp4"),
            ):
                _save_sim_video(
                    video_dir=joystick_eval_dir,
                    video_fps=int(round(1.0 / env_dt)),
                    frames=_sim_frames,
                    slowdown_factor=slow,
                    name_video=name,
                )
        except Exception as exc:
            print(f"[finalize] failed to save video: {exc}")
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

        with mujoco.viewer.launch_passive(
            viewer_model,
            viewer_data,
            key_callback=key_callback,
        ) as viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.distance = 8.0
            viewer.cam.elevation = -15.0
            viewer.cam.azimuth = 60.0
            viewer.sync()

            i = 0
            while viewer.is_running():
                tic = timer()

                done = _step_once(debug=debug_cmd)
                sync_viewer_data(state, viewer_model, viewer_data)
                if i % 2 == 0:
                    _record_frame()

                # TITA republishes these in info["robot"]; elsewhere read them
                # straight off the env's sensors, which every locomotion env has.
                if "robot" in state.info:
                    local_linvel = np.asarray(
                        state.info["robot"]["local_linvel"][0]
                    )
                    gyro = np.asarray(state.info["robot"]["gyro"][0])
                else:
                    data0 = jax.tree_util.tree_map(lambda x: x[0], state.data)
                    local_linvel = np.asarray(env.get_local_linvel(data0))
                    gyro = np.asarray(env.get_gyro(data0))

                # One "actual" entry per command entry, so the HUD lines up:
                # [vx, wz] for TITA, [vx, vy, wz] for a quadruped.
                if cmd_dim == 2:
                    actual_vec = np.array([local_linvel[0], gyro[2]])
                else:
                    actual_vec = np.array(
                        [local_linvel[0], local_linvel[1], gyro[2]]
                    )

                if has_height_cmd:
                    actual_height = float(
                        np.asarray(state.info["robot"]["com_height"][0])
                    )
                    target_height = float(
                        np.asarray(state.info["base_height_target"][0])
                    )
                else:
                    actual_height = None
                    target_height = None

                viewer.set_texts(build_hud(
                    command_vec=np.asarray(state.info["command"][0]),
                    target_vec=np.asarray(state.info["target_command"][0]),
                    actual_vec=actual_vec,
                    actual_height=actual_height,
                    target_height=target_height,
                    command_handle=command_handle,
                    lateral_step=LATERAL_STEP if has_lateral_cmd else None,
                    lateral_limits=lateral_limits,
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
    parser.add_argument("--name", type=str, default="TitaJoystickFlatTerrain")
    parser.add_argument(
        "--load", nargs="?", const="best", default="best", metavar="RUN_OR_SUFFIX"
    )
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints")
    parser.add_argument(
        "--scene",
        type=str,
        default=DEFAULT_SCENE,
        metavar="SCENE",
        help="Terrain XML to load with the SAME joystick env selected by "
             f"--name. Aliases: {', '.join(sorted(SCENE_ALIASES))} "
             f"(default: {DEFAULT_SCENE}). Also accepts a file name inside "
             "the robot's xmls/ directory, or a path to an XML.",
    )
    parser.add_argument("--steps", type=int, default=EPISODE_LENGTH)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--zero", action="store_true",
        help="Force zero actions (ignore policy network), like train_srbd.py --zero.",
    )
    parser.add_argument(
        "--baseline", action="store_true",
        help="MPC + WBC only: no checkpoint is loaded and no policy network is "
             "built, and on envs that support it the residual branch is "
             "switched off at the env (enable_residual=False). Unlike --zero, "
             "this needs no checkpoint on disk.",
    )
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
        "lite": "Lite3JoystickFlatTerrain",
        "lite3": "Lite3JoystickFlatTerrain",
        "litee2e": "Lite3JoystickE2EFlatTerrain",
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
        zero=args.zero,
        baseline=args.baseline,
        scene=args.scene,
    )