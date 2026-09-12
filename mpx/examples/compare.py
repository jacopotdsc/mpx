"""Compare an MPC baseline and a residual policy for velocity tracking.

The environment is selected with --name (default: the Lite3 joystick env). The
script runs one continuous sequence (five seconds per fixed command) for the
baseline, residual policy, and optional end-to-end policy. It saves tracking
data, one CSV plus individual plots for every reward term, and one complete
video per controller:

    <run>/baseline/rewards/<test>_rewards.csv
    <run>/baseline/rewards/<test>/<reward>.png
    <run>/residual/rewards/...
    <run>/end_to_end/rewards/...             # unless --no-e2e
    <run>/compare_rewards/<test>/<reward>_comparison.png

The whole sequence is repeated on every scene in DEFAULT_SCENES, so <test> above
reads "<command>__<scene>", e.g. "vx_1p5__flat_terrain". The scenes are run one
at a time (all the tests of one, then the next), because switching reloads the
MuJoCo model.

The command layout (which components exist and in which order) is read from the
selected env's command_config.names, so it is never hardcoded here; an env that
does not publish those names gets a numbered v1 ... vn layout instead.

Examples:
    python compare.py
    python compare.py --name tita
    python compare.py --load
    python compare.py --load 20260907_101530
    python compare.py --load saved/my_run
    python compare.py --load final
    python compare.py --load --load-e2e saved/my_e2e_run
    python compare.py --load --no-e2e
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import time
from datetime import datetime
from pathlib import Path

# Configure MuJoCo before importing it so off-screen rendering works headlessly.
if not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.sac import networks as sac_networks

import train_srbd


# Short aliases for the environments this script can evaluate, mirroring the
# mapping used in train_srbd.py so the same names work in both places.
_NAME_SHORTCUTS = {
    "go1": "Go1JoystickFlatTerrain",
    "aliengo": "AliengoJoystickE2EFlatTerrain",
    "tita": "TitaJoystickFlatTerrain",
    "titae2e": "TitaJoystickE2EFlatTerrain",
    "lite3": "Lite3JoystickFlatTerrain",
    "lite3e2e": "Lite3JoystickE2EFlatTerrain",
}
DEFAULT_ENV_NAME = "Lite3JoystickFlatTerrain"

TEST_DURATION_SECONDS = 5.0

# How each command component is measured, keyed by the names the env publishes
# in command_config.names: linear-velocity commands are read from the base-local
# linear velocity, angular ones from the gyroscope. Kept as a lookup (not a
# fixed [vx, vy, wz] assumption) so the pipeline follows whatever command layout
# the selected env declares.
_MEASURED_SOURCE = {
    "vx": ("linvel", 0, "m/s"),
    "vy": ("linvel", 1, "m/s"),
    "vz": ("linvel", 2, "m/s"),
    "wx": ("gyro", 0, "rad/s"),
    "wy": ("gyro", 1, "rad/s"),
    "wz": ("gyro", 2, "rad/s"),
}

# Positional fallback for a command component whose name is not in
# _MEASURED_SOURCE: the [vx, vy, vz, wx, wy, wz] layout, read by index.
_MEASURED_BY_INDEX = list(_MEASURED_SOURCE.values())


def _command_names(config, fallback_length: int) -> list[str]:
    """Names of the command components, in the order the env uses them.

    Envs normally publish them in command_config.names. When that field (or the
    whole command_config) is missing, fall back to a numbered layout v1 ... vn,
    so the CSV columns, plot labels and video HUD still have one label per
    component. The length then comes from command_config.a (one amplitude per
    component) and, failing that, from the width of the declared tests."""
    command_config = getattr(config, "command_config", None)
    if command_config is None:
        return [f"v{i + 1}" for i in range(fallback_length)]

    names = command_config.get("names", None)
    if names:
        return list(names)

    length = len(command_config.get("a", ())) or fallback_length
    return [f"v{i + 1}" for i in range(length)]


def _measured_source(name: str, index: int) -> tuple[str, int, str]:
    """Sensor, channel and unit a command component is compared against.

    A named component is looked up in _MEASURED_SOURCE; an unnamed one (the
    v1 ... vn fallback) is read positionally from the [vx, vy, vz, wx, wy, wz]
    layout, which is what the numbered order implies."""
    if name in _MEASURED_SOURCE:
        return _MEASURED_SOURCE[name]
    if index >= len(_MEASURED_BY_INDEX):
        raise ValueError(
            f"Command component '{name}' (index {index}) has no known measured "
            f"source: name it in command_config.names, or add it to "
            f"_MEASURED_SOURCE."
        )
    return _MEASURED_BY_INDEX[index]

# Scenes the whole test sequence is repeated over, as the env's `task` names.
# Add or remove entries to change the terrains every test is run on; each one
# multiplies the run time, and the length of the videos, by roughly one pass.
DEFAULT_SCENES = ("flat_terrain", "rough_terrain", "perlin_terrain")

# Fixed command sequence to run, per environment. Each entry is a
# (name, command) pair, optionally extended to (name, command, scene).
#
# - A 1D command runs a single 5-second block (one episode).
# - A 2D command (several rows) runs as ONE episode that lasts
#   5 * n_rows seconds: the command target switches to the next row every
#   TEST_DURATION_SECONDS, WITHOUT resetting the robot in between. This is the
#   way to probe a real command change (e.g. going from a velocity down to 0).
# Each command row must match the env's command_config layout (order and length
# of command_config.names).
# - The optional third element is the scene, i.e. the env's `task` name
#   ("flat_terrain", "rough_terrain", "perlin_terrain", ...). The env is built
#   with the scene its registry name implies, so a test asking for a different
#   one reloads the MuJoCo model in place before its episode starts (this also
#   re-jits reset/step and rebuilds the renderer). Omit it, or use None, to keep
#   whatever scene the env name implies.
#   It takes either a single scene name or a list of them (DEFAULT_SCENES): with
#   a list the test is repeated once per scene, and each repetition is named
#   "<test>__<scene>" so their outputs stay apart. See _normalize_tests.
TESTS = {
    "Lite3JoystickFlatTerrain": (
        ("vx_1p5", np.array([1.5, 0.0, 0.0], dtype=np.float32), DEFAULT_SCENES),
        ("vx_1p0", np.array([1.0, 0.0, 0.0], dtype=np.float32), DEFAULT_SCENES),
        ("vy_0p4", np.array([0.0, 0.4, 0.0], dtype=np.float32), DEFAULT_SCENES),
        ("wz_0p6", np.array([0.0, 0.0, 0.6], dtype=np.float32), DEFAULT_SCENES),
        ("vx_1p0_vy_0p4", np.array([1.0, 0.4, 0.0], dtype=np.float32), DEFAULT_SCENES),
        ("vx_1p0_wz_0p6", np.array([1.0, 0.0, 0.6], dtype=np.float32), DEFAULT_SCENES),
        ("vy_0p4_wz_0p6", np.array([0.0, 0.4, 0.6], dtype=np.float32), DEFAULT_SCENES),
        ("vx_0p5_then_0", np.array([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32), DEFAULT_SCENES),
    ),
    "TitaJoystickFlatTerrain": (
        ("vx_1p5", np.array([1.5, 0.0], dtype=np.float32), DEFAULT_SCENES),
        ("wz_0p6", np.array([0.0, 0.6], dtype=np.float32), DEFAULT_SCENES),
        ("vx_1p0_wz_0p6", np.array([1.0, 0.6], dtype=np.float32), DEFAULT_SCENES),
        ("vx_2p0_then_0", np.array([[2.0, 0.0], [0.0, 0.0]], dtype=np.float32), DEFAULT_SCENES),
        ("vx_2p5_then_0", np.array([[2.5, 0.0], [0.0, 0.0]], dtype=np.float32), DEFAULT_SCENES),
        ("vx_2p0_wz_0p4_then_0", np.array([[2.0, 0.4], [0.0, 0.0]], dtype=np.float32), DEFAULT_SCENES),
        ("vx_3p0_then_0", np.array([[3.0, 0.0], [0.0, 0.0]], dtype=np.float32), DEFAULT_SCENES),
        ("vx_3p0_wz_0p8", np.array([3.0, 0.8], dtype=np.float32), DEFAULT_SCENES),
        ("vx_3p0_wz_0p8_then_0", np.array([[3.0, 0.8], [0.0, 0.0]], dtype=np.float32), DEFAULT_SCENES),
    )
}


def _test_block_counts(tests: tuple) -> list[int]:
    """Number of 5-second blocks each test spans (rows of its command array).

    A 1D command is a single block; a 2D command is one block per row."""
    return [
        np.atleast_2d(np.asarray(command, dtype=np.float32)).shape[0]
        for _, command, *_ in tests
    ]


def _build_blocks(tests: tuple) -> list[tuple[np.ndarray, bool, int]]:
    """Flatten the test list into a per-5-second-block schedule.

    A test whose command is a 1D array is a single block. A test whose command
    is a 2D array (several rows) contributes one block per row, and all of those
    blocks belong to the SAME episode: the command target changes every
    TEST_DURATION_SECONDS but the robot is NOT reset between them. Only the first
    block of each test starts a new episode (i.e. triggers a reset).

    Returns a list of (command_row, starts_new_test, test_index) tuples."""
    blocks: list[tuple[np.ndarray, bool, int]] = []
    for test_index, (_, command, *_) in enumerate(tests):
        rows = np.atleast_2d(np.asarray(command, dtype=np.float32))
        for row_index in range(rows.shape[0]):
            blocks.append((rows[row_index], row_index == 0, test_index))
    return blocks


def _unwrap_env(env):
    """Strip the training wrappers (e.g. SAC) to reach the concrete env."""
    base_env = env
    while hasattr(base_env, "env"):
        base_env = base_env.env
    return base_env


def _default_scene(env_name: str) -> str:
    """Scene (task name) a registry env name implies, e.g.

    "TitaJoystickFlatTerrain"  -> "flat_terrain"
    "Lite3JoystickE2ERoughTerrain" -> "rough_terrain"
    """
    suffix = env_name.split("Joystick")[-1].replace("E2E", "")
    return re.sub(r"(?<!^)(?=[A-Z])", "_", suffix).lower()


def _scene_tuple(scene, default_scene: str) -> tuple[str, ...]:
    """The scenes one TESTS entry declares, as a tuple.

    Accepts a single scene name, a list/tuple of them, or None (and a missing
    third element, which the caller passes as None) to mean the scene the env
    name implies."""
    if scene is None:
        return (default_scene,)
    if isinstance(scene, str):
        return (scene,)
    scenes = tuple(scene)
    if not scenes:
        raise ValueError(
            "A test's scene list is empty: name at least one scene, or drop the "
            "element to use the one the env name implies."
        )
    duplicates = {s for s in scenes if scenes.count(s) > 1}
    if duplicates:
        raise ValueError(
            f"A test's scene list repeats {sorted(duplicates)}: each scene may "
            f"appear once, otherwise the repeats would overwrite each other."
        )
    return scenes


def _normalize_tests(tests: tuple, default_scene: str) -> tuple:
    """Expand every TESTS entry into one (name, command, scene) test per scene.

    An entry naming several scenes (DEFAULT_SCENES) is run once on each of them,
    as a separate episode named "<test>__<scene>". The suffix is what keeps the
    repetitions apart: test names are the file names of every CSV, plot and
    reward folder this script writes, so without it each scene would overwrite
    the previous one's results.

    The expansion is grouped by scene rather than by test. Switching scene
    reloads the MuJoCo model and re-jits reset/step (see ensure_scene), so
    running all the tests of one scene before moving to the next keeps that to a
    single reload per scene instead of one per test."""
    declared = [
        (
            test[0],
            test[1],
            _scene_tuple(test[2] if len(test) > 2 else None, default_scene),
        )
        for test in tests
    ]

    # Scene order = order of first appearance across the entries, so the run
    # follows the order DEFAULT_SCENES is written in.
    ordered_scenes = list(
        dict.fromkeys(scene for _, _, scenes in declared for scene in scenes)
    )

    return tuple(
        (f"{name}__{scene}", command, scene)
        for scene in ordered_scenes
        for name, command, scenes in declared
        if scene in scenes
    )


def load_scene(env, scene: str) -> bool:
    """Reload the env's MuJoCo model for `scene` (the env's `task` name).

    The env is re-initialised in place (same object, so the training wrappers
    around it stay valid), which reloads the XML and re-runs _post_init().
    Returns True when the model actually changed, so the caller knows it has to
    rebuild everything that captured the old model (jitted fns, renderer)."""
    base_env = _unwrap_env(env)
    if getattr(base_env, "_compare_scene", None) == scene:
        return False

    print(f"[scene] reloading the env model for scene '{scene}'")
    type(base_env).__init__(base_env, task=scene, config=base_env._config)
    base_env._compare_scene = scene
    return True


def mark_scene(env, scene: str) -> None:
    """Record the scene an env was built with, without reloading anything."""
    _unwrap_env(env)._compare_scene = scene


def copy_joystick_source(env, run_dir: Path) -> None:
    """Copy the env's joystick.py into run_dir (the timestamped run folder that
    holds baseline/, residual/ and compare_graphics/), as compare_joystick.txt."""
    import inspect
    import shutil

    # Unwrap training wrappers (e.g. SAC) to reach the concrete env class.
    base_env = _unwrap_env(env)

    source = Path(inspect.getfile(type(base_env))).resolve()
    if source.name != "joystick.py":
        source = source.parent / "joystick.py"
    if not source.exists():
        print(f"[WARN] joystick.py not found at {source}")
        return

    destination = (run_dir / "compare_joystick.txt").resolve()
    shutil.copy2(source, destination)
    print(f"[copy] joystick.py from: {source}")
    print(f"[copy]              to:  {destination}")

def _to_e2e_name(env_name: str) -> str:
    """Best-effort end-to-end counterpart of a base joystick env name."""
    if "E2E" in env_name:
        return env_name
    return env_name.replace("Joystick", "JoystickE2E")


def add_video_hud(
    frame: np.ndarray,
    controller_name: str,
    command: np.ndarray,
    measured: np.ndarray,
    component_names: list[str],
    frozen: bool = False,
) -> np.ndarray:
    """Overlay controller name and command/measured velocity values."""
    command_text = "   ".join(
        f"{name}={value:+.2f}"
        for name, value in zip(component_names, np.asarray(command))
    )
    measured_text = "   ".join(
        f"{name}={value:+.2f}"
        for name, value in zip(component_names, np.asarray(measured))
    )

    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    status = " | TERMINATED - FRAME FROZEN" if frozen else ""
    lines = (
        f"{controller_name.upper()}{status}",
        f"COMMAND  {command_text}",
        f"ACTUAL   {measured_text}",
    )
    line_height = 16
    panel_height = 8 + line_height * len(lines)
    draw.rectangle((0, 0, image.width, panel_height), fill=(0, 0, 0, 180))
    for line_index, line in enumerate(lines):
        draw.text(
            (8, 5 + line_index * line_height),
            line,
            fill=(255, 255, 255, 255),
            font=font,
        )
    return np.asarray(image)


def create_comparison_video(
    video_paths: dict[str, Path],
    source_fps: dict[str, float],
    output_path: Path,
    fps: float,
    total_blocks: int,
) -> None:
    """Create normal-speed and x2 slow 2x2 time-synchronized videos."""
    layout = {
        "baseline": (0, 0),
        "residual": (0, 1),
        "end_to_end": (1, 0),
    }
    readers = {
        name: imageio.get_reader(path)
        for name, path in video_paths.items()
        if path.exists()
    }
    normal_writer = imageio.get_writer(
        output_path,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=None,
    )
    slow_output_path = output_path.with_name(
        f"{output_path.stem}_slowed_x2{output_path.suffix}"
    )
    slow_writer = imageio.get_writer(
        slow_output_path,
        fps=fps / 2.0,
        codec="libx264",
        quality=8,
        macro_block_size=None,
    )

    tile_width, tile_height = 640, 480
    duration_seconds = TEST_DURATION_SECONDS * total_blocks
    output_frame_count = int(round(duration_seconds * fps))
    try:
        for output_frame_index in range(output_frame_count):
            time_seconds = output_frame_index / fps
            current_frames = {}
            for name, reader in readers.items():
                source_frame_index = min(
                    int(round(time_seconds * source_fps[name])),
                    int(round(duration_seconds * source_fps[name])) - 1,
                )
                try:
                    current_frames[name] = reader.get_data(source_frame_index)
                except (IndexError, RuntimeError):
                    # Keep the tile black if a source video is unexpectedly short.
                    continue

            canvas = np.zeros(
                (2 * tile_height, 2 * tile_width, 3), dtype=np.uint8
            )
            for name, frame in current_frames.items():
                row, column = layout[name]
                if frame.shape[:2] != (tile_height, tile_width):
                    frame = np.asarray(
                        Image.fromarray(frame).resize(
                            (tile_width, tile_height), Image.Resampling.BILINEAR
                        )
                    )
                y0, x0 = row * tile_height, column * tile_width
                canvas[y0:y0 + tile_height, x0:x0 + tile_width] = frame[:, :, :3]
            normal_writer.append_data(canvas)
            slow_writer.append_data(canvas)
    finally:
        normal_writer.close()
        slow_writer.close()
        for reader in readers.values():
            reader.close()

    print(f"Combined comparison video saved to: {output_path}")
    print(f"Slow combined comparison video saved to: {slow_output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a baseline and residual policy velocity tracking."
    )
    parser.add_argument(
        "--name",
        type=str,
        default=DEFAULT_ENV_NAME,
        help=(
            "Environment to evaluate. Accepts a MuJoCo Playground env name or "
            "one of the shortcuts (go1, aliengo, tita, titae2e, lite3, "
            f"lite3e2e). Default: {DEFAULT_ENV_NAME}."
        ),
    )
    parser.add_argument(
        "--load",
        nargs="?",
        const="best",
        default="best",
        metavar="RUN_OR_SUFFIX",
        help=(
            "Residual checkpoint to load. With no value, load the best checkpoint "
            "from the latest run. Accepts a timestamp/prefix, saved run name, "
            "relative run path, or best/final/crash."
        ),
    )
    parser.add_argument(
        "--load-e2e",
        nargs="?",
        const="best",
        default="best",
        metavar="RUN_OR_SUFFIX",
        help=(
            "End-to-end checkpoint to load. With no value, load the best "
            "checkpoint from the latest E2E run."
        ),
    )
    parser.add_argument(
        "--no-e2e",
        action="store_true",
        help="Skip the end-to-end evaluation and compare only baseline/residual.",
    )
    parser.add_argument(
        "--ckpt-dir",
        default="checkpoints",
        help="Checkpoint root used by train_srbd.py (default: checkpoints).",
    )
    parser.add_argument(
        "--algo",
        choices=("ppo", "sac"),
        default="ppo",
        help="Algorithm used to train the residual checkpoint.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_fixed_command(state, command: jax.Array, baseline: bool):
    """Pin the command and prevent the environment command sampler from changing it."""
    batch_command = command[None, :]
    info = {
        **state.info,
        #"command": jnp.zeros_like(batch_command),
        "target_command": batch_command,
    }
    if "steps_until_next_cmd" in info:
        info["steps_until_next_cmd"] = jnp.full_like(
            info["steps_until_next_cmd"], 1_000_000
        )
    # Compatibility with environments that expose an explicit MPC-only flag.
    # The provided joystick environment does not need it: its baseline is
    # obtained by passing a zero residual action.
    if "use_only_mpc" in info:
        info["use_only_mpc"] = jnp.full_like(
            info["use_only_mpc"], baseline, dtype=jnp.bool_
        )
    return state.replace(info=info)


def state_observation(state):
    if isinstance(state.obs, dict):
        return {key: value[0] for key, value in state.obs.items()}
    return state.obs[0]


def save_tracking_plot(
    output_path: Path,
    test_title: str,
    time_values: np.ndarray,
    target_commands: np.ndarray,
    commands: np.ndarray,
    measured: np.ndarray,
    reset_flags: np.ndarray,
    frozen_flags: np.ndarray,
    component_labels: list[tuple[str, str]],
) -> None:
    fig, axes = plt.subplots(
        len(component_labels), 1, figsize=(10, 8), sharex=True
    )
    axes = np.atleast_1d(axes)
    frozen_mask = np.asarray(frozen_flags, dtype=bool)
    for index, (axis, (label, unit)) in enumerate(zip(axes, component_labels)):
        axis.step(
            time_values, target_commands[:, index], where="post", color="black",
            linestyle="--", linewidth=1.5, label="target",
        )
        axis.plot(
            time_values,
            commands[:, index],
            color="gray",
            linestyle="--",
            linewidth=1.5,
            label="command",
        )
        axis.plot(
            time_values,
            np.where(frozen_mask, np.nan, measured[:, index]),
            color="tab:blue",
            label="measured",
        )
        reset_times = time_values[reset_flags]
        for reset_index, reset_time in enumerate(reset_times):
            axis.axvline(
                reset_time,
                color="red",
                linewidth=1.2,
                alpha=0.8,
                label="reset / command change" if reset_index == 0 else None,
            )
        rmse = float(np.sqrt(np.mean((measured[:, index] - commands[:, index]) ** 2)))
        axis.set_ylabel(f"{label} [{unit}]")
        axis.set_title(f"{label} tracking - RMSE {rmse:.3f} {unit}")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best")
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle(test_title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_csv(
    output_path: Path,
    time_values: np.ndarray,
    commands: np.ndarray,
    measured: np.ndarray,
    reset_flags: np.ndarray,
    frozen_flags: np.ndarray,
    component_names: list[str],
    q_values: np.ndarray,
    dq_values: np.ndarray,
) -> None:
    """The full state closes every row: the whole qpos as q0 ... q(nq-1),
    then the whole qvel as dq0 ... dq(nv-1), floating base included."""
    header = (
        ["time_s"]
        + [f"cmd_{name}" for name in component_names]
        + list(component_names)
        + ["reset", "frozen"]
        + [f"q{index}" for index in range(q_values.shape[1])]
        + [f"dq{index}" for index in range(dq_values.shape[1])]
    )
    with output_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for time_s, command, velocity, reset, frozen, q, dq in zip(
            time_values, commands, measured, reset_flags, frozen_flags,
            q_values, dq_values,
        ):
            writer.writerow(
                (
                    time_s, *command.tolist(), *velocity.tolist(),
                    int(reset), int(frozen),
                    *q.tolist(), *dq.tolist(),
                )
            )


def save_rewards_csv(
    output_path: Path,
    time_values: np.ndarray,
    rewards: np.ndarray,
    frozen_flags: np.ndarray,
    reward_names: list[str],
) -> None:
    """Save the total reward and every weighted reward term for one test."""
    with output_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["time_s", *reward_names, "frozen"])
        for time_s, reward_values, frozen in zip(
            time_values, rewards, frozen_flags
        ):
            writer.writerow(
                (time_s, *reward_values.tolist(), int(frozen))
            )


def save_reward_plots(
    output_dir: Path,
    test_title: str,
    time_values: np.ndarray,
    rewards: np.ndarray,
    frozen_flags: np.ndarray,
    reward_names: list[str],
) -> None:
    """Save one plot per reward term, including the environment total."""
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen_mask = np.asarray(frozen_flags, dtype=bool)
    for index, reward_name in enumerate(reward_names):
        values = np.where(frozen_mask, np.nan, rewards[:, index])
        fig, axis = plt.subplots(figsize=(10, 4.5))
        axis.plot(time_values, values, color="tab:blue", linewidth=1.3)
        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        axis.set_xlabel("Time [s]")
        axis.set_ylabel("Weighted reward")
        axis.set_title(f"{test_title} - {reward_name}")
        axis.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(
            output_dir / f"{reward_name}.png",
            dpi=160,
            bbox_inches="tight",
        )
        plt.close(fig)


def compare_rewards(comparison_dir: Path, env_name: str) -> None:
    """Read controller reward CSVs and overlay every available reward term."""
    controller_specs = [
        ("baseline", "tab:blue"),
        ("residual", "tab:orange"),
    ]
    if (comparison_dir / "end_to_end").is_dir():
        controller_specs.append(("end_to_end", "tab:green"))

    output_dir = comparison_dir / "compare_rewards"
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_rewards_dir = comparison_dir / "baseline" / "rewards"

    for baseline_csv in sorted(baseline_rewards_dir.glob("*_rewards.csv")):
        controller_data = {}
        for controller_name, _ in controller_specs:
            csv_path = (
                comparison_dir / controller_name / "rewards" / baseline_csv.name
            )
            if not csv_path.exists():
                print(f"[WARN] Missing reward comparison CSV: {csv_path}")
                continue
            controller_data[controller_name] = np.atleast_1d(
                np.genfromtxt(csv_path, delimiter=",", names=True, dtype=float)
            )

        if "baseline" not in controller_data or "residual" not in controller_data:
            continue

        # E2E may expose a different reward set. Plot the union and simply omit
        # a controller from terms that its environment does not publish.
        reward_names = sorted(
            {
                field
                for data in controller_data.values()
                for field in (data.dtype.names or ())
                if field not in {"time_s", "frozen"}
            }
        )
        test_name = baseline_csv.name.removesuffix("_rewards.csv")
        test_output_dir = output_dir / test_name
        test_output_dir.mkdir(parents=True, exist_ok=True)

        for reward_name in reward_names:
            fig, axis = plt.subplots(figsize=(10, 4.5))
            plotted = False
            for controller_name, color in controller_specs:
                data = controller_data.get(controller_name)
                if data is None or reward_name not in (data.dtype.names or ()):
                    continue
                frozen = data["frozen"] > 0.5
                reward_values = np.where(frozen, np.nan, data[reward_name])

                reward_mean = float(np.nanmean(reward_values))
                reward_std = float(np.nanstd(reward_values))

                axis.plot(
                    data["time_s"],
                    reward_values,
                    color=color,
                    linewidth=1.3,
                    label=f"{controller_name}: {reward_mean:.4f} ± {reward_std:.4f}",
                )
                plotted = True
            if plotted:
                axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
                axis.set_xlabel("Time [s]")
                axis.set_ylabel("Weighted reward")
                axis.set_title(f"{env_name} - {test_name} - {reward_name}")
                axis.grid(True, alpha=0.3)
                axis.legend(loc="best")
                fig.tight_layout()
                fig.savefig(
                    test_output_dir / f"{reward_name}_comparison.png",
                    dpi=160,
                    bbox_inches="tight",
                )
            plt.close(fig)

    print(f"Reward comparison graphics saved to: {output_dir}")


def compare_graphics(
    comparison_dir: Path,
    env_name: str,
    component_labels: list[tuple[str, str]],
) -> None:
    """Create overlaid baseline, residual, and optional E2E tracking plots."""
    controller_specs = [
        ("baseline", "tab:blue"),
        ("residual", "tab:orange"),
    ]
    if (comparison_dir / "end_to_end").is_dir():
        controller_specs.append(("end_to_end", "tab:green"))

    output_dir = comparison_dir / "compare_graphics"
    output_dir.mkdir(parents=True, exist_ok=True)

    velocity_fields = tuple(
        (name, f"cmd_{name}", f"{name} [{unit}]")
        for name, unit in component_labels
    )

    baseline_dir = comparison_dir / "baseline"
    for baseline_csv in sorted(baseline_dir.glob("*.csv")):
        controller_data = {}
        for controller_name, _ in controller_specs:
            csv_path = comparison_dir / controller_name / baseline_csv.name
            if not csv_path.exists():
                print(f"[WARN] Missing comparison CSV: {csv_path}")
                continue
            controller_data[controller_name] = np.atleast_1d(
                np.genfromtxt(csv_path, delimiter=",", names=True, dtype=float)
            )

        if "baseline" not in controller_data or "residual" not in controller_data:
            continue

        fig, axes = plt.subplots(
            len(velocity_fields), 1, figsize=(10, 8), sharex=True
        )
        axes = np.atleast_1d(axes)
        for axis, (velocity_key, command_key, ylabel) in zip(
            axes, velocity_fields
        ):
            baseline_data = controller_data["baseline"]
            baseline_time = baseline_data["time_s"]
            baseline_command = baseline_data[command_key]
            axis.step(
                baseline_time,
                baseline_command,
                where="post",
                color="black",
                linestyle="--",
                linewidth=1.5,
                label="command",
            )

            for controller_name, color in controller_specs:
                if controller_name not in controller_data:
                    continue
                data = controller_data[controller_name]
                time_data = data["time_s"]
                velocity = data[velocity_key]
                command = data[command_key]
                frozen = data["frozen"] > 0.5
                rmse = float(np.sqrt(np.mean((velocity - command) ** 2)))

                axis.plot(
                    time_data,
                    np.where(frozen, np.nan, velocity),
                    color=color,
                    linewidth=1.3,
                    label=f"{controller_name} - RMSE {rmse:.3f}",
                )
            axis.set_ylabel(ylabel)
            axis.grid(True, alpha=0.3)
            axis.legend(loc="best")

        axes[-1].set_xlabel("Time [s]")
        names = " vs ".join(name for name, _ in controller_specs)
        fig.suptitle(f"{env_name} {names} - {baseline_csv.stem}")
        fig.tight_layout()
        fig.savefig(
            output_dir / f"{baseline_csv.stem}_comparison.png",
            dpi=160,
            bbox_inches="tight",
        )
        plt.close(fig)

    print(f"Comparison graphics saved to: {output_dir}")


def run_sequence(
    env,
    controller_name: str,
    reset_fn,
    step_fn,
    velocity_fn,
    policy_fn,
    video_writers,
    residual: bool,
    seed: int,
    tests: tuple,
    component_names: list[str],
    get_runtime,
    on_scene_end=None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
    np.ndarray,
    np.ndarray,
]:
    """Run the whole test sequence for one controller.

    `on_scene_end`, when given, is called as
    on_scene_end(scene, test_indices, results) every time the last test of a
    scene finishes -- so a scene's CSVs and plots are on disk before the next
    one starts loading, and an interrupted run keeps whatever it had already
    completed. `results` is the same 9-tuple this function returns, holding the
    series recorded so far; the test indices are absolute, so the caller slices
    it with the same offsets it would use at the end."""
    # Flatten the tests into 5-second blocks. Consecutive rows of a 2D command
    # share a single episode: the command target changes between them but the
    # robot is NOT reset. A reset happens only when the next block starts a new
    # test (its starts_new_test flag is True).
    blocks = _build_blocks(tests)
    num_blocks = len(blocks)

    renderer = None
    render_data = None
    get_obs_fn = None

    def ensure_scene(test_index: int) -> None:
        """Load the scene the given test declares, if it is not loaded already.

        A scene swap replaces the env's MuJoCo model, so everything that
        captured the previous one -- the jitted runtime and the renderer -- is
        rebuilt against the new model."""
        nonlocal reset_fn, step_fn, velocity_fn, get_obs_fn, renderer, render_data
        scene = tests[test_index][2]
        changed = load_scene(env, scene)
        if changed or get_obs_fn is None:
            reset_fn, step_fn, velocity_fn, get_obs_fn = get_runtime(env, scene)
        if changed or renderer is None:
            if renderer is not None:
                renderer.close()
            # Off-screen rendering works without a viewer and therefore also in
            # headless runs (train_srbd configures EGL when DISPLAY is missing).
            renderer = mujoco.Renderer(env.mj_model, height=480, width=640)
            render_data = mujoco.MjData(env.mj_model)

    ensure_scene(blocks[0][2])

    reset_index = 0
    state = reset_fn(jax.random.PRNGKey(seed + reset_index)[None, :])
    state = state.replace(
        info={
            **state.info,
            "command": jnp.zeros_like(state.info["command"]),
        }
    )
    first_command = jnp.asarray(blocks[0][0])
    state = set_fixed_command(state, first_command, baseline=not residual)

    # Rebuild the observation after replacing the command at reset.
    zero_action = jnp.zeros((1, env.action_size), dtype=jnp.float32)
    state = state.replace(obs=get_obs_fn(state.data, state.info, zero_action))

    steps_per_command = max(1, int(round(TEST_DURATION_SECONDS / float(env.dt))))
    num_steps = steps_per_command * num_blocks
    measured = []
    target_commands = []
    commands = []
    reset_flags = []
    frozen_flags = []
    # Full state of the robot at every step: the whole qpos and qvel, floating
    # base included. They close every row of the tracking CSV.
    q_values = []
    dq_values = []
    # Use the same source as train_srbd.py --eval: that evaluator flattens
    # state.info and plots the fields below reward_terms/. These values are
    # already multiplied by reward_config.scales inside the environment.
    reward_terms = state.info.get("reward_terms", {})
    reward_names = ["total", *reward_terms.keys()]
    reward_values = []
    rng = jax.random.PRNGKey(seed + 10_000)

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 4.0
    camera.elevation = -15.0
    camera.azimuth = 135.0

    frozen = False
    last_velocity = None
    last_frame = None
    last_q = None
    last_dq = None

    # Wall-clock report at every test boundary: how long the test just finished
    # took, and the time of day it ended.
    test_start_time = time.monotonic()

    def print_test_time(test_index: int) -> None:
        nonlocal test_start_time
        now = time.monotonic()
        minutes, seconds = divmod(int(now - test_start_time), 60)
        test_start_time = now
        print(
            f"  [{controller_name}] test {tests[test_index][0]} "
            f"| elapsed {minutes:02d}:{seconds:02d} | "
            f"time {datetime.now().strftime('%H:%M:%S')}"
        )

    def collect() -> tuple:
        """The series recorded so far, in the shape this function returns."""
        measured_array = np.asarray(measured, dtype=np.float64)
        return (
            np.arange(1, len(measured_array) + 1, dtype=np.float64)
            * float(env.dt),
            np.asarray(target_commands, dtype=np.float64),
            np.asarray(commands, dtype=np.float64),
            measured_array,
            np.asarray(reset_flags, dtype=bool),
            np.asarray(frozen_flags, dtype=bool),
            np.asarray(reward_values, dtype=np.float64),
            reward_names,
            np.asarray(q_values, dtype=np.float64),
            np.asarray(dq_values, dtype=np.float64),
        )

    # First test of the scene currently running: the tests from here up to the
    # one that just ended are what a scene boundary hands to on_scene_end.
    scene_first_test = 0

    def flush_scene(last_test_index: int) -> None:
        """Report the scene ending at `last_test_index` as finished."""
        nonlocal scene_first_test
        if on_scene_end is not None:
            on_scene_end(
                tests[last_test_index][2],
                list(range(scene_first_test, last_test_index + 1)),
                collect(),
            )
        scene_first_test = last_test_index + 1

    for step_index in range(num_steps):
        block_index = min(step_index // steps_per_command, num_blocks - 1)
        command_row = blocks[block_index][0]
        command_jax = jnp.asarray(command_row)
        block_finished = (
            (step_index + 1) % steps_per_command == 0
            and step_index + 1 < num_steps
        )
        # A finished block either switches the command inside the same test
        # (no reset) or moves on to the next test (reset). Only the latter is a
        # genuine reset; the former is a pure command change on a live episode.
        next_block_is_new_test = block_finished and blocks[block_index + 1][1]

        if frozen:
            # After an early termination, preserve the final state visually
            # and numerically until the current episode ends with a reset.
            measured.append(last_velocity.copy())
            q_values.append(last_q.copy())
            dq_values.append(last_dq.copy())
            current_target_command = np.asarray(
                command_row,
                dtype=np.float64,
            )

            current_command = np.asarray(
                jax.device_get(state.info["command"][0]),
                dtype=np.float64,
            )

            target_commands.append(current_target_command.copy())
            commands.append(current_command.copy())
            reset_flags.append(block_finished)
            frozen_flags.append(True)
            reward_values.append(np.full(len(reward_names), np.nan))
            for writer in video_writers:
                writer.append_data(last_frame)
        else:
            state = set_fixed_command(state, command_jax, baseline=not residual)
            # Make the current command immediately visible to the policy.
            state = state.replace(
                obs=get_obs_fn(state.data, state.info, state.info["last_act"])
            )
            if residual:
                rng, action_rng = jax.random.split(rng)
                action, _ = policy_fn(state_observation(state), action_rng)
                action = action[None, :]
            else:
                action = zero_action

            state = step_fn(state, action)
            state = set_fixed_command(state, command_jax, baseline=not residual)
            # Match train_srbd.py --eval exactly by reading
            # state.info["reward_terms"], then transfer every term at once.
            step_rewards = jnp.stack(
                [state.reward, *[
                    state.info["reward_terms"][name]
                    for name in reward_names[1:]
                ]],
                axis=-1,
            )[0]
            reward_values.append(
                np.asarray(jax.device_get(step_rewards), dtype=np.float64)
            )
            last_velocity = np.asarray(
                jax.device_get(velocity_fn(state.data)[0])
            )
            measured.append(last_velocity.copy())
            current_target_command = np.asarray(
                jax.device_get(state.info["target_command"][0])
            )

            current_command = np.asarray(
                jax.device_get(state.info["command"][0])
            )

            target_commands.append(current_target_command.copy())
            commands.append(current_command.copy())

            terminated = bool(np.asarray(jax.device_get(state.done[0])))

            last_q = np.asarray(
                jax.device_get(state.data.qpos[0]), dtype=np.float64
            )
            last_dq = np.asarray(
                jax.device_get(state.data.qvel[0]), dtype=np.float64
            )
            q_values.append(last_q.copy())
            dq_values.append(last_dq.copy())

            render_data.qpos[:] = last_q
            render_data.qvel[:] = last_dq
            mujoco.mj_forward(env.mj_model, render_data)
            base_position = render_data.qpos[:3]
            camera.lookat[:] = 0.9 * camera.lookat + 0.1 * base_position
            renderer.update_scene(render_data, camera=camera)
            last_frame = add_video_hud(
                renderer.render().copy(),
                controller_name=controller_name,
                command=current_command,
                measured=last_velocity,
                component_names=component_names,
                frozen=terminated,
            )
            for writer in video_writers:
                writer.append_data(last_frame)

            # Do not reset here. Freeze this terminal state until the regular
            # reset at the end of the current episode (i.e. the next test).
            frozen = terminated
            reset_flags.append(block_finished)
            frozen_flags.append(False)

        if block_finished and next_block_is_new_test:
            # Boundary between two different tests: start a fresh episode.
            finished_test = blocks[block_index][2]
            next_test = blocks[block_index + 1][2]
            print_test_time(finished_test)
            # When the next test also changes scene, the scene just ended: save
            # its results before the reload, which is the slow part.
            if tests[next_test][2] != tests[finished_test][2]:
                flush_scene(finished_test)
            ensure_scene(next_test)
            reset_index += 1
            state = reset_fn(jax.random.PRNGKey(seed + reset_index)[None, :])
            state = state.replace(
                info={
                    **state.info,
                    "command": jnp.zeros_like(state.info["command"]),
                }
            )
            next_command = jnp.asarray(blocks[block_index + 1][0])
            state = set_fixed_command(state, next_command, baseline=not residual)
            state = state.replace(obs=get_obs_fn(state.data, state.info, zero_action))
            frozen = False
            last_velocity = None
            last_frame = None
            last_q = None
            last_dq = None
        # else, if block_finished within the same test: only the command target
        # changes (applied at the top of the next step); the robot state and any
        # frozen/terminated status are preserved -- no reset.

    # Last test of the sequence: no boundary follows it, so report it here.
    print_test_time(blocks[-1][2])

    renderer.close()

    # The scene of the last test never hits a boundary either: close it here.
    flush_scene(len(tests) - 1)

    return collect()


def main() -> None:
    args = parse_args()
    train_srbd.ALGO = args.algo
    train_srbd.ALGO_PARAMS = (
        train_srbd.SAC_PARAMS if args.algo == "sac" else train_srbd.PPO_PARAMS
    )

    env_name = _NAME_SHORTCUTS.get(args.name.lower(), args.name)
    e2e_env_name = _to_e2e_name(env_name)

    if env_name not in TESTS:
        raise KeyError(
            f"No test sequence defined for '{env_name}'. "
            f"Add an entry to TESTS (available: {sorted(TESTS)})."
        )
    # Keep the tests as declared: a 2D command stays one test (one episode with
    # in-episode command changes), it is NOT split into independent tests.
    # Entries without an explicit scene inherit the one the env name implies.
    tests = _normalize_tests(TESTS[env_name], _default_scene(env_name))
    block_counts = _test_block_counts(tests)
    total_blocks = sum(block_counts)

    _, env, _ = train_srbd.make_envs(env_name=env_name)
    # registry.load already built the env with the scene its name implies, so
    # record it: a test asking for that same scene must not trigger a reload.
    mark_scene(env, _default_scene(env_name))

    # Command layout comes from the env itself: order and names of the command
    # components are read from command_config.names (numbered v1 ... vn when the
    # env does not publish them), and each name's measured source / unit from
    # _MEASURED_SOURCE.
    component_names = _command_names(
        env._config, np.atleast_2d(tests[0][1]).shape[1]
    )
    component_sources = [
        _measured_source(name, index)
        for index, name in enumerate(component_names)
    ]
    component_units = [unit for _, _, unit in component_sources]
    component_labels = list(zip(component_names, component_units))

    env_base_dir = os.path.join(args.ckpt_dir, env_name)
    run_dir, suffix = train_srbd._resolve_load(env_base_dir, args.load)
    params = train_srbd.load_params(run_dir, suffix=suffix)
    if params is None:
        raise FileNotFoundError(f"No residual checkpoint found in '{run_dir}'.")

    networks = train_srbd._build_fresh_networks(env)
    make_inference_fn = (
        ppo_networks.make_inference_fn(networks)
        if args.algo == "ppo"
        else sac_networks.make_inference_fn(networks)
    )
    policy_fn = jax.jit(make_inference_fn(params, deterministic=True))

    def build_runtime(eval_env):
        reset = jax.jit(jax.vmap(eval_env.reset))
        step = jax.jit(jax.vmap(eval_env.step))
        local_velocity = jax.jit(jax.vmap(eval_env.get_local_linvel))
        gyro = jax.jit(jax.vmap(eval_env.get_gyro))

        @jax.jit
        def velocity(data):
            linear = local_velocity(data)
            angular = gyro(data)
            columns = []
            for source, index, _ in component_sources:
                series = linear if source == "linvel" else angular
                columns.append(series[:, index])
            return jnp.stack(columns, axis=-1)

        return reset, step, velocity, jax.jit(jax.vmap(eval_env._get_obs))

    # One jitted runtime per (env, scene): a scene reload swaps the MuJoCo model
    # that the jitted functions captured, so each scene needs its own set. The
    # cache keeps a scene that is visited again from recompiling.
    runtime_cache: dict[tuple[int, str], tuple] = {}

    def get_runtime(eval_env, scene: str):
        key = (id(eval_env), scene)
        if key not in runtime_cache:
            runtime_cache[key] = build_runtime(eval_env)
        return runtime_cache[key]

    reset_fn, step_fn, velocity_fn, _ = get_runtime(env, _default_scene(env_name))

    comparison_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_dir = (
        Path(run_dir) / f"comparison_{env_name}" / comparison_timestamp
    )
    comparison_dir.mkdir(parents=True, exist_ok=True)
    copy_joystick_source(env, comparison_dir)
    controller_modes = [
        ("baseline", env, reset_fn, step_fn, velocity_fn, policy_fn, False),
        ("residual", env, reset_fn, step_fn, velocity_fn, policy_fn, True),
    ]

    if not args.no_e2e:
        _, e2e_env, _ = train_srbd.make_envs(env_name=e2e_env_name)
        mark_scene(e2e_env, _default_scene(e2e_env_name))
        e2e_base_dir = os.path.join(args.ckpt_dir, e2e_env_name)
        e2e_run_dir, e2e_suffix = train_srbd._resolve_load(
            e2e_base_dir, args.load_e2e
        )
        e2e_params = train_srbd.load_params(e2e_run_dir, suffix=e2e_suffix)
        if e2e_params is None:
            raise FileNotFoundError(
                f"No end-to-end checkpoint found in '{e2e_run_dir}'."
            )

        e2e_networks = train_srbd._build_fresh_networks(e2e_env)
        e2e_make_inference_fn = (
            ppo_networks.make_inference_fn(e2e_networks)
            if args.algo == "ppo"
            else sac_networks.make_inference_fn(e2e_networks)
        )
        e2e_policy_fn = jax.jit(
            e2e_make_inference_fn(e2e_params, deterministic=True)
        )
        e2e_reset_fn, e2e_step_fn, e2e_velocity_fn, _ = get_runtime(
            e2e_env, _default_scene(e2e_env_name)
        )
        controller_modes.append(
            (
                "end_to_end",
                e2e_env,
                e2e_reset_fn,
                e2e_step_fn,
                e2e_velocity_fn,
                e2e_policy_fn,
                True,
            )
        )
        print(f"End-to-end checkpoint: {e2e_run_dir}")

    comparison_dir.mkdir(parents=True, exist_ok=True)
    controller_video_paths = {}
    controller_video_fps = {}

    print(f"Environment: {env_name}")
    scene_order = list(dict.fromkeys(scene for _, _, scene in tests))
    print(
        f"Scenes: {scene_order} "
        f"({len(tests)} tests, {total_blocks} x {TEST_DURATION_SECONDS:g} s "
        f"= {TEST_DURATION_SECONDS * total_blocks:g} s per controller)"
    )
    print(f"Command components: {component_names}")
    print(f"Residual checkpoint: {run_dir}")
    print(f"Output directory: {comparison_dir}")
    print(f"Start time: {datetime.now().strftime('%H:%M:%S')}")
    last_mode_time = time.monotonic()
    for (
        mode_name,
        mode_env,
        mode_reset_fn,
        mode_step_fn,
        mode_velocity_fn,
        mode_policy_fn,
        uses_policy,
    ) in controller_modes:
        mode_dir = comparison_dir / mode_name
        mode_dir.mkdir(parents=True, exist_ok=True)
        video_path = mode_dir / f"{mode_name}_complete.mp4"
        controller_video_paths[mode_name] = video_path
        controller_video_fps[mode_name] = 1.0 / float(mode_env.dt)
        normal_video_writer = imageio.get_writer(
            video_path,
            fps=1.0 / float(mode_env.dt),
            codec="libx264",
            quality=8,
            macro_block_size=None,
        )
        slow_video_path = video_path.with_name(
            f"{video_path.stem}_slowed_x2{video_path.suffix}"
        )
        slow_video_writer = imageio.get_writer(
            slow_video_path,
            fps=(1.0 / float(mode_env.dt)) / 2.0,
            codec="libx264",
            quality=8,
            macro_block_size=None,
        )
        total_seconds = int(round(TEST_DURATION_SECONDS * total_blocks))
        print(
            f"[{mode_name}] Starting the continuous {total_seconds}-second sequence"
        )

        rewards_dir = mode_dir / "rewards"
        rewards_dir.mkdir(parents=True, exist_ok=True)
        steps_per_command = max(
            1, int(round(TEST_DURATION_SECONDS / float(mode_env.dt)))
        )
        # Cumulative block offsets so each test is sliced by its own length
        # (a 2D-command test spans several blocks / 5-second segments).
        block_offsets = np.concatenate(([0], np.cumsum(block_counts)))

        def save_finished_scene(scene: str, test_indices, results) -> None:
            """Write the CSVs and plots of a scene as soon as it finishes.

            run_sequence calls this at every scene boundary, so results land on
            disk while the next scene is still running and an interrupted run
            keeps the scenes it completed. `results` carries the series recorded
            so far and the test indices are absolute, so the slicing below is
            the same either way."""
            (
                time_values,
                target_commands,
                commands,
                measured,
                reset_flags,
                frozen_flags,
                rewards,
                reward_names,
                q_values,
                dq_values,
            ) = results
            for test_index in test_indices:
                test_name = tests[test_index][0]
                start = int(block_offsets[test_index]) * steps_per_command
                stop = min(
                    int(block_offsets[test_index + 1]) * steps_per_command,
                    len(time_values),
                )
                # Each CSV has local time starting at 0 for easier comparison; a
                # multi-block test therefore runs from 0 to 5 * n_blocks seconds.
                local_time = time_values[start:stop] - start * float(mode_env.dt)
                save_csv(
                    mode_dir / f"{test_name}.csv",
                    local_time,
                    commands[start:stop],
                    measured[start:stop],
                    reset_flags[start:stop],
                    frozen_flags[start:stop],
                    component_names,
                    q_values[start:stop],
                    dq_values[start:stop],
                )
                save_tracking_plot(
                    mode_dir / f"{test_name}_tracking.png",
                    f"{env_name} {mode_name} - {test_name}",
                    local_time,
                    target_commands[start:stop],
                    commands[start:stop],
                    measured[start:stop],
                    reset_flags[start:stop],
                    frozen_flags[start:stop],
                    component_labels,
                )
                save_rewards_csv(
                    rewards_dir / f"{test_name}_rewards.csv",
                    local_time,
                    rewards[start:stop],
                    frozen_flags[start:stop],
                    reward_names,
                )
                save_reward_plots(
                    rewards_dir / test_name,
                    f"{env_name} {mode_name} - {test_name}",
                    local_time,
                    rewards[start:stop],
                    frozen_flags[start:stop],
                    reward_names,
                )
            print(
                f"  [{mode_name}] scene '{scene}' finished: saved "
                f"{len(test_indices)} tests to {mode_dir}"
            )

        (
            time_values,
            target_commands,
            commands,
            measured,
            reset_flags,
            frozen_flags,
            rewards,
            reward_names,
            q_values,
            dq_values,
        ) = run_sequence(
            env=mode_env,
            controller_name=mode_name,
            reset_fn=mode_reset_fn,
            step_fn=mode_step_fn,
            velocity_fn=mode_velocity_fn,
            policy_fn=mode_policy_fn,
            video_writers=[normal_video_writer, slow_video_writer],
            residual=uses_policy,
            seed=args.seed,
            tests=tests,
            component_names=component_names,
            get_runtime=get_runtime,
            on_scene_end=save_finished_scene,
        )
        normal_video_writer.close()
        slow_video_writer.close()

        # The per-test CSVs and plots are already on disk: save_finished_scene
        # wrote each scene's as it ended. What is left is the plot of the whole
        # sequence, which only exists once every scene has run.
        save_tracking_plot(
            mode_dir / "tracking_complete.png",
            f"{env_name} {mode_name} - complete command sequence",
            time_values,
            target_commands,
            commands,
            measured,
            reset_flags,
            frozen_flags,
            component_labels,
        )
        print(
            f"  completed: {time_values[-1]:.3f} s, "
            f"block boundaries={int(np.sum(reset_flags))}"
        )
        print(f"  video saved to: {video_path}")
        print(f"  slow video saved to: {slow_video_path}")
        now = time.monotonic()
        minutes, secs = divmod(int(now - last_mode_time), 60)
        last_mode_time = now
        print(
            f"  [{mode_name}] elapsed {minutes:02d}:{secs:02d}\n"
            f"  [{mode_name}] time {datetime.now().strftime('%H:%M:%S')}\n"
            "-------------------------------"
        )

    compare_graphics(comparison_dir, env_name, component_labels)
    compare_rewards(comparison_dir, env_name)
    create_comparison_video(
        video_paths=controller_video_paths,
        source_fps=controller_video_fps,
        output_path=(
            comparison_dir
            / "compare_graphics"
            / "controllers_comparison_2x2.mp4"
        ),
        fps=1.0 / float(env.dt),
        total_blocks=total_blocks,
    )

    print(f"Comparison completed: {comparison_dir}")


if __name__ == "__main__":
    main()