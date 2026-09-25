"""Compare an MPC baseline and a residual policy for velocity tracking.

The environment is selected with --name (default: the Lite3 joystick env). The
script runs one continuous sequence (configurable duration per fixed command) for the
baseline and residual policy, plus the end-to-end policy only when --use-e2e is
passed. It saves tracking data, one CSV plus individual plots for every reward
term, and one complete video per controller:

    <run>/baseline/rewards/<test>_rewards.csv
    <run>/baseline/rewards/<test>/<reward>.png
    <run>/residual/rewards/...
    <run>/end_to_end/rewards/...             # only with --use-e2e
    <run>/compare_rewards/<test>/<reward>_comparison.png
    <run>/compare_torque/<test>_comparison.png
    <run>/compare_joint_velocity/<test>_comparison.png
    <run>/compare_joint_position/<test>_comparison.png

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
    python compare.py --load --use-e2e
    python compare.py --load --use-e2e --load-e2e saved/my_e2e_run
    python compare.py --load 20260907_101530 --redo 20260907_123000
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import time
import json
import re
from datetime import datetime
from pathlib import Path

# Configure MuJoCo before importing it so off-screen rendering works headlessly.
_DEVICE = "cpu" #os.environ.get("COMPARE_DEVICE", "gpu").lower()
if _DEVICE == "cpu":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
else:
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

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import train_srbd


# Wall-clock reference for every "elapsed" report: the moment this script
# started, so the reports show total time rather than per-test time.
SCRIPT_START_TIME = time.monotonic()


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

# The video is rendered at the control rate itself (one frame per control step),
# so the video FPS is derived from the env control period (env.dt = ctrl_dt) and
# is never a hardcoded target. This keeps the rendered frames (and their HUD
# values) at the exact same time resolution as the tracking CSVs and plots: every
# CSV sample has a matching video frame, so a value read off the overlay always
# matches the plotted trace (no decimated peaks). The MuJoCo renderer runs on the
# GPU under EGL, so this costs more device time than decimating, but it is the
# only way overlay and plots stay consistent.


def _render_stride(dt: float) -> int:
    """Control steps per rendered video frame.

    Always 1: one frame per control step, so the video FPS equals the control
    rate (1 / ctrl_dt), derived from the env and not from a hardcoded target.
    Overlay/video therefore show exactly the same samples as the CSVs and plots."""
    del dt  # kept for a stable signature; the stride no longer depends on it.
    return 1

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
# (name, command) or (name, command, duration_seconds).
#
# - A 1D command runs one episode.
# - A 2D command runs one episode with a command change after each row.
# - An optional third item sets the duration in seconds: one number applies
#   to every row, or a sequence specifies each row separately. The default
#   is TEST_DURATION_SECONDS per row. A row switch does not reset the robot.
#   Examples: ("vx_1p0", command, 8.0) or
#   ("vx_1p0_then_0", two_row_command, (5.0, 7.0)).
# Each command row must match the env's command_config layout (order and length
# of command_config.names).
# The scene is specified by the outer key in TESTS; it is not a test item.
LITE3_FLAT_TERRAIN_TESTS = (
    ("vx_1p0", np.array([1.0, 0.0, 0.0], dtype=np.float32)),
    ("vx_1p5", np.array([1.5, 0.0, 0.0], dtype=np.float32)),
    ("vx_2p0", np.array([2.0, 0.0, 0.0], dtype=np.float32)),
    ("vy_0p6", np.array([0.0, 0.6, 0.0], dtype=np.float32)),
    ("vy_0p8", np.array([0.0, 0.8, 0.0], dtype=np.float32)),
    ("vx_1p0_vy_0p4", np.array([1.0, 0.4, 0.0], dtype=np.float32)),
    ("vx_1p0_wz_0p6", np.array([1.0, 0.0, 0.6], dtype=np.float32)),
    ("vy_0p4_wz_0p6_then_0", np.array([[0.0, 0.4, 0.6], [0.0, 0.0, 0.0]], dtype=np.float32)),
)

LITE3_ROUGH_TERRAIN_TESTS = LITE3_FLAT_TERRAIN_TESTS
LITE3_PERLIN_TERRAIN_TESTS = LITE3_FLAT_TERRAIN_TESTS

TITA_FLAT_TERRAIN_TESTS = (
    #("vx_1p0", np.array([1.0, 0.0], dtype=np.float32)),
    #("vx_1p5", np.array([1.5, 0.0], dtype=np.float32)),
    ("vx_2p0", np.array([2.0, 0.0], dtype=np.float32)),
    #("wz_0p6", np.array([0.0, 0.6], dtype=np.float32)),
    ("wz_0p8", np.array([0.0, 0.8], dtype=np.float32)),
    #("vx_1p0_wz_0p6", np.array([1.0, 0.6], dtype=np.float32)),
    ("vx_1p5_wz_0p6", np.array([1.5, 0.6], dtype=np.float32)),
    #("vx_3p0_wz_0p8", np.array([3.0, 0.8], dtype=np.float32)),
    ("vx_2p0", np.array([[2.0, 0.0]], dtype=np.float32)),
    ("vx_2p0_wz_0p6", np.array([[2.0, 0.0], [0.0, 0.6]], dtype=np.float32)),
    #("vx_2p5_then_0", np.array([[2.5, 0.0], [0.0, 0.0]], dtype=np.float32)),
    #("vx_0p5_wz_0p8_then_vx_0p5_wz_n0p8", np.array([[0.5, 0.8], [0.5, -0.8]], dtype=np.float32)),
    #("vx_1p0_wz_0p8_then_vx_1p0_wz_n0p8", np.array([[1.0, 0.8], [1.0, -0.8]], dtype=np.float32)),
    #("vx_1p5_wz_0p8_then_vx_1p5_wz_n0p8", np.array([[1.5, 0.8], [1.5, -0.8]], dtype=np.float32)),
    #("vx_2p0_wz_0p4_then_0", np.array([[2.0, 0.4], [0.0, 0.0]], dtype=np.float32)),
    #("vx_2p5_then_vx_1p0_wz_0p8", np.array([[2.5, 0.0], [1.0, 0.8]], dtype=np.float32)),
    #("vx_2p0_wz_0p8_then_vx_1p0_wz_n0p8", np.array([[2.0, 0.8], [1.0, -0.8]], dtype=np.float32)),
    #("vx_2p5_wz_0p8_then_vx_0p5_wz_n0p8", np.array([[2.5, 0.8], [0.5, -0.8]], dtype=np.float32)),
    #("vx_1p0_wz_0p8_then_vx_n1p0_wz_n0p8", np.array([[1.0, 0.8], [-1.0, -0.8]], dtype=np.float32)),
    #("vx_2p0_wz_0p8_then_vx_n1p0_wz_n0p8", np.array([[2.0, 0.8], [-1.0, -0.8]], dtype=np.float32)),
    #("vx_3p0_wz_0p8_then_0", np.array([[3.0, 0.8], [0.0, 0.0]], dtype=np.float32)),
)

TITA_ROUGH_TERRAIN_TESTS = TITA_FLAT_TERRAIN_TESTS
TITA_PERLIN_TERRAIN_TESTS = (
    ("vx_0p5", np.array([0.5, 0.0], dtype=np.float32)),
    ("vx_0p7", np.array([0.7, 0.0], dtype=np.float32)),
    ("vx_1p0", np.array([1.0, 0.0], dtype=np.float32)),
    ("wz_0p5", np.array([0.0, 0.5], dtype=np.float32)),
    ("vx_0p5_wz_0p3", np.array([0.5, 0.3], dtype=np.float32)),
    ("vx_0p7_wz_0p3", np.array([0.7, 0.3], dtype=np.float32)),
    ("vx_1p0_wz_0p3", np.array([1.0, 0.3], dtype=np.float32)),
    ("vx_1p0_then_0", np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)),
    ("vx_1p0_wz_0p3_then_vx_0p0_wz_0p0", np.array([[1.0, 0.3], [0.0, 0.0]], dtype=np.float32)),
)

TESTS = {
    "Lite3JoystickFlatTerrain": {
        "flat_terrain": LITE3_FLAT_TERRAIN_TESTS,
        "rough_terrain": LITE3_ROUGH_TERRAIN_TESTS,
        "perlin_terrain": LITE3_PERLIN_TERRAIN_TESTS,
    },
    "TitaJoystickFlatTerrain": {
        "flat_terrain": TITA_FLAT_TERRAIN_TESTS,
        "rough_terrain": TITA_ROUGH_TERRAIN_TESTS,
        "perlin_terrain": TITA_PERLIN_TERRAIN_TESTS,
    },
}


def _test_block_counts(tests: tuple) -> list[int]:
    """Number of 5-second blocks each test spans (rows of its command array).

    A 1D command is a single block; a 2D command is one block per row."""
    return [
        np.atleast_2d(np.asarray(command, dtype=np.float32)).shape[0]
        for _, command, *_ in tests
    ]


def _build_blocks(tests: tuple) -> list[tuple[np.ndarray, bool, int]]:
    """Flatten the test list into a per-command-row schedule.

    A test whose command is a 1D array is a single block. A test whose command
    is a 2D array (several rows) contributes one block per row, and all of those
    blocks belong to the SAME episode: the command target changes after the
    configured row duration, without resetting the robot. Only the first
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


def _normalize_tests(tests_by_scene: dict[str, tuple]) -> tuple[tuple, tuple]:

    def _validate_test_names(tests_by_scene: dict[str, tuple]) -> None:
        """Check duplicate names and consistency between names and commands."""
        axes = {"vx": 0, "vy": 1, "wz": 2}
        component = re.compile(r"(vx|vy|wz)_(-?\d+p\d+|-?\d+)")

        def parse_command(label: str, test_name: str) -> np.ndarray:
            if label == "0":
                return np.zeros(3, dtype=np.float32)

            expected = np.zeros(3, dtype=np.float32)
            seen_axes = set()
            position = 0

            while position < len(label):
                match = component.match(label, position)
                if match is None:
                    raise ValueError(
                        f"{test_name}: invalid command in name: {label!r}"
                    )

                axis, value = match.groups()
                if axis in seen_axes:
                    raise ValueError(
                        f"{test_name}: {axis} appears more than once"
                    )
                seen_axes.add(axis)
                expected[axes[axis]] = float(value.replace("p", "."))

                position = match.end()
                if position < len(label):
                    if not label.startswith("_", position):
                        raise ValueError(
                            f"{test_name}: invalid command in name: {label!r}"
                        )
                    position += 1

            return expected

        for scene, scene_tests in tests_by_scene.items():
            seen_names = set()

            for entry in scene_tests:
                test_name, command = entry[:2]

                if test_name in seen_names:
                    raise ValueError(
                        f"{scene}: duplicate test name {test_name!r}"
                    )
                seen_names.add(test_name)

                rows = np.atleast_2d(
                    np.asarray(command, dtype=np.float32)
                )
                if rows.shape[1] != 3:
                    raise ValueError(
                        f"{scene}__{test_name}: expected 3 command values per row, "
                        f"got shape {rows.shape}"
                    )

                parts = test_name.split("_then_")
                if len(parts) > 2:
                    raise ValueError(
                        f"{scene}__{test_name}: more than one '_then_'"
                    )

                if len(parts) != rows.shape[0]:
                    raise ValueError(
                        f"{scene}__{test_name}: name describes {len(parts)} "
                        f"commands, but array has {rows.shape[0]} rows"
                    )

                for row_index, (part, actual) in enumerate(zip(parts, rows)):
                    expected = parse_command(part, test_name)
                    if not np.allclose(actual, expected, rtol=0, atol=1e-6):
                        raise ValueError(
                            f"{scene}__{test_name}, row {row_index}: "
                            f"name expects {expected.tolist()}, "
                            f"array contains {actual.tolist()}"
                        )
        
    """Return (test_name, command, scene) tests and row durations in seconds."""
    _validate_test_names(tests_by_scene)
    normalized = []
    durations_by_test = []
    for scene, scene_tests in tests_by_scene.items():
        for entry in scene_tests:
            if len(entry) == 2:
                test_name, command = entry
                duration = TEST_DURATION_SECONDS
            elif len(entry) == 3:
                test_name, command, duration = entry
            else:
                raise ValueError(f"Invalid test entry: {entry!r}")

            rows = np.atleast_2d(np.asarray(command, dtype=np.float32))
            durations = np.atleast_1d(np.asarray(duration, dtype=np.float64))
            if durations.size == 1:
                durations = np.repeat(durations, rows.shape[0])
            if durations.size != rows.shape[0]:
                raise ValueError(
                    f"{test_name}: expected {rows.shape[0]} row durations, "
                    f"got {durations.size}"
                )
            if not np.all(np.isfinite(durations)) or np.any(durations <= 0):
                raise ValueError(f"{test_name}: durations must be positive and finite")

            normalized.append((f"{scene}__{test_name}", command, scene))
            durations_by_test.append(tuple(float(d) for d in durations))
    return tuple(normalized), tuple(durations_by_test)


def _block_step_counts(test_durations: tuple, dt: float) -> np.ndarray:
    """Convert each row duration to control steps using the controller's dt."""
    return np.asarray(
        [max(1, int(round(duration / dt)))
         for durations in test_durations for duration in durations],
        dtype=np.int64,
    )


def _test_step_offsets(test_durations: tuple, dt: float) -> np.ndarray:
    """Step offsets at test boundaries, including the sequence end."""
    block_counts = _block_step_counts(test_durations, dt)
    row_offsets = np.concatenate(([0], np.cumsum(block_counts)))
    test_row_offsets = np.concatenate(
        ([0], np.cumsum([len(durations) for durations in test_durations]))
    )
    return row_offsets[test_row_offsets]


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

def readable_test_name(
    test_name: str,
    wrap_commands: bool = False,
    component_names: list[str] | None = None,
) -> tuple[str, str]:
    """Return readable terrain and command.

    When `component_names` is given (the env's command_config.names), every
    declared component is shown in order, so one that is zero in this test
    (and therefore absent from the test name) still appears (e.g. wz=0 for a
    pure vx/vy test). Without it, only the components present in the name are
    shown. Valid names and their units come from _MEASURED_SOURCE.
    """
    first, second = test_name.split("__", 1)

    if first.endswith("_terrain"):
        terrain, command = first, second
    else:
        command, terrain = first, second

    token_re = re.compile(rf"({'|'.join(_MEASURED_SOURCE)})_(n?\d+(?:p\d+)?)")

    def _unit(name: str, index: int) -> str:
        if name in _MEASURED_SOURCE:
            return _MEASURED_SOURCE[name][2]
        return _measured_source(name, index)[2]

    blocks = []
    for block in command.split("_then_"):
        parsed = {
            name: float(value.replace("n", "-").replace("p", "."))
            for name, value in token_re.findall(block)
        }
        if component_names:
            parts = [
                f"{name}={parsed.get(name, 0.0):.1f} {_unit(name, index)}"
                for index, name in enumerate(component_names)
            ]
            blocks.append(", ".join(parts))
        else:
            parts = [
                f"{name}={val:.1f} {_MEASURED_SOURCE[name][2]}"
                for name, val in parsed.items()
            ]
            blocks.append(", ".join(parts) if parts else "0")

    readable_command = blocks[0]
    for index, block in enumerate(blocks[1:], start=1):
        separator = "\nto " if wrap_commands and index % 2 == 0 else " to "
        readable_command += separator + block

    return terrain.replace("_", " "), readable_command


def readable_title(
    env_name: str,
    test_name: str,
    plot_type: str,
    component_names: list[str] | None = None,
) -> str:
    terrain, command = readable_test_name(
        test_name, wrap_commands=True, component_names=component_names
    )

    return (
        f"{env_name} - {terrain} - "
        f"{plot_type.replace('_', ' ')}\n"
        f"{command}"
    )

def add_video_hud(
    frame: np.ndarray,
    controller_name: str,
    target_command: np.ndarray,
    command: np.ndarray,
    measured: np.ndarray,
    component_names: list[str],
    frozen: bool = False,
) -> np.ndarray:
    """Overlay controller name and command/measured velocity values."""
    target_text = "   ".join(
        f"{name}={value:+.2f}"
        for name, value in zip(component_names, np.asarray(target_command))
    )
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
        f"TARGET   {target_text}",
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

def _write_comparison_video_index(
    output_path: Path,
    tests: tuple,
    test_durations: tuple,
    controller_test_success: dict[str, dict[str, bool]],
) -> None:
    """Write test start times and controller completion tables."""
    lines = []
    elapsed_seconds = 0.0
    current_scene = None

    for (full_test_name, command, scene), durations in zip(tests, test_durations):
        if scene != current_scene:
            if lines:
                lines.append("")
            lines.append(scene.replace("_", " ").upper())
            current_scene = scene

        _, test_name = readable_test_name(full_test_name)

        minutes, seconds = divmod(int(round(elapsed_seconds)), 60)
        hours, minutes = divmod(minutes, 60)

        lines.append(
            f"{test_name} - {hours:02d}:{minutes:02d}:{seconds:02d}"
        )

        elapsed_seconds += sum(durations)

    lines.extend(["", "TEST COMPLETION"])

    controller_names = [
        name
        for name in ("baseline", "residual", "end_to_end")
        if name in controller_test_success
    ]

    scenes = list(dict.fromkeys(scene for _, _, scene in tests))

    for scene in scenes:
        scene_tests = [
            (full_test_name, full_test_name.removeprefix(f"{scene}__"))
            for full_test_name, _, test_scene in tests
            if test_scene == scene
        ]

        headers = ["task", *controller_names]
        rows = []

        for full_test_name, _ in scene_tests:
            _, test_name = readable_test_name(full_test_name)
            row = [test_name]

            for controller_name in controller_names:
                completed = controller_test_success[controller_name].get(
                    full_test_name,
                    False,
                )
                row.append("V" if completed else "")

            rows.append(row)

        totals = [
            str(
                sum(
                    controller_test_success[controller_name].get(
                        full_test_name,
                        False,
                    )
                    for full_test_name, _ in scene_tests
                )
            )
            for controller_name in controller_names
        ]

        total_row = ["TOTAL", *totals]

        all_rows = [*rows, total_row]
        column_widths = [
            max(
                len(headers[column_index]),
                max(
                    len(row[column_index])
                    for row in all_rows
                ),
            )
            for column_index in range(len(headers))
        ]

        separator = "-+-".join("-" * width for width in column_widths)

        lines.extend(["", scene.replace("_", " ").upper()])
        lines.append(
            " | ".join(
                value.ljust(column_widths[index])
                for index, value in enumerate(headers)
            )
        )
        lines.append(separator)

        for row in rows:
            lines.append(
                " | ".join(
                    value.ljust(column_widths[index])
                    for index, value in enumerate(row)
                )
            )

        lines.append(separator)
        lines.append(
            " | ".join(
                value.ljust(column_widths[index])
                for index, value in enumerate(total_row)
            )
        )

    index_path = output_path.with_name("comparison_video.txt")
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Comparison video index saved to: {index_path}")

def _build_realtime_tracking_tile(
    comparison_graphics_dir: Path,
    tests: tuple,
    test_durations: tuple,
    time_seconds: float,
    tile_width: int,
    tile_height: int,
    plot_cache: dict[str, np.ndarray],
) -> np.ndarray:
    """Show the current comparison plot up to the current simulation time."""
    elapsed = 0.0
    selected_test_name = None
    selected_duration = 0.0
    local_time = 0.0

    for (test_name, _, _), durations in zip(tests, test_durations):
        test_duration = sum(durations)

        if time_seconds < elapsed + test_duration:
            selected_test_name = test_name
            selected_duration = test_duration
            local_time = time_seconds - elapsed
            break

        elapsed += test_duration

    if selected_test_name is None and tests:
        selected_test_name = tests[-1][0]
        selected_duration = sum(test_durations[-1])
        local_time = selected_duration

    if selected_test_name is None:
        return np.full(
            (tile_height, tile_width, 3),
            255,
            dtype=np.uint8,
        )

    if selected_test_name not in plot_cache:
        plot_path = (
            comparison_graphics_dir
            / f"{selected_test_name}_comparison.png"
        )

        if not plot_path.exists():
            print(f"[WARN] Missing comparison plot for video: {plot_path}")
            plot_cache[selected_test_name] = np.full(
                (tile_height, tile_width, 3),
                255,
                dtype=np.uint8,
            )
        else:
            plot_image = Image.open(plot_path).convert("RGB")
            plot_image.thumbnail(
                (tile_width, tile_height),
                Image.Resampling.LANCZOS,
            )

            # Center the plot without changing its aspect ratio.
            background = Image.new(
                "RGB",
                (tile_width, tile_height),
                "white",
            )
            background.paste(
                plot_image,
                (
                    (tile_width - plot_image.width) // 2,
                    (tile_height - plot_image.height) // 2,
                ),
            )
            plot_cache[selected_test_name] = np.asarray(background)

    tile = Image.fromarray(
        plot_cache[selected_test_name].copy()
    ).convert("RGBA")
    draw = ImageDraw.Draw(tile, "RGBA")

    # Approximate the axes area in the rendered Matplotlib plot.
    plot_left = int(0.10 * tile_width)
    plot_right = int(0.97 * tile_width)
    plot_top = int(0.08 * tile_height)
    plot_bottom = int(0.91 * tile_height)

    progress = np.clip(
        local_time / max(selected_duration, 1e-9),
        0.0,
        1.0,
    )
    cursor_x = int(
        plot_left + progress * (plot_right - plot_left)
    )

    # Cover the future portion of the curves.
    if cursor_x < plot_right:
        draw.rectangle(
            (cursor_x, plot_top, plot_right, plot_bottom),
            fill=(255, 255, 255, 255),
        )

    draw.line(
        (cursor_x, plot_top, cursor_x, plot_bottom),
        fill=(255, 0, 0, 255),
        width=3,
    )

    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", 15)
    except OSError:
        font = ImageFont.load_default()

    label = (
        f"TRACKING COMPARISON | "
        f"t={local_time:.2f}/{selected_duration:.2f} s"
    )
    draw.rectangle(
        (0, 0, tile_width, 25),
        fill=(0, 0, 0, 190),
    )
    draw.text(
        (8, 5),
        label,
        fill=(255, 255, 255, 255),
        font=font,
    )

    return np.asarray(tile.convert("RGB"))

def _diag_array(host: dict, name: str, fallback: np.ndarray) -> np.ndarray:
    """A tau_* diagnostic from the single per-step host transfer, or a fallback.

    Mirrors the previous per-field reader: a batched value (one extra leading
    axis over the fallback) is unbatched with [0]."""
    if name not in host:
        return np.asarray(fallback, dtype=np.float64)
    array = np.asarray(host[name], dtype=np.float64)
    return array[0] if array.ndim > np.asarray(fallback).ndim else array


def _diag_scalar(host: dict, name: str) -> float:
    """A scalar diagnostic from the per-step host transfer, or NaN if absent."""
    if name not in host:
        return float("nan")
    return float(np.asarray(host[name]).reshape(-1)[0])

def _save_comparison_meta(
    comparison_dir: Path,
    env_dt: float,
    tests: tuple,
    test_durations: tuple,
    controller_video_fps: dict[str, float],
    controller_test_success: dict[str, dict[str, bool]],
) -> None:
    """Tiny sidecar describing the per-controller videos, so --redo can rebuild
    the 2x2 comparison video from them without loading an environment. Only a
    handful of numbers and names: no frames are stored here."""
    meta = {
        "comparison_fps": (1.0 / float(env_dt)) / _render_stride(float(env_dt)),
        "source_duration_s": float(sum(map(sum, test_durations))),
        "tests": [
            {"name": name, "scene": scene, "durations": [float(d) for d in durs]}
            for (name, _, scene), durs in zip(tests, test_durations)
        ],
        "controllers": {
            name: {"fps": float(fps), "video": f"{name}/{name}_complete.mp4"}
            for name, fps in controller_video_fps.items()
        },
        "controller_test_success": {
            ctrl: {t: bool(v) for t, v in table.items()}
            for ctrl, table in controller_test_success.items()
        },
    }
    (comparison_dir / "comparison_video_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

def _write_slowed_video(source_path: Path, output_path: Path, fps: float) -> None:
    """Write a 2x-slow copy of `source_path` by halving the container frame rate.

    The frames are identical to the normal video; only playback speed changes.
    Generated in one pass after the rollout so the hot loop encodes a single
    video stream instead of two."""
    reader = imageio.get_reader(source_path)
    writer = imageio.get_writer(
        output_path,
        fps=fps / 2.0,
        codec="libx264",
        quality=8,
        macro_block_size=None,
    )
    try:
        for frame in reader:
            writer.append_data(frame)
    finally:
        writer.close()
        reader.close()

def create_comparison_video(
    video_paths: dict[str, Path],
    source_fps: dict[str, float],
    output_path: Path,
    fps: float,
    tests: tuple,
    test_durations: tuple,
    controller_test_success: dict[str, dict[str, bool]],
) -> None:
    """Create normal-speed and x2 slow time-synchronized comparison videos."""
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

    # Use one row for baseline/residual and two rows when E2E is present.
    num_rows = 2
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

    tile_width, tile_height = 640, 480
    plot_cache: dict[str, np.ndarray] = {}
    comparison_graphics_dir = output_path.parent
    duration_seconds = sum(map(sum, test_durations))
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
                (num_rows * tile_height, 2 * tile_width, 3),
                dtype=np.uint8,
            )
            for name, frame in current_frames.items():
                row, column = layout[name]
                if frame.shape[:2] != (tile_height, tile_width):
                    frame = np.asarray(
                        Image.fromarray(frame).resize(
                            (tile_width, tile_height),
                            Image.Resampling.BILINEAR,
                        )
                    )

                y0 = row * tile_height
                x0 = column * tile_width
                canvas[
                    y0:y0 + tile_height,
                    x0:x0 + tile_width,
                ] = frame[:, :, :3]
            
            tracking_width = tile_width if "end_to_end" in readers else 2 * tile_width

            tracking_tile = _build_realtime_tracking_tile(
                comparison_graphics_dir=comparison_graphics_dir,
                tests=tests,
                test_durations=test_durations,
                time_seconds=time_seconds,
                tile_width=tile_width,
                tile_height=tile_height,
                plot_cache=plot_cache,
            )

            if "end_to_end" not in readers:
                canvas[tile_height:2 * tile_height, :, :] = 255

            x0 = tile_width if "end_to_end" in readers else tile_width // 2
            canvas[
                tile_height:2 * tile_height,
                x0:x0 + tile_width,
            ] = tracking_tile


            normal_writer.append_data(canvas)
    finally:
        normal_writer.close()
        for reader in readers.values():
            reader.close()

    # Same frames, half the frame rate: re-time in one pass rather than
    # encoding a parallel slow stream inside the compositing loop.
    _write_slowed_video(output_path, slow_output_path, fps)
    _write_comparison_video_index(output_path, tests, test_durations, controller_test_success)
    print(f"Files txt for test summary saved to: {output_path.with_name('comparison_video.txt')}")
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
        "--use-e2e",
        action="store_true",
        help="Also evaluate the end-to-end policy. Off by default: only "
             "baseline and residual are compared unless this flag is passed.",
    )
    parser.add_argument(
        "--redo",
        type=str,
        default=None,
        metavar="COMPARISON_RUN",
        help=(
            "Regenerate plots from CSV files without executing environments. "
            "The argument selects the comparison run directory inside the "
            "checkpoint selected by --load, for example 20260907_123000."
        ),
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
        # Error mean +- std over the non-frozen samples (error = measured - command).
        # Using the error, not the raw signal, keeps this meaningful across an
        # in-episode command change (the command is per-timestep). Same window as
        # the RMSE and consistent with it: RMSE^2 = mean_err^2 + std_err^2
        # (bias^2 + variance). Frozen rows (post-fall) are excluded.
        valid = ~frozen_mask
        err = (measured[:, index] - commands[:, index])
        err = err[valid] if valid.any() else err
        err_mean, err_std = float(np.mean(err)), float(np.std(err))
        axis.set_ylabel(f"{label} [{unit}]")
        axis.set_title(
            f"{label} tracking - RMSE {rmse:.3f} {unit} | "
            f"ME {err_mean:+.3f} ± std {err_std:.3f} {unit}"
        )
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
    target_commands: np.ndarray,
    commands: np.ndarray,
    measured: np.ndarray,
    reset_flags: np.ndarray,
    frozen_flags: np.ndarray,
    component_names: list[str],
    q_values: np.ndarray,
    dq_values: np.ndarray,
    torque_values: np.ndarray,
    actuator_force_values: np.ndarray,
    tau_nominal_values: np.ndarray,
    tau_residual_values: np.ndarray,
    mpc_tau_values: np.ndarray,
    tau_saturated_values: np.ndarray,
    mpc_bad_values: np.ndarray,
) -> None:
    """The full state closes every row: the whole qpos as q0 ... q(nq-1),
    then the whole qvel as dq0 ... dq(nv-1), floating base included."""
    header = (
        ["time_s"]
        + [f"target_{name}" for name in component_names]
        + [f"cmd_{name}" for name in component_names]
        + list(component_names)
        + ["reset", "frozen"]
        + [f"q{index}" for index in range(q_values.shape[1])]
        + [f"dq{index}" for index in range(dq_values.shape[1])]
        + [f"joint_pos_{index}" for index in range(torque_values.shape[1])]
        + [f"joint_vel_{index}" for index in range(torque_values.shape[1])]
        + [f"tau_total_{index}" for index in range(torque_values.shape[1])]
        + [f"actuator_force_{index}" for index in range(torque_values.shape[1])]
        + [f"tau_nominal_{index}" for index in range(torque_values.shape[1])]
        + [f"tau_residual_{index}" for index in range(torque_values.shape[1])]
        + [f"mpc_tau_{index}" for index in range(torque_values.shape[1])]
        + ["tau_saturated_frac", "mpc_bad"]
    )
    with output_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for (
            time_s, target, command, velocity, reset, frozen, q, dq,
            torque, actuator_force, tau_nominal, tau_residual, mpc_tau,
            tau_saturated, mpc_bad,
        ) in zip(
            time_values, target_commands, commands, measured, reset_flags, frozen_flags,
            q_values, dq_values,
            torque_values, actuator_force_values, tau_nominal_values,
            tau_residual_values, mpc_tau_values, tau_saturated_values,
            mpc_bad_values,
        ):
            writer.writerow(
                (
                    time_s, *target.tolist(), *command.tolist(), *velocity.tolist(),
                    int(reset), int(frozen),
                    *q.tolist(), *dq.tolist(),
                    *q[7:].tolist(), *dq[6:].tolist(),
                    *torque.tolist(), *actuator_force.tolist(),
                    *tau_nominal.tolist(), *tau_residual.tolist(),
                    *mpc_tau.tolist(), tau_saturated, mpc_bad,
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
        title_header, separator, command = test_title.partition("\n")
        title_header = title_header.removesuffix(" - rewards")

        axis.set_title(
            f"{title_header} - {reward_name.replace('_', ' ')}"
            f"{separator}{command}"
        )
        axis.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(
            output_dir / f"{reward_name}.png",
            dpi=160,
            bbox_inches="tight",
        )
        plt.close(fig)


def compare_rewards(
    comparison_dir: Path,
    env_name: str,
    component_names: list[str] | None = None,
) -> None:
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
                axis.set_title(
                    readable_title(
                        env_name,
                        test_name,
                        reward_name,
                        component_names,
                    )
                )
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
            # Use the command from the controller that remained active the longest.
            reference_name, reference_data = max(
                controller_data.items(),
                key=lambda item: np.count_nonzero(item[1]["frozen"] <= 0.5),
            )

            command_time = reference_data["time_s"]
            reference_command = reference_data[command_key]
            axis.step(
                command_time,
                reference_command,
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
                # Error mean +- std (error = measured - command), robust to an
                # in-episode command change; RMSE^2 = mean_err^2 + std_err^2.
                valid = ~frozen
                err = velocity - command
                err = err[valid] if np.any(valid) else err
                err_mean, err_std = float(np.mean(err)), float(np.std(err))

                axis.plot(
                    time_data,
                    np.where(frozen, np.nan, velocity),
                    color=color,
                    linewidth=1.3,
                    label=(
                        f"{controller_name} - RMSE {rmse:.3f} | "
                        f"err mean {err_mean:+.3f} ± std {err_std:.3f}"
                    ),
                )
            axis.set_ylabel(ylabel)
            axis.grid(True, alpha=0.3)
            axis.legend(loc="best")

        axes[-1].set_xlabel("Time [s]")
        fig.suptitle(
            readable_title(
                env_name,
                baseline_csv.stem,
                "velocity tracking",
                [name for name, _ in component_labels],
            )
        )
        fig.tight_layout()
        fig.savefig(
            output_dir / f"{baseline_csv.stem}_comparison.png",
            dpi=160,
            bbox_inches="tight",
        )
        plt.close(fig)

    print(f"Comparison graphics saved to: {output_dir}")

def _quaternion_to_roll_pitch_yaw_deg(
    qw: np.ndarray,
    qx: np.ndarray,
    qy: np.ndarray,
    qz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert MuJoCo scalar-first quaternions to roll, pitch and yaw in degrees."""
    quaternions = np.column_stack((qw, qx, qy, qz)).astype(
        float,
        copy=False,
    )
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)

    valid = np.isfinite(norms[:, 0]) & (norms[:, 0] > 1e-12)
    normalized = np.full_like(quaternions, np.nan)
    normalized[valid] = quaternions[valid] / norms[valid]

    w, x, y, z = normalized.T

    roll = np.arctan2(
        2.0 * (w * x + y * z),
        1.0 - 2.0 * (x * x + y * y),
    )
    pitch = np.arcsin(
        np.clip(
            2.0 * (w * y - z * x),
            -1.0,
            1.0,
        )
    )
    yaw = np.arctan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )

    yaw = np.unwrap(yaw)

    return (
        np.rad2deg(roll),
        np.rad2deg(pitch),
        np.rad2deg(yaw),
    )


def compare_attitude(
    comparison_dir: Path,
    env_name: str,
    component_names: list[str] | None = None,
) -> None:
    """Compare base roll, pitch and yaw for all available controllers."""
    controller_specs = [
        ("baseline", "tab:blue"),
        ("residual", "tab:orange"),
    ]

    if (comparison_dir / "end_to_end").is_dir():
        controller_specs.append(("end_to_end", "tab:green"))

    output_dir = comparison_dir / "compare_attitude"
    output_dir.mkdir(parents=True, exist_ok=True)

    required_fields = {
        "time_s",
        "frozen",
        "q3",
        "q4",
        "q5",
        "q6",
    }
    missing_fields_reported = False

    baseline_dir = comparison_dir / "baseline"

    for baseline_csv in sorted(baseline_dir.glob("*.csv")):
        controller_data = {}

        for controller_name, _ in controller_specs:
            csv_path = (
                comparison_dir
                / controller_name
                / baseline_csv.name
            )

            if not csv_path.exists():
                print(
                    f"[WARN] Missing attitude comparison CSV: {csv_path}"
                )
                continue

            data = np.atleast_1d(
                np.genfromtxt(
                    csv_path,
                    delimiter=",",
                    names=True,
                    dtype=float,
                )
            )

            if not required_fields.issubset(data.dtype.names or ()):
                if not missing_fields_reported:
                    print(
                        "[WARN] Cannot generate compare_attitude: "
                        "the CSVs must contain time_s, frozen and q3..q6."
                    )
                    missing_fields_reported = True
                continue

            controller_data[controller_name] = data

        if (
            "baseline" not in controller_data
            or "residual" not in controller_data
        ):
            continue

        fig, axes = plt.subplots(
            3,
            1,
            figsize=(10, 9),
            sharex=True,
        )

        for controller_name, color in controller_specs:
            data = controller_data.get(controller_name)

            if data is None:
                continue

            roll, pitch, yaw = _quaternion_to_roll_pitch_yaw_deg(
                data["q3"],
                data["q4"],
                data["q5"],
                data["q6"],
            )

            frozen = data["frozen"] > 0.5
            time_values = data["time_s"]

            # Roll/pitch have an implicit target of 0 deg (base kept level), so
            # the error is the value itself; RMSE^2 = mean^2 + std^2 over the
            # non-frozen samples. Yaw is a free heading, so no stats are shown.
            attitude_values = (
                ("Roll [deg]", roll, True),
                ("Pitch [deg]", pitch, True),
                ("Yaw [deg]", yaw, False),
            )

            for axis, (ylabel, values, show_stats) in zip(
                axes,
                attitude_values,
            ):
                if show_stats:
                    valid = ~frozen
                    stat_values = values[valid] if np.any(valid) else values
                    rmse = float(np.sqrt(np.mean(stat_values ** 2)))
                    mean = float(np.mean(stat_values))
                    std = float(np.std(stat_values))
                    label = (
                        f"{controller_name}: RMSE {rmse:.2f}° | "
                        f"{mean:+.2f} ± {std:.2f}°"
                    )
                else:
                    label = controller_name

                axis.plot(
                    time_values,
                    np.where(frozen, np.nan, values),
                    color=color,
                    linewidth=1.3,
                    label=label,
                )
                axis.set_ylabel(ylabel)

        axes[-1].set_xlabel("Time [s]")

        for axis in axes:
            axis.axhline(
                0.0,
                color="black",
                linewidth=0.8,
                alpha=0.5,
            )
            axis.grid(True, alpha=0.3)
            axis.legend(loc="best")

        fig.suptitle(
            readable_title(
                env_name,
                baseline_csv.stem,
                "base attitude",
                component_names,
            )
        )
        fig.tight_layout()
        fig.savefig(
            output_dir / f"{baseline_csv.stem}_comparison.png",
            dpi=160,
            bbox_inches="tight",
        )
        plt.close(fig)

    print(f"Attitude comparison graphics saved to: {output_dir}")

def compare_joint_data(
    comparison_dir: Path,
    env_name: str,
    component_names: list[str] | None = None,
) -> None:
    """Compare torque, velocity and position of TITA's eight actuators.

    Each figure uses four rows and two columns: left-leg joints 0..3 are in
    the left column and right-leg joints 4..7 in the right column. Row four is
    the wheel on each side.
    """
    controller_specs = [
        ("baseline", "tab:blue"),
        ("residual", "tab:orange"),
    ]
    if (comparison_dir / "end_to_end").is_dir():
        controller_specs.append(("end_to_end", "tab:green"))

    plot_specs = (
        (
            "compare_torque",
            ("tau_total",),
            "Torque [N m]",
        ),
        ("compare_joint_velocity", ("joint_vel",), "Velocity [rad/s]"),
        ("compare_joint_position", ("joint_pos",), "Position [rad]"),
    )
    torque_styles = {
        "tau_nominal": ("--", "nominal"),
        "tau_residual": (":", "residual"),
        "tau_total": ("-", "total"),
    }
    # For the compare_torque figure, which torque component each controller
    # plots: the baseline shows its full applied torque (tau_total), the
    # residual controller shows ONLY its residual term (tau_residual) -- the
    # policy's own contribution -- not its total applied torque. E2E has no
    # residual, so it falls back to total.
    torque_field_by_controller = {
        "baseline": "tau_total",
        "residual": "tau_residual",
        "end_to_end": "tau_total",
    }
    joint_names = ("hip", "thigh", "knee", "wheel")
    baseline_dir = comparison_dir / "baseline"

    for output_name, field_prefixes, ylabel in plot_specs:
        output_dir = comparison_dir / output_name
        output_dir.mkdir(parents=True, exist_ok=True)
        missing_fields_reported = False

        for baseline_csv in sorted(baseline_dir.glob("*.csv")):
            controller_data = {}
            for controller_name, _ in controller_specs:
                csv_path = comparison_dir / controller_name / baseline_csv.name
                if not csv_path.exists():
                    continue
                controller_data[controller_name] = np.atleast_1d(
                    np.genfromtxt(
                        csv_path, delimiter=",", names=True, dtype=float
                    )
                )

            if "baseline" not in controller_data or "residual" not in controller_data:
                continue
            # Fields each controller contributes to this figure. compare_torque
            # picks a per-controller component (baseline -> total, residual ->
            # residual term); every other plot uses the same field for all.
            if output_name == "compare_torque":
                controller_prefixes = {
                    name: (torque_field_by_controller.get(name, "tau_total"),)
                    for name, _ in controller_specs
                }
            else:
                controller_prefixes = {
                    name: field_prefixes for name, _ in controller_specs
                }
            if any(
                f"{field_prefix}_{index}" not in (data.dtype.names or ())
                for name, data in controller_data.items()
                for field_prefix in controller_prefixes[name]
                for index in range(8)
            ):
                if not missing_fields_reported:
                    print(
                        f"[WARN] Cannot regenerate {output_name} for this run: "
                        "the required joint fields are not present in its CSVs."
                    )
                    missing_fields_reported = True
                continue

            fig, axes = plt.subplots(
                4, 2, figsize=(14, 13), sharex=True, constrained_layout=True
            )
            for row, joint_name in enumerate(joint_names):
                for column, side in enumerate(("left", "right")):
                    joint_index = row + 4 * column
                    axis = axes[row, column]
                    for controller_name, color in controller_specs:
                        data = controller_data.get(controller_name)
                        if data is None:
                            continue
                        frozen = data["frozen"] > 0.5
                        for field_prefix in controller_prefixes[controller_name]:
                            field = f"{field_prefix}_{joint_index}"
                            if output_name == "compare_torque":
                                linestyle, component_name = torque_styles[
                                    field_prefix
                                ]
                                # Avoid "residual residual" when the controller
                                # name already matches the torque component.
                                label = (
                                    component_name
                                    if component_name == controller_name
                                    else f"{controller_name} {component_name}"
                                )
                            else:
                                linestyle = "-"
                                label = controller_name
                            axis.plot(
                                data["time_s"],
                                np.where(frozen, np.nan, data[field]),
                                color=color,
                                linestyle=linestyle,
                                linewidth=1.2,
                                label=label,
                            )
                    axis.set_title(f"{side} {joint_name}")
                    axis.set_ylabel(ylabel)
                    axis.grid(True, alpha=0.3)
                    axis.legend(loc="best")
            axes[-1, 0].set_xlabel("Time [s]")
            axes[-1, 1].set_xlabel("Time [s]")
            plot_type = output_name.removeprefix("compare_")

            fig.suptitle(
                readable_title(
                    env_name,
                    baseline_csv.stem,
                    plot_type,
                    component_names,
                )
            )
            fig.savefig(
                output_dir / f"{baseline_csv.stem}_comparison.png",
                dpi=160,
                bbox_inches="tight",
            )
            plt.close(fig)

        print(f"{output_name} graphics saved to: {output_dir}")


def _resolve_redo_dir(
    run_dir: str | Path,
    env_name: str,
    comparison_run: str,
) -> Path:
    """Resolve one exact comparison run inside the selected checkpoint."""
    comparison_dir = (
        Path(run_dir) / f"comparison_{env_name}" / comparison_run
    )
    if not comparison_dir.is_dir():
        raise FileNotFoundError(
            f"Comparison run not found: '{comparison_dir}'."
        )
    if not (comparison_dir / "baseline").is_dir():
        raise FileNotFoundError(
            f"Missing baseline directory in '{comparison_dir}'."
        )
    return comparison_dir


def _tracking_layout_from_csv(comparison_dir: Path) -> tuple[list[str], list[tuple[str, str]]]:
    """Infer command names and plot units using only a saved tracking CSV."""
    csv_paths = sorted((comparison_dir / "baseline").glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(
            f"No baseline tracking CSV found in '{comparison_dir / 'baseline'}'."
        )
    data = np.atleast_1d(
        np.genfromtxt(csv_paths[0], delimiter=",", names=True, dtype=float)
    )
    fields = data.dtype.names or ()
    component_names = [
        field.removeprefix("cmd_")
        for field in fields
        if field.startswith("cmd_")
    ]
    if not component_names:
        raise ValueError(f"No cmd_* columns found in '{csv_paths[0]}'.")
    component_labels = [
        (name, _measured_source(name, index)[2])
        for index, name in enumerate(component_names)
    ]
    return component_names, component_labels

def _load_comparison_meta(comparison_dir: Path) -> dict | None:
    path = comparison_dir / "comparison_video_meta.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_selected_tests(all_names: list[str], selection: str) -> list[str]:
    """Turn a comma-separated --redo-tests string into full test names.

    Accepts the full 'scene__test' name, or just the 'test' part when it is
    unambiguous across scenes, or a substring as a last-resort convenience."""
    resolved = []
    for token in (t.strip() for t in selection.split(",") if t.strip()):
        if token in all_names:
            matches = [token]
        else:
            matches = [n for n in all_names if n.split("__", 1)[-1] == token]
            if not matches:
                matches = [n for n in all_names if token in n]
        if not matches:
            raise KeyError(f"No test matches '{token}'. Available: {all_names}")
        if len(matches) > 1:
            raise KeyError(
                f"'{token}' is ambiguous ({matches}); use the full "
                f"'scene__test' name."
            )
        resolved.append(matches[0])
    seen, ordered = set(), []
    for name in resolved:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def rebuild_comparison_video(
    comparison_dir: Path,
    meta: dict,
    selected_names: list[str] | None,
    output_path: Path,
) -> None:
    """Recompose the 2x2 comparison video from the per-controller videos already
    on disk, optionally keeping only `selected_names`. Streams frame by frame:
    the per-controller mp4s are the (compressed) frame store, so nothing large
    is held in memory and no raw frames are dumped."""
    all_names = [t["name"] for t in meta["tests"]]
    scene_by_name = {t["name"]: t["scene"] for t in meta["tests"]}
    durations_by_name = {t["name"]: tuple(t["durations"]) for t in meta["tests"]}

    # Start time of every test in the ORIGINAL full sequence (the layout of the
    # per-controller source videos). The output video may keep only a subset.
    src_start, acc = {}, 0.0
    for name in all_names:
        src_start[name] = acc
        acc += sum(durations_by_name[name])
    source_duration = float(meta.get("source_duration_s", acc))

    names = selected_names or all_names
    comparison_fps = float(meta["comparison_fps"])
    source_fps = {c: float(v["fps"]) for c, v in meta["controllers"].items()}

    readers = {}
    for controller, spec in meta["controllers"].items():
        video_path = comparison_dir / spec["video"]
        if video_path.exists():
            readers[controller] = imageio.get_reader(video_path)
        else:
            print(f"[WARN] Missing per-controller video, skipping: {video_path}")
    if "baseline" not in readers or "residual" not in readers:
        print("[WARN] Cannot rebuild comparison video: baseline/residual missing.")
        return

    layout = {"baseline": (0, 0), "residual": (0, 1), "end_to_end": (1, 0)}
    tile_width, tile_height, num_rows = 640, 480, 2

    # Selected tests in the requested order, for the tracking tile and the index.
    # The tile keys its plots by test name and walks the OUTPUT timeline, so the
    # subset shows exactly those tests. (command is unused by both helpers.)
    sel_tests = tuple((name, None, scene_by_name[name]) for name in names)
    sel_durations = tuple(durations_by_name[name] for name in names)

    # Output timeline: selected tests played back to back.
    out_bounds, t = [], 0.0
    for name in names:
        d = float(sum(durations_by_name[name]))
        out_bounds.append((t, t + d, name))
        t += d
    total_out = t

    max_src_index = {
        c: max(0, int(round(source_duration * fps)) - 1)
        for c, fps in source_fps.items()
    }

    normal_writer = imageio.get_writer(
        output_path, fps=comparison_fps, codec="libx264",
        quality=8, macro_block_size=None,
    )
    slow_output_path = output_path.with_name(
        f"{output_path.stem}_slowed_x2{output_path.suffix}"
    )
    plot_cache: dict[str, np.ndarray] = {}
    comparison_graphics_dir = output_path.parent
    output_frame_count = int(round(total_out * comparison_fps))
    try:
        for output_frame_index in range(output_frame_count):
            t_out = output_frame_index / comparison_fps

            # Which selected test this instant belongs to, and where the matching
            # frame sits in the ORIGINAL source video.
            name = names[-1]
            local = float(sum(durations_by_name[name]))
            for lo, hi, candidate in out_bounds:
                if t_out < hi:
                    name, local = candidate, t_out - lo
                    break
            src_time = src_start[name] + local

            current_frames = {}
            for controller, reader in readers.items():
                idx = min(
                    int(round(src_time * source_fps[controller])),
                    max_src_index[controller],
                )
                try:
                    current_frames[controller] = reader.get_data(idx)
                except (IndexError, RuntimeError):
                    continue

            canvas = np.zeros(
                (num_rows * tile_height, 2 * tile_width, 3), dtype=np.uint8
            )
            for controller, frame in current_frames.items():
                row, column = layout[controller]
                if frame.shape[:2] != (tile_height, tile_width):
                    frame = np.asarray(
                        Image.fromarray(frame).resize(
                            (tile_width, tile_height), Image.Resampling.BILINEAR
                        )
                    )
                y0, x0 = row * tile_height, column * tile_width
                canvas[y0:y0 + tile_height, x0:x0 + tile_width] = frame[:, :, :3]

            tracking_tile = _build_realtime_tracking_tile(
                comparison_graphics_dir=comparison_graphics_dir,
                tests=sel_tests,
                test_durations=sel_durations,
                time_seconds=t_out,
                tile_width=tile_width,
                tile_height=tile_height,
                plot_cache=plot_cache,
            )
            if "end_to_end" not in readers:
                canvas[tile_height:2 * tile_height, :, :] = 255
            x0 = tile_width if "end_to_end" in readers else tile_width // 2
            canvas[tile_height:2 * tile_height, x0:x0 + tile_width] = tracking_tile

            normal_writer.append_data(canvas)
    finally:
        normal_writer.close()
        for reader in readers.values():
            reader.close()

    _write_slowed_video(output_path, slow_output_path, comparison_fps)
    _write_comparison_video_index(
        output_path, sel_tests, sel_durations,
        meta.get("controller_test_success", {}),
    )
    print(f"Rebuilt comparison video ({len(names)} tests): {output_path}")
    print(f"Slow rebuilt comparison video: {slow_output_path}")

def redo_plots(
    comparison_dir: Path,
    env_name: str,
    tests: tuple,
    video_tests: str | None = None,
) -> None:
    """Regenerate every CSV-backed plot without loading an environment.

    New CSVs contain target_* columns. For older CSVs, which only persisted
    cmd_*, the filtered command is also used as the target so historical runs
    remain usable.
    """
    component_names, component_labels = _tracking_layout_from_csv(comparison_dir)
    controller_names = ["baseline", "residual"]
    if (comparison_dir / "end_to_end").is_dir():
        controller_names.append("end_to_end")

    # Keep declaration order but avoid processing an accidentally duplicated
    # test name twice (its CSV path is necessarily the same).
    # The CSV files are the source of truth for --redo. Historical runs may
    # use a different naming convention from the current _normalize_tests
    # implementation (for example test__scene instead of scene__test), and
    # TESTS itself may have changed after the run was produced.
    available_test_names = {
        path.stem
        for path in (comparison_dir / "baseline").glob("*.csv")
    }

    # Preserve the declared execution order where possible, accepting both
    # historical scene suffixes and current scene prefixes. Any CSV not known
    # to the current TESTS definition is still processed afterwards.
    ordered_test_names = []
    for test_name, _, scene in tests:
        candidates = [test_name]
        scene_prefix = f"{scene}__"
        if test_name.startswith(scene_prefix):
            candidates.append(f"{test_name[len(scene_prefix):]}__{scene}")
        for candidate in candidates:
            if (
                candidate in available_test_names
                and candidate not in ordered_test_names
            ):
                ordered_test_names.append(candidate)
                break
    ordered_test_names.extend(
        sorted(available_test_names.difference(ordered_test_names))
    )

    if not ordered_test_names:
        raise FileNotFoundError(
            f"No tracking CSV found in '{comparison_dir / 'baseline'}'."
        )

    for controller_name in controller_names:
        controller_dir = comparison_dir / controller_name
        complete_parts = []
        for test_name in ordered_test_names:
            csv_path = controller_dir / f"{test_name}.csv"
            if not csv_path.exists():
                print(
                    f"[WARN] Test '{test_name}' exists for baseline but its "
                    f"CSV is missing for {controller_name}: {csv_path}"
                )
                continue
            data = np.atleast_1d(
                np.genfromtxt(csv_path, delimiter=",", names=True, dtype=float)
            )
            fields = data.dtype.names or ()
            commands = np.column_stack(
                [data[f"cmd_{name}"] for name in component_names]
            )
            target_commands = np.column_stack([
                data[f"target_{name}"]
                if f"target_{name}" in fields else data[f"cmd_{name}"]
                for name in component_names
            ])
            measured = np.column_stack(
                [data[name] for name in component_names]
            )
            reset_flags = data["reset"] > 0.5
            frozen_flags = data["frozen"] > 0.5
            time_values = np.asarray(data["time_s"], dtype=float)
            save_tracking_plot(
                controller_dir / f"{test_name}_tracking.png",
                readable_title(
                    env_name,
                    test_name,
                    "velocity tracking",
                    component_names,
                ),
                time_values,
                target_commands,
                commands,
                measured,
                reset_flags,
                frozen_flags,
                component_labels,
            )
            complete_parts.append(
                (time_values, target_commands, commands, measured,
                 reset_flags, frozen_flags)
            )

            rewards_csv = (
                controller_dir / "rewards" / f"{test_name}_rewards.csv"
            )
            if rewards_csv.exists():
                reward_data = np.atleast_1d(
                    np.genfromtxt(
                        rewards_csv, delimiter=",", names=True, dtype=float
                    )
                )
                reward_names = [
                    field for field in (reward_data.dtype.names or ())
                    if field not in {"time_s", "frozen"}
                ]
                rewards = np.column_stack(
                    [reward_data[name] for name in reward_names]
                )
                save_reward_plots(
                    controller_dir / "rewards" / test_name,
                    readable_title(
                        env_name,
                        test_name,
                        "rewards",
                        component_names,
                    ),
                    np.asarray(reward_data["time_s"], dtype=float),
                    rewards,
                    reward_data["frozen"] > 0.5,
                    reward_names,
                )

        if complete_parts:
            complete_time = []
            elapsed = 0.0
            for part in complete_parts:
                local_time = part[0]
                complete_time.append(local_time - local_time[0] + elapsed)
                if len(local_time) > 1:
                    elapsed = complete_time[-1][-1] + float(
                        np.median(np.diff(local_time))
                    )
                else:
                    elapsed = complete_time[-1][-1]
            save_tracking_plot(
                controller_dir / "tracking_complete.png",
                f"{env_name} {controller_name} - complete command sequence",
                np.concatenate(complete_time),
                np.concatenate([part[1] for part in complete_parts]),
                np.concatenate([part[2] for part in complete_parts]),
                np.concatenate([part[3] for part in complete_parts]),
                np.concatenate([part[4] for part in complete_parts]),
                np.concatenate([part[5] for part in complete_parts]),
                component_labels,
            )

    compare_graphics(comparison_dir, env_name, component_labels)
    compare_rewards(comparison_dir, env_name, component_names)
    compare_joint_data(comparison_dir, env_name, component_names)
    compare_attitude(comparison_dir, env_name, component_names)
    print(f"Plots regenerated from CSV files: {comparison_dir}")

    if video_tests is not None:
        meta = _load_comparison_meta(comparison_dir)
        if meta is None:
            print(
                "[WARN] No comparison_video_meta.json in this run: cannot "
                "rebuild the comparison video. Re-run a full comparison first."
            )
        else:
            all_names = [t["name"] for t in meta["tests"]]
            selected = (
                None
                if video_tests.strip().lower() == "all"
                else _resolve_selected_tests(all_names, video_tests)
            )
            rebuild_comparison_video(
                comparison_dir,
                meta,
                selected,
                comparison_dir / "compare_graphics"
                / "controllers_comparison_2x2_selected.mp4",
            )


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
    test_durations: tuple,
    component_names: list[str],
    get_runtime,
    on_scene_end=None,
) -> tuple:
    """Run the whole test sequence for one controller.

    `on_scene_end`, when given, is called as
    on_scene_end(scene, test_indices, results) every time the last test of a
    scene finishes -- so a scene's CSVs and plots are on disk before the next
    one starts loading, and an interrupted run keeps whatever it had already
    completed. `results` is the same tuple this function returns, holding the
    series recorded so far; the test indices are absolute, so the caller slices
    it with the same offsets it would use at the end."""
    # Flatten the tests into 5-second blocks. Consecutive rows of a 2D command
    # share a single episode: the command target changes between them but the
    # robot is NOT reset. A reset happens only when the next block starts a new
    # test (its starts_new_test flag is True).
    blocks = _build_blocks(tests)
    num_blocks = len(blocks)
    # One video frame per control step (render_stride == 1): the video FPS equals
    # the control rate (1 / env.dt), so the render/HUD/encode run every step and
    # the overlay matches the full-resolution CSVs and plots exactly.
    render_stride = _render_stride(env.dt)

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

    block_step_ends = np.cumsum(
        _block_step_counts(test_durations, float(env.dt))
    )
    if len(block_step_ends) != num_blocks:
        raise ValueError("Every command row needs one duration")
    num_steps = int(block_step_ends[-1])
    measured = []
    target_commands = []
    commands = []
    reset_flags = []
    frozen_flags = []
    # Full state of the robot at every step: the whole qpos and qvel, floating
    # base included. They close every row of the tracking CSV.
    q_values = []
    dq_values = []
    torque_values = []
    actuator_force_values = []
    tau_nominal_values = []
    tau_residual_values = []
    mpc_tau_values = []
    tau_saturated_values = []
    mpc_bad_values = []
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
    # Stand-off from the tracked robot (metres, world units). Was 4 m, which sat
    # too close on every scene and cropped the (larger) Tita out of frame. 6.5 m
    # gives every robot some breathing room. NB: a fixed metre value is used on
    # purpose -- model.stat.extent is unreliable here because the rough/perlin
    # scenes set a terrain-sized <statistic extent> (4-5 m), which would push the
    # camera out to 30 m and shrink the robot to a dot.
    camera.distance = 2.5 if type(env).__name__.startswith("Lite3") else 6.0
    camera.elevation = -15.0
    camera.azimuth = 135.0

    frozen = False
    last_velocity = None
    last_frame = None
    last_q = None
    last_dq = None
    last_torque = None
    last_actuator_force = None
    last_tau_nominal = None
    last_tau_residual = None
    last_mpc_tau = None
    last_tau_saturated = None
    last_mpc_bad = None

    def append_reset_sample(command_row) -> None:
        """Log the freshly reset state as a test's t=0 sample, before any
        control step runs.

        Both controllers reset from the same PRNGKey(seed + reset_index), so
        this row is byte-for-byte identical between baseline and residual (base
        pose, attitude and velocities). It anchors every per-test plot at the
        shared initial state, making the first-step divergence explicit instead
        of hiding it in a post-step first sample. No control has run yet, so the
        torques are zero and the MPC/residual split does not exist: those
        diagnostics are marked NaN (skipped in the plots), exactly as the frozen
        branch treats a missing term.
        """
        nonlocal last_velocity, last_q, last_dq, last_torque
        nonlocal last_actuator_force, last_tau_nominal, last_tau_residual
        nonlocal last_mpc_tau, last_tau_saturated, last_mpc_bad
        host = jax.device_get({
            "velocity": velocity_fn(state.data)[0],
            "qpos": state.data.qpos[0],
            "qvel": state.data.qvel[0],
            "actuator_force": state.data.actuator_force[0],
        })
        last_velocity = np.asarray(host["velocity"], dtype=np.float64)
        last_q = np.asarray(host["qpos"], dtype=np.float64)
        last_dq = np.asarray(host["qvel"], dtype=np.float64)
        last_actuator_force = np.asarray(
            host["actuator_force"], dtype=np.float64
        )
        last_torque = np.zeros_like(last_actuator_force)
        nan_torque = np.full_like(last_torque, np.nan, dtype=np.float64)
        last_tau_nominal = nan_torque.copy()
        last_tau_residual = nan_torque.copy()
        last_mpc_tau = nan_torque.copy()
        last_tau_saturated = np.nan
        last_mpc_bad = np.nan

        command_np = np.asarray(command_row, dtype=np.float64)
        current_command = np.asarray(
            jax.device_get(state.info["command"][0]), dtype=np.float64
        )
        measured.append(last_velocity.copy())
        target_commands.append(command_np.copy())
        commands.append(current_command.copy())
        reset_flags.append(False)
        frozen_flags.append(False)
        reward_values.append(np.full(len(reward_names), np.nan))
        q_values.append(last_q.copy())
        dq_values.append(last_dq.copy())
        torque_values.append(last_torque.copy())
        actuator_force_values.append(last_actuator_force.copy())
        tau_nominal_values.append(last_tau_nominal.copy())
        tau_residual_values.append(last_tau_residual.copy())
        mpc_tau_values.append(last_mpc_tau.copy())
        tau_saturated_values.append(last_tau_saturated)
        mpc_bad_values.append(last_mpc_bad)

    # Wall-clock report at every test boundary: how long the run has taken in
    # total so far, and the time of day the test ended.
    def print_test_time(test_index: int) -> None:
        now = time.monotonic()
        minutes, seconds = divmod(int(now - SCRIPT_START_TIME), 60)
        print(
            f"  [{controller_name}] test {tests[test_index][0]} "
            f"| elapsed {minutes:02d}:{seconds:02d} | "
            f"time {datetime.now().strftime('%H:%M:%S')}"
        )

    def collect() -> tuple:
        """The series recorded so far, in the shape this function returns."""
        measured_array = np.asarray(measured, dtype=np.float64)
        return (
            # Index 0 is each test's reset sample (append_reset_sample), so the
            # time base starts at 0: sample i is the state after i control
            # steps, t = i * dt.
            np.arange(0, len(measured_array), dtype=np.float64)
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
            np.asarray(torque_values, dtype=np.float64),
            np.asarray(actuator_force_values, dtype=np.float64),
            np.asarray(tau_nominal_values, dtype=np.float64),
            np.asarray(tau_residual_values, dtype=np.float64),
            np.asarray(mpc_tau_values, dtype=np.float64),
            np.asarray(tau_saturated_values, dtype=np.float64),
            np.asarray(mpc_bad_values, dtype=np.float64),
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

    # Anchor the first test at its shared reset state (t = 0), before stepping.
    append_reset_sample(blocks[0][0])

    for step_index in range(num_steps):
        block_index = int(np.searchsorted(block_step_ends, step_index, side="right"))
        command_row = blocks[block_index][0]
        command_jax = jnp.asarray(command_row)
        block_finished = (
            step_index + 1 == block_step_ends[block_index]
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
            torque_values.append(last_torque.copy())
            actuator_force_values.append(last_actuator_force.copy())
            tau_nominal_values.append(last_tau_nominal.copy())
            tau_residual_values.append(last_tau_residual.copy())
            mpc_tau_values.append(last_mpc_tau.copy())
            tau_saturated_values.append(last_tau_saturated)
            mpc_bad_values.append(last_mpc_bad)
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
            if last_frame is not None and step_index % render_stride == 0:
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

            # Everything read back from the device this step is gathered into a
            # single pytree and pulled over with one jax.device_get. A dict
            # transfer is one blocking round-trip instead of one per field, so
            # the device is not stalled between each small read.
            transfer = {
                "reward": step_rewards,
                "velocity": velocity_fn(state.data)[0],
                "target_command": state.info["target_command"][0],
                "command": state.info["command"][0],
                "done": state.done[0],
                "qpos": state.data.qpos[0],
                "qvel": state.data.qvel[0],
                "ctrl": state.data.ctrl[0],
                "actuator_force": state.data.actuator_force[0],
            }
            for _diag_name in (
                "tau_total", "tau_nominal", "tau_residual", "mpc_tau",
                "tau_saturated_frac", "mpc_bad",
            ):
                _diag_value = state.info.get(_diag_name)
                if _diag_value is not None:
                    transfer[_diag_name] = _diag_value
            host = jax.device_get(transfer)

            reward_values.append(np.asarray(host["reward"], dtype=np.float64))
            last_velocity = np.asarray(host["velocity"], dtype=np.float64)
            measured.append(last_velocity.copy())
            current_target_command = np.asarray(
                host["target_command"], dtype=np.float64
            )
            current_command = np.asarray(host["command"], dtype=np.float64)
            target_commands.append(current_target_command.copy())
            commands.append(current_command.copy())

            terminated = bool(host["done"])

            last_q = np.asarray(host["qpos"], dtype=np.float64)
            last_dq = np.asarray(host["qvel"], dtype=np.float64)
            q_values.append(last_q.copy())
            dq_values.append(last_dq.copy())

            last_torque = _diag_array(
                host, "tau_total",
                np.asarray(host["ctrl"], dtype=np.float64),
            )
            last_actuator_force = np.asarray(
                host["actuator_force"], dtype=np.float64
            )
            nan_torque = np.full_like(last_torque, np.nan, dtype=np.float64)
            last_tau_nominal = _diag_array(host, "tau_nominal", nan_torque)
            last_tau_residual = _diag_array(host, "tau_residual", nan_torque)
            last_mpc_tau = _diag_array(host, "mpc_tau", nan_torque)
            last_tau_saturated = _diag_scalar(host, "tau_saturated_frac")
            last_mpc_bad = _diag_scalar(host, "mpc_bad")
            torque_values.append(last_torque.copy())
            actuator_force_values.append(last_actuator_force.copy())
            tau_nominal_values.append(last_tau_nominal.copy())
            tau_residual_values.append(last_tau_residual.copy())
            mpc_tau_values.append(last_mpc_tau.copy())
            tau_saturated_values.append(last_tau_saturated)
            mpc_bad_values.append(last_mpc_bad)

            # Render and encode only every render_stride control steps (always on
            # the terminating step, to capture the fall). The EGL renderer runs
            # on the GPU, so decimating it returns those cycles to the env step;
            # the physics itself still advances every step above.
            if step_index % render_stride == 0 or terminated:
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
                    target_command=current_target_command,
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
            last_torque = None
            last_actuator_force = None
            last_tau_nominal = None
            last_tau_residual = None
            last_mpc_tau = None
            last_tau_saturated = None
            last_mpc_bad = None
            # Anchor the next test at its reset state (t = 0), before stepping.
            append_reset_sample(blocks[block_index + 1][0])
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
    #
    # Only the scenes listed in DEFAULT_SCENES are actually run: commenting a
    # scene out there drops it from the sequence even though TESTS still
    # declares it (order follows TESTS).
    scenes_to_run = {
        scene: scene_tests
        for scene, scene_tests in TESTS[env_name].items()
        if scene in DEFAULT_SCENES
    }
    if not scenes_to_run:
        raise ValueError(
            f"None of DEFAULT_SCENES {DEFAULT_SCENES} are declared in "
            f"TESTS['{env_name}'] (declares {tuple(TESTS[env_name])})."
        )
    tests, test_durations = _normalize_tests(scenes_to_run)
    block_counts = _test_block_counts(tests)
    total_blocks = sum(block_counts)

    # --redo is deliberately handled before make_envs: --load selects the
    # checkpoint and --redo selects one exact comparison run within it. This
    # path reads CSVs only and therefore neither creates nor steps an environment.
    if args.redo is not None:
        env_base_dir = os.path.join(args.ckpt_dir, env_name)
        run_dir, _ = train_srbd._resolve_load(env_base_dir, args.load)
        source_comparison_dir = _resolve_redo_dir(
            run_dir, env_name, args.redo,
        )

        redo_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        redo_comparison_dir = (
            source_comparison_dir.parent
            / f"{redo_timestamp}_redo"
        )

        shutil.copytree(
            source_comparison_dir,
            redo_comparison_dir,
        )

        print(f"Original comparison: {source_comparison_dir}")
        print(f"Redo copy created: {redo_comparison_dir}")

        redo_plots(
            redo_comparison_dir,
            env_name,
            tests,
        )
        return

    _, env, _ = train_srbd.make_envs(env_name=env_name)
    env._config.randomize_reset = 1.0
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

    # The MPC baseline needs the residual branch OFF, not just a zero action.
    # In a residual env a ZERO action is NOT the plain MPC: at action 0 the
    # low-level controller still adds tau_rl = residual_gain*(Kp*(default_pose-q)
    # - Kd*qvel), a PD holding the joints at the nominal stance, on top of
    # tau_mpc. That stance PD fights the MPC gait and, at a 1 m/s forward command,
    # collapses tracking and tips the base over (measured on this env: vx command
    # 1.0 -> 0.01 m/s and roll -> pi, i.e. a fall), while the same MPC with the
    # branch disabled tracks 1.02 m/s upright -- matching the standalone
    # lite3_srbd.py baseline. The env exposes this exact gate via
    # config.enable_residual (-> _residual_gain 0.0); use a dedicated env so the
    # residual/e2e modes keep their branch on. Setting it on _config too keeps it
    # off across a scene reload (load_scene re-__init__s from _config).
    from mujoco_playground import registry as _registry
    try:
        baseline_env = _registry.load(
            env_name,
            config_overrides={
                "residual_config.enabled": False,
                "randomize_reset": 1.0,
            },
        )
    except (KeyError, AttributeError, ValueError, TypeError):
        baseline_env = _registry.load(
            env_name,
            config_overrides={
                "enable_residual": False,
                "randomize_reset": 1.0,
            },
        )
    mark_scene(baseline_env, _default_scene(env_name))
    # enable_residual=False sets _residual_gain=0.0 in the env __init__ (and, being
    # baked into _config, it survives a scene reload, which re-__init__s from
    # _config). Assert it so a silent config change can't let the stance PD back
    # into the "baseline". _residual_gain is captured by the jit built just below.
    try:
        residual_enabled = baseline_env._config.residual_config.enabled
    except AttributeError:
        residual_enabled = baseline_env._config.enable_residual

    assert not residual_enabled, (
        "baseline env still has the residual branch enabled"
    )
    baseline_reset_fn, baseline_step_fn, baseline_velocity_fn, _ = get_runtime(
        baseline_env, _default_scene(env_name)
    )

    comparison_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_dir = (
        Path(run_dir) / f"comparison_{env_name}" / comparison_timestamp
    )
    comparison_dir.mkdir(parents=True, exist_ok=True)
    copy_joystick_source(env, comparison_dir)
    controller_modes = [
        ("baseline", baseline_env, baseline_reset_fn, baseline_step_fn,
         baseline_velocity_fn, policy_fn, False),
        ("residual", env, reset_fn, step_fn, velocity_fn, policy_fn, True),
    ]

    if args.use_e2e:
        _, e2e_env, _ = train_srbd.make_envs(env_name=e2e_env_name)
        e2e_env._config.randomize_reset = 1.0
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
    controller_test_success = {}

    print(f"Environment: {env_name}")
    scene_order = list(dict.fromkeys(scene for _, _, scene in tests))
    print(
        f"Scenes: {scene_order} "
        f"({len(tests)} tests, {total_blocks} command segments, "
        f"{sum(map(sum, test_durations)):g} s per controller)"
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
        # Videos are written at the decimated render rate, not the control rate.
        video_fps = (1.0 / float(mode_env.dt)) / _render_stride(mode_env.dt)
        controller_video_fps[mode_name] = video_fps
        normal_video_writer = imageio.get_writer(
            video_path,
            fps=video_fps,
            codec="libx264",
            quality=8,
            macro_block_size=None,
        )
        slow_video_path = video_path.with_name(
            f"{video_path.stem}_slowed_x2{video_path.suffix}"
        )
        total_seconds = sum(map(sum, test_durations))
        print(
            f"[{mode_name}] Starting the continuous {total_seconds:g}-second sequence"
        )

        rewards_dir = mode_dir / "rewards"
        rewards_dir.mkdir(parents=True, exist_ok=True)
        # Test offsets match the per-row step counts used by run_sequence, plus
        # the one reset sample append_reset_sample() prepends to each test: test
        # k is preceded by k such samples, so shift every boundary by its index
        # (offset[0] stays 0; the final entry becomes the padded total length).
        test_step_offsets = _test_step_offsets(
            test_durations, float(mode_env.dt)
        )
        test_step_offsets = test_step_offsets + np.arange(len(test_step_offsets))

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
                torque_values,
                actuator_force_values,
                tau_nominal_values,
                tau_residual_values,
                mpc_tau_values,
                tau_saturated_values,
                mpc_bad_values,
            ) = results
            for test_index in test_indices:
                test_name = tests[test_index][0]
                start = int(test_step_offsets[test_index])
                stop = min(
                    int(test_step_offsets[test_index + 1]),
                    len(time_values),
                )
                # Each CSV has local time starting at 0 for easier comparison; a
                # multi-block test therefore runs from 0 to 5 * n_blocks seconds.
                local_time = time_values[start:stop] - start * float(mode_env.dt)
                save_csv(
                    mode_dir / f"{test_name}.csv",
                    local_time,
                    target_commands[start:stop],
                    commands[start:stop],
                    measured[start:stop],
                    reset_flags[start:stop],
                    frozen_flags[start:stop],
                    component_names,
                    q_values[start:stop],
                    dq_values[start:stop],
                    torque_values[start:stop],
                    actuator_force_values[start:stop],
                    tau_nominal_values[start:stop],
                    tau_residual_values[start:stop],
                    mpc_tau_values[start:stop],
                    tau_saturated_values[start:stop],
                    mpc_bad_values[start:stop],
                )
                save_tracking_plot(
                    mode_dir / f"{test_name}_tracking.png",
                    readable_title(
                        env_name,
                        test_name,
                        "velocity tracking",
                        component_names,
                    ),
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
                    readable_title(
                        env_name,
                        test_name,
                        "rewards",
                        component_names,
                    ),
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
            torque_values,
            actuator_force_values,
            tau_nominal_values,
            tau_residual_values,
            mpc_tau_values,
            tau_saturated_values,
            mpc_bad_values,
        ) = run_sequence(
            env=mode_env,
            controller_name=mode_name,
            reset_fn=mode_reset_fn,
            step_fn=mode_step_fn,
            velocity_fn=mode_velocity_fn,
            policy_fn=mode_policy_fn,
            video_writers=[normal_video_writer],
            residual=uses_policy,
            seed=args.seed,
            tests=tests,
            test_durations=test_durations,
            component_names=component_names,
            get_runtime=get_runtime,
            on_scene_end=save_finished_scene,
        )

        controller_test_success[mode_name] = {}

        for test_index, (test_name, _, _) in enumerate(tests):
            start = int(test_step_offsets[test_index])
            stop = min(
                int(test_step_offsets[test_index + 1]),
                len(frozen_flags),
            )

            controller_test_success[mode_name][test_name] = not np.any(
                frozen_flags[start:stop]
            )

        normal_video_writer.close()
        # The 2x-slow video is the same frames re-timed; produce it in one pass
        # afterwards instead of encoding a second stream during the rollout.
        _write_slowed_video(video_path, slow_video_path, video_fps)

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
    compare_rewards(comparison_dir, env_name, component_names)
    compare_joint_data(comparison_dir, env_name, component_names)
    compare_attitude(comparison_dir, env_name, component_names)
    create_comparison_video(
        video_paths=controller_video_paths,
        source_fps=controller_video_fps,
        output_path=(
            comparison_dir
            / "compare_graphics"
            / "controllers_comparison_2x2.mp4"
        ),
        fps=(1.0 / float(env.dt)) / _render_stride(env.dt),
        tests=tests,
        test_durations=test_durations,
        controller_test_success=controller_test_success,
    )
    _save_comparison_meta(
        comparison_dir, env.dt, tests, test_durations,
        controller_video_fps, controller_test_success,
    )

    print(f"Comparison completed: {comparison_dir}")


if __name__ == "__main__":
    main()