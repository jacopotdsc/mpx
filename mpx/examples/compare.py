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

The command layout (which components exist and in which order) is read from the
selected env's command_config.names, so it is never hardcoded here.

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

# Fixed command sequence to run, per environment. Each entry is a
# (name, command) pair; the command vector must match the env's command_config
# layout (order and length of command_config.names).
TESTS = {
    "Lite3JoystickFlatTerrain": (
        ("vx_1p5", np.array([1.5, 0.0, 0.0], dtype=np.float32)),
        ("vy_0p4", np.array([0.0, 0.4, 0.0], dtype=np.float32)),
        ("wz_0p6", np.array([0.0, 0.0, 0.6], dtype=np.float32)),
        ("vx_1p0_vy_0p4", np.array([1.0, 0.4, 0.0], dtype=np.float32)),
        ("vx_1p0_wz_0p6", np.array([1.0, 0.0, 0.6], dtype=np.float32)),
        ("vy_0p4_wz_0p6", np.array([0.0, 0.4, 0.6], dtype=np.float32)),
        ("vx_0p5_then_0", np.array([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)),
    ),
    "TitaJoystickFlatTerrain": (
        ("vx_1p5", np.array([1.5, 0.0], dtype=np.float32)),
        ("wz_0p6", np.array([0.0, 0.6], dtype=np.float32)),
        ("vx_1p0_wz_0p6", np.array([1.0, 0.6], dtype=np.float32)),
        ("vx_3p0_wz_0p8", np.array([3.0, 0.8], dtype=np.float32)),
        ("vx_2p0_then_0", np.array([[2.0, 0.0], [0.0, 0.0]], dtype=np.float32)),
        ("vx_2p0_wz_0p4_then_0", np.array([[2.0, 0.4], [0.0, 0.0]], dtype=np.float32)),
        ("vx_3p0_then_0", np.array([[3.0, 0.0], [0.0, 0.0]], dtype=np.float32)),
        ("vx_3p0_wz_0p8_then_0", np.array([[3.0, 0.8], [0.0, 0.0]], dtype=np.float32)),
    )
}

def _expand_tests(tests: tuple) -> tuple:
    """Expand any entry whose command is an array of arrays into one entry per
    row, so a single test issues several commands in sequence (one command
    change every TEST_DURATION_SECONDS). The run then lasts 5 * total_blocks s."""
    expanded = []
    for name, command in tests:
        command = np.asarray(command, dtype=np.float32)
        if command.ndim == 1:
            expanded.append((name, command))
        else:
            for i, sub in enumerate(command):
                expanded.append((f"{name}_{i}", np.asarray(sub, dtype=np.float32)))
    return tuple(expanded)

def copy_joystick_source(env, run_dir: Path) -> None:
    """Copy the env's joystick.py into run_dir (the timestamped run folder that
    holds baseline/, residual/ and compare_graphics/), as compare_joystick.txt."""
    import inspect
    import shutil

    # Unwrap training wrappers (e.g. SAC) to reach the concrete env class.
    base_env = env
    while hasattr(base_env, "env"):
        base_env = base_env.env

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
    tests: tuple,
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
    duration_seconds = TEST_DURATION_SECONDS * len(tests)
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
        "command": batch_command,
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
            time_values, commands[:, index], where="post", color="black",
            linestyle="--", linewidth=1.5, label="command",
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
) -> None:
    header = (
        ["time_s"]
        + [f"cmd_{name}" for name in component_names]
        + list(component_names)
        + ["reset", "frozen"]
    )
    with output_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for time_s, command, velocity, reset, frozen in zip(
            time_values, commands, measured, reset_flags, frozen_flags
        ):
            writer.writerow(
                (
                    time_s, *command.tolist(), *velocity.tolist(),
                    int(reset), int(frozen),
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
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
]:
    reset_index = 0
    state = reset_fn(jax.random.PRNGKey(seed + reset_index)[None, :])
    first_command = jnp.asarray(tests[0][1])
    state = set_fixed_command(state, first_command, baseline=not residual)

    # Rebuild the observation after replacing the command at reset.
    zero_action = jnp.zeros((1, env.action_size), dtype=jnp.float32)
    get_obs_fn = jax.jit(jax.vmap(env._get_obs))
    state = state.replace(obs=get_obs_fn(state.data, state.info, zero_action))

    steps_per_command = max(1, int(round(TEST_DURATION_SECONDS / float(env.dt))))
    num_steps = steps_per_command * len(tests)
    measured = []
    commands = []
    reset_flags = []
    frozen_flags = []
    # Use the same source as train_srbd.py --eval: that evaluator flattens
    # state.info and plots the fields below reward_terms/. These values are
    # already multiplied by reward_config.scales inside the environment.
    reward_terms = state.info.get("reward_terms", {})
    reward_names = ["total", *reward_terms.keys()]
    reward_values = []
    rng = jax.random.PRNGKey(seed + 10_000)

    # Off-screen rendering works without a viewer and therefore also in
    # headless runs (train_srbd configures EGL when DISPLAY is unavailable).
    renderer = mujoco.Renderer(env.mj_model, height=480, width=640)
    render_data = mujoco.MjData(env.mj_model)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 4.0
    camera.elevation = -15.0
    camera.azimuth = 135.0

    frozen = False
    last_velocity = None
    last_frame = None

    for step_index in range(num_steps):
        command_index = min(step_index // steps_per_command, len(tests) - 1)
        command_jax = jnp.asarray(tests[command_index][1])
        command_block_finished = (
            (step_index + 1) % steps_per_command == 0
            and step_index + 1 < num_steps
        )

        if frozen:
            # After an early termination, preserve the final state visually
            # and numerically until this command's five-second block ends.
            measured.append(last_velocity.copy())
            current_command = np.asarray(tests[command_index][1])
            commands.append(current_command)
            reset_flags.append(command_block_finished)
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
            current_command = np.asarray(tests[command_index][1])
            commands.append(current_command)

            terminated = bool(np.asarray(jax.device_get(state.done[0])))

            render_data.qpos[:] = np.asarray(jax.device_get(state.data.qpos[0]))
            render_data.qvel[:] = np.asarray(jax.device_get(state.data.qvel[0]))
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
            # reset at the end of the current five-second command block.
            frozen = terminated
            reset_flags.append(command_block_finished)
            frozen_flags.append(False)

        if command_block_finished:
            reset_index += 1
            state = reset_fn(jax.random.PRNGKey(seed + reset_index)[None, :])
            next_command_index = min(
                (step_index + 1) // steps_per_command, len(tests) - 1
            )
            next_command = jnp.asarray(tests[next_command_index][1])
            state = set_fixed_command(state, next_command, baseline=not residual)
            state = state.replace(obs=get_obs_fn(state.data, state.info, zero_action))
            frozen = False
            last_velocity = None
            last_frame = None

    renderer.close()

    measured_array = np.asarray(measured, dtype=np.float64)
    commands_array = np.asarray(commands, dtype=np.float64)
    reset_flags_array = np.asarray(reset_flags, dtype=bool)
    frozen_flags_array = np.asarray(frozen_flags, dtype=bool)
    rewards_array = np.asarray(reward_values, dtype=np.float64)
    time_values = np.arange(1, len(measured_array) + 1, dtype=np.float64) * float(env.dt)
    return (
        time_values,
        commands_array,
        measured_array,
        reset_flags_array,
        frozen_flags_array,
        rewards_array,
        reward_names,
    )


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
    tests = _expand_tests(TESTS[env_name])

    _, env, _ = train_srbd.make_envs(env_name=env_name)

    # Command layout comes from the env itself: order and names of the command
    # components are read from command_config.names, and each name's measured
    # source / unit from _MEASURED_SOURCE.
    component_names = list(env._config.command_config.names)
    component_units = [_MEASURED_SOURCE[name][2] for name in component_names]
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
            for name in component_names:
                source, index, _ = _MEASURED_SOURCE[name]
                series = linear if source == "linvel" else angular
                columns.append(series[:, index])
            return jnp.stack(columns, axis=-1)

        return reset, step, velocity

    reset_fn, step_fn, velocity_fn = build_runtime(env)

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
        e2e_reset_fn, e2e_step_fn, e2e_velocity_fn = build_runtime(e2e_env)
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
    print(f"Command components: {component_names}")
    print(f"Residual checkpoint: {run_dir}")
    print(f"Output directory: {comparison_dir}")
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
        total_seconds = int(round(TEST_DURATION_SECONDS * len(tests)))
        print(
            f"[{mode_name}] Starting the continuous {total_seconds}-second sequence"
        )
        (
            time_values,
            commands,
            measured,
            reset_flags,
            frozen_flags,
            rewards,
            reward_names,
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
        )
        normal_video_writer.close()
        slow_video_writer.close()

        rewards_dir = mode_dir / "rewards"
        rewards_dir.mkdir(parents=True, exist_ok=True)

        steps_per_command = max(
            1, int(round(TEST_DURATION_SECONDS / float(mode_env.dt)))
        )
        for test_index, (test_name, _) in enumerate(tests):
            start = test_index * steps_per_command
            stop = min((test_index + 1) * steps_per_command, len(time_values))
            # Each CSV has local time from 0 to 5 seconds for easier comparison.
            local_time = time_values[start:stop] - test_index * TEST_DURATION_SECONDS
            save_csv(
                mode_dir / f"{test_name}.csv",
                local_time,
                commands[start:stop],
                measured[start:stop],
                reset_flags[start:stop],
                frozen_flags[start:stop],
                component_names,
            )
            save_tracking_plot(
                mode_dir / f"{test_name}_tracking.png",
                f"{env_name} {mode_name} - {test_name}",
                local_time,
                commands[start:stop],
                measured[start:stop],
                reset_flags[start:stop],
                frozen_flags[start:stop],
                component_labels,
            )
            reward_csv_path = rewards_dir / f"{test_name}_rewards.csv"
            save_rewards_csv(
                reward_csv_path,
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

        save_tracking_plot(
            mode_dir / "tracking_complete.png",
            f"{env_name} {mode_name} - complete command sequence",
            time_values,
            commands,
            measured,
            reset_flags,
            frozen_flags,
            component_labels,
        )
        print(
            f"  completed: {time_values[-1]:.3f} s, "
            f"automatic resets={int(np.sum(reset_flags))}"
        )
        print(f"  video saved to: {video_path}")
        print(f"  slow video saved to: {slow_video_path}")

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
        tests=tests,
    )

    print(f"Comparison completed: {comparison_dir}")


if __name__ == "__main__":
    main()
