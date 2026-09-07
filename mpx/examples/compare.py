"""Compare Lite3 MPC baseline and residual policy velocity tracking.

The script runs one continuous 30-second sequence (six five-second commands)
for the baseline, residual policy, and optional end-to-end policy. It saves six
separate CSV files, six per-test tracking plots, one complete tracking plot,
and one complete video per controller:

    <selected checkpoint>/comparison_lite3/{baseline,residual}/

Examples:
    python compare_lite3.py
    python compare_lite3.py --load
    python compare_lite3.py --load 20260907_101530
    python compare_lite3.py --load saved/my_lite3_run
    python compare_lite3.py --load final
    python compare_lite3.py --load --load-e2e saved/my_e2e_run
    python compare_lite3.py --load --no-e2e
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


ENV_NAME = "Lite3JoystickFlatTerrain"
E2E_ENV_NAME = "Lite3JoystickE2EFlatTerrain"
TEST_DURATION_SECONDS = 5.0
TESTS = (
    ("vx_1p5", np.array([1.5, 0.0, 0.0], dtype=np.float32)),
    ("vy_0p4", np.array([0.0, 0.4, 0.0], dtype=np.float32)),
    ("wz_0p6", np.array([0.0, 0.0, 0.6], dtype=np.float32)),
    ("vx_1p0_vy_0p4", np.array([1.0, 0.4, 0.0], dtype=np.float32)),
    ("vx_1p0_wz_0p6", np.array([1.0, 0.0, 0.6], dtype=np.float32)),
    ("vy_0p4_wz_0p6", np.array([0.0, 0.4, 0.6], dtype=np.float32)),
)


def add_video_hud(
    frame: np.ndarray,
    controller_name: str,
    command: np.ndarray,
    measured: np.ndarray,
    frozen: bool = False,
) -> np.ndarray:
    """Overlay controller name and command/measured velocity values."""
    component_names = ("vx", "vy", "omega")
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
    duration_seconds = TEST_DURATION_SECONDS * len(TESTS)
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
        description="Compare Lite3 baseline and residual velocity tracking."
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
            "checkpoint from the latest Lite3 E2E run."
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
    # The provided Lite3 joystick environment does not need it: its baseline is
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
) -> None:
    labels = (("vx", "m/s"), ("vy", "m/s"), ("wz", "rad/s"))
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    frozen_mask = np.asarray(frozen_flags, dtype=bool)
    for axis, (label, unit), index in zip(axes, labels, range(3)):
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
) -> None:
    with output_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            (
                "time_s", "cmd_vx", "cmd_vy", "cmd_wz",
                "vx", "vy", "wz", "reset", "frozen",
            )
        )
        for time_s, command, velocity, reset, frozen in zip(
            time_values, commands, measured, reset_flags, frozen_flags
        ):
            writer.writerow(
                (
                    time_s, *command.tolist(), *velocity.tolist(),
                    int(reset), int(frozen),
                )
            )


def compare_graphics(comparison_dir: Path) -> None:
    """Create overlaid baseline, residual, and optional E2E tracking plots."""
    controller_specs = [
        ("baseline", "tab:blue"),
        ("residual", "tab:orange"),
    ]
    if (comparison_dir / "end_to_end").is_dir():
        controller_specs.append(("end_to_end", "tab:green"))

    output_dir = comparison_dir / "compare_graphics"
    output_dir.mkdir(parents=True, exist_ok=True)

    velocity_fields = (
        ("vx", "cmd_vx", "vx [m/s]"),
        ("vy", "cmd_vy", "vy [m/s]"),
        ("wz", "cmd_wz", "wz [rad/s]"),
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

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
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
        fig.suptitle(f"Lite3 {names} - {baseline_csv.stem}")
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    reset_index = 0
    state = reset_fn(jax.random.PRNGKey(seed + reset_index)[None, :])
    first_command = jnp.asarray(TESTS[0][1])
    state = set_fixed_command(state, first_command, baseline=not residual)

    # Rebuild the observation after replacing the command at reset.
    zero_action = jnp.zeros((1, env.action_size), dtype=jnp.float32)
    get_obs_fn = jax.jit(jax.vmap(env._get_obs))
    state = state.replace(obs=get_obs_fn(state.data, state.info, zero_action))

    steps_per_command = max(1, int(round(TEST_DURATION_SECONDS / float(env.dt))))
    num_steps = steps_per_command * len(TESTS)
    measured = []
    commands = []
    reset_flags = []
    frozen_flags = []
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
        command_index = min(step_index // steps_per_command, len(TESTS) - 1)
        command_jax = jnp.asarray(TESTS[command_index][1])
        command_block_finished = (
            (step_index + 1) % steps_per_command == 0
            and step_index + 1 < num_steps
        )

        if frozen:
            # After an early termination, preserve the final state visually
            # and numerically until this command's five-second block ends.
            measured.append(last_velocity.copy())
            current_command = np.asarray(TESTS[command_index][1])
            commands.append(current_command)
            reset_flags.append(command_block_finished)
            frozen_flags.append(True)
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
            last_velocity = np.asarray(
                jax.device_get(velocity_fn(state.data)[0])
            )
            measured.append(last_velocity.copy())
            current_command = np.asarray(TESTS[command_index][1])
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
                (step_index + 1) // steps_per_command, len(TESTS) - 1
            )
            next_command = jnp.asarray(TESTS[next_command_index][1])
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
    time_values = np.arange(1, len(measured_array) + 1, dtype=np.float64) * float(env.dt)
    return (
        time_values,
        commands_array,
        measured_array,
        reset_flags_array,
        frozen_flags_array,
    )


def main() -> None:
    args = parse_args()
    train_srbd.ALGO = args.algo
    train_srbd.ALGO_PARAMS = (
        train_srbd.SAC_PARAMS if args.algo == "sac" else train_srbd.PPO_PARAMS
    )

    _, env, _ = train_srbd.make_envs(env_name=ENV_NAME)
    env_base_dir = os.path.join(args.ckpt_dir, ENV_NAME)
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
            return jnp.stack(
                (linear[:, 0], linear[:, 1], angular[:, 2]), axis=-1
            )

        return reset, step, velocity

    reset_fn, step_fn, velocity_fn = build_runtime(env)

    comparison_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_dir = Path(run_dir) / "comparison_lite3" / comparison_timestamp
    controller_modes = [
        ("baseline", env, reset_fn, step_fn, velocity_fn, policy_fn, False),
        ("residual", env, reset_fn, step_fn, velocity_fn, policy_fn, True),
    ]

    if not args.no_e2e:
        _, e2e_env, _ = train_srbd.make_envs(env_name=E2E_ENV_NAME)
        e2e_base_dir = os.path.join(args.ckpt_dir, E2E_ENV_NAME)
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
        print(f"[{mode_name}] Starting the continuous 30-second sequence")
        time_values, commands, measured, reset_flags, frozen_flags = run_sequence(
            env=mode_env,
            controller_name=mode_name,
            reset_fn=mode_reset_fn,
            step_fn=mode_step_fn,
            velocity_fn=mode_velocity_fn,
            policy_fn=mode_policy_fn,
            video_writers=[normal_video_writer, slow_video_writer],
            residual=uses_policy,
            seed=args.seed,
        )
        normal_video_writer.close()
        slow_video_writer.close()

        steps_per_command = max(
            1, int(round(TEST_DURATION_SECONDS / float(mode_env.dt)))
        )
        for test_index, (test_name, _) in enumerate(TESTS):
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
            )
            save_tracking_plot(
                mode_dir / f"{test_name}_tracking.png",
                f"Lite3 {mode_name} - {test_name}",
                local_time,
                commands[start:stop],
                measured[start:stop],
                reset_flags[start:stop],
                frozen_flags[start:stop],
            )

        save_tracking_plot(
            mode_dir / "tracking_complete.png",
            f"Lite3 {mode_name} - complete command sequence",
            time_values,
            commands,
            measured,
            reset_flags,
            frozen_flags,
        )
        print(
            f"  completed: {time_values[-1]:.3f} s, "
            f"automatic resets={int(np.sum(reset_flags))}"
        )
        print(f"  video saved to: {video_path}")
        print(f"  slow video saved to: {slow_video_path}")

    compare_graphics(comparison_dir)
    create_comparison_video(
        video_paths=controller_video_paths,
        source_fps=controller_video_fps,
        output_path=(
            comparison_dir
            / "compare_graphics"
            / "controllers_comparison_2x2.mp4"
        ),
        fps=1.0 / float(env.dt),
    )

    print(f"Comparison completed: {comparison_dir}")


if __name__ == "__main__":
    main()
