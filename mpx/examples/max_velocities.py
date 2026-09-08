"""Find the commandable-velocity envelope for MPC baseline, MPC + residual and
end-to-end policies.

The environment is selected with --name (default: the Lite3 joystick env),
exactly like compare.py. This script reuses compare.py/train_srbd.py for env
selection, checkpoint loading, PPO/SAC policy construction, the fixed-command
injection (set_fixed_command), reset, termination and fall detection, so the
tracking semantics match the comparison pipeline.

For every controller it sweeps commands outward from the zero command along a
set of directions and, for each direction, keeps the last command that the robot
can actually track (survives the whole trial without falling / NaN and whose
velocity converges to the command within a tolerance). The collection of those
last-valid points per direction is the velocity envelope.

The command layout (which components exist and in which order) is read from the
selected env's command_config.names, so it is never hardcoded here: the script
works for Lite3 (["vx", "vy", "wz"]), Tita (["vx", "wz"]) or any other layout,
including a single-component env.

Outputs (created next to baseline/, residual/, end_to_end/, compare_graphics/,
compare_rewards/):

    <comparison_run>/max_velocities/
        baseline_velocities/     residual_velocities/
        end_to_end_velocities/   (unless --no-e2e)
        comparison/

Each controller folder holds one CSV with every tested point (so the plots can
be regenerated from the CSVs alone), one summary CSV with the last valid command
per direction, the 2D envelope plots, an optional 3D representation and a
max-velocity histogram. comparison/ overlays the controllers.

Examples:
    python max_velocities.py
    python max_velocities.py --name tita
    python max_velocities.py --no-e2e
    python max_velocities.py --velocity-step 0.1 --settling-time 2.0 --evaluation-time 3.0
    python max_velocities.py --plot-only path/to/max_velocities
    # quick smoke test (reduced ranges):
    python max_velocities.py --name tita --no-e2e --num-directions 8 \
        --max-vx 1.0 --max-wz 1.0 --settling-time 0.5 --evaluation-time 0.5
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
from datetime import datetime
from pathlib import Path
import time

# Configure MuJoCo before it is imported (matches compare.py / train_srbd.py).
if not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless-safe; must precede pyplot import.
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
#  Colours: same mapping used by compare.py, defined once and reused everywhere.
# ─────────────────────────────────────────────────────────────────────────────
CONTROLLER_COLORS = {
    "baseline": "tab:blue",
    "residual": "tab:orange",
    "end_to_end": "tab:green",
}
CONTROLLER_ORDER = ("baseline", "residual", "end_to_end")

# Unit lookup fallback (kept in sync with compare._MEASURED_SOURCE) so --plot-only
# can label axes without importing jax/mujoco. When the simulation stack is
# importable, compare._MEASURED_SOURCE is preferred as the single source.
_UNIT_FALLBACK = {
    "vx": "m/s", "vy": "m/s", "vz": "m/s",
    "wx": "rad/s", "wy": "rad/s", "wz": "rad/s",
}

MAX_VELOCITY_TARGETS = {
    "Lite3JoystickFlatTerrain": (
        ("vx_3p0",              np.array([3.0, 0.0, 0.0], np.float32), dict(vx_inc=0.5)),
        ("vy_1p0",              np.array([0.0, 1.0, 0.0], np.float32), dict(vy_inc=0.1)),
        ("wz_3p0",              np.array([0.0, 0.0, 3.0], np.float32), dict(wz_inc=0.5)),
        ("vx_3p0_vy_1p0",       np.array([3.0, 1.0, 0.0], np.float32), dict(vx_inc=0.5, vy_inc=0.1)),
        ("vx_3p0_wz_0p5",       np.array([3.0, 0.0, 0.5], np.float32), dict(vx_inc=0.5, wz_inc=0.1)),
        ("vy_1p0_wz_0p5",       np.array([0.0, 1.0, 0.5], np.float32), dict(vy_inc=0.1, wz_inc=0.1)),
        ("vx_3p0_vy_1p0_wz_0p5",np.array([3.0, 1.0, 0.5], np.float32), dict(vx_inc=0.5, vy_inc=0.1, wz_inc=0.5)),
    ),
    "TitaJoystickFlatTerrain": (
        ("vx_3p0",        np.array([3.0, 0.0], np.float32), dict(vx_inc=0.1)),
        ("wz_0p8",        np.array([0.0, 0.8], np.float32), dict(wz_inc=0.)),
        ("vx_3p0_wz_0p8", np.array([3.0, 0.8], np.float32), dict(vx_inc=0.1, wz_inc=0.1)),
    ),
}

def _march_targets(command_names, target, increments, default_step):
    """Commands from ~0 toward `target`. Each active component ramps by its own
    increment (abs value; direction = sign of target) and is clamped at its
    target, so components with different increments/targets each converge to
    their own value. The exact target is always the last command."""
    target = np.asarray(target, dtype=np.float64)
    num = len(command_names)
    step = np.array([
        abs(increments.get(f"{command_names[c]}_inc", default_step)) for c in range(num)
    ], dtype=np.float64)
    step = np.where(step <= 0.0, default_step, step)
    active = np.abs(target) > 1e-12

    steps_needed = 1
    for c in range(num):
        if active[c]:
            steps_needed = max(steps_needed, int(np.ceil(abs(target[c]) / step[c])))

    commands = []
    for k in range(1, steps_needed + 1):
        cmd = np.zeros(num)
        for c in range(num):
            if active[c]:
                cmd[c] = np.sign(target[c]) * min(abs(target[c]), k * step[c])
        commands.append(cmd)
    if not np.allclose(commands[-1], target):
        commands.append(target.copy())
    return commands


def _fmt_elapsed(seconds: float) -> str:
    minutes, secs = divmod(int(round(seconds)), 60)
    return f"{minutes:02d}:{secs:02d}"

def _units_for(name: str) -> str:
    try:
        from compare import _MEASURED_SOURCE  # noqa: WPS433 (lazy on purpose)

        return _MEASURED_SOURCE[name][2]
    except Exception:
        return _UNIT_FALLBACK.get(name, "")


def _is_angular(name: str) -> bool:
    return _units_for(name) == "rad/s"


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    # DEFAULT_ENV_NAME lives in compare.py; import lazily to keep --plot-only
    # free of the heavy stack, with a literal fallback for the help string.
    try:
        from compare import DEFAULT_ENV_NAME
    except Exception:
        DEFAULT_ENV_NAME = "Lite3JoystickFlatTerrain"

    parser = argparse.ArgumentParser(
        description="Find the commandable-velocity envelope per controller."
    )
    # --- reused from compare.py (env / checkpoints / controllers) ---
    parser.add_argument(
        "--name", type=str, default=DEFAULT_ENV_NAME,
        help=(
            "Environment to evaluate. Accepts a MuJoCo Playground env name or a "
            "compare.py shortcut (go1, aliengo, tita, titae2e, lite3, lite3e2e)."
        ),
    )
    parser.add_argument(
        "--load", nargs="?", const="best", default="best", metavar="RUN_OR_SUFFIX",
        help="Residual checkpoint to load (see compare.py).",
    )
    parser.add_argument(
        "--load-e2e", nargs="?", const="best", default="best", metavar="RUN_OR_SUFFIX",
        help="End-to-end checkpoint to load (see compare.py).",
    )
    parser.add_argument(
        "--no-e2e", action="store_true",
        help="Skip the end-to-end policy; sweep only baseline and residual.",
    )
    parser.add_argument(
        "--ckpt-dir", default="checkpoints",
        help="Checkpoint root used by train_srbd.py (default: checkpoints).",
    )
    parser.add_argument(
        "--algo", choices=("ppo", "sac"), default="ppo",
        help="Algorithm used to train the checkpoints.",
    )
    parser.add_argument("--seed", type=int, default=42)

    # --- search configuration ---
    parser.add_argument(
        "--velocity-step", type=float, default=0.1,
        help="Magnitude increment while marching outward (default: 0.1).",
    )
    parser.add_argument(
        "--settling-time", type=float, default=2.0,
        help="Seconds to let the robot reach the command before measuring.",
    )
    parser.add_argument(
        "--evaluation-time", type=float, default=3.0,
        help="Seconds used to judge stability / convergence (after settling).",
    )
    parser.add_argument(
        "--max-tracking-error", type=float, default=0.30,
        help="Max RMSE (m/s) on linear components for a command to count valid.",
    )
    parser.add_argument(
        "--max-tracking-error-ang", type=float, default=0.50,
        help="Max RMSE (rad/s) on angular components for a command to count valid.",
    )
    # Safety caps (avoid infinite marching). These are NOT physical limits and are
    # NOT taken from command_config.a (which is only the training sampling range).
    parser.add_argument("--max-vx", type=float, default=4.0,
                        help="Safety cap |vx| (m/s). Default 4.0.")
    parser.add_argument("--max-vy", type=float, default=2.0,
                        help="Safety cap |vy| (m/s). Default 2.0.")
    parser.add_argument("--max-wz", type=float, default=4.0,
                        help="Safety cap |wz| (rad/s). Default 4.0.")
    parser.add_argument("--max-linear", type=float, default=4.0,
                        help="Safety cap for any other linear component (m/s).")
    parser.add_argument("--max-angular", type=float, default=4.0,
                        help="Safety cap for any other angular component (rad/s).")

    parser.add_argument(
        "--num-directions", type=int, default=24,
        help="Directions around the circle for 2-component combinations.",
    )
    parser.add_argument(
        "--num-directions-3d", type=int, default=60,
        help="Sphere directions for 3-component combinations.",
    )
    parser.add_argument(
        "--num-directions-nd", type=int, default=120,
        help="Random unit directions for combinations of 4+ components.",
    )

    parser.add_argument(
        "--into", type=str, default=None, metavar="COMPARISON_DIR",
        help=(
            "Existing comparison run directory to drop max_velocities/ into. "
            "By default a fresh comparison_<env>/<timestamp>/ is created."
        ),
    )
    parser.add_argument(
        "--plot-only", nargs="?", const="__auto__", default=None,
        metavar="MAX_VELOCITIES_DIR",
        help=(
            "Regenerate plots from existing CSVs without running any simulation. "
            "Optionally give the max_velocities directory; otherwise the most "
            "recent one is used."
        ),
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  CSV schema (writer and reader agree on these builders)
# ─────────────────────────────────────────────────────────────────────────────
def _points_header(names: list[str]) -> list[str]:
    return (
        ["controller", "combination", "direction_index"]
        + [f"{n}_command" for n in names]
        + [f"{n}_measured_mean" for n in names]
        + [f"{n}_rmse" for n in names]
        + ["command_norm", "stable", "fell", "terminated", "converged",
           "finite", "valid", "settling_time_s", "evaluation_time_s"]
    )


def _summary_header(names: list[str]) -> list[str]:
    return (
        ["controller", "combination", "direction_index"]
        + [f"{n}_command" for n in names]
        + ["command_norm", "cap_limited", "found_valid",
           "settling_time_s", "evaluation_time_s"]
    )


def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _read_csv_dicts(path: Path) -> list[dict]:
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def _names_from_header(fieldnames: list[str]) -> list[str]:
    return [f[: -len("_command")] for f in fieldnames if f.endswith("_command")]


# ─────────────────────────────────────────────────────────────────────────────
#  Direction generation
# ─────────────────────────────────────────────────────────────────────────────
def _fibonacci_sphere(count: int) -> np.ndarray:
    """`count` roughly uniform unit vectors on the sphere (both hemispheres)."""
    if count <= 1:
        return np.array([[0.0, 0.0, 1.0]])
    indices = np.arange(count)
    y = 1.0 - 2.0 * indices / (count - 1)
    radius = np.sqrt(np.clip(1.0 - y * y, 0.0, 1.0))
    phi = np.pi * (3.0 - np.sqrt(5.0))
    theta = phi * indices
    return np.stack([np.cos(theta) * radius, y, np.sin(theta) * radius], axis=1)


def make_directions(combo: tuple[int, ...], num_components: int,
                    args: argparse.Namespace) -> list[np.ndarray]:
    """Unit directions for a combination, each a length-`num_components` vector
    with zeros outside `combo`. Covers positive and negative directions."""
    order = len(combo)
    directions: list[np.ndarray] = []

    if order == 1:
        for sign in (1.0, -1.0):
            vec = np.zeros(num_components)
            vec[combo[0]] = sign
            directions.append(vec)
    elif order == 2:
        num = max(4, args.num_directions)
        for step in range(num):
            angle = 2.0 * np.pi * step / num
            vec = np.zeros(num_components)
            vec[combo[0]] = np.cos(angle)
            vec[combo[1]] = np.sin(angle)
            directions.append(vec)
    elif order == 3:
        for point in _fibonacci_sphere(max(6, args.num_directions_3d)):
            vec = np.zeros(num_components)
            for local_index, global_index in enumerate(combo):
                vec[global_index] = point[local_index]
            directions.append(vec)
    else:
        rng = np.random.default_rng(0)  # deterministic
        gauss = rng.standard_normal((max(8, args.num_directions_nd), order))
        gauss /= np.linalg.norm(gauss, axis=1, keepdims=True)
        for row in gauss:
            vec = np.zeros(num_components)
            for local_index, global_index in enumerate(combo):
                vec[global_index] = row[local_index]
            directions.append(vec)
    return directions


def _component_cap(name: str, args: argparse.Namespace) -> float:
    explicit = {"vx": args.max_vx, "vy": args.max_vy, "wz": args.max_wz}
    if name in explicit:
        return abs(explicit[name])
    return abs(args.max_angular if _is_angular(name) else args.max_linear)


def _tol_per_name(names: list[str], args: argparse.Namespace) -> dict[str, float]:
    return {
        n: (args.max_tracking_error_ang if _is_angular(n) else args.max_tracking_error)
        for n in names
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Simulation stack (lazy imports so --plot-only never needs jax/mujoco)
# ─────────────────────────────────────────────────────────────────────────────
def build_controllers(args: argparse.Namespace):
    """Mirror compare.py's setup: pick env from --name, build PPO/SAC policies,
    load residual (and optional E2E) checkpoints, and JIT reset/step/velocity/
    upvector once per env so nothing recompiles per command.

    Returns (controllers, component_names, component_units, comparison_dir).
    """
    import jax
    import jax.numpy as jnp
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.sac import networks as sac_networks

    import compare
    import train_srbd

    train_srbd.ALGO = args.algo
    train_srbd.ALGO_PARAMS = (
        train_srbd.SAC_PARAMS if args.algo == "sac" else train_srbd.PPO_PARAMS
    )

    env_name = compare._NAME_SHORTCUTS.get(args.name.lower(), args.name)
    e2e_env_name = compare._to_e2e_name(env_name)

    _, env, _ = train_srbd.make_envs(env_name=env_name)

    # Command layout is read from the env itself (never hardcoded).
    component_names = list(env._config.command_config.names)
    component_units = [compare._MEASURED_SOURCE[n][2] for n in component_names]

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
        get_obs = jax.jit(jax.vmap(eval_env._get_obs))
        local_velocity = jax.jit(jax.vmap(eval_env.get_local_linvel))
        gyro = jax.jit(jax.vmap(eval_env.get_gyro))
        up = jax.jit(jax.vmap(eval_env.get_upvector))

        @jax.jit
        def velocity(data):
            linear = local_velocity(data)
            angular = gyro(data)
            columns = []
            for name in component_names:
                source, index, _ = compare._MEASURED_SOURCE[name]
                series = linear if source == "linvel" else angular
                columns.append(series[:, index])
            return jnp.stack(columns, axis=-1)

        @jax.jit
        def upvector_z(data):
            return up(data)[:, 2]

        return reset, step, get_obs, velocity, upvector_z

    reset_fn, step_fn, get_obs_fn, velocity_fn, upvector_fn = build_runtime(env)

    controllers = [
        dict(name="baseline", env=env, reset_fn=reset_fn, step_fn=step_fn,
             get_obs_fn=get_obs_fn, velocity_fn=velocity_fn, upvector_fn=upvector_fn,
             policy_fn=policy_fn, uses_policy=False,
             action_size=int(env.action_size), dt=float(env.dt)),
        dict(name="residual", env=env, reset_fn=reset_fn, step_fn=step_fn,
             get_obs_fn=get_obs_fn, velocity_fn=velocity_fn, upvector_fn=upvector_fn,
             policy_fn=policy_fn, uses_policy=True,
             action_size=int(env.action_size), dt=float(env.dt)),
    ]

    if not args.no_e2e:
        _, e2e_env, _ = train_srbd.make_envs(env_name=e2e_env_name)
        e2e_base_dir = os.path.join(args.ckpt_dir, e2e_env_name)
        e2e_run_dir, e2e_suffix = train_srbd._resolve_load(e2e_base_dir, args.load_e2e)
        e2e_params = train_srbd.load_params(e2e_run_dir, suffix=e2e_suffix)
        if e2e_params is None:
            raise FileNotFoundError(f"No end-to-end checkpoint found in '{e2e_run_dir}'.")
        e2e_networks = train_srbd._build_fresh_networks(e2e_env)
        e2e_make_inference_fn = (
            ppo_networks.make_inference_fn(e2e_networks)
            if args.algo == "ppo"
            else sac_networks.make_inference_fn(e2e_networks)
        )
        e2e_policy_fn = jax.jit(e2e_make_inference_fn(e2e_params, deterministic=True))
        e_reset, e_step, e_obs, e_vel, e_up = build_runtime(e2e_env)
        controllers.append(
            dict(name="end_to_end", env=e2e_env, reset_fn=e_reset, step_fn=e_step,
                 get_obs_fn=e_obs, velocity_fn=e_vel, upvector_fn=e_up,
                 policy_fn=e2e_policy_fn, uses_policy=True,
                 action_size=int(e2e_env.action_size), dt=float(e2e_env.dt))
        )
        print(f"End-to-end checkpoint: {e2e_run_dir}")

    if args.into:
        comparison_dir = Path(args.into)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_dir = Path(run_dir) / f"comparison_{env_name}" / stamp
    comparison_dir.mkdir(parents=True, exist_ok=True)

    print(f"Environment: {env_name}")
    print(f"Command components: {component_names}")
    print(f"Residual checkpoint: {run_dir}")
    print(f"Output directory: {comparison_dir / 'max_velocities'}")
    return controllers, component_names, component_units, comparison_dir


def evaluate_command(controller, command_vec, *, settling_steps, eval_steps,
                     seed, component_names, tol_per_name):
    """Run one command from a clean, deterministic reset and judge it.

    Uses compare.set_fixed_command / compare.state_observation so the command
    injection, reset, termination and fall detection match compare.py exactly.
    Statistics (mean, RMSE, stability, convergence) use ONLY the evaluation
    window, i.e. the steps after the settling phase.
    """
    import jax
    import jax.numpy as jnp

    from compare import set_fixed_command, state_observation

    residual = controller["uses_policy"]
    num_components = len(component_names)
    command_jax = jnp.asarray(command_vec, dtype=jnp.float32)
    zero_action = jnp.zeros((1, controller["action_size"]), dtype=jnp.float32)
    get_obs_fn = controller["get_obs_fn"]

    # Deterministic reset: same seed for every point isolates the effect of the
    # command and stops one command's failure from leaking into the next.
    state = controller["reset_fn"](jax.random.PRNGKey(seed)[None, :])
    state = set_fixed_command(state, command_jax, baseline=not residual)
    state = state.replace(obs=get_obs_fn(state.data, state.info, zero_action))

    rng = jax.random.PRNGKey(seed + 10_000)
    total_steps = settling_steps + eval_steps
    eval_measured: list[np.ndarray] = []
    terminated = False
    fell = False
    finite = True

    for step_index in range(total_steps):
        state = set_fixed_command(state, command_jax, baseline=not residual)
        state = state.replace(
            obs=get_obs_fn(state.data, state.info, state.info["last_act"])
        )
        if residual:
            rng, action_rng = jax.random.split(rng)
            action, _ = controller["policy_fn"](state_observation(state), action_rng)
            action = action[None, :]
        else:
            action = zero_action

        state = controller["step_fn"](state, action)
        state = set_fixed_command(state, command_jax, baseline=not residual)

        done = bool(np.asarray(jax.device_get(state.done[0])))
        up_z = float(np.asarray(jax.device_get(controller["upvector_fn"](state.data)[0])))
        velocity = np.asarray(
            jax.device_get(controller["velocity_fn"](state.data)[0]), dtype=np.float64
        )

        if not np.all(np.isfinite(velocity)):
            finite = False
        if up_z < 0.0:
            fell = True
        if done:
            terminated = True
        if step_index >= settling_steps:
            eval_measured.append(velocity)
        if done or fell or (not finite):
            break

    survived = (not terminated) and (not fell) and finite
    if eval_measured:
        window = np.asarray(eval_measured, dtype=np.float64)
        measured_mean = np.nanmean(window, axis=0)
        command = np.asarray(command_vec, dtype=np.float64)
        rmse = np.sqrt(np.nanmean((window - command[None, :]) ** 2, axis=0))
    else:
        measured_mean = np.full(num_components, np.nan)
        rmse = np.full(num_components, np.nan)

    tol = np.array([tol_per_name[name] for name in component_names], dtype=np.float64)
    converged = (
        survived
        and bool(np.all(np.isfinite(rmse)))
        and bool(np.all(rmse <= tol))
    )
    stable = survived
    valid = stable and converged
    return dict(
        measured_mean=measured_mean, rmse=rmse, stable=stable, fell=fell,
        terminated=terminated, converged=converged, finite=finite, valid=valid,
    )


def run_search(controller, component_names, args):
    """March toward each named target in MAX_VELOCITY_TARGETS, recording every
    tested point and the last valid one per target."""
    import time
    import compare

    env_name = compare._NAME_SHORTCUTS.get(args.name.lower(), args.name)
    if env_name not in MAX_VELOCITY_TARGETS:
        raise KeyError(
            f"No max-velocity targets defined for '{env_name}'. "
            f"Add an entry to MAX_VELOCITY_TARGETS (available: "
            f"{sorted(MAX_VELOCITY_TARGETS)})."
        )
    tests = MAX_VELOCITY_TARGETS[env_name]

    tol_per_name = _tol_per_name(component_names, args)
    dt = controller["dt"]
    settling_steps = max(1, int(round(args.settling_time / dt)))
    eval_steps = max(1, int(round(args.evaluation_time / dt)))

    point_rows, summary_rows = [], []
    controller_name = controller["name"]
    combo_counter: dict[str, int] = {}

    for test_name, target, increments in tests:
        target = np.asarray(target, dtype=np.float64)
        active = [i for i in range(len(component_names)) if abs(target[i]) > 1e-12]
        combo_name = "+".join(component_names[i] for i in active) or test_name
        direction_index = combo_counter.get(combo_name, 0)
        combo_counter[combo_name] = direction_index + 1

        commands = _march_targets(component_names, target, increments, args.velocity_step)
        last_valid = np.zeros(len(component_names))
        found_valid = False
        reached_target = False
        last_read = np.full(len(component_names), np.nan)   # <-- aggiungi
        target_start = time.perf_counter()  

        for command in commands:
            point_start = time.perf_counter()
            result = evaluate_command(
                controller, command, settling_steps=settling_steps,
                eval_steps=eval_steps, seed=args.seed,
                component_names=component_names, tol_per_name=tol_per_name,
            )
            point_seconds = time.perf_counter() - point_start

            point_rows.append(
                [controller_name, combo_name, direction_index]
                + [f"{v:.6f}" for v in command]
                + [f"{v:.6f}" for v in result["measured_mean"]]
                + [f"{v:.6f}" for v in result["rmse"]]
                + [f"{float(np.linalg.norm(command)):.6f}",
                   int(result["stable"]), int(result["fell"]),
                   int(result["terminated"]), int(result["converged"]),
                   int(result["finite"]), int(result["valid"]),
                   f"{args.settling_time:.3f}", f"{args.evaluation_time:.3f}"]
            )

            cmd_text = " ".join(f"{n}={c:+.2f}" for n, c in zip(component_names, command))
            status = "VALID" if result["valid"] else (
                "FALL" if result["fell"] else
                "TERM" if result["terminated"] else
                "NAN" if not result["finite"] else "NOCONV"
            )
            if command is commands[0]:
                print(f"[{controller_name}] {test_name}")
            print(f"       cmd  {cmd_text} | {status}")

            # keep the data for the final summary lines
            last_read = result["measured_mean"]
            final_status = status                 # <-- aggiungi
            final_command = command 

            if result["valid"]:
                last_valid = command.copy()
                found_valid = True
                reached_target = bool(np.allclose(command, target))
            else:
                reached_target = False
                break

        target_seconds = time.perf_counter() - target_start
        read_text = " ".join(f"{n}={m:+.2f}" for n, m in zip(component_names, last_read))
        last_text = " ".join(f"{n}={c:+.2f}" for n, c in zip(component_names, last_valid))
        minutes, secs = divmod(int(round(target_seconds)), 60)
        if final_status != "VALID":
            reached_text = " ".join(
                f"{n}={c:+.2f}" for n, c in zip(component_names, final_command)
            )
            summary_line += (
                f"   | {final_status} at {reached_text}"
                f" after {minutes:02d}:{secs:02d}"
            )
        print(
            f"       read {read_text}\n"
            f"{summary_line}\n"
            f"       elapsed {minutes:02d}:{secs:02d}\n"
            f"       time {datetime.now().strftime('%H:%M:%S')}\n"
            "-------------------------------"
        )

        summary_rows.append(
            [controller_name, combo_name, direction_index]
            + [f"{v:.6f}" for v in last_valid]
            + [f"{float(np.linalg.norm(last_valid)):.6f}",
               int(reached_target), int(found_valid),
               f"{args.settling_time:.3f}", f"{args.evaluation_time:.3f}"]
        )

    return point_rows, summary_rows

# ─────────────────────────────────────────────────────────────────────────────
#  Plotting (numpy + matplotlib only, reads exclusively from CSVs)
# ─────────────────────────────────────────────────────────────────────────────
def _order_boundary(points_xy: np.ndarray) -> np.ndarray:
    """Order per-direction farthest-valid points into a closed boundary.

    The reachable set is assumed star-shaped w.r.t. the zero command (true for
    velocity envelopes: if a command is trackable, smaller ones in the same
    direction are too), so ordering the farthest-valid point of each direction
    by polar angle around the origin recovers the envelope boundary. This is
    preferred over a convex hull, which would inflate concave envelopes, and it
    degrades gracefully with few points.
    """
    points_xy = np.asarray(points_xy, dtype=np.float64)
    if points_xy.size == 0:
        return points_xy.reshape(0, 2)
    keep = np.linalg.norm(points_xy, axis=1) > 1e-9
    points_xy = points_xy[keep]
    if len(points_xy) == 0:
        return points_xy
    angles = np.arctan2(points_xy[:, 1], points_xy[:, 0])
    ordered = points_xy[np.argsort(angles)]
    if len(ordered) >= 3:
        ordered = np.vstack([ordered, ordered[0]])  # close the polygon
    return ordered


def _pair_arrays(points, summary, combo_name, name_i, name_j):
    """Extract boundary / stable / failed (x, y) arrays for one component pair."""
    key_i, key_j = f"{name_i}_command", f"{name_j}_command"

    boundary = np.array(
        [
            [float(r[key_i]), float(r[key_j])]
            for r in summary
            if r["combination"] == combo_name and int(r["found_valid"]) == 1
        ],
        dtype=np.float64,
    ).reshape(-1, 2)

    stable, failed = [], []
    for row in points:
        if row["combination"] != combo_name:
            continue
        xy = [float(row[key_i]), float(row[key_j])]
        if int(row["valid"]) == 1:
            stable.append(xy)
        elif np.linalg.norm(xy) > 1e-9:
            failed.append(xy)
    return (
        boundary,
        np.array(stable, dtype=np.float64).reshape(-1, 2),
        np.array(failed, dtype=np.float64).reshape(-1, 2),
    )


def _draw_pair(axis, boundary, color, label, stable=None, failed=None):
    if failed is not None and len(failed):
        axis.scatter(failed[:, 0], failed[:, 1], s=9, c="0.7", marker="x",
                     alpha=0.35, linewidths=0.6, zorder=1)
    if stable is not None and len(stable):
        axis.scatter(stable[:, 0], stable[:, 1], s=9, color=color,
                     alpha=0.22, zorder=2)
    ordered = _order_boundary(boundary)
    if len(ordered) >= 2:
        axis.plot(ordered[:, 0], ordered[:, 1], color=color, lw=2.0,
                  label=label, zorder=3)
    elif len(ordered) == 1:
        axis.scatter(ordered[:, 0], ordered[:, 1], color=color, s=45,
                     label=label, zorder=3)


def _draw_projection(axis, points_xy, color, label, scatter=True):
    """Draw the 2D "shadow" of a projected 3D stable-point cloud as its convex
    hull. A projection of a reachable volume is a filled 2D region, so its outer
    outline (convex hull) is an honest, unambiguous representation - unlike
    ordering projected boundary points by angle, which produces a misleading
    spiky curve. Falls back to the star-ordering only when scipy/enough points
    are unavailable."""
    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    if scatter and len(points):
        axis.scatter(points[:, 0], points[:, 1], s=8, color=color, alpha=0.22, zorder=1)
    if len(points) >= 3:
        try:
            from scipy.spatial import ConvexHull

            hull = ConvexHull(points)
            loop = np.append(hull.vertices, hull.vertices[0])
            axis.plot(points[loop, 0], points[loop, 1], color=color, lw=2.0,
                      label=label, zorder=3)
            return
        except Exception:
            pass
    ordered = _order_boundary(points)
    if len(ordered) >= 2:
        axis.plot(ordered[:, 0], ordered[:, 1], color=color, lw=2.0, label=label, zorder=3)
    elif len(ordered) == 1:
        axis.scatter(ordered[:, 0], ordered[:, 1], color=color, s=45, label=label, zorder=3)


def _axis_labels(axis, name_i, name_j):
    axis.set_xlabel(f"{name_i} [{_units_for(name_i)}]")
    axis.set_ylabel(f"{name_j} [{_units_for(name_j)}]")
    axis.axhline(0.0, color="black", lw=0.6, alpha=0.4)
    axis.axvline(0.0, color="black", lw=0.6, alpha=0.4)
    axis.grid(True, alpha=0.3)
    axis.set_aspect("equal", adjustable="datalim")


def plot_controller_envelopes(folder: Path, controller_name: str, names: list[str],
                              points: list[dict], summary: list[dict]) -> None:
    color = CONTROLLER_COLORS.get(controller_name, "tab:blue")
    num = len(names)

    if num == 1:  # 1D: max positive and negative reachable velocity.
        name = names[0]
        values = sorted(
            float(r[f"{name}_command"]) for r in summary if int(r["found_valid"]) == 1
        )
        pos = max([v for v in values if v > 0], default=0.0)
        neg = min([v for v in values if v < 0], default=0.0)
        fig, axis = plt.subplots(figsize=(7, 3.5))
        axis.barh([0], [pos], color=color, alpha=0.85, label="max +")
        axis.barh([0], [neg], color=color, alpha=0.45, label="max -")
        axis.axvline(0.0, color="black", lw=0.8)
        axis.set_yticks([])
        axis.set_xlabel(f"{name} [{_units_for(name)}]")
        axis.set_title(f"{controller_name} - {name} reachable range")
        axis.legend(loc="best")
        fig.tight_layout()
        fig.savefig(folder / "envelope_1d.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
    else:
        for i, j in itertools.combinations(range(num), 2):
            combo_name = f"{names[i]}+{names[j]}"
            boundary, stable, failed = _pair_arrays(points, summary, combo_name,
                                                    names[i], names[j])
            fig, axis = plt.subplots(figsize=(6, 6))
            _draw_pair(axis, boundary, color, controller_name, stable, failed)
            _axis_labels(axis, names[i], names[j])
            axis.set_title(f"{controller_name} envelope - {names[i]} vs {names[j]}")
            axis.legend(loc="best")
            fig.tight_layout()
            fig.savefig(folder / f"envelope_{names[i]}_{names[j]}.png",
                        dpi=160, bbox_inches="tight")
            plt.close(fig)

        if num == 3:
            _plot_3d_single(folder, controller_name, names, points, summary, color)

    _plot_histogram(folder, controller_name, names, summary, single=True)


def _stable_points_3d(points, names):
    keys = [f"{n}_command" for n in names]
    stable = [[float(r[k]) for k in keys] for r in points if int(r["valid"]) == 1]
    failed = [
        [float(r[k]) for k in keys]
        for r in points
        if int(r["valid"]) == 0 and float(r["command_norm"]) > 1e-9
    ]
    return (np.array(stable).reshape(-1, 3), np.array(failed).reshape(-1, 3))


def _plot_3d_single(folder, controller_name, names, points, summary, color):
    triple = "+".join(names)
    stable, failed = _stable_points_3d(
        [r for r in points if r["combination"] == triple], names
    )

    fig = plt.figure(figsize=(7, 6))
    axis = fig.add_subplot(111, projection="3d")
    if len(failed):
        axis.scatter(failed[:, 0], failed[:, 1], failed[:, 2], s=8, c="0.7",
                     marker="x", alpha=0.3)
    if len(stable):
        axis.scatter(stable[:, 0], stable[:, 1], stable[:, 2], s=12, color=color,
                     alpha=0.6, label=controller_name)
    axis.set_xlabel(f"{names[0]} [{_units_for(names[0])}]")
    axis.set_ylabel(f"{names[1]} [{_units_for(names[1])}]")
    axis.set_zlabel(f"{names[2]} [{_units_for(names[2])}]")
    axis.set_title(f"{controller_name} - stable {triple} commands")
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(folder / "envelope_3d_scatter.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Projections of the stable 3D cloud onto the three coordinate planes,
    # drawn as convex-hull shadows (see _draw_projection).
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, (i, j) in zip(axes, itertools.combinations(range(3), 2)):
        projected = stable[:, [i, j]] if len(stable) else np.empty((0, 2))
        _draw_projection(axis, projected, color, controller_name, scatter=True)
        _axis_labels(axis, names[i], names[j])
        axis.set_title(f"{names[i]} vs {names[j]}")
    fig.suptitle(f"{controller_name} - {triple} projections")
    fig.tight_layout()
    fig.savefig(folder / "envelope_3d_projections.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _histogram_series(names, summary):
    """Build labelled max-velocity bars. Singles keep sign (max +, max -);
    multi-component combinations use the command-vector norm ||command||."""
    labels, values = [], []
    for name in names:  # singles, split by sign
        pos = max(
            [float(r[f"{name}_command"]) for r in summary
             if r["combination"] == name and float(r[f"{name}_command"]) > 0],
            default=0.0,
        )
        neg = min(
            [float(r[f"{name}_command"]) for r in summary
             if r["combination"] == name and float(r[f"{name}_command"]) < 0],
            default=0.0,
        )
        labels.append(f"{name} +"); values.append(pos)
        labels.append(f"{name} -"); values.append(abs(neg))

    num = len(names)
    for order in range(2, num + 1):  # pairs, triples, ...: max ||command||
        for combo in itertools.combinations(range(num), order):
            combo_name = "+".join(names[k] for k in combo)
            norms = [
                float(r["command_norm"]) for r in summary
                if r["combination"] == combo_name and int(r["found_valid"]) == 1
            ]
            labels.append(f"||{combo_name}||")
            values.append(max(norms, default=0.0))
    return labels, values


def _plot_histogram(folder, controller_name, names, summary, single):
    labels, values = _histogram_series(names, summary)
    color = CONTROLLER_COLORS.get(controller_name, "tab:blue")
    fig, axis = plt.subplots(figsize=(max(7, 0.6 * len(labels) + 3), 4.5))
    axis.bar(range(len(labels)), values, color=color, alpha=0.85)
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=45, ha="right")
    axis.set_ylabel("max reachable (signed value / ||command||)")
    axis.set_title(f"{controller_name} - max velocities per command combination")
    axis.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(folder / "histogram_max_velocities.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(mv_dir: Path, names: list[str],
                    controller_data: dict[str, tuple[list, list]]) -> None:
    """Overlay every controller. Reads only the per-controller CSVs already
    written. Uses CONTROLLER_COLORS everywhere; works with --no-e2e."""
    out_dir = mv_dir / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    present = [c for c in CONTROLLER_ORDER if c in controller_data]
    num = len(names)

    if num == 1:
        name = names[0]
        fig, axis = plt.subplots(figsize=(8, 4))
        for row_index, controller in enumerate(present):
            _, summary = controller_data[controller]
            vals = [float(r[f"{name}_command"]) for r in summary
                    if int(r["found_valid"]) == 1]
            pos = max([v for v in vals if v > 0], default=0.0)
            neg = min([v for v in vals if v < 0], default=0.0)
            color = CONTROLLER_COLORS[controller]
            axis.barh([row_index], [pos], color=color, alpha=0.85)
            axis.barh([row_index], [neg], color=color, alpha=0.45)
        axis.set_yticks(range(len(present)))
        axis.set_yticklabels(present)
        axis.axvline(0.0, color="black", lw=0.8)
        axis.set_xlabel(f"{name} [{_units_for(name)}]")
        axis.set_title(f"Reachable {name} range per controller")
        fig.tight_layout()
        fig.savefig(out_dir / "envelope_1d.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
    else:
        for i, j in itertools.combinations(range(num), 2):
            combo_name = f"{names[i]}+{names[j]}"
            fig, axis = plt.subplots(figsize=(6, 6))
            for controller in present:
                points, summary = controller_data[controller]
                boundary, _, _ = _pair_arrays(points, summary, combo_name,
                                              names[i], names[j])
                _draw_pair(axis, boundary, CONTROLLER_COLORS[controller], controller)
            _axis_labels(axis, names[i], names[j])
            axis.set_title(f"Envelope comparison - {names[i]} vs {names[j]}")
            axis.legend(loc="best")
            fig.tight_layout()
            fig.savefig(out_dir / f"envelope_{names[i]}_{names[j]}.png",
                        dpi=160, bbox_inches="tight")
            plt.close(fig)

        if num == 3:
            triple = "+".join(names)
            fig = plt.figure(figsize=(7, 6))
            axis = fig.add_subplot(111, projection="3d")
            for controller in present:
                points, _ = controller_data[controller]
                stable, _ = _stable_points_3d(
                    [r for r in points if r["combination"] == triple], names
                )
                if len(stable):
                    axis.scatter(stable[:, 0], stable[:, 1], stable[:, 2], s=10,
                                 color=CONTROLLER_COLORS[controller], alpha=0.5,
                                 label=controller)
            axis.set_xlabel(f"{names[0]} [{_units_for(names[0])}]")
            axis.set_ylabel(f"{names[1]} [{_units_for(names[1])}]")
            axis.set_zlabel(f"{names[2]} [{_units_for(names[2])}]")
            axis.set_title(f"Stable {triple} commands per controller")
            axis.legend(loc="best")
            fig.tight_layout()
            fig.savefig(out_dir / "envelope_3d_scatter.png", dpi=160,
                        bbox_inches="tight")
            plt.close(fig)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            for axis, (i, j) in zip(axes, itertools.combinations(range(3), 2)):
                for controller in present:
                    points, _ = controller_data[controller]
                    stable, _ = _stable_points_3d(
                        [r for r in points if r["combination"] == triple], names
                    )
                    projected = stable[:, [i, j]] if len(stable) else np.empty((0, 2))
                    _draw_projection(axis, projected, CONTROLLER_COLORS[controller],
                                     controller, scatter=False)
                _axis_labels(axis, names[i], names[j])
                axis.set_title(f"{names[i]} vs {names[j]}")
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center", ncol=len(present))
            fig.suptitle(f"{triple} projections per controller")
            fig.tight_layout(rect=(0, 0.07, 1, 0.96))
            fig.savefig(out_dir / "envelope_3d_projections.png", dpi=160,
                        bbox_inches="tight")
            plt.close(fig)

    # Grouped comparison histogram.
    labels_ref, _ = _histogram_series(names, next(iter(controller_data.values()))[1])
    width = 0.8 / max(1, len(present))
    fig, axis = plt.subplots(figsize=(max(8, 0.7 * len(labels_ref) + 3), 4.8))
    for offset, controller in enumerate(present):
        _, summary = controller_data[controller]
        _, values = _histogram_series(names, summary)
        positions = np.arange(len(labels_ref)) + (offset - (len(present) - 1) / 2) * width
        axis.bar(positions, values, width=width, color=CONTROLLER_COLORS[controller],
                 alpha=0.85, label=controller)
    axis.set_xticks(range(len(labels_ref)))
    axis.set_xticklabels(labels_ref, rotation=45, ha="right")
    axis.set_ylabel("max reachable (signed value / ||command||)")
    axis.set_title("Max velocities per command combination - controller comparison")
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "histogram_max_velocities.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  Orchestration
# ─────────────────────────────────────────────────────────────────────────────
def _load_controller_data(mv_dir: Path, no_e2e: bool):
    """Read the per-controller CSVs back for plotting. Returns
    (controller_data, names)."""
    controller_data: dict[str, tuple[list, list]] = {}
    names: list[str] | None = None
    for controller in CONTROLLER_ORDER:
        if no_e2e and controller == "end_to_end":
            continue
        folder = mv_dir / f"{controller}_velocities"
        points_path = folder / f"{controller}_points.csv"
        summary_path = folder / f"{controller}_summary.csv"
        if not (points_path.exists() and summary_path.exists()):
            continue
        points = _read_csv_dicts(points_path)
        summary = _read_csv_dicts(summary_path)
        controller_data[controller] = (points, summary)
        if names is None:
            with points_path.open() as handle:
                header = next(csv.reader(handle))
            names = _names_from_header(header)
    if names is None:
        raise FileNotFoundError(f"No controller CSVs found under '{mv_dir}'.")
    return controller_data, names


def _auto_find_max_velocities(ckpt_dir: str) -> Path:
    candidates = list(Path(".").glob("**/max_velocities"))
    candidates += list(Path(ckpt_dir).glob("**/max_velocities"))
    candidates = [c for c in candidates if c.is_dir()]
    if not candidates:
        raise FileNotFoundError("No max_velocities directory found for --plot-only.")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def regenerate_plots(mv_dir: Path, no_e2e: bool) -> None:
    controller_data, names = _load_controller_data(mv_dir, no_e2e)
    for controller, (points, summary) in controller_data.items():
        folder = mv_dir / f"{controller}_velocities"
        plot_controller_envelopes(folder, controller, names, points, summary)
    plot_comparison(mv_dir, names, controller_data)
    print(f"Plots regenerated from CSVs in: {mv_dir}")


def main() -> None:
    args = parse_args()

    if args.plot_only is not None:
        mv_dir = (
            _auto_find_max_velocities(args.ckpt_dir)
            if args.plot_only == "__auto__"
            else Path(args.plot_only)
        )
        if mv_dir.name != "max_velocities" and (mv_dir / "max_velocities").is_dir():
            mv_dir = mv_dir / "max_velocities"
        print(f"[plot-only] using: {mv_dir}")
        regenerate_plots(mv_dir, args.no_e2e)
        return

    controllers, names, _units, comparison_dir = build_controllers(args)
    mv_dir = comparison_dir / "max_velocities"

    try:
        import compare
        compare.copy_joystick_source(controllers[0]["env"], comparison_dir)
    except Exception as error:  # non-fatal: the snapshot is a convenience only.
        print(f"[WARN] could not copy joystick source: {error}")

    controller_data: dict[str, tuple[list, list]] = {}
    global_summary: list[list] = []
    for controller in controllers:
        controller_name = controller["name"]
        folder = mv_dir / f"{controller_name}_velocities"
        folder.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Searching envelope: {controller_name} ===")
        points, summary = run_search(controller, names, args)
        _write_csv(folder / f"{controller_name}_points.csv",
                   _points_header(names), points)
        _write_csv(folder / f"{controller_name}_summary.csv",
                   _summary_header(names), summary)
        controller_data[controller_name] = (
            _read_csv_dicts(folder / f"{controller_name}_points.csv"),
            _read_csv_dicts(folder / f"{controller_name}_summary.csv"),
        )
        global_summary.extend(summary)

    (mv_dir / "comparison").mkdir(parents=True, exist_ok=True)
    _write_csv(mv_dir / "comparison" / "comparison_summary.csv",
               _summary_header(names), global_summary)

    for controller_name, (points, summary) in controller_data.items():
        folder = mv_dir / f"{controller_name}_velocities"
        plot_controller_envelopes(folder, controller_name, names, points, summary)
    plot_comparison(mv_dir, names, controller_data)

    print(f"\nEnvelope search completed: {mv_dir}")


if __name__ == "__main__":
    main()
