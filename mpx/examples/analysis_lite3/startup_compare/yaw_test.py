"""Test: con l'x0-seeding ATTIVO, la caduta dipende dallo yaw di reset?

Se cade solo con |yaw| grande -> il solver e' sensibile al warm-start a yaw!=0
(la mia ipotesi). Se cade per ogni yaw -> il seeding e' rotto a prescindere.
Sweep di seed, logga yaw di reset e roll_max[0,1.2] (fall se >0.4).
"""
import os, sys, math
import jax, jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import train_srbd
from compare import set_fixed_command

ENV_NAME = "Lite3JoystickFlatTerrain"
CMD = jnp.array([1.0, 0.4, 0.0], dtype=jnp.float32)
seeds = [int(s) for s in sys.argv[1:]] or [0, 1, 2, 3, 4, 5, 6, 7]

_, env, _ = train_srbd.make_envs(env_name=ENV_NAME)
env._config.randomize_reset = 1.0
reset = jax.jit(jax.vmap(env.reset))
step = jax.jit(jax.vmap(env.step))
get_obs = jax.jit(jax.vmap(env._get_obs))


def yaw_of(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def roll_of(q):
    w, x, y, z = q
    return math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))


n = int(round(1.2 / float(env.dt)))
print(f"{'seed':>5} {'reset_yaw(rad)':>14} {'roll_max[0,1.2]':>16} {'fall?':>6}")
for sd in seeds:
    rng = jax.random.split(jax.random.PRNGKey(sd), 1)
    state = reset(rng)
    # Match compare.py: it zeroes the command at reset (compare.py ~2357) so the
    # robot starts from rest and ramps to the test command -- NOT the random reset
    # command. Without this the probe injects a random startup command (artifact).
    state = state.replace(info={**state.info,
                                "command": jnp.zeros_like(state.info["command"])})
    ry = yaw_of(np.asarray(state.data.qpos[0])[3:7])
    zero_action = jnp.zeros((1, env.action_size))
    rmax = 0.0
    for _ in range(n):
        state = set_fixed_command(state, CMD, baseline=True)
        state = state.replace(obs=get_obs(state.data, state.info, state.info["last_act"]))
        state = step(state, zero_action)
        state = set_fixed_command(state, CMD, baseline=True)
        rmax = max(rmax, abs(roll_of(np.asarray(state.data.qpos[0])[3:7])))
    print(f"{sd:>5} {ry:>14.3f} {rmax:>16.3f} {'FALL' if rmax > 0.4 else 'ok':>6}")
