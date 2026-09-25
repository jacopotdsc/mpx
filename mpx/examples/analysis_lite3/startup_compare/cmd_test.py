"""Il comando CASUALE al reset (joystick.py:310, non azzerato da compare) e' la
causa dell'inciampo/caduta d'avvio?

Per ogni seed, due rollout baseline con lo STESSO reset:
  A) command parte dal valore casuale del reset (comportamento attuale di compare).
  B) command forzato al target dal primo step (info["command"] = CMD).
Se B non cade / roll piu' basso -> e' il comando casuale la causa.
"""
import os, sys, math
import jax, jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import train_srbd
from compare import set_fixed_command

ENV = "Lite3JoystickFlatTerrain"
CMD = jnp.array([1.0, 0.4, 0.0], dtype=jnp.float32)
seeds = [int(s) for s in sys.argv[1:]] or [0, 1, 2, 4, 5, 10]

_, env, _ = train_srbd.make_envs(env_name=ENV)
env._config.randomize_reset = 1.0
reset = jax.jit(jax.vmap(env.reset))
step = jax.jit(jax.vmap(env.step))
get_obs = jax.jit(jax.vmap(env._get_obs))
n = int(round(1.2 / float(env.dt)))
za = jnp.zeros((1, env.action_size))


def roll_of(q):
    w, x, y, z = q
    return math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))


def rollout(sd, force_cmd):
    rng = jax.random.split(jax.random.PRNGKey(sd), 1)
    state = reset(rng)
    reset_cmd = np.asarray(state.info["command"][0]).copy()
    if force_cmd:
        state = state.replace(info={**state.info, "command": CMD[None, :]})
    rmax = 0.0
    for _ in range(n):
        state = set_fixed_command(state, CMD, baseline=True)
        if force_cmd:  # keep command pinned to target too (not just target_command)
            state = state.replace(info={**state.info, "command": CMD[None, :]})
        state = state.replace(obs=get_obs(state.data, state.info, state.info["last_act"]))
        state = step(state, za)
        state = set_fixed_command(state, CMD, baseline=True)
        rmax = max(rmax, abs(roll_of(np.asarray(state.data.qpos[0])[3:7])))
    return reset_cmd, rmax


print(f"{'seed':>5} {'reset_cmd(vx,vy,wz)':>22} {'A rollmax':>10} {'B rollmax':>10}")
for sd in seeds:
    rc, ra = rollout(sd, False)
    _, rb = rollout(sd, True)
    fa = "FALL" if ra > 0.4 else "ok"
    fb = "FALL" if rb > 0.4 else "ok"
    print(f"{sd:>5} {str(np.round(rc,2)):>22} {ra:>7.3f} {fa:>4} {rb:>7.3f} {fb:>4}")
