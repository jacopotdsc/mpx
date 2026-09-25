"""Sonda d'avvio sull'ENV vero (joystick Lite3), non sullo standalone CPU.

Motivo: lo standalone lite3_srbd NON riproduce l'avvio di compare.py (conclusioni
ribaltate). Questa sonda usa lo stesso env di compare.py (train_srbd.make_envs) e
lo stesso baseline (azione residua = 0), quindi il transitorio d'avvio è
affidabile. Sweep di Qomega_z a parità di reset per trovare il compromesso
wz-vs-inciampo.

Uso:
    python env_probe.py <qz1> <qz2> ...      # es. 10 100 200
Stampa roll_max e vy nei primi ~1.2 s del comando [1.0, 0.4, 0] (il caso peggiore).
"""
import os, sys, math
import jax, jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import mpx.config.config_lite3 as config
import train_srbd
from compare import set_fixed_command

ENV_NAME = "Lite3JoystickFlatTerrain"
CMD = jnp.array([1.0, 0.4, 0.0], dtype=jnp.float32)
STEPS = 240        # 240 * ctrl_dt(0.02) = 4.8 s ... ctrl_dt is 0.02 -> use 60 for ~1.2s
SEED = 42

qz_list = [float(a) for a in sys.argv[1:]] or [10.0, 100.0, 200.0]


def roll_of(qpos):
    w, x, y, z = qpos[3], qpos[4], qpos[5], qpos[6]
    return math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))


def probe(qz):
    config.Qomega = jnp.diag(jnp.array([1.0, 1.0, qz / 10.0])) * 1e1
    config.W = jax.scipy.linalg.block_diag(
        config.Qp, config.Qrot, config.Qdp, config.Qomega, config.Qgrf
    )
    _, env, _ = train_srbd.make_envs(env_name=ENV_NAME)
    env._config.randomize_reset = 1.0

    reset = jax.jit(jax.vmap(env.reset))
    step = jax.jit(jax.vmap(env.step))
    get_obs = jax.jit(jax.vmap(env._get_obs))
    local_vel = jax.jit(jax.vmap(env.get_local_linvel))

    rng = jax.random.split(jax.random.PRNGKey(SEED), 1)
    state = reset(rng)
    cmd = CMD[None, :]
    zero_action = jnp.zeros((1, env.action_size))

    n = int(round(1.2 / float(env.dt)))  # ~1.2 s
    rolls, vys, ts = [], [], []
    for i in range(n):
        state = set_fixed_command(state, CMD, baseline=True)
        state = state.replace(obs=get_obs(state.data, state.info, state.info["last_act"]))
        state = step(state, zero_action)
        state = set_fixed_command(state, CMD, baseline=True)
        q = np.asarray(state.data.qpos[0])
        rolls.append(abs(roll_of(q)))
        vys.append(float(np.asarray(local_vel(state.data))[0, 1]))
        ts.append(i * float(env.dt))
    # finestra prima falcata [0,0.5s] (inciampo) vs tutto [0,1.2s]
    r05 = max(r for r, t in zip(rolls, ts) if t <= 0.5)
    v05 = max((v for v, t in zip(vys, ts) if t <= 0.5), key=abs)
    return r05, v05, max(rolls), float(env.dt)


print(f"{'Qomega_z':>9} {'roll[0,.5]':>11} {'vy[0,.5]':>9} {'roll[0,1.2]':>12}")
for qz in qz_list:
    r05, v05, rmax, dt = probe(qz)
    print(f"{qz:>9.0f} {r05:>11.3f} {v05:>9.3f} {rmax:>12.3f}")
