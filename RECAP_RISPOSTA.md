# Recap dettagliato delle differenze — fix comando simultaneo vx + omega (DFCIP MPC/WBC)

Documento hunk per hunk: per ogni modifica c'è il **codice vecchio**, il **codice nuovo**, poi
**cosa fa** e **cosa risolve**. Ordinato per file.

Diff completo: `mpx/COMBINED_COMMAND_FIX.diff` · Report: `mpx/COMBINED_COMMAND_FIX.md` ·
Validazione: `mpx/mpx/examples/validate_dfcip_controller.py` → `validation_results.json`.

> Nota di attribuzione: `mu = 0.9`, la riga `Fz >= 5 N` nel cono d'attrito del QP, il wiring di
> `h_fz` in `objectives.py` e i commenti su `w_pcomxy/w_v` erano già presenti nel working tree da
> una passata precedente (vedi `FINAL_REPORT.md`). Sono elencati a fine documento per completezza;
> il resto sono le modifiche di questa sessione, che risolvono il comando combinato.

---

# 1. `mpx/mpx/utils/timing.py` — FILE NUOVO

Nessun file esisteva prima. Aggiunto un modulo unico per derivare la timing dai valori primitivi e
verificarla. Contenuto essenziale:

```python
def derive_timing(config) -> dict:
    sim_f = int(config.simulation_frequency)
    mpc_f = int(config.mpc_frequency)
    wbc_f = int(config.whole_body_frequency)
    dt_mpc = float(config.dt_mpc); N = int(config.N)
    horizon_required = float(getattr(config, "mpc_horizon_s", 0.5))

    if sim_f % mpc_f != 0:
        raise ValueError(f"simulation_frequency ({sim_f}) must be a multiple of mpc_frequency ({mpc_f})")
    if sim_f % wbc_f != 0:
        raise ValueError(f"simulation_frequency ({sim_f}) must be a multiple of whole_body_frequency ({wbc_f})")
    horizon_s = N * dt_mpc
    if abs(horizon_s - horizon_required) > 1e-9:
        raise ValueError(f"N*dt_mpc = {horizon_s} differs from the required {horizon_required} s")
    shift_exact = (1.0 / mpc_f) / dt_mpc
    shift = int(round(shift_exact))
    if shift < 1 or abs(shift - shift_exact) > 1e-9:
        raise ValueError("MPC update period is not an integer multiple of dt_mpc (>=1 node)")
    return dict(sim_f=sim_f, dt_sim=1.0/sim_f, mpc_f=mpc_f, wbc_f=wbc_f, dt_wbc=1.0/wbc_f,
                dt_mpc=dt_mpc, N=N, horizon_s=horizon_s,
                mpc_period_sim_steps=sim_f//mpc_f, wbc_period_sim_steps=sim_f//wbc_f,
                mpc_shift_nodes=shift)

def check_model_timestep(model_timestep, timing, tol=1e-12):
    if abs(float(model_timestep) - timing["dt_sim"]) > tol:
        raise ValueError("model.opt.timestep != 1/simulation_frequency")
```

**Cosa fa.** Calcola `dt_sim`, `dt_wbc`, i passi di simulazione tra due update MPC/WBC (`5` e `5`) e
lo shift del warm start in **nodi MPC** (`1`). Solleva eccezione se le frequenze non sono multipli
interi, se `N·dt_mpc ≠ 0.5`, se lo shift non è intero o è `< 1`, o se il timestep del modello non
è `1/simulation_frequency`.

**Cosa risolve.** Prima le frequenze/timestep erano sparse e ricalcolate a mano in ogni file, con
il rischio di arrotondamenti silenziosi (es. `int(1/(dt_mpc·mpc_frequency))`). Ora c'è un'unica
fonte verificata; qualsiasi configurazione temporale incoerente fallisce subito all'import invece
di sbagliare in silenzio.

---

# 2. `mpx/mpx/config/config_dfcip.py`

### 2.1 Blocco timing

**Vecchio**
```python
# Time and stage parameters
#dt = 0.002  # Time step in seconds
dt_mpc = 0.002
N = 250        # Number of stages
T_TRAJECTORY = 60
mpc_frequency = 100  # Frequency of MPC updates in Hz
grav = 9.81
whole_body_frequency = 500
dt_ref = 1.0 / whole_body_frequency
```

**Nuovo**
```python
simulation_frequency = 500   # Hz  -> dt_sim = 0.002 s (must equal the XML timestep)
whole_body_frequency = 100   # Hz  -> dt_wbc = 0.01 s (WBC update period)
mpc_frequency = 100          # Hz  -> MPC update period 0.01 s
dt_mpc = 0.01                # s   MPC prediction step
N = 50                       # MPC stages -> horizon N * dt_mpc = 0.5 s
mpc_horizon_s = 0.5          # s   required prediction horizon (checked)
mpc_iterations = 1
wbc_lookahead_dt = 0.0
T_TRAJECTORY = 60
grav = 9.81
```

**Cosa fa.** Introduce `simulation_frequency` come variabile indipendente (500 Hz), separata da
`whole_body_frequency` (ora 100 Hz). Porta `dt_mpc` a 0.01 e `N` a 50 (orizzonte 0.5 s). Aggiunge
due parametri: `mpc_iterations` (iterazioni FDDP per update, default 1) e `wbc_lookahead_dt` (usato
al punto 3.6 e 2.x). Rimuove `dt_ref` (non serve più: niente reference densa da decimare).

**Cosa risolve.** Prima `whole_body_frequency` faceva sia da frequenza WBC sia da frequenza di
simulazione (MuJoCo integrava a 500 Hz solo perché coincidevano). Ora simulazione 500 Hz e
MPC/WBC 100 Hz sono concettualmente distinti; tra due update il comando è mantenuto per 5 step.

### 2.2 In coda al file: timing derivata e controllata all'import

**Nuovo** (non c'era)
```python
import types as _types
from mpx.utils.timing import derive_timing as _derive_timing
_timing = _derive_timing(_types.SimpleNamespace(
    simulation_frequency=simulation_frequency, mpc_frequency=mpc_frequency,
    whole_body_frequency=whole_body_frequency, dt_mpc=dt_mpc, N=N,
    mpc_horizon_s=mpc_horizon_s))
dt_sim = _timing["dt_sim"]                              # 0.002 s
dt_wbc = _timing["dt_wbc"]                              # 0.01 s
mpc_period_sim_steps = _timing["mpc_period_sim_steps"]  # 5
wbc_period_sim_steps = _timing["wbc_period_sim_steps"]  # 5
mpc_shift_nodes = _timing["mpc_shift_nodes"]            # 1 MPC node per update
```

**Cosa fa.** Espone `dt_sim`, `dt_wbc`, `mpc_period_sim_steps`, `wbc_period_sim_steps`,
`mpc_shift_nodes` come attributi del config, calcolati e validati.
**Cosa risolve.** Rende disponibili i valori derivati a tutti i consumatori senza ricalcolarli, e
impone il check `dt_mpc·N == 0.5` e `shift == 1` a ogni import.

### 2.3 `w_eq` (CAUSA PRINCIPALE)

**Vecchio**
```python
w_eq     = 1e8    # momento + contatto
```
**Nuovo**
```python
w_eq     = 1e6    # (con lungo commento: 1e8 rende FDDP non convergente sotto rotazione del riferimento)
```

**Cosa fa.** Abbassa di due ordini di grandezza il peso condiviso dei vincoli soft (bilancio dei
momenti, contatto piatto, `Fz≥0`, stabilità terminale).
**Cosa risolve.** Con `1e8` la curvatura Gauss-Newton del residuo di momento (bilineare, ~1e11
contro pesi 5-300) è così mal condizionata che l'unica iterazione FDDP viene rifiutata dalla line
search appena il riferimento ruota (v+ω insieme): il piano non converge → QP WBC NaN → caduta. Con
`1e6` converge (0 step rifiutati), violazione 0.011 N·m. Misure: 1e8 → cadute; 1e6/1e5 → tracking
pulito.

### 2.4 `w_base`

**Vecchio**
```python
w_base      = 1e-2
```
**Nuovo**
```python
w_base      = 1e0
```

**Cosa fa.** Aumenta il peso del task di orientamento base (roll/pitch a zero, yaw all'asse ruote)
nel QP WBC.
**Cosa risolve.** Con `1e-2` la base rolla ~15 mrad in curva sotto la forza laterale: il CoM finisce
5 mm dentro il base point e, via il vincolo terminale `pcom(N)=c(N)`, l'MPC piega il percorso
(ω +5%). Con `1.0` il roll resta ≤ 2 mrad.

### 2.5 `w_posture` + `posture_joint_ids` (nuovo)

**Vecchio**
```python
w_posture = 1e-1
```
**Nuovo**
```python
w_posture = 0.1
posture_joint_ids = (0, 4)   # joint_left_leg_1, joint_right_leg_1 (indici nei 8 giunti attuati)
```

**Cosa fa.** Mantiene il peso della postura ma la **restringe ai soli giunti di abduzione** (indici
0 e 4), tramite il nuovo parametro `posture_joint_ids` letto dal WBC (punto 3.5).
**Cosa risolve.** I giunti di abduzione sono la ridondanza cinematica della carreggiata e nessun
task li ancora: sotto carico laterale derivano (19 mrad a 1.0/0.8), spostando `c` di 3-5 mm e
piegando la base via vincolo terminale (ω +5%). Regolare **tutti** i giunti verso `q0` combatterebbe
il task di altezza CoM (pitch −0.02 rad); regolare solo l'abduzione ancora la carreggiata senza
quel conflitto.

---

# 3. `mpx/mpx/utils/mpc_wrapper_dfcip.py`

### 3.1 Import del modulo timing

**Nuovo**
```python
from mpx.utils.timing import derive_timing, check_model_timestep, describe_timing
```

### 3.2 `__init__`: timing derivata, shift, iterazioni, lookahead

**Vecchio**
```python
self.mpc_frequency = config.mpc_frequency
self.shift = int(1 / (config.dt_mpc * config.mpc_frequency))
print(f"MPC update every {self.shift} simulation steps (mpc_frequency={self.mpc_frequency} Hz, dt={config.dt_mpc} s)")
```
**Nuovo**
```python
self.timing = derive_timing(config)
check_model_timestep(model.opt.timestep, self.timing)
self.mpc_frequency = config.mpc_frequency
# Warm-start shift in MPC nodes per MPC update (one update period).
self.shift = self.timing["mpc_shift_nodes"]
self.mpc_iterations = int(getattr(config, "mpc_iterations", 1))
self.wbc_lookahead_dt = float(getattr(config, "wbc_lookahead_dt", 0.0))
print("[MPC/WBC timing] " + describe_timing(self.timing, self.mpc_iterations)
      + f" | WBC reference lookahead {self.wbc_lookahead_dt} s")
```

**Cosa fa.** Deriva la timing verificata; lo shift diventa `mpc_shift_nodes` (= 1 **nodo MPC**);
legge `mpc_iterations` e `wbc_lookahead_dt`.
**Cosa risolve.** Prima lo shift era `int(1/(dt_mpc·mpc_frequency))` con messaggio "simulation steps"
**sbagliato** (era in step di simulazione, non nodi). Con la vecchia config valeva 5 e coincideva
per caso; il messaggio ora è corretto e il valore è verificato.

### 3.3 `__init__`: loop di iterazioni FDDP

**Vecchio**
```python
work = partial(optimizers.fddp_mpc, self.cost, self.dynamics, self.hessian_approx, False)
```
**Nuovo**
```python
_fddp = partial(optimizers.fddp_mpc, self.cost, self.dynamics, self.hessian_approx, False)

def work(reference, parameter, W, x0, X_init, U_init):
    X_it, U_it, D_it = X_init, U_init, None
    for _ in range(self.mpc_iterations):
        X_it, U_it, D_it = _fddp(reference, parameter, W, x0, X_it, U_it)
    return X_it, U_it, D_it
```

**Cosa fa.** Permette di eseguire N iterazioni FDDP per update (default 1, invariato rispetto al
comportamento RTI).
**Cosa risolve.** Non serve al fix (con `w_eq=1e6` basta 1 iterazione), ma è la leva usata per
**dimostrare** che il problema era il condizionamento e non il budget di iterazioni (a `w_eq=1e8`
nemmeno 3 iterazioni salvavano le curve veloci).

### 3.4 `__init__`: reference alla discretizzazione MPC

**Vecchio**
```python
self.ref_substeps = int(round(config.dt_mpc / config.dt_ref))
self.N_dense = config.N * self.ref_substeps
reference_generator = partial(mpc_utils.reference_generator_dfcip_online, self.N_dense, config.dt_ref, config.mass, config.grav)
```
**Nuovo**
```python
reference_generator = partial(mpc_utils.reference_generator_dfcip_online,
                              config.N, config.dt_mpc, config.mass, config.grav)
```

E nella `run`:

**Vecchio**
```python
reference_full = jnp.concatenate([x_ref, u_ref], axis=-1)
reference = reference_full[:, ::self.ref_substeps, :]
```
**Nuovo**
```python
reference = jnp.concatenate([x_ref, u_ref], axis=-1)   # (n_env, N+1, nx+nu)
```

**Cosa fa.** Genera la reference direttamente con `(N, dt_mpc)`: N+1 nodi, niente generazione densa
e successiva decimazione.
**Cosa risolve.** Il generatore online produce esattamente la discretizzazione richiesta dall'MPC
(come da requisito). Con la vecchia config `dt_ref == dt_mpc` e la decimazione era un no-op degenere;
ora il meccanismo è esplicito e coerente.

### 3.5 `__init__`: costruzione della maschera di postura

**Nuovo** (non c'era)
```python
_pj = getattr(config, "posture_joint_ids", None)
if _pj is None:
    _posture_mask = None
else:
    _posture_mask = jnp.zeros(nj).at[jnp.array(list(_pj))].set(1.0)
self._posture_mask = _posture_mask
```
e la chiamata al QP passa `posture_mask=_posture_mask`.

**Cosa fa.** Traduce `posture_joint_ids` in una maschera 0/1 di lunghezza `nj` e la passa al WBC.
**Cosa risolve.** Abilita la regolazione di postura sui soli giunti scelti (punto 2.5). Con `None`
il comportamento è quello vecchio (tutti i giunti tranne le ruote).

### 3.6 `_build_desired_impl`: lookahead dei riferimenti WBC

**Vecchio**
```python
dt = 1.0 / self.config.whole_body_frequency
acc_com_ = (fcl + fcr) / self.config.mass + g_vec[None, :]
```
**Nuovo**
```python
dt = self.wbc_lookahead_dt
acc_com_ = (fcl + fcr) / self.config.mass + g_vec[None, :]
```

**Cosa fa.** Il `dt` con cui si costruiscono `pos_ref = p + dt·v` e `vel_ref = v + dt·a` diventa
`wbc_lookahead_dt` (= 0), invece del periodo WBC.
**Cosa risolve.** Con `dt ≠ 0` il termine PD di task diventa un feedback positivo `Kp·dt·v` il cui
guadagno scala con dt: a `dt = dt_wbc = 0.01` fa divergere lo yaw (ω misurata 5 rad/s per comando
0.8, QP NaN). Con `dt = 0` i termini PD di CoM/ruote si annullano identicamente e il WBC insegue le
accelerazioni MPC come puro feedforward di dinamica inversa (il feedback lo chiude l'MPC a 100 Hz).

### 3.7 Diagnostica MPC e WBC (`run_diag`, `whole_body_run_diag`, `mpc_diag`)

**Nuovo** (non c'era) — funzioni parallele a `run`/`whole_body_run` che oltre al risultato
restituiscono un dizionario di diagnostica: costo prima/dopo il passo FDDP, norma dei difetti di
shooting, se il passo è stato accettato, residuo di momento del piano, lean del CoM, e per il QP i
residui dei task, i margini d'attrito, lo stato di convergenza. Le firme pubbliche di `run` e
`whole_body_run` **restano invariate**.

**Cosa fa.** Espone, senza toccare la legge di controllo, tutte le grandezze interne necessarie a
localizzare dove il comando combinato veniva perso.
**Cosa risolve.** È lo strumento con cui è stata dimostrata la catena causale (line search che
rifiuta → difetti che crescono → QP NaN) e con cui si sono ordinate le cause. Consumata solo dal
test headless.

---

# 4. `mpx/mpx/utils/mpc_utils.py`

### 4.1 `reference_generator_dfcip_online`: integrazione coerente col modello MPC

**Vecchio**
```python
# C++-style unicycle Euler integration
# vx/vy are velocities, not displacements
vx_next = v_next * jnp.cos(theta)      # theta del nodo CORRENTE
vy_next = v_next * jnp.sin(theta)

p_xy_next = p_xy + jnp.array([
    vx_next * dt,
    vy_next * dt,
])

theta_next = theta + omega_next * dt
```
**Nuovo**
```python
p_xy_next = p_xy + jnp.array([
    v * jnp.cos(theta) * dt,
    v * jnp.sin(theta) * dt,
])

theta_next = theta + omega * dt

vx_next = v_next * jnp.cos(theta_next)   # theta del nodo SUCCESSIVO
vy_next = v_next * jnp.sin(theta_next)
```

**Cosa fa.** Il base point avanza con velocità/heading del nodo corrente; la velocità di CoM di
riferimento al nodo k+1 è espressa lungo il heading del nodo k+1 (coerente con
`models.wheeled_dfcip_dynamics`).
**Cosa risolve.** Prima la velocità del CoM al nodo k+1 usava θ del nodo k: la direzione del CoM di
riferimento restava indietro di un nodo rispetto al heading. Via il vincolo terminale
`pcom(N)=c(N)` questo lag (`N·dt²·v·ω`, 4 mm a v=1, ω=0.8, dt=0.01) faceva barattare all'MPC il
tracking di ω per l'allineamento CoM/base (ω 0.75 per comando 0.8).

### 4.2 Docstring del command layout corretta

**Vecchio**
```
cmd[0] = desired forward velocity v_des
cmd[1] = desired yaw rate omega_des
cmd[2] = desired vertical CoM velocity vz_des, ignored here for flat walking
```
**Nuovo**
```
cmd[0] = desired forward velocity v_des [m/s]
cmd[1] = desired vertical CoM velocity vz_des [m/s], ignored for flat ground
cmd[2] = desired yaw rate omega_des [rad/s]
cmd[3] = desired CoM height [m], currently ignored (z_com_ref is fixed below)
```

**Cosa fa.** Corregge la documentazione: `omega` è `cmd[2]`, non `cmd[1]`.
**Cosa risolve.** Il codice leggeva già correttamente `omega_des = cmd[2]` e `v_des = cmd[0]`; era
la **docstring** a dichiarare indici sbagliati (`omega` in `cmd[1]`). Il layout effettivo
`[vx, vz, omega, height]` è ora documentato coerente con `mpc_wheeled_input` e con l'env RL. Nessun
bug di indicizzazione nel comando (verificato).

### 4.3 QP WBC: parametro `posture_mask` e selezione dei giunti

**Vecchio** (firma + selezione)
```python
def whole_body_interface_wheeled_legged_qp(
    ...
    qpos, qvel, desired
):
    ...
    # matrix_with_no_wheel: Identity(nj) with wheel dof zeroed out (joints 3 and 7 = wheels)
    matrix_with_no_wheel = jnp.eye(nj).at[3, 3].set(0.0).at[7, 7].set(0.0)
```
**Nuovo**
```python
def _whole_body_interface_wheeled_legged_qp_impl(
    ...
    qpos, qvel, desired,
    posture_mask=None,
):
    ...
    if posture_mask is None:
        matrix_with_no_wheel = jnp.eye(nj).at[3, 3].set(0.0).at[7, 7].set(0.0)
    else:
        matrix_with_no_wheel = jnp.diag(jnp.asarray(posture_mask, dtype=qpos.dtype))
```

Inoltre la funzione è stata rinominata in `_..._impl` e restituisce anche un dizionario di
diagnostica; sono stati aggiunti due wrapper pubblici:
```python
def whole_body_interface_wheeled_legged_qp(*args, **kwargs):
    tau, qddot, fl, fr, _ = _whole_body_interface_wheeled_legged_qp_impl(*args, **kwargs)
    return tau, qddot, fl, fr           # firma pubblica invariata

def whole_body_interface_wheeled_legged_qp_diag(*args, **kwargs):
    return _whole_body_interface_wheeled_legged_qp_impl(*args, **kwargs)   # + diag
```

**Cosa fa.** La matrice di selezione del task di postura ora può essere una maschera qualsiasi
(`posture_mask`), invece dell'identità con le sole ruote azzerate. Il wrapper pubblico mantiene la
firma e il valore di ritorno `(tau, qddot, fl, fr)` inalterati per chi già la chiama (env RL
inclusa).
**Cosa risolve.** È il meccanismo che realizza la regolazione di postura sui soli giunti di
abduzione (punto 2.5), senza rompere l'interfaccia esistente. Il diagnostico separato dà i residui
dei task usati nell'analisi.

---

# 5. `mpx/mpx/examples/mjx_tita.py`

### 5.1 Precedenza di import

**Vecchio**
```python
sys.path.append(os.path.abspath(os.path.join(dir_path, "..")))
...
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, "..")))
```
**Nuovo**
```python
# Insert at the front so that THIS working tree wins over any editable/site-packages install of mpx.
sys.path.insert(0, os.path.abspath(os.path.join(dir_path, "..", "..")))
```

**Cosa fa.** Mette la root del repo in testa a `sys.path`.
**Cosa risolve.** Nell'env `mjpl` `mpx` è installato editable da un altro checkout; con
`sys.path.append` la `.pth` vinceva e girava l'altra copia. Ora gira questo codice.

### 5.2 Timing e controllo del timestep

**Vecchio**
```python
sim_frequency = float(config.whole_body_frequency)
model.opt.timestep = 1.0 / sim_frequency
```
**Nuovo**
```python
timing = derive_timing(config)
check_model_timestep(model.opt.timestep, timing)
sim_frequency = float(timing["sim_f"])
model.opt.timestep = timing["dt_sim"]
print("[timing] " + describe_timing(timing, int(getattr(config, "mpc_iterations", 1))))
```

**Cosa fa.** La frequenza di simulazione viene da `simulation_frequency` (500 Hz), non più da
`whole_body_frequency`; verifica che il timestep XML coincida.
**Cosa risolve.** MuJoCo continua a integrare a 500 Hz mentre MPC/WBC vanno a 100 Hz; il legame
implicito precedente è rotto in modo controllato.

### 5.3 Aggiornamenti MPC e WBC separati

**Vecchio**
```python
period = int(sim_frequency / config.mpc_frequency)
```
**Nuovo**
```python
period = timing["mpc_period_sim_steps"]        # simulation steps between MPC updates (5)
wbc_period = timing["wbc_period_sim_steps"]    # simulation steps between WBC updates (5)
```
e il blocco WBC è ora dentro `if counter % wbc_period == 0:` (prima era nello stesso blocco
dell'MPC).

**Cosa fa.** Rende i due periodi concettualmente distinti anche se qui valgono entrambi 5.
**Cosa risolve.** Se in futuro le due frequenze divergono, MPC e WBC si aggiornano ciascuno alla
propria cadenza senza modifiche strutturali.

### 5.4 Lettura di stato coerente: `mj_step1` / `mj_step2`

**Vecchio**
```python
init_start = timer()
qpos = data.qpos.copy()
qvel = data.qvel.copy()
...
mujoco.mj_step(model, data)
```
**Nuovo**
```python
init_start = timer()
mujoco.mj_step1(model, data)   # forward: grandezze derivate coerenti con qpos/qvel dati al WBC
qpos = data.qpos.copy()
qvel = data.qvel.copy()
...
mujoco.mj_step2(model, data)   # integra (mj_step1 chiamato in testa allo step)
```

**Cosa fa.** Fa un `mj_step1` (forward) prima di leggere lo stato, e integra con `mj_step2` dopo
aver scritto `ctrl`.
**Cosa risolve.** Dopo `mj_step` le grandezze derivate (`geom_xpos`, `subtree_com`,
`mj_objectVelocity`) descrivono lo stato pre-integrazione mentre `qpos/qvel` sono già integrati:
`x0` è di un passo indietro rispetto all'input del WBC (2 mm e 2 mm/s a 1 m/s), che con Kp 50 vale
0.1 m/s² di accelerazione di task spuria. Ora lo stato letto coincide con quello passato al WBC.

### 5.5 PD esterno con `dt_sim` esplicito

**Vecchio**
```python
dt = model.opt.timestep
...
dq_desired = qvel_joint + qddot_joint * dt
q_desired  = qpos_joint + qvel_joint * dt + 0.5 * qddot_joint * dt**2
```
**Nuovo**
```python
dt_sim = model.opt.timestep
...
dq_desired = qvel_joint + qddot_joint * dt_sim
q_desired  = qpos_joint + qvel_joint * dt_sim + 0.5 * qddot_joint * dt_sim**2
```

**Cosa fa.** Rinomina esplicitamente il passo del PD esterno in `dt_sim`.
**Cosa risolve.** Chiarisce che il PD esterno predice **un passo di simulazione** (0.002 s), non il
passo WBC (0.01 s): elimina l'ambiguità richiesta dai vincoli sul timing.

### 5.6 Crash pre-esistente nel ramo headless

**Vecchio**
```python
for _ in range(steps):
    cmd = command_handle.get_command()      # metodo inesistente -> AttributeError
    mpc_state, tau, qddot, ... = step_controller(...)
```
**Nuovo**
```python
for _ in range(steps):
    mpc_state, tau, qddot, ... = step_controller(...)
```

**Cosa fa.** Rimuove la chiamata a `get_command()` che non esiste su `KeyboardVelocityCommand`.
**Cosa risolve.** Il ramo headless andava in `AttributeError` all'inizio del loop; ora
`mjx_tita.py --headless` gira (verificato 1500 step).

---

# 6. `mpx/mpx/examples/tita.py` (driver legacy, modifiche minime)

**Vecchio → Nuovo**
```python
MAX_STEPS = round(int((2 + config.T_TRAJECTORY) / config.dt_ref))     # -> config.dt_sim
sim_frequency = float(config.whole_body_frequency)                    # -> config.simulation_frequency
"dt_mpc", "dt_ref", "N", ... "whole_body_frequency",                  # -> "dt_sim", "simulation_frequency", "whole_body_frequency"
```

**Cosa fa.** Aggiorna i riferimenti ai nomi rimossi/rinominati (`dt_ref` → `dt_sim`, frequenza di
simulazione da `simulation_frequency`).
**Cosa risolve.** Evita `AttributeError` su `config.dt_ref` ora che non esiste più, mantenendo il
driver legacy importabile.

---

# 7. `mpx/mpx/examples/validate_dfcip_controller.py` (test headless, riscritto)

Non è codice di controllo, ma è lo strumento con cui tutto sopra è stato misurato. Riscritto per:
guidare la stessa legge di controllo di `mjx_tita.py` con timing corretta; `--set chiave=valore` per
override di qualunque parametro (ablazioni una causa alla volta); `--stale-state` /
`--no-outer-pd` / `--outer-pd-mode` / `--wbc-every` per isolare gli effetti; `--save-logs` per
salvare le serie temporali. Registra per ogni step: target, comando filtrato, reference `v/ω/vcom`,
stato misurato, `a/α/Fl/Fr`, riferimenti delle ruote, accelerazioni richieste al WBC, residui e
stato del QP, margini d'attrito, `qddot`, coppie, altezza/roll/pitch, NaN. I sei casi obbligatori
sono nella lista `CASES`.

---

# Tabella riassuntiva

| Problema | Causato da | Risolto con |
|---|---|---|
| Le modifiche non avevano effetto | `mpx`/`mujoco_playground` editable da un altro checkout; `sys.path.append` perde contro la `.pth` | `sys.path.insert(0, <repo>)` in `mjx_tita.py` e nel test (5.1) |
| Caduta + NaN col comando combinato v+ω | `w_eq = 1e8`: FDDP a 1 iterazione non converge quando il riferimento ruota | `w_eq = 1e6` (2.3) |
| Yaw instabile / robot che gira su sé stesso | Lookahead riferimenti WBC = feedback positivo `Kp·dt·v` | `wbc_lookahead_dt = 0.0` (2.1, 3.6) |
| ω sotto-tracciata (0.75 per 0.8) | Reference: velocità CoM al nodo k+1 con θ del nodo k → lag di un nodo, penalizzato dal vincolo terminale | Integrazione reference coerente col modello MPC (4.1) |
| Errori di task spuri (~0.1 m/s²) | Stato letto dopo `mj_step`: derivate un passo più vecchie di qpos/qvel | Split `mj_step1` / lettura / `mj_step2` (5.4) |
| ω +5% residuo, base che rolla in curva | Giunti di abduzione non ancorati + `w_base` troppo basso → CoM lean | `w_base = 1.0` + postura solo su abduzione (2.4, 2.5, 3.5, 4.3) |
| Timing accoppiato, shift mal descritto, reference densa | `whole_body_frequency` usata anche come freq. sim.; shift in "step" invece di nodi; densa+decimazione | `simulation_frequency` separata; `timing.py` con controlli; shift = 1 nodo; reference alla discretizzazione MPC (1, 2.1, 2.2, 3.2, 3.4, 5.2, 5.3) |
| Docstring del comando con indici errati | `omega` documentata in `cmd[1]` invece di `cmd[2]` | Docstring corretta (4.2) — nessun bug nel codice |
| `mjx_tita --headless` crashava | Chiamata a `command_handle.get_command()` inesistente | Rimossa (5.6) |
| Impossibile isolare le cause / validare | Nessun test non interattivo con diagnostica | Harness riscritto + `run_diag`/`whole_body_run_diag`/QP diag (3.7, 4.3, 7) |

## Nota: modifiche già presenti nel tree da una passata precedente (non di questa sessione)

| Modifica | File | Effetto |
|---|---|---|
| `mu = 0.6 → 0.9` | `config_dfcip.py` | budget d'attrito allineato al runtime C++ |
| riga `Fz ≥ 5 N` nel cono d'attrito (pyramid 4→5 righe) | `mpc_utils.py` | evita l'infeasibility del QP con `Fz` piccola/negativa |
| wiring di `h_fz` nello stage cost e nella Hessiana GN | `objectives.py` | penalizza `Fz < 0` nel piano MPC |

Queste sono documentate in `FINAL_REPORT.md`; le riporto perché compaiono nel diff (HEAD è stato
spostato indietro di un commit), ma non fanno parte del fix del comando combinato di questa sessione.
