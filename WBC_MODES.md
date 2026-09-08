# WBC del Tita — le tre modalità e come si runnano

Il wrapper DFCIP (`mpx/utils/mpc_wrapper_dfcip.py`) può chiudere il loop sul piano
MPC con tre controllori whole-body diversi, scelti da `config.wbc_type`. Tutti e
tre espongono le stesse due closure, quindi `whole_body_run` / `whole_body_run_diag`
e tutti i chiamanti (`tita.py`, `mjx_tita.py`, env RL) non sanno quale è attivo.

| `wbc_type` | implementazione | cosa risolve | esito |
|---|---|---|---|
| `"qp"` | `mpc_utils.whole_body_interface_wheeled_legged_qp` | gerarchia di task come QP con disequazioni (qpax) | ✅ baseline |
| `"model_based"` | `mpx/utils/wbc_model_based.py` | stessi task, una sola solve lineare (pseudo-inversa jacobiana) | ✅ **default**, pari al QP e 5× più veloce |
| `"wheeled"` | `mpc_utils.whole_body_interface_wheeled` | proiezione delle forze MPC sulla jacobiana di contatto (stile quadrupede) | ❌ cade (vedi §6) |

---

## 1. Ambiente

Env conda **`mjpl`** (jax 0.6.2, mujoco 3.8.0, qpax). `mpx_env` non ha `qpax`,
`mpx` non ha `jax`.

```bash
conda activate mjpl
cd ~/Desktop/repo_rl/tita_rl/test/mpx
```

Oppure senza attivare nulla:

```bash
/home/jacopo/miniconda3/envs/mjpl/bin/python <script>
```

`JAX_PLATFORMS=cpu` per forzare la CPU (test rapidi e riproducibili).

---

## 2. Run rapido — harness di validazione

Il modo più veloce per vedere un controllore in funzione: niente GUI, una riga di
riepilogo per comando.

```bash
cd mpx/examples

JAX_PLATFORMS=cpu python validate_dfcip_controller.py --wbc qp           --hold 4
JAX_PLATFORMS=cpu python validate_dfcip_controller.py --wbc model_based  --hold 4
JAX_PLATFORMS=cpu python validate_dfcip_controller.py --wbc wheeled      --hold 4
```

Output atteso:

```
[WBC] model-based whole-body controller (Jacobian inversion) | rolling constraint on | contact forces from dynamics | damping 1e-06
[  vx=1.0,omega=0.8] fell=False nan=False vx=+0.997/+1.0 wz=+0.805/+0.8 ... |tau|max=20.8 ... wall=5.7s
```

Flag utili:

| flag | effetto |
|---|---|
| `--wbc {qp,model_based,wheeled}` | sceglie il controllore (scorciatoia per `--set wbc_type=...`) |
| `--cases 0,3,4` | esegue solo alcuni comandi (default: tutti e 6) |
| `--hold 4` | secondi di mantenimento dopo la rampa (default 8) |
| `--ramp 1.0` | durata della rampa del comando |
| `--set KEY=VALUE` | sovrascrive qualsiasi attributo del config, ripetibile |
| `--save-logs out.npz` | salva le serie temporali complete |
| `--out results.json` | dove scrivere il riepilogo (usa un path assoluto) |
| `--verbose-every N` | una riga di diagnostica ogni N passi MPC |

I sei comandi di `CASES`: `vx=1.0/ω=0`, `vx=0/ω=0.8`, `0.3/0.2`, `0.6/0.4`,
`1.0/0.8`, `1.0/-0.8`.

---

## 3. Run in simulazione (viewer / rollout completo)

`tita.py` e `mjx_tita.py` leggono il config direttamente, quindi usano il default
del config (`model_based`). Per cambiarlo si imposta `wbc_type` (§4) e poi

```bash
cd mpx/examples
python tita.py     --steps 2000 --scene flat       # viewer + plot
python tita.py     --steps 2000 --headless
python mjx_tita.py --steps 2000 --scene flat       # versione più snella
```

Senza toccare il file di config:

```python
import mpx.config.config_dfcip as config
config.wbc_type = "model_based"
mpc = mpc_wrapper_dfcip.BatchedMPCControllerWrapper(config, n_env=1)
```

---

## 4. Config

In `mpx/config/config_dfcip.py`:

```python
wbc_type = "model_based"            # "qp" | "model_based" | "wheeled"  (default: model_based)

# usate solo da "model_based":
wbc_mb_enforce_rolling = True       # vincolo di rotolamento hard (KKT)
wbc_mb_force_source = "dynamics"    # "dynamics" | "mpc"
wbc_mb_damping = 1e-6               # smorzamento dell'Hessiana dei task
```

Sovrascrivibili da CLI:

```bash
python validate_dfcip_controller.py --wbc model_based \
    --set wbc_mb_force_source=mpc \
    --set wbc_mb_enforce_rolling=false \
    --set wbc_mb_damping=1e-4
```

---

## 5. `model_based` — inversione della jacobiana

`mpx/utils/wbc_model_based.py`. Gli stessi task del QP (CoM, ruota sx/dx,
orientamento base, postura, regolarizzazione `w_qddot`) impilati come minimi
quadrati pesati in `qddot`. La stazionarietà è un unico sistema lineare:

```
H qddot = g      H = w_qddot I + Σ wᵢ Jᵢ' Jᵢ     g = Σ wᵢ Jᵢ' (aᵢ_total − aᵢ_drift)
```

cioè una pseudo-inversa smorzata e pesata della jacobiana impilata: lo stesso
ottimo che darebbe il QP con nessun vincolo attivo (`H`, `g` sono letteralmente
`H_acc`, `−f_acc` del QP).

**Vincolo di rotolamento** (`wbc_mb_enforce_rolling=True`, default): i minimi
quadrati sono risolti *soggetti* al rotolamento delle ruote, tramite il KKT

```
[ H    Ac' ] [ qddot  ]   [ g  ]
[ Ac   -eI ] [ lambda ] = [ bc ]
```

una solve di dimensione `nv+6 = 20`. Con `False` si ottiene la pseudo-inversa
nuda `qddot = H⁻¹g`, che però può violare il rotolamento non-olonomo: solo per
ablazioni.

**Forze di contatto** (`wbc_mb_force_source`):

- `"dynamics"` (default) — ricavate dalle sei righe non attuate della dinamica
  dato `qddot`: `Mu qddot + cu = B [fl; fr]`. La mappa è quadrata → ancora una
  solve lineare. È l'uguaglianza che impone anche il QP.
- `"mpc"` — prese direttamente dall'MPC (`fcl`/`fcr`), la trasposizione letterale
  del quadrupede. **Fa cadere il robot** a `vx=1.0` e a `vx=0.6/ω=0.4`: `qddot` e
  le forze DFCIP insieme non soddisfano le righe del floating base
  (`dyn_res_norm ≈ 1.3 N` già da fermo), quindi la coppia non produce
  l'accelerazione per cui è stata calcolata.

Coppia finale, identica al QP:
`tau = Ma qddot + ca − Jla' T_l fl − Jra' T_r fr`.

**Cosa si perde**: le disequazioni non sono rappresentabili con una solve lineare.
Coni d'attrito, floor sulla forza normale e limiti di giunto sono **solo misurati**
e riportati nella diagnostica (`fric_margin_l/r`, `ineq_slack_min_joint`), non
imposti.

---

## 6. `wheeled` — la trasposizione del quadrupede

`mpc_utils.whole_body_interface_wheeled`, accanto a `whole_body_interface` (la
versione a 4 zampe). Stessa legge di controllo, due contatti invece di quattro:

```python
tau_mpc    = -(J @ grf)[6:]                                    # gambe in stance
tau_PD     = (J @ cartesian_space_action)[6:]                  # gambe in volo
tau_fb_lin = D[6:] + (M @ pinv(J.T) @ cartesian_space_action)[6:]
tau = tau_mpc*mask + (1-mask)*(tau_PD + tau_fb_lin)
```

La funzione è riga per riga identica a `whole_body_interface`; le uniche
differenze sono i due contatti invece di quattro e due correzioni necessarie:

1. `J = concatenate([J_F, J_R])` → `J_L` (`J_F` non era definito).
2. La maschera di contatto. Nella versione quadrupede è scritta per coordinata di
   task perché lì `n_joints = 3*n_contact = 12` e le due cose coincidono per caso.
   Qui no (`n_joints = 8`, `3*n_contact = 6`), quindi è scritta con le stesse
   parentesi esplicite ma su 4 giunti per gamba: altrimenti il prodotto con `tau`
   non torna nemmeno di dimensione.

```bash
diff <(sed -n '/^def whole_body_interface(model/,/^    return tau , J$/p' mpx/utils/mpc_utils.py) \
     <(sed -n '/^def whole_body_interface_wheeled(model/,/^    return tau , J$/p' mpx/utils/mpc_utils.py)
```

### Argomenti passati dal wrapper

Gli stessi che `mpc_wrapper_srbd` passa a `whole_body_interface`, letti dal vettore
`desired` comune a tutte le modalità (così il costruttore dei riferimenti resta
uno solo):

| argomento | valore |
|---|---|
| `model`, `mjx_model`, `contact_id`, `body_id` | come per il QP |
| `sim_frequency` | `config.whole_body_frequency` (come nel wrapper SRBD; non usato dentro) |
| `Kp`, `Kd` | `config.Kp` / `config.Kd`, 6×6 = `3*n_contact` |
| `grf` | forze di contatto MPC `[fcl; fcr]` = `state.sol.grf` |
| `foot_ref` | riferimenti di posizione dei centri ruota (`_REF_LW_POS`, `_REF_RW_POS`) |
| `foot_ref_dot` | riferimenti di velocità dei centri ruota |
| `contact` | `ones(2)` — entrambe le ruote sempre in appoggio |

`foot_ref` e il `current_leg` letto dentro la funzione sono coerenti: entrambi
sono centri ruota (`geom_xpos` del geom di collisione della ruota, e
`pl_z + wheel_radius` nel riferimento), la stessa coppia che usa il WBC QP.

### Perché non regge sul Tita

Sul Tita **entrambe le ruote sono sempre in contatto**, quindi la maschera seleziona
sempre il ramo di stance e la coppia si riduce a `tau = -(J grf)`: puro feedforward
delle forze MPC, senza nessun feedback proprio. Il quadrupede se lo può permettere
perché su quattro piedi puntiformi è staticamente stabile; il Tita è un pendolo
inverso su ruote e il pitch della base non è stabilizzato da nulla.

Misurato: cade su tutti i comandi, pitch fino a **1.22 rad** (~70°), anche a
velocità nulla con solo `ω=0.8`.

Ho provato anche l'ablazione ovvia — sommare sempre `tau_PD + tau_fb_lin` a
`tau_mpc` invece di mascherare, dato che il ramo di swing è codice morto qui — e
cade lo stesso (pitch 1.25 rad): il task sulle ruote vincola la lunghezza delle
gambe, non l'assetto della base. Per stare in piedi servono i task su CoM e
orientamento base, che è esattamente ciò che aggiunge `model_based`.

La funzione resta quindi utile come riferimento/confronto, non come controllore
operativo.

---

## 7. Confronto misurato

`validate_dfcip_controller.py`, 6 comandi, `--hold 4`, scena piatta, default:

| | `qp` | `model_based` | `wheeled` |
|---|---|---|---|
| vx=1.0, ω=0.8 | 0.997 / 0.805 | 0.997 / 0.805 | caduta |
| vx=0.6, ω=0.4 | 0.598 / 0.404 | 0.598 / 0.403 | caduta |
| cadute / NaN | 0 / 0 | 0 / 0 | 6 / 0 |
| \|tau\|max | 20.5 N m | 20.8 N m | 26.6 N m |
| pitch max (rampa) | 0.006 rad | 0.16 rad | 1.22 rad |
| tempo per chiamata (CPU) | 1.59 ms | 0.29 ms | — |

`qp` e `model_based` hanno tracking identico a tre decimali su tutti i casi.
L'unica differenza reale è il transitorio di pitch durante la rampa: le
disequazioni del QP frenano la partenza, la solve lineare no. Se dà fastidio,
alza `Kd_motion` o allunga `--ramp`.

Benchmark dei tempi:

```bash
JAX_PLATFORMS=cpu python - <<'EOF'
import numpy as np, jax, jax.numpy as jnp, types, mujoco, time
import mpx.config.config_dfcip as cfg_mod
import mpx.utils.mpc_wrapper_dfcip as W

def make_cfg(**over):
    c = types.SimpleNamespace()
    for k in dir(cfg_mod):
        if not k.startswith("__"): setattr(c, k, getattr(cfg_mod, k))
    for k, v in over.items(): setattr(c, k, v)
    return c

model = mujoco.MjModel.from_xml_path(cfg_mod.model_path)
data = mujoco.MjData(model)
data.qpos[:3] = [0, 0, cfg_mod.robot_height]; data.qpos[3:7] = [1, 0, 0, 0]
data.qpos[7:] = np.asarray(cfg_mod.q0); mujoco.mj_forward(model, data)
qpos = jnp.array(data.qpos)[None]; qvel = jnp.array(data.qvel)[None]
com = np.asarray(data.subtree_com[0])
x0 = jnp.array(np.concatenate([com, [0, 0, 0], [com[0], com[1], 0.], [0.], [0.], [0.], [0.]]))
pl = jnp.array([[0.0, 0.2835, 0.0]]); pr = jnp.array([[0.0, -0.2835, 0.0]]); z = jnp.zeros((1, 3))

for wbc, src in [("qp", None), ("model_based", "dynamics"), ("model_based", "mpc"), ("wheeled", None)]:
    over = dict(wbc_type=wbc)
    if src: over["wbc_mb_force_source"] = src
    mpc = W.BatchedMPCControllerWrapper(make_cfg(**over), n_env=1)
    st = mpc.init_state()
    st, _ = mpc.run(st, x0[None, :], jnp.array([[0.5, 0.0, 0.0, 0.0]]))
    f = lambda: mpc.whole_body_run(st, x0, qpos, qvel, pl, pr, z, z)
    r = f(); jax.block_until_ready(r[1])
    N = 200; t0 = time.perf_counter()
    for _ in range(N):
        r = f(); jax.block_until_ready(r[1])
    print(f">>> {wbc}/{src}: {(time.perf_counter()-t0)/N*1e3:.3f} ms/call")
EOF
```

---

## 8. Chiamare i controllori a mano

### `model_based`

Drop-in per il QP: stessa firma, stessi ritorni `(tau, qddot, fl, fr)`, più le
opzioni specifiche in coda.

```python
import mpx.utils.wbc_model_based as wbc_mb

tau, qddot, fl, fr = wbc_mb.whole_body_interface_wheeled_legged_model_based(
    mjx_model, mass, grav, d,
    contact_id, body_id, base_body_id,
    wheel_radius, dt_wbc, n_contacts,
    Kp_motion, Kd_motion, Kp_wheel, Kd_wheel, Kp_reg, Kd_reg,
    w_posture, w_qddot, w_com, w_lwheel, w_rwheel, w_base, mu,
    qpos, qvel, desired,
    posture_mask=posture_mask,
    enforce_rolling=True, force_source="dynamics", damping=1e-6,
)

# variante con dict di diagnostica
tau, qddot, fl, fr, diag = wbc_mb.whole_body_interface_wheeled_legged_model_based_diag(...)
```

Chiavi di `diag` (le stesse del QP, più una):

- `res_com`, `res_lwheel`, `res_rwheel`, `res_base` — residui dei task
- `a_*_total`, `err_*` — accelerazioni richieste e termini PD
- `eq_res_norm` / `roll_res_norm` — violazione del vincolo di rotolamento
- `dyn_res_norm` *(nuova)* — residuo delle 6 righe del floating base (≈0 con
  `force_source="dynamics"`, ≈1.3 N con `"mpc"`)
- `fric_margin_l/r`, `ineq_slack_min_joint` — margini **misurati**, non imposti
- `converged=True`, `iters=0` — solo per parità di interfaccia col QP

### `wheeled`

```python
tau, J = mpc_utils.whole_body_interface_wheeled(
    model, mjx_model, contact_id, body_id, simulation_frequency,
    Kp, Kd,                       # (3*n_contact) quadrate, config.Kp / config.Kd
    qpos, qvel,
    grf,                          # (6,)  forze MPC [fcl; fcr]
    foot_ref, foot_ref_dot,       # (6,)  posizioni / velocità ruote
    contact,                      # (2,)  ones sul Tita
)
```

Non produce `qddot`: il wrapper restituisce zeri per compatibilità di firma,
quindi `--outer-pd-mode plan` non è utilizzabile in questa modalità. Anche la
diagnostica è ridotta (residui dei task e margini a zero: non esistono task né
vincoli qui).

### Vettore `desired`

Mantiene il layout di `mpc_utils.pack_reference` (CoM / ruote / base / giunti) e
accoda `fcl` (3) e `fcr` (3) in fondo: tutti gli offset esistenti sono invariati e
il QP ignora la coda. Helper in `wbc_model_based.py`:

```python
wbc_mb.desired_size(nj)              # lunghezza totale
wbc_mb.contact_force_offsets(nj)     # (offset_fl, offset_fr)
wbc_mb.unpack_contact_forces(desired, nj)
```

---

## 9. Problemi noti

- **`ModuleNotFoundError: No module named 'jax'`** — env sbagliato, usa `mjpl`.
- **`PermissionError: '/val.json'`** — `--out` risolto rispetto alla root perché
  la shell non era nella cartella attesa; passa un path assoluto.
- **Prima chiamata lenta (~15-17 s)** — compilazione XLA, non il solver; a regime
  sono 0.29 ms (`model_based`) / 1.59 ms (`qp`).
- **`wheeled` cade sempre** — non è un bug, è §6.
