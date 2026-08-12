# TITA end-to-end PPO — record sperimentale del task di stazione

Ambito: ricostruire lo stato attuale e la storia sperimentale del bipede
bilanciato a due ruote **TITA**, addestrato end-to-end con **Brax PPO su MuJoCo
Playground / MJX**. Niente MPC, niente WBC, niente controllore residuale — la
policy pilota direttamente gli attuatori MuJoCo.

Questo è un record sperimentale, non un tutorial. Ogni valore è tracciato dai file
**caricati** (`train_srbd.py`, `joystickE2E.py`) e dal modello (`base.py`,
`tita.xml`), seguendo il **percorso di codice realmente attivo**. Dove il codice
attuale potrebbe differire da ciò che era attivo al lancio di uno specifico run, è
segnalato con una di:

- **CONFERMATO ATTIVO** — presente nel codice attuale e coerente col run.
- **CODICE ATTUALE — SUCCESSIVO / DIVERSO** — presente ora, ma non è ciò che il run
  ha usato (o non necessariamente).
- **CRONOLOGIA INCERTA** — non stabilibile dal solo sorgente.

Gli identificatori di codice restano in inglese; il resto è in italiano.

> **Nota di versione — i file caricati differiscono dall'analisi precedente.**
> Il sorgente ora riflette gran parte del setup di successo: `discounting=0.99`,
> randomizzazione del reset **attiva**, altezza del critic sul **sensore CoM**.
> Inoltre `Kd` gambe è ora `1.0` (era 10) e `noise_config.level = 0.0` (rumore
> d'osservazione **disattivato**). Le cautele residue sono in §13, §15, §17, §19–21,
> §28.

---

## 1. Panoramica dell'esperimento

- Robot: TITA, gamba a 3-DOF × 2 + una ruota motorizzata per gamba → **8 DOF
  controllati**.
- Metodo: RL puro E2E, Brax PPO, fisica MJX. Policy → attuatori MuJoCo diretti.
- Task (deliberatamente semplice):
  ```text
  velocità avanti comandata = 0 m/s
  yaw rate comandato        = 0 rad/s
  altezza CoM target        = 0.4 m
  ```
- Equilibrio desiderato: `vx≈0, yaw_rate≈0, CoM_z≈0.4 m`, in verticale, bilanciato.
- Stato: la stazione di base è appresa (reward di eval ~14–14.5). Il senso del
  record è documentare — in modo conservativo — come, dato che la stessa
  formulazione E2E inizialmente falliva.

---

## 2. Cronologia sperimentale

Attribuzione conservativa: un cambiamento è accreditato solo dove la cronologia lo
supporta.

### Stage 0 — setup iniziale fallimentare
```text
num_envs    ≈ 8192
batch_size  ≈ 256
discounting = 0.97
reward eval ≈ 3–4   → nessuna stazione robusta (cade presto o non regge l'equilibrio)
```

### Stage 1 — batching PPO + osservazione di altezza (più cambiamenti insieme)
```text
num_envs   : 8192 -> 4096
batch_size : 256  -> 512
```
Contestualmente è cambiata l'informazione di altezza data alla rete:
- **Actor**: aggiunto un segnale esplicito di altezza CoM (prima l'actor non aveva
  feedback utile esplicito sull'altezza).
- **Critic**: `qpos[2:3]` (base flottante) → z del sensore CoM (`base_subtree_com`).

✅ Nel sorgente caricato **entrambi** i cambiamenti sono presenti (§8/§9): actor e
critic usano ora `current_com_height` dal sensore CoM.

Risultato dopo questa fase: `reward eval ≈ 6`, stazione breve.

> Qui sono cambiate più variabili. **Non attribuire il miglioramento 3–4 → ~6 a una
> sola di esse** (`num_envs`, `batch_size`, o il segnale di altezza). Trattale come
> un blocco.

### Stage 2 — fattore di sconto (effetto osservato maggiore)
```text
discounting : 0.97 -> 0.99   (resto della formulazione ~invariato)
prima : reward ≈ 6,  sta in piedi solo brevemente
dopo  : reward ≈ 14+, stazione stabile appresa
~13M env steps -> reward ≈ 13, stazione già chiaramente appresa
~26M env steps -> reward ≈ 14.5
```
✅ Il sorgente caricato ha `discounting = 0.99` (§17): coerente col run di successo.

> **Evidenza empirica principale:** aumentare `discounting` da `0.97` a `0.99` è
> stato associato al miglioramento osservato maggiore — dallo *stare in piedi
> brevemente* al *mantenere l'equilibrio*. È un risultato empirico **per questo
> ambiente**, non un'affermazione universale su PPO.

---

## 3. Avvertenza di cronologia (vale ovunque)

Il sorgente contiene alcune modifiche fatte **dopo** che run di successo erano già
stati lanciati — in particolare la config comandi (`command_config.b`, §16) e
possibilmente `Kd` gambe e `noise_config.level` (§28). Un run di successo (~reward
14) è stato lanciato con la randomizzazione dello stato iniziale abilitata; il suo
successo **non** va accreditato a edit introdotti dopo l'avvio. Classificazioni
per-parametro inline.

---

## 4. DOF del robot e azioni della policy

Layout indici `action` / `qpos[7:]` (da `tita_constants.py` + `joystickE2E.py`):

| Idx azione | Giunto | Tipo | Significato azione | Scala | Controllo low-level effettivo |
|-----------:|--------|------|--------------------|-------|-------------------------------|
| 0 | left leg 1 (hip/spread) | position | offset dal target di default | 0.5 rad | `τ = 35·(q_tgt−q) − 1·q̇` |
| 1 | left leg 2 (thigh) | position | offset dal target di default | 0.5 rad | stesso PD |
| 2 | left leg 3 (knee) | position | offset dal target di default | 0.5 rad | stesso PD |
| 3 | left wheel | velocity | target di velocità | 5.0 rad/s | `τ = 0.5·(5a − ω) = 2.5a − 0.5ω` |
| 4 | right leg 1 (hip/spread) | position | offset dal target di default | 0.5 rad | stesso PD |
| 5 | right leg 2 (thigh) | position | offset dal target di default | 0.5 rad | stesso PD |
| 6 | right leg 3 (knee) | position | offset dal target di default | 0.5 rad | stesso PD |
| 7 | right wheel | velocity | target di velocità | 5.0 rad/s | `τ = 2.5a − 0.5ω` |

```text
LEG_DOF_IDS   = [0, 1, 2, 4, 5, 6]   # attuatori position
WHEEL_DOF_IDS = [3, 7]               # attuatori velocity
```
Distribuzione `tanh_normal` ⇒ ogni azione ∈ (−1, 1).

---

## 5. Controllo realmente eseguito

### 5.1 Gambe
```text
azione di rete a  ->  q_target = default_pose + 0.5·a  ->  attuatore <position>  ->  τ
```
I guadagni **non** sono quelli grezzi dell'XML: `base.py` li sovrascrive dalla
config:
```python
# base.py — gambe
actuator_gainprm[leg,0] =  config.Kp   # 35
actuator_biasprm[leg,1] = -config.Kp   # -35
actuator_biasprm[leg,2] = -config.Kd   # -1   (Kd ora = 1.0, era 10)
```
Legge effettiva delle gambe:
```text
τ = Kp·(q_target − q) − Kd·q̇ = 35·(q_target − q) − 1·q̇
```
- `config.Kp = 35`, `config.Kd = 1.0` controllano davvero gli attuatori (via
  `base.py`), quindi config e guadagni effettivi coincidono. Nessuna incoerenza.
- ⚠️ **Cambiamento vs analisi precedente:** `Kd` gambe è **1.0** (era 10). Valore
  vicino al riferimento DDT (~1); riduce lo smorzamento dei giunti (facilita il
  rialzo della base). Classificazione: **CODICE ATTUALE**; presenza nel run di
  successo = **CRONOLOGIA INCERTA** (non citato nella cronologia).

### 5.2 Ruote
```python
# base.py — ruote (attuatore velocity)
actuator_gainprm[wheel,0] =  config.Kd_wheel   # 0.5
actuator_biasprm[wheel,1] =  0.0               # nessun feedback di posizione
actuator_biasprm[wheel,2] = -config.Kd_wheel   # -0.5
```
```text
v_target = 5.0·a
τ = 0.5·(v_target − ω) = 0.5·(5a − ω) = 2.5·a − 0.5·ω
```
Comando di **coppia smorzato** (coppia feed-forward ∝ azione + smorzamento di
velocità), *non* un blocco rigido di velocità — corretto per una ruota di
bilanciamento; identico alla legge di ruota del riferimento DDT Isaac-Gym.
Limite di coppia (da `tita.xml`, tutti gli 8 attuatori): **`forcerange = ±120 N·m`**.

---

## 6. Timing di controllo e simulazione

```text
sim_dt  = 0.002 s   -> fisica 500 Hz
ctrl_dt = 0.01  s   -> policy  100 Hz
n_substeps = ctrl_dt / sim_dt = 5
```
```text
la policy emette un'azione ogni 10 ms
   -> target di controllo tenuto costante
   -> MuJoCo integra 5 × 2 ms
```
- `action_repeat = 1` (config **e** PPO).
- Env `episode_length = 1000`; PPO `episode_length = 1000` — **coincidono**.
- Durata fisica episodio ≈ `1000 × 0.01 = 10 s`. Nessun wrapper riscala
  (`EpisodeWrapper(episode_length=1000, action_repeat=1)`) → **10 s confermati**.

---

## 7. Osservazione dell'actor (`state`) — dal `jp.hstack` attivo

| Componente | Dim | Frame | Significato | Rumore | Note |
|------------|----:|-------|-------------|--------|------|
| vel. lineare locale | 3 | corpo | vel. lin. base (incl. vz) | 0 (level=0) | velocimeter @ imu |
| gyro | 3 | corpo | vel. angolare base | 0 (level=0) | |
| gravità proiettata | 3 | corpo | tilt (pitch/roll) | 0 (level=0) | `site_xmat.T·ẑ` |
| errore pos. gambe | 6 | giunto | `(q−q_def)` per 6 giunti gamba | 0 (level=0) | ruote escluse |
| velocità giunti | 8 | giunto | tutti i DOF incl. **entrambe le ruote** | 0 (level=0) | |
| azione precedente | 8 | – | ultima azione della policy | nessuno | vedi sotto |
| comando | 2 | – | `[vx_cmd, yaw_cmd]` | nessuno | sempre `[0,0]` (§16) |
| **altezza CoM** | 1 | mondo | `current_com_height` (z CoM, **grezza** ~0.4) | nessuno | `base_subtree_com`[2] |

```text
dim state = 3+3+3+6+8+8+2+1 = 34
```
- ⚠️ **Nota:** l'actor riceve la **z del CoM grezza** (`current_com_height`), non
  l'errore `com_z − 0.4` (la riga `com_height_err[None]` è commentata). Il commento
  dice "centrata vicino a 0", ma il codice usa il valore grezzo; con
  `normalize_observations=True` (running normalizer) la differenza è ininfluente in
  pratica. Dimensione invariata (34).
- Tutti i rumori sono **0** perché `noise_config.level = 0.0` (§15); le colonne
  "rumore" riportano l'ampiezza *effettiva*, non le `scale` configurate.
- **L'azione precedente è davvero l'azione precedente** dal punto di vista della
  decisione successiva: `step()` passa l'`action` corrente in `_get_obs` per l'obs
  restituita questo passo e aggiorna `info["last_act"]=action` dopo. ✓

---

## 8. Il cambiamento dell'osservazione CoM

| Segnale | Prima | Dopo (sorgente caricato) | Stato |
|---------|-------|--------------------------|-------|
| Altezza **actor** | nessun feedback CoM utile esplicito | `current_com_height` in `state` (dim 33→34), da `base_subtree_com`[2] | **CONFERMATO ATTIVO** |
| Altezza **critic** | `qpos[2:3]` (base flottante) | `current_com_height` (z sensore CoM) | **CONFERMATO ATTIVO** (cambiamento presente) |

`z root/base` vs `z CoM del subtree`:
```text
qpos[2]            = z dell'origine della base_link      (~0.44 alla posa nominale)
base_subtree_com.z = z del CoM dell'intero robot         (~0.399 alla posa nominale)
```
Il target della reward è definito su **CoM z** (`base_height_target = 0.4`); ora
sia actor sia critic percepiscono la stessa grandezza fisica che la reward
ottimizza — variabile di feedback coerente. **Non affermare che il solo segnale CoM
abbia prodotto la stazione** — era uno dei vari cambiamenti dello Stage 1.

---

## 9. Osservazione del critic (`privileged_state`) — dimensione reale

I commenti inline sono **obsoleti** (dicono `state # 33` e `total: 84`). Ricalcolo
dal codice:

| Componente | Dim |
|------------|----:|
| `state` (obs completa dell'actor) | 34 |
| gyro | 3 |
| accelerometer | 3 |
| gravità | 3 |
| linvel locale | 3 |
| angvel globale | 3 |
| errore pos. giunti gamba `(q−q_def)[leg_ids]` | 6 |
| velocità giunti | 8 |
| forze attuatori | 8 |
| ultimo contatto (piedi) | 2 |
| velocità piedi | 6 |
| feet air time | 2 |
| **altezza CoM `current_com_height`** (z sensore CoM) | 1 |
| forza esterna sul torso | 3 |
| **TOTALE** | **85** |

```text
dim privileged_state = 34 + 51 = 85   (NON 84 come dichiara il commento)
input actor  = obs["state"]            (34)
input critic = obs["privileged_state"] (85)
```
Lo split actor/critic è confermato dal percorso di rete attivo (§20/§21): la policy
legge `state`, la value net legge `privileged_state`, tramite i default delle
chiavi obs di `make_ppo_networks` del fork Brax installato.

---

## 10. Funzione di reward

Termini × scala, sommati, poi `× dt (0.01)` e clip a `±1e4`.
`only_positive_rewards = False` → percorso a somma semplice:
```python
reward = clip(sum(scale_k · term_k) · dt, -1e4, 1e4)
```

| Termine | Peso | Grandezza grezza | Ottimo | Scopo |
|---------|-----:|------------------|--------|-------|
| tracking_lin_vel | +1.0 | `exp(−(vx_cmd−vx)²/σ)` | +1.0 | tracciare vel. avanti (solo x) |
| tracking_ang_vel | +0.5 | `exp(−(yaw_cmd−ωz)²/σ)` | +0.5 | tracciare yaw rate |
| orientation | −2.0 | `‖g_xy‖²` | 0 | restare in piano |
| ang_vel_xy | −0.3 | `ωx²+ωy²` | 0 | smorzare rate roll/pitch |
| base_height | −1.0 | `1−exp(−((h−0.4)/0.05)²)` | 0 | tenere CoM z=0.4 (§11) |
| posture | −5.0 | `Σ w·(q−q_def)²·gate` | 0 | gambe neutre in stazione (§12) |
| torques | −1e−4 | `Σ τ²` | →0 | sforzo |
| action_rate | −0.01 | `Σ(a−a_prev)²` | →0 | fluidità (adimensionale) |
| dof_pos_limits | −1.0 | hinge soft-limit | 0 | tenere le gambe lontane dai limiti |
| termination | −5.0 | `done` | 0 | penalizzare la caduta |

```text
tracking_sigma       = 0.25
base_height_target   = 0.4
posture_cmd_sigma    = 0.25
only_positive_rewards= False
```
In stazione ideale la reward per passo ≈ `(1.0 + 0.5 − piccolo)·0.01 ≈ +0.015`, di
cui **+1.5 (pre-dt) è tracking indipendente dall'altezza**.

---

## 11. Reward di altezza

```python
err  = body_height - 0.4          # body_height = base_subtree_com.z
cost = 1.0 - exp(-(err/0.05)**2)  # scala -1.0
```
```text
vecchia:  deadzone a gradiente nullo ≈ [0.38, 0.42] m  (primi ~2 cm gratis)
attuale:  costo liscio, gradiente non nullo esattamente a 0.40, satura (->1) per err grande
```
Valori campione: 0.40→0.000, 0.39→0.039, 0.38→0.148, 0.36→0.473, 0.30→0.982
(×scala −1.0).

> ⚠️ Come da brief: **non accreditare la rimozione della deadzone per il successo
> del run `gamma=0.99` a meno che la sua presenza al lancio non sia confermata.**
> Sorgente attuale = no-deadzone; presenza al lancio = **CRONOLOGIA INCERTA**.

---

## 12. Regolarizzazione della postura

```python
weights = [1.0, 0.5, 0.5,  1.0, 0.5, 0.5]     # [hip, thigh, knee] × 2
err     = (q - default_pose)[leg_ids]
raw     = Σ weights · err²
gate    = exp(-‖command‖² / posture_cmd_sigma)  # posture_cmd_sigma = 0.25
cost    = raw · gate                             # scala -5.0
```
- I giunti hip/spread hanno il peso maggiore (1.0).
- Il **gate** rende la postura più forte in stazione (`command≈0 → gate≈1`) e la
  attenua al crescere della velocità comandata, così non contrasta il moto dinamico
  delle gambe in locomozione.
- Classificazione: presente nel sorgente attuale; **attivo nel run di successo =
  CRONOLOGIA INCERTA**.

---

## 13. Reset e randomizzazione dello stato iniziale

Inizializzazione (`reset`, sorgente caricato) — **randomizzazione ATTIVA**:
```python
qpos = self._init_q                       # home keyframe, z base = 0.4435
dxy = U(-0.5, 0.5)
qpos = qpos.at[0:2].set(qpos[0:2] + dxy)                     # xy ATTIVA
yaw = U(-3.14, 3.14)
qpos = qpos.at[3:7].set(quat_mul(qpos[3:7], yaw_quat))       # yaw ATTIVA
vx = U(-0.2, 0.2); vy = U(-0.1, 0.1)
qvel = qvel.at[0:2].set([vx, vy])                            # vx,vy ATTIVE
ctrl[leg_ids] = qpos[7:][leg_ids]         # target gambe = angoli iniziali; ruote 0
```
```text
qpos iniziale : home keyframe (z base 0.4435) + offset xy U(±0.5) + yaw U(±π)
qvel iniziale : vx U(±0.2), vy U(±0.1), resto 0 (ruote 0)
orientamento  : verticale ruotato di yaw casuale
rumore obs    : INATTIVO (noise_config.level = 0.0)  (§15)
perturbazioni : INATTIVE (pert_config.enable = False)
```

> ✅ **Coerente col run di successo:** il brief afferma che il run ~14 è stato
> lanciato **con la randomizzazione dello stato iniziale abilitata**; il sorgente
> caricato ha quelle assegnazioni **attive**. Classificazione: **CONFERMATO
> ATTIVO**. (Nota: in una versione precedente dei file queste righe erano
> commentate; ora sono attive.)

---

## 15. Perturbazioni e rumore d'osservazione

```text
pert_config.enable = False        -> perturbazioni INATTIVE (i range sotto sono dormienti)
  velocity_kick    = [0.0, 3.0]
  kick_durations   = [0.05, 0.2]
  kick_wait_times  = [1.0, 3.0]

noise_config.level = 0.0          -> rumore d'osservazione INATTIVO (ampiezza effettiva = 0)
  scale (dormienti, moltiplicate per level=0):
  joint_pos = 0.01 ; joint_vel = 1.5 ; gyro = 0.2 ; gravity = 0.05 ; linvel = 0.1
```
- ⚠️ **Cambiamento vs analisi precedente:** `level` è **0.0** (era 1.0). Poiché
  `rumore = level · scale`, con `level=0` **il rumore d'osservazione effettivo è
  zero**, nonostante le `scale` siano definite. Non riportare le `scale` come
  ampiezze attive.
- Classificazione `level=0`: **CODICE ATTUALE**; presenza nel run di successo =
  **CRONOLOGIA INCERTA** (non citato nella cronologia).
- Perturbazioni: definite ma **non abilitate**.

---

## 16. Sistema di comandi

```text
command = [velocità_avanti, yaw_rate]            (dim 2)
command_config.a = [0.0, 0.0]     # ampiezza (semi-range) per componente
command_config.b = [0.75, 0.75]   # prob. che una componente ricampionata resti non nulla
command_config.p_stand = 0.2      # probabilità di stand esplicito
ricampionamento: steps_until_next_cmd ~ round(Exponential()·5 s / dt)
```
Poiché l'ampiezza `a = [0, 0]`, ogni comando campionato è **`[0, 0]`**
indipendentemente da `b`/`p_stand` — task di pura stazione. L'esperimento di
successo ha usato `avanti=0, yaw=0`.

> ⚠️ **Edit successivo:** `command_config.b` è stato modificato da `[0.75, 0.5]` a
> `[0.75, 0.75]` (unica differenza tra la snapshot precedente e i file caricati).
> Essendo `a = [0, 0]`, la modifica è **inerte** sulla distribuzione effettiva dei
> comandi. Classificazione: **CODICE ATTUALE — NON RESPONSABILE DEL SUCCESSO GIÀ
> OSSERVATO** (esempio concreto di edit comandi post-lancio). Vedi
> `command_dim_generalization.md`.

---

## 17. Configurazione PPO (chiamata attiva a `ppo.train`)

Percorso attivo: `make_train_fn("ppo")` → `functools.partial(ppo.train,
**PPO_PARAMS, progress_fn=progress)`. Gli scalari di `PPO_PARAMS` alimentano
direttamente `ppo.train` (eccezione: `network_factory`, sovrascritto — §20).

| Parametro PPO | Valore attivo (sorgente caricato) | Nota |
|---------------|----------------------------------:|------|
| Algoritmo | PPO | |
| Timesteps totali | 100.000.000 | |
| N. valutazioni | 10 | |
| Episode length | 1000 | = env (10 s) |
| N. ambienti | **4096** | `num_envs=NUM_ENVS`, `NUM_ENVS=4096` |
| Batch size | **512** | |
| Unroll length | 20 | |
| N. minibatch | 32 | |
| Update per batch | 4 | |
| **Fattore di sconto** | **0.99** | ✅ coincide col run di successo |
| Learning rate | 3e-4 | |
| Entropy cost | 1e-2 | |
| Reward scaling | 1.0 | (l'env già ×dt internamente) |
| Max grad norm | 1.0 | |
| Normalize observations | True | running_statistics |
| Action repeat | 1 | |
| Reset per eval | 10 | |
| Seed | 0 | |

> Tutti i valori qui coincidono col run di successo descritto (incluso
> `discounting=0.99`). `PPO_PARAMS["network_factory"]` **non** è elencato come
> attivo — è sovrascritto (§20).

---

## 18. Architettura di rete (costruzione attiva)

Costruita da `_build_fresh_networks` (branch PPO):
```python
ppo_networks.make_ppo_networks(
    observation_size=env.observation_size,      # dict: state=34, privileged_state=85
    action_size=env.action_size,                # 8
    policy_hidden_layer_sizes=(512, 256, 128),  # ATTIVO
    preprocess_observations_fn=running_statistics.normalize,  # ATTIVO (norm obs on)
    distribution_type="tanh_normal",            # ATTIVO  -> output policy 2·8 = 16
    # activation=linen.elu,                      <-- COMMENTATA (usato default Brax)
    # policy_network_kernel_init_fn=...,         <-- COMMENTATA (init default Brax)
    # init_noise_std=INIT_STD,                   <-- COMMENTATA (INIT_STD scollegato)
)
```
- **Hidden layer policy**: `(512, 256, 128)` — attivo.
- **Hidden layer value/critic**: **default Brax** — `value_hidden_layer_sizes`
  *non* è passato. Il `(512,256,128)` del value vive solo nel dict morto
  `PPO_PARAMS["network_factory"]` (§20), quindi **non** si applica.
- **Distribuzione**: `tanh_normal`. **Preprocessing**: running-statistics normalize.
- **Attivazione / init**: default del fork Brax (ELU e init custom commentati per
  PPO). I default esatti dipendono dal fork; il fork non è importabile qui, quindi
  sono riportati come "default Brax", non asseriti.

---

## 19. `INIT_STD`, `ZERO_INIT_OUTPUT_LAYER`, ELU — NON attivi in PPO

Definiti nel file:
```python
INIT_STD = 0.03
ZERO_INIT_OUTPUT_LAYER = False
```
Nel branch **PPO** di `_build_fresh_networks` gli argomenti che li userebbero sono
commentati (verificato nel file caricato):
```python
# activation=linen.elu,
# policy_network_kernel_init_fn=_policy_kernel_init_factory,
# init_noise_std=INIT_STD
```

- **`INIT_STD`** — **non passato** a `make_ppo_networks`. Il run PPO di successo ha
  usato la **std iniziale di default di Brax**, non `0.03`.
  ```text
  Std iniziale PPO: default Brax (la variabile INIT_STD non è collegata)
  ```
- **`ZERO_INIT_OUTPUT_LAYER`** — passato a
  `_build_fresh_networks(zero_init_output_layer=...)`, ma il meccanismo che lo usa
  (`policy_network_kernel_init_fn` → ramo zero-init) è **commentato per PPO** →
  nessun effetto.
  ```text
  ZERO_INIT_OUTPUT_LAYER: definito ma inattivo per PPO
  ```
- **ELU** — commentata per PPO → PPO usa l'**attivazione di default di Brax**. Non
  riportare ELU come attiva.
- **Distinzione SAC** — questi argomenti custom sono **attivi solo nel branch SAC**
  (`sac_networks.make_sac_networks`). Questo esperimento è PPO; non mescolare le
  impostazioni di rete SAC.

---

## 20. Percorso reale di `network_factory` (morto vs. attivo)

```text
main -> run_train
          built_networks = _build_fresh_networks(env, zero_init_output_layer=False)
          selected_network_factory = lambda *a, **k: built_networks
          train_fn = make_train_fn("ppo")  =  partial(ppo.train, **PPO_PARAMS, progress_fn=progress)
          train_fn(environment=..., eval_env=..., wrap_env_fn=...,
                   network_factory=selected_network_factory,   # <-- sovrascrive PPO_PARAMS
                   policy_params_fn=...)
```
`functools.partial` lega `network_factory=PPO_PARAMS["network_factory"]`, ma il
keyword **al momento della chiamata** `network_factory=selected_network_factory` lo
**sovrascrive** (le kwargs di chiamata hanno priorità sulle kwargs del partial).
Pertanto:

```text
ATTIVO  : selected_network_factory -> built_networks (da _build_fresh_networks)
MORTO   : PPO_PARAMS["network_factory"] = dict(
              policy_hidden_layer_sizes=(512,256,128),   # morto (stesso valore ri-fornito a _build_fresh_networks)
              value_hidden_layer_sizes =(512,256,128),   # MORTO -> il critic usa il default Brax
              policy_obs_key="state",                    # MORTO -> risolto via default di make_ppo_networks
              value_obs_key ="privileged_state",         # MORTO -> risolto via default di make_ppo_networks
          )
```
Il dizionario `network_factory` dentro `PPO_PARAMS` è **configurazione
sovrascritta**: non va riportato come attivo. In particolare la larghezza del critic
è un **default Brax**, non `(512,256,128)`.

---

## 21. Chiavi di osservazione actor vs. critic (risoluzione attiva)

La chiamata attiva a `make_ppo_networks` in `_build_fresh_networks` **non** passa
`policy_obs_key` / `value_obs_key` → si risolvono ai **default** di quella funzione
nel fork Brax. `run_train` legge esattamente quei default dalla signature:
```python
policy_obs_key = signature(make_ppo_networks).parameters["policy_obs_key"].default
value_obs_key  = signature(make_ppo_networks).parameters["value_obs_key"].default
```
Poiché l'env espone `obs = {"state": 34, "privileged_state": 85}` e il run addestra
correttamente, questi default si risolvono in:
```text
chiave obs policy (actor)  -> "state"
chiave obs value  (critic) -> "privileged_state"
```
Lo split è reale, ma proviene dai **default del fork Brax**, **non** dal dict morto
`PPO_PARAMS["network_factory"]`. Le stringhe esatte dipendono dal fork (non
importabile qui); la risoluzione è dedotta dall'obs dict funzionante.

---

## 22. Comportamento di valutazione

- **Eval interna di PPO**: governata da `DETERMINISTIC_EVAL = False` → la
  valutazione periodica **campiona** la policy stocastica (non la media).
- **Viewer/rollout custom**: costruisce esplicitamente
  `policy_fn = inference_fn(params, deterministic=True)` → usa l'azione **media**.

Quindi i reward della curva di training (eval stocastica) e il comportamento del
viewer (deterministico) non sono prodotti con lo stesso rumore di azione; ci si
aspetta che il viewer appaia leggermente più stabile.

---

## 23. Interpretazione di gamma (solo intuizione)

Orizzonte effettivo approssimato `≈ 1/(1−γ)` (non un orizzonte fisico formale):
```text
γ = 0.97 -> ~33 passi di policy -> ~0.33 s  (a 100 Hz)
γ = 0.99 -> ~100 passi di policy -> ~1.0 s
```
Perché può contare per un robot bilanciato:
- le azioni correttive hanno conseguenze **ritardate** (un'azione che costa reward
  immediato può evitare una caduta dopo);
- la dinamica wheel-legged / pendolo inverso evolve su centinaia di ms;
- mantenere l'equilibrio è intrinsecamente un obiettivo **a lungo orizzonte**.

Osservazione empirica (il punto chiave):
```text
γ = 0.97 -> fatica / stazione breve
γ = 0.99 -> apprende stazione stabile
```

---

## 24. Comportamento qualitativo attuale (cosa funziona)

La policy E2E attuale può:
- restare in verticale e mantenere l'equilibrio;
- soddisfare approssimativamente il task a velocità nulla;
- raggiungere reward di eval ~14–14.5;
- apprendere una stazione utile entro ~13M env steps.

Questo mostra che la formulazione E2E di base
osservazione/azione/controllo **è** capace di equilibrio stazionario. Non implica
che l'ambiente sia ottimale.

---

## 25. Problema residuo — deriva lenta in avanti

Sebbene `comando avanti = 0`, il robot scivola/deriva lentamente in avanti.
Irrisolto. Direzioni candidate (nessuna causa asserita): attrito ruota-suolo, un
piccolo bias di velocità della ruota, offset di equilibrio del pitch, un minimo
bias di azione della policy, tolleranza del tracking di velocità
(`tracking_sigma`), asimmetria degli attuatori, comportamento di contatto MJX.
```text
il bilanciamento funziona ; l'equilibrio stazionario perfetto non ancora
```

---

## 26. Problema residuo — glitch della gamba

Occasionalmente una gamba raggiunge una configurazione anomala/glitchata. Solo
ipotesi — cause possibili: instabilità fisica/di contatto, gestione delle
collisioni, limiti di giunto, numerica MJX, saturazione attuatori, artefatti di
simulazione. **Non** classificato come fallimento di PPO senza evidenza. Tenuto
separato dal risultato dell'addestramento di stazione.

---

## 27. Tabella dei cambiamenti sperimentali (attribuzione conservativa)

| Parametro / feature | Prima | Dopo | Stage | Risultato osservato | Confidenza attribuzione |
|---------------------|-------|------|-------|---------------------|-------------------------|
| num_envs | 8192 | 4096 | 1 | parte di 3–4 → ~6 | **mista** (bundle) |
| batch_size | 256 | 512 | 1 | parte di 3–4 → ~6 | **mista** (bundle) |
| altezza actor | assente | CoM `current_com_height` | 1 | parte di 3–4 → ~6 | **mista** (bundle) |
| altezza critic | `qpos[2:3]` | `current_com_height` (sensore CoM) | 1 | parte di 3–4 → ~6 | **mista** (bundle) |
| discounting | 0.97 | 0.99 | 2 | ~6 → ~14–14.5, breve→stabile | **forte (empirica)** |

Progressione della reward:
```text
iniziale (Stage 0)              : ~3–4   stazione fallita
batch + osservazione (Stage 1)  : ~6     stazione breve   (attribuzione mista)
discount 0.99 (Stage 2)         : ~14–14.5 stazione stabile (empirica forte)
```

---

## 28. Tabella delle modifiche del sorgente attuale

| Feature | Vecchio valore/comportamento | Sorgente caricato | Testato nel run di successo? |
|---------|------------------------------|-------------------|------------------------------|
| `action_scale_vel` ruota | 30 | **5** (`τ=2.5a−0.5ω`) | probabilmente sì (fix precede altezza/γ per `02`/`03`) — cronologia non esplicita |
| reward altezza | deadzone ±2 cm | **liscia, no deadzone** | **CRONOLOGIA INCERTA** |
| altezza actor | mancante | **`current_com_height` (CoM)** | riportata presente (Stage 1) → **probabilmente sì** |
| altezza critic | `qpos[2:3]` | **`current_com_height` (CoM)** | riportata presente (Stage 1) → **probabilmente sì** |
| termine posture | (precedente) | **postura command-gated** | **CRONOLOGIA INCERTA** |
| randomizzazione reset | (commentata in ver. precedente) | **ATTIVA** (xy, yaw, vx, vy) | run: abilitata → **CONFERMATO ATTIVO** |
| `Kd` gambe | 10 | **1.0** | **CRONOLOGIA INCERTA** (non citato) |
| `noise_config.level` | 1.0 | **0.0** (rumore obs off) | **CRONOLOGIA INCERTA** (non citato) |
| `command_config.b` | [0.75, 0.5] | **[0.75, 0.75]** (inerte, `a=[0,0]`) | **NON responsabile del successo** |
| discounting | 0.97 | **0.99** | **CONFERMATO ATTIVO** |

Dove il valore di lancio non è recuperabile dal sorgente: **sconosciuto / serve
verifica git-history.**

---

# Snapshot setup di successo / attuale

Riepilogo di un minuto (valori del sorgente caricato).

## Robot
```text
tipo       : bipede bilanciato TITA a due ruote
azioni     : 8
giunti gamba : 6 (hip/thigh/knee × 2), attuatori position
ruote      : 2, attuatori velocity
massa totale : ~27.7 kg
```

## Controllo
```text
gambe : q_target = default_pose + 0.5·action ;  τ = 35·(q_tgt−q) − 1·q̇   (Kd=1)
ruote : v_target = 5.0·action               ;  τ = 0.5·(5a − ω) = 2.5a − 0.5ω
scale : action_scale_pos = 0.5 rad ; action_scale_vel = 5.0 rad/s
gains : Kp=35, Kd=1 (gambe, via base.py) ; Kd_wheel=0.5
limite coppia : ±120 N·m (tutti gli attuatori)
```

## Timing
```text
sim_dt=0.002 ; ctrl_dt=0.01 ; fisica=500 Hz ; policy=100 Hz ; substeps=5
episode_length=1000 ; durata fisica ≈ 10 s ; action_repeat=1
```

## Osservazioni actor (state, dim 34)
```text
linvel locale(3) + gyro(3) + gravità proiettata(3) + leg_pos_err(6)
+ joint_vel incl. ruote(8) + azione prec.(8) + comando(2) + current_com_height(1)
(rumore d'osservazione = 0 perché noise_config.level = 0.0)
```

## Osservazioni critic (privileged_state, dim 85)
```text
state(34) + gyro(3)+accel(3)+gravità(3)+linvel(3)+angvel(3)
+ leg_pos_err(6)+joint_vel(8)+actuator_force(8)+last_contact(2)
+ feet_vel(6)+feet_air_time(2)+ current_com_height(1) + ext_force(3)
NOTA: altezza critic = z CoM (current_com_height).  Il commento inline "84" è obsoleto (reale 85).
```

## Reward (coefficienti attuali)
```text
tracking_lin_vel +1.0 | tracking_ang_vel +0.5 | orientation -2.0 | ang_vel_xy -0.3
base_height -1.0 | posture -5.0 | torques -1e-4 | action_rate -0.01
dof_pos_limits -1.0 | termination -5.0
tracking_sigma=0.25 ; base_height_target=0.4 ; posture_cmd_sigma=0.25
only_positive_rewards=False ; total = clip(Σ scale·term · dt, ±1e4)
```

## Reset / randomizzazione (sorgente caricato)
```text
rand. posizione : ATTIVA  (x,y += U(±0.5))
rand. yaw       : ATTIVA  (yaw ~ U(±π))
rand. vx,vy     : ATTIVE  (vx ~ U(±0.2), vy ~ U(±0.1))
rumore obs      : INATTIVO (noise_config.level = 0.0)
perturbazioni   : INATTIVE (enable=False)
z base al reset = 0.4435 (ruote appena a contatto)
```

## PPO (attivo)
```text
timesteps=1e8 ; evals=10 ; episode_length=1000 ; num_envs=4096 ; batch_size=512
unroll=20 ; minibatches=32 ; updates/batch=4 ; discounting=0.99
lr=3e-4 ; entropy=1e-2 ; reward_scaling=1.0 ; max_grad_norm=1.0
normalize_obs=True ; action_repeat=1 ; resets/eval=10 ; seed=0
```

## Rete (realmente attiva in PPO)
```text
hidden layer policy : (512, 256, 128)
hidden layer value  : default Brax (NON (512,256,128); quel valore è in config morta)
distribuzione       : tanh_normal (output policy dim 16)
normalizzazione obs : running_statistics.normalize (ON)
attivazione         : default Brax (ELU custom commentata)
inizializzazione    : default Brax

INIT_STD = 0.03            : NON ATTIVO IN PPO
ZERO_INIT_OUTPUT_LAYER     : NON ATTIVO IN PPO
ELU custom                 : NON ATTIVA IN PPO
(questi tre sono attivi solo nel branch SAC)
```

## Risultati sperimentali
| Stage di configurazione | Reward | Comportamento |
|-------------------------|-------:|---------------|
| Iniziale | ~3–4 | stazione fallita |
| Cambi batch + osservazione | ~6 | stazione breve |
| `discounting=0.99` | ~13 a ~13M | stazione stabile appresa |
| Training successivo | ~14.5 a ~26M | stazione forte |

---

## 30. Conclusioni

> Il controllore end-to-end PPO di TITA è capace di apprendere il bilanciamento
> stazionario. Le prestazioni iniziali scarse **non** erano prova che la
> formulazione E2E azione/osservazione fosse fondamentalmente incapace di
> bilanciare. I cambi al sampling PPO (`num_envs 8192→4096`, `batch_size 256→512`)
> e l'aggiunta di un'osservazione di altezza CoM (actor e critic) hanno migliorato
> moderatamente l'apprendimento (~3–4 → ~6), mentre aumentare il fattore di sconto
> da `0.97` a `0.99` è stato associato al miglioramento comportamentale osservato
> maggiore — dallo stare in piedi brevemente alla stazione stabile (~6 → ~14–14.5).

> Gli esperimenti PPO di successo **non** hanno usato i custom `INIT_STD`,
> l'inizializzazione a output-layer zero, o le impostazioni ELU esplicite definite
> in `train_srbd.py`; quegli hook sono commentati / scollegati nella costruzione di
> rete PPO attiva (attivi solo nel branch SAC).

**Note di integrità del sorgente (verificare prima del riuso):**
1. `PPO_PARAMS.discounting = 0.99` (coerente col run). ✅
2. Randomizzazione del reset **attiva** (coerente col run). ✅
3. Altezza actor **e** critic ora sul **sensore CoM** (`current_com_height`). ✅
4. `privileged_state` è di dim **85** (il commento inline dice 84).
5. L'actor usa la **z CoM grezza**, non l'errore centrato (`com_height_err`
   commentato) — ininfluente con normalizzazione obs.
6. `PPO_PARAMS["network_factory"]` è morto (sovrascritto da
   `selected_network_factory`); larghezza critic = default Brax, non `(512,256,128)`.
7. Da confermare per il run di successo (non in cronologia): `Kd=1` gambe,
   `noise_config.level=0.0`, `command_config.b=[0.75,0.75]` (inerte).

**Lavoro residuo:**
- eliminare la deriva lenta in avanti (§25);
- investigare l'occasionale glitch gamba/fisica (§26);
- validare la stazione su più seed casuali;
- estendere in seguito i comandi oltre la stazione stazionaria.

# TITA — tabelle parametri (reference compatto)

Valori dal sorgente caricato (`train_srbd.py`, `joystickE2E.py`, `base.py`,
`tita.xml`). Riferito al percorso di codice **realmente attivo**.

## Robot / DOF
| Voce | Valore |
|------|--------|
| Tipo | bipede bilanciato a due ruote (segway-like) |
| DOF controllati | 8 |
| Giunti gamba | 6 → `LEG_DOF_IDS = [0,1,2,4,5,6]` (hip/thigh/knee × 2) |
| Ruote | 2 → `WHEEL_DOF_IDS = [3,7]` |
| Massa totale | ~27.7 kg |
| z base al reset | 0.4435 m (CoM ~0.399 m) |

## Controllo e guadagni
| Parametro | Valore | Note |
|-----------|--------|------|
| `action_scale_pos` | 0.5 rad | `q_tgt = q_def + 0.5·a` |
| `action_scale_vel` | 5.0 rad/s | `v_tgt = 5·a` |
| `Kp` (gambe) | 35 | attuatore position |
| `Kd` (gambe) | 1.0 | (era 10) |
| `Kd_wheel` (ruote) | 0.5 | attuatore velocity |
| Legge gambe | `τ = 35·(q_tgt − q) − 1·q̇` | |
| Legge ruote | `τ = 0.5·(5a − ω) = 2.5·a − 0.5·ω` | comando coppia smorzato |
| Limite coppia | ±120 N·m | tutti gli attuatori (`forcerange`) |
| Distribuzione azione | `tanh_normal` → a ∈ (−1,1) | |

## Timing
| Parametro | Valore |
|-----------|--------|
| `sim_dt` | 0.002 s → fisica 500 Hz |
| `ctrl_dt` | 0.01 s → policy 100 Hz |
| substeps | 5 |
| `action_repeat` | 1 |
| `episode_length` | 1000 → ~10 s |

## Reward (`reward_config`)
| Termine | Scala | Formula grezza | Ottimo |
|---------|------:|----------------|--------|
| tracking_lin_vel | +1.0 | `exp(−(vx_cmd−vx)²/σ)` | +1.0 |
| tracking_ang_vel | +0.5 | `exp(−(yaw_cmd−ωz)²/σ)` | +0.5 |
| orientation | −2.0 | `‖g_xy‖²` | 0 |
| ang_vel_xy | −0.3 | `ωx²+ωy²` | 0 |
| base_height | −1.0 | `1 − exp(−((h−0.4)/0.05)²)` | 0 |
| posture | −5.0 | `Σ w·(q−q_def)²·gate` | 0 |
| torques | −1e−4 | `Σ τ²` | →0 |
| action_rate | −0.01 | `Σ(a−a_prev)²` | →0 |
| dof_pos_limits | −1.0 | hinge soft-limit | 0 |
| termination | −5.0 | `done` | 0 |

| Param reward | Valore |
|--------------|--------|
| `tracking_sigma` | 0.25 |
| `base_height_target` | 0.4 m (CoM) |
| `posture_cmd_sigma` | 0.25 |
| `posture_weights` | `[1.0, 0.5, 0.5, 1.0, 0.5, 0.5]` |
| `only_positive_rewards` | False |
| Aggregazione | `clip(Σ scale·term · dt, ±1e4)` |

## Reset / rumore / perturbazioni
| Voce | Stato | Valore |
|------|-------|--------|
| rand. posizione xy | ATTIVA | `x,y += U(±0.5)` |
| rand. yaw | ATTIVA | `yaw ~ U(±π)` |
| rand. vx, vy | ATTIVE | `vx ~ U(±0.2)`, `vy ~ U(±0.1)` |
| rumore obs (`noise_config.level`) | INATTIVO | 0.0 (scale ininfluenti) |
| perturbazioni (`pert_config.enable`) | INATTIVE | False |
| scale rumore (dormienti) | — | joint_pos 0.01 · joint_vel 1.5 · gyro 0.2 · gravity 0.05 · linvel 0.1 |

## Comando
| Parametro | Valore | Note |
|-----------|--------|------|
| dim | 2 | `[vx, yaw_rate]` |
| `a` (ampiezza) | `[0.0, 0.0]` | ⇒ comando sempre `[0,0]` |
| `b` | `[0.75, 0.75]` | inerte (a=0) |
| `p_stand` | 0.2 | |

## Osservazioni
| Input | Chiave | Dim | Nota altezza |
|-------|--------|----:|--------------|
| Actor | `state` | 34 | `current_com_height` (z CoM grezza) |
| Critic | `privileged_state` | 85 | `current_com_height` (commento inline "84" obsoleto) |

## PPO (`PPO_PARAMS`, chiamata attiva a `ppo.train`)
| Parametro | Valore |
|-----------|--------|
| num_timesteps | 100_000_000 |
| num_evals | 10 |
| episode_length | 1000 |
| num_envs | 4096 (`NUM_ENVS`) |
| batch_size | 512 |
| unroll_length | 20 |
| num_minibatches | 32 |
| num_updates_per_batch | 4 |
| **discounting** | **0.99** |
| learning_rate | 3e-4 |
| entropy_cost | 1e-2 |
| reward_scaling | 1.0 |
| max_grad_norm | 1.0 |
| normalize_observations | True |
| action_repeat | 1 |
| num_resets_per_eval | 10 |
| seed | 0 |
| deterministic_eval | False |

## Rete (realmente attiva in PPO)
| Voce | Valore |
|------|--------|
| hidden layer policy | (512, 256, 128) — attivo |
| hidden layer value | **default Brax** (il (512,256,128) nel `network_factory` è morto) |
| distribuzione | `tanh_normal` (output 16) |
| normalizzazione obs | `running_statistics.normalize` (ON) |
| attivazione | default Brax (ELU commentata) |
| inizializzazione | default Brax |
| policy_obs_key / value_obs_key | `state` / `privileged_state` (da default `make_ppo_networks`) |
| `INIT_STD = 0.03` | **NON attivo in PPO** (solo SAC) |
| `ZERO_INIT_OUTPUT_LAYER` | **NON attivo in PPO** (solo SAC) |
| ELU custom | **NON attiva in PPO** (solo SAC) |
| `PPO_PARAMS["network_factory"]` | **morto** (sovrascritto da `selected_network_factory`) |

## Risultati
| Stage | Reward | Comportamento |
|-------|-------:|---------------|
| Iniziale (8192/256/γ0.97) | ~3–4 | stazione fallita |
| Batch + osservazione (4096/512 + CoM) | ~6 | stazione breve |
| `discounting=0.99` | ~13 @ ~13M | stazione stabile |
| Training successivo | ~14.5 @ ~26M | stazione forte |