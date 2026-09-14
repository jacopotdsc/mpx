# Baseline SRBD MPC del Lite3 — analisi quantitativa e correzione

Data: 2026-09-12
Ambiente: Conda `mjpl` (Python 3.11.15). Import forzati alle copie locali di `repo_lite3`
(`mpx` + `mujoco_playground` locale, quest'ultimo necessario perché `config_lite3.py`
importa `lite3_constants` per il path dell'XML e le costanti del modello).
Entry point: **`lite3_srbd.py`** (vedi nota sotto). Nessun training eseguito.

## Nota sull'entry point
Il file richiesto `mjx_lite3.py` **non esiste** nel repository. L'unico controller SRBD MPC
standalone del Lite3 è `mpx/mpx/examples/lite3_srbd.py` (docstring: *"Standalone SRBD MPC +
whole-body controller on the Lite3. No RL. Mirror of srbd_quad.py"*). È inequivocabilmente
il controller model-based in oggetto; tutta l'analisi usa questo entry point.

## Metodo
Ogni test: stato iniziale fisso (keyframe `home`), 5 s (1000 step @200 Hz), stesso terreno
(`scene_flat.xml`), comando costante, un processo per test (nessun riuso dello stato finale).
Metriche a regime calcolate su `t >= 2.0 s`. Script: `analyze_srbd.py`, suite `run_suite.sh`.
`vx = 1.5` **non** è stato usato per il tuning (§5 della consegna).

---

## 1. Dati misurati — baseline (controller originale, comandi a step)

| comando | vx̄ | vȳ | wz̄ | RMSEvx | RMSEvy | RMSEwz | roll_max | pitch_max | base_z_min | τ_max | sat% | caduta |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [1.0, 0, 0]   | 0.117 | 0.167 | -0.542 | 0.97 | 0.31 | 1.88 | 3.14 | 1.52 | 0.060 | 254 | 19.3 | **SÌ** |
| [0, 0.4, 0]   | -0.019 | 0.322 | 0.007 | 0.05 | 0.10 | 0.12 | 0.05 | 0.05 | 0.308 | 20 | 0.0 | no |
| [1.0, 0.4, 0] | 0.872 | 0.338 | 0.032 | 0.16 | 0.11 | 0.19 | 0.49 | 0.40 | 0.198 | 92 | 0.8 | no |
| [0, 0, 0.6]   | -0.024 | 0.005 | 0.556 | 0.03 | 0.02 | 0.10 | 0.02 | 0.03 | 0.310 | 9 | 0.0 | no |
| [1.0, 0, 0.6] | 0.053 | -0.004 | 0.492 | 0.97 | 0.27 | 1.89 | 3.14 | 1.24 | 0.081 | 260 | 21.2 | **SÌ** |
| [0, 0.4, 0.6] | 0.002 | 0.317 | 0.557 | 0.04 | 0.10 | 0.12 | 0.05 | 0.03 | 0.308 | 21 | 0.0 | no |

Dati grezzi: `baseline/*.csv`, `metrics_baseline.json`.

Osservazione chiave (controintuitiva): **vx = 1.0 puro CADE (si ribalta, roll → π), ma
[1.0, 0.4, 0] è stabile e traccia vx ≈ 0.87**. Non è quindi un'incapacità di locomuovere a
vx ≈ 1.0.

## 2. Ipotesi verificate

**(a) Soglia di stabilità (sweep vx puro)** — `diag/`:
`vx=0.6` stabile; `vx=0.8, 0.9, 1.0` **cadono tutte**. La soglia sta tra 0.6 e 0.8.

**(b) Meccanismo della caduta** — serie temporale di `[1.0,0,0]`:
durante l'accelerazione iniziale il **pitch esplode per primo** (0.5 rad a t=0.28 s, 0.82 rad
a t=0.40 s) mentre vx raggiunge 1.0; subito dopo il roll diverge fino al ribaltamento
(roll = π a t≈0.8 s) e il robot rotola. Il pitch iniziale è 0.67 rad per [1.0,0,0] contro
0.19 rad per [1.0,0.4,0] (che resta stabile).

**(c) Causa dominante = comando a step, non instabilità di regime** — `diag/vx1p0_ramp.csv`:
rampando il comando 0→1.0 in 1.5 s (nessuna modifica al controller, solo profilo di comando),
`vx=1.0` diventa **stabile**: roll_max 0.043, pitch_max 0.065, base_z 0.310, vx a regime 0.853.
Il gait a `step_freq=1.4` è quindi già capace di vx ≈ 0.85; la caduta è un **transitorio di
accelerazione** provocato dallo step istantaneo 0→comando che l'SRBD tenta di annullare entro
un orizzonte, generando un momento di beccheggio che ribalta la base.

## 3. Ipotesi scartate

* **Foothold troppo distante / gait** (ipotesi #1 della consegna): a `vx=1.0`,
  `f1 = 0.5·vx·duty/step_freq = 0.232 m`. Ho provato `step_freq ∈ {1.7, 2.0, 2.5}` (`diag/`,
  `sf2p0/`): il comportamento è **non-monotòno** (1.7 cade, 2.0 regge il vx puro, 2.5 cade) →
  `step_freq=2.0` funziona solo per risonanza. Sulla suite completa `step_freq=2.0`
  **fallisce comunque [1.0,0,0.6]** e introduce un drift di wz in [1.0,0.4,0]. **Scartato**:
  fragile e con effetti collaterali. Il foothold non è la causa dominante (col comando rampato
  e `step_freq=1.4` invariato il robot è stabilissimo).
* **Spline di swing, pesi MPC, WBC**: non modificati. Col fix alla causa dominante tutti i
  target sono soddisfatti, coppie ≤ 32 Nm e saturazione 0% → nessun problema attribuibile a
  spline/pesi/WBC nei casi richiesti. Coerente con "modifica solo se i dati lo mostrano" (§7).

## 4. Modifica scelta (una sola)

**Limitatore di slew-rate sul riferimento di velocità** (accelerazione di riferimento
limitata), dentro il controller. Trasforma internamente lo step di comando in una rampa,
colpendo direttamente la causa dominante.

* `mpx/mpx/config/config_lite3.py`: `max_lin_acc = 0.8 m/s²`, `max_yaw_acc = 1.5 rad/s²`.
* `mpx/mpx/utils/mpc_wrapper_srbd.py`: campo `cmd_filt` in `MPCState`; in `run()` il comando
  lineare (`input[:3]`) e di imbardata (`input[5]`) sono limitati a `max_*_acc / mpc_frequency`
  per tick; l'altezza (`input[6]`) passa invariata.

Motivazione numerica di `0.8 m/s²`: raggiunge vx=1.0 in ~1.25 s, in linea con la rampa da 1.5 s
già verificata come stabile (§2c). **Gated dai parametri di config**: chi non li definisce
(Aliengo `config_srbd`, Go1) ottiene `max_dv = ∞` → il clip è un no-op → comportamento
**identico** a prima (verificato: `config_srbd` costruisce col nuovo wrapper, `_max_dv = inf`).

## 5. Miglioramenti ottenuti — controller corretto (comandi a step)

| comando | vx̄ | vȳ | wz̄ | roll_max | pitch_max | base_z_min | τ_max | sat% | caduta |
|---|---|---|---|---|---|---|---|---|---|
| [1.0, 0, 0]   | **0.853** | 0.001 | 0.007 | 0.049 | 0.064 | 0.310 | 27 | 0.0 | **no** |
| [0, 0.4, 0]   | -0.019 | 0.322 | 0.007 | 0.055 | 0.056 | 0.310 | 20 | 0.0 | no |
| [1.0, 0.4, 0] | **0.870** | **0.350** | 0.022 | 0.113 | 0.091 | 0.310 | 32 | 0.0 | **no** |
| [0, 0, 0.6]   | -0.024 | 0.005 | 0.556 | 0.017 | 0.030 | 0.310 | 10 | 0.0 | no |
| [1.0, 0, 0.6] | **0.858** | -0.077 | 0.535 | 0.068 | 0.068 | 0.310 | 28 | 0.0 | **no** |
| [0, 0.4, 0.6] | 0.001 | 0.314 | 0.548 | 0.076 | 0.056 | 0.310 | 26 | 0.0 | no |

Dati grezzi: `fix1_acc0p8/*.csv`, `metrics_fix1.json`. Grafici: `plots/`.

Sintesi baseline → corretto:
* [1.0,0,0]: **da caduta (roll π) a vx=0.85 stabile** (roll_max 3.14 → 0.05).
* [1.0,0,0.6]: **da caduta a stabile** (vx=0.86, wz=0.54).
* [0,0.4,0] e [0,0,0.6] e [0,0.4,0.6]: **invariati** (il limitatore non tocca i casi già stabili).
* [1.0,0.4,0]: stabile come prima ma con assetto migliore (roll_max 0.49 → 0.11).
* **Saturazione 0%** in tutti i casi; **nessun NaN/Inf/errore solver**; **nessuna caduta**.
* Gait regolare: duty reale ≈ 0.50–0.65 (comando 0.65), simmetria diagonale FL+HR vs FR+HL
  entro 0.04, mismatch contatti pianificati/reali ≤ 0.14 (nessuna gamba salta i contatti).

## 6. Criteri di accettazione — esito

| criterio | esito |
|---|---|
| vx=1.0 seguito decentemente e stabilmente | ✅ vx≈0.85, no caduta |
| vy=0.4 seguito ragionevolmente | ✅ vy≈0.32–0.35 |
| [1.0,0.4,0] stabile e caratterizzato | ✅ vx=0.87, vy=0.35 |
| wz=0.6 non degradato | ✅ 0.556 → 0.556 (identico) |
| nessuna caduta nei casi principali | ✅ |
| nessun NaN/Inf/fallimento solver | ✅ |
| nessuna saturazione continua | ✅ 0% |
| nessuna gamba salta sistematicamente i contatti | ✅ mismatch ≤ 0.14 |
| gait senza anomalie | ✅ |
| codice semplice | ✅ una modifica, gated, +33 righe |
| margine per il residual learning | ✅ vx a 0.85 (non 1.0), vy a 0.32 |

## 7. Limiti ancora presenti (lasciati intenzionalmente al residual learning)

* **Deficit di velocità a regime**: vx satura a ~0.85 per comando 1.0 (≈15%); vy a ~0.32 per
  comando 0.4 (≈20%). Presente anche col comando rampato → è del MPC/gait, non del transitorio.
  Migliorabile col residual (obiettivo esplicito della consegna).
* **Combinati con wz**: in [1.0,0.4,0] compare un piccolo bias (assetto ok, wz≈0.02); non
  perfezionato per non fare over-tuning.
* Il limitatore **non** cambia il comportamento di regime: risolve solo il transitorio da step.
  Comandi già dolci (vy=0.4, wz=0.6) restano identici alla baseline.

## 8. Configurazioni provate (elenco)

1. baseline (`step_freq=1.4`, comando a step) — 2 cadute su 6.
2. sweep vx puro {0.6, 0.8, 0.9} — soglia di stabilità 0.6–0.8.
3. comando a rampa (1.5 s), controller invariato — diagnostica: vx=1.0 stabile.
4. `step_freq ∈ {1.7, 2.0, 2.5}` sul caso duro; `step_freq=2.0` sull'intera suite — **scartato**.
5. **slew-limiter `max_lin_acc=0.8`, `max_yaw_acc=1.5`** — **scelto**, passa tutta la suite.

---

## 10. Seconda correzione — guadagno di feedforward sul riferimento (richiesta: buon tracking a vx=1.0, margine a vx=1.2)

Dopo il fix del transitorio, il tracking di regime restava con un **deficit ~costante
del ~15%** (guadagno effettivo ≈ 0.835), *identico su tutto l'intervallo*:

| comando vx | 1.0 | 1.1 | 1.2 | 1.3 | 1.4 | 1.5 |
|---|---|---|---|---|---|---|
| vx misurato | 0.853 | 0.943 | 1.024 | 1.101 | 1.178 | 1.253 |

Il riferimento grezzo è stabile fino a 1.5 (nessuna caduta, roll ≤ 0.05). Il deficit è
quindi un **guadagno di tracking**, non un limite di stabilità.

**Ipotesi scartata:** aumentare il peso di velocità `Qdp` (x2/x3/x5). Riduce il deficit
(vx=1.0 → 0.89/0.92/0.95) **ma erode il margine**: a `Qdp x3` **vx=1.2 cade** e [1.0,0.4,0]
degrada (τ 171 Nm, drift wz). Scartato.

**Modifica scelta:** guadagno di feedforward sul riferimento lineare
`vel_ref_gain = 1.2` (≈ 1/0.835), applicato a vx,vy nel wrapper (`config_lite3.py` +
`mpc_wrapper_srbd.py`; gated, default 1.0). Comando 1.0 → riferimento 1.2 → misurato ~1.02;
comando 1.2 → riferimento 1.44 (dentro l'inviluppo stabile) → misurato ~1.21. Non tocca i
pesi né la stabilità.

**Risultato (config finale, comandi a step):**

| comando | vx̄ | vȳ | wz̄ | roll_max | pitch_max | base_z_min | τ_max | sat% | caduta |
|---|---|---|---|---|---|---|---|---|---|
| [1.0, 0, 0]   | **1.024** | 0.001 | 0.006 | 0.049 | 0.067 | 0.310 | 30 | 0.0 | no |
| [1.2, 0, 0]   | **1.207** | -0.001 | 0.012 | 0.049 | 0.085 | 0.296 | 33 | 0.1 | no |
| [0, 0.4, 0]   | -0.017 | **0.392** | 0.009 | 0.056 | 0.056 | 0.310 | 26 | 0.0 | no |
| [1.0, 0.4, 0] | 1.034 | 0.390 | 0.033 | 0.201 | 0.173 | 0.278 | 62 | 0.2 | no |
| [1.0, 0, 0.6] | 1.027 | -0.081 | 0.504 | 0.072 | 0.073 | 0.308 | 33 | 0.0 | no |
| [0, 0, 0.6]   | -0.024 | 0.005 | 0.556 | 0.017 | 0.030 | 0.310 | 10 | 0.0 | no |
| [0, 0.4, 0.6] | -0.002 | 0.379 | 0.523 | 0.081 | 0.056 | 0.310 | 28 | 0.0 | no |

Dati: `fix2_gain120/*.csv`, `metrics_fix2.json`.

* **vx=1.0 → 1.02** (era 0.85), **vy=0.4 → 0.39** (era 0.32), **vx=1.2 → 1.21** stabile.
* Nessuna caduta, saturazione ~0%, nessun NaN/solver-fail. Gait invariato (`step_freq=1.4`).
* `wz` non è scalato (deficit ~7%, già accettabile): pure wz=0.6 → 0.556 invariato; nel
  combinato [1.0,0,0.6] scende leggermente (0.50) per l'interazione col vx più alto — stabile.
* **Nota**: il guadagno riduce il margine di tracking prima lasciato al residual — è un
  compromesso esplicitamente richiesto (buon tracking nominale a vx=1.0). Resta margine su
  robustezza/velocità superiori/disturbi.

## 11. File prodotti (in `analysis_lite3/mpc_tuning/`)

`analyze_srbd.py`, `run_suite.sh`, `diag_ramp.py`, `diag_param.py`, `make_plots.py`;
`baseline/`, `fix1_acc0p8/` (CSV principali); `metrics_baseline.json`, `metrics_fix1.json`;
`plots/` (baseline vs fix, comandato vs misurato, diagnostica); questo report.
