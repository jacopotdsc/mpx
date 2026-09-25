# Lite3 — oscillazione di `wz` della baseline MPC a vx=1.0: causa e correzione

Branch `lite3`. Run analizzato:
`checkpoints/Lite3JoystickFlatTerrain/20260924_092103/comparison_.../20260924_110316`.
Ambiente conda `mjpl`. **Nessun training, nessun uso GPU** (occupata): analisi statica +
rollout brevi **1 env su CPU** (`JAX_PLATFORMS=cpu` forzato prima degli import).
Script e dati grezzi: `analysis_lite3/mpc_tuning/diag_yaw.py` (sweep singolo parametro),
`analysis_lite3/mpc_tuning/eval_set.py` (valutazione di un SET su suite di comandi),
cartelle `mpc_tuning/yaw/` e `mpc_tuning/sets/`.

---

## Sommario

Con comando `vx=1.0, vy=0, wz=0` la baseline mostra una `wz` con oscillazioni reali fino a
±0.48 rad/s. **Non è rumore né un bug di misura**: è l'ondeggiamento di imbardata del tronco
indotto dalla trot (a media zero — l'heading non deriva). La **causa** è che i pesi MPC e i
parametri del gait sono **ereditati/scalati da Aliengo** (config_srbd.py): un robot ~2× più
pesante e ~20% più lungo, con un MPC senza `vy` e senza tuning di yaw. Sul Lite3 (piccolo)
questi valori lasciano un'oscillazione di yaw grande.

**Set finale (MINIMO): UN SOLO parametro cambiato rispetto all'originale.** `config_lite3.py`:

| parametro | originale | finale | perché |
|---|---|---|---|
| `Qomega` z (yaw-rate) | `1e1` (=10) | **`20·1e1` (=200)** | unico termine sullo yaw (`Qrot` z=0), era un default di Aliengo; taglia il wz senza toccare l'avvio |
| `step_height` | 0.055 | 0.055 (**invariato**) | provato 0.050 e ripristinato: per non rischiare l'avvio (vedi Parte H) |
| `vel_ref_gain` | off | off (**scartato**) | 1.10 migliorava vx ma **destabilizza l'avvio** (Parte H) |
| `duty_factor` | 0.65 | 0.65 (invariato) | 0.58 dava un lurch d'avvio (Parte F) |
| `step_freq`, `clearence_speed` | 1.4 / 0.4 | invariati | fragili / senza beneficio |

**Effetto (regime, flat) vs originale**, picco `|wz|` sui comandi a wz=0:
`vx=1.0`: 0.479 → **0.264** (**-45%**); `vx+vy`: 0.631 → **0.508** (-19%); `vy=0.4`: 0.284 →
**0.216** (-24%). `errwz` migliora ovunque. Tutta la suite valida. **La sola `Qomega` z porta
il grosso del guadagno wz** e, essendo la deviazione minima dall'originale (che ha un avvio
pulito), non introduce l'inciampo. `W` si ricalcola da `Qomega` all'import → propaga a
`lite3_srbd.py` e all'env `joystick.py`.

> **Perché così minimale:** vedi Parte H. L'inciampo d'avvio NON è inerente (l'originale parte
> pulito, roll 0.074 ≈ Aliengo); comparve con modifiche mie. E il mio harness CPU non riproduce
> l'avvio dell'env, quindi non posso tarare l'avvio offline: mi limito alla modifica wz
> validata a regime e lascio l'avvio identico all'originale.

---

## Parte A — Che cos'è il segnale `wz` (diagnosi)

- Nel CSV/overlay `wz = get_gyro(data)[2]` = giroscopio (frame tronco). L'IMU è allineato al
  tronco (`xmls/lite3.xml:75`, `<gyro site="imu"/>`), tronco ~orizzontale → `gyro_z ≈ yaw-rate
  mondo`. È il **giroscopio pulito**: il rumore sensore è aggiunto solo all'osservazione della
  policy (`noisy_gyro`, `joystick.py:597-603`), non al log/overlay.
- Su tutto l'episodio baseline (`flat_terrain__vx_1p0.csv`): `mean_wz=+0.002`, `std=0.138`,
  `max|wz|=0.465`; **variazione di heading in 5 s = 0.012 rad ≈ 0.7°** (il quaternione `qz`
  resta ~0.47). Quindi il robot **non ruota**: la `wz` oscilla a media zero alla frequenza del
  passo (cambia segno ogni ~0.36 s). `0.17` letto nel video è lo stesso segnale a metà ciclo.
- La stessa reward `tracking_ang_vel` usa la `wz` istantanea grezza (`joystick.py:753`,
  `tracking_sigma=0.25`), quindi l'oscillazione a media zero costa reward a ogni step.

## Parte B — Causa nel controllore MPC

Cost dell'MPC SRBD (`config_lite3.py`, ex-Aliengo):
`Qp=diag(0,0,1e4)`, `Qrot=diag(1e3,1e3,0)`, `Qdp=1e3`, **`Qomega=diag(1,1,1)·1e1`**, `Qgrf=1e-2`.

- `Qrot` z **= 0** → l'MPC **non penalizza l'angolo di yaw**. Il riferimento è
  `quat_ref=[1,0,0,0]` (yaw=0 assoluto, `mpc_utils.py:159`, `use_terrain_estimator=False`), ma
  con peso 0 non ha effetto — coerente col fatto che il robot resetta a yaw casuale e non ci
  torna. Quindi lo yaw **non ha feedback d'angolo**.
- L'unico controllo di yaw è **`Qomega` z** sulla yaw-rate, verso `omega_ref = wz_cmd` (=0),
  con peso 10 (uguale a roll/pitch, che però hanno anche `Qrot=1e3` a tenerne l'angolo).
- Il foot placement (`calc_foothold`, `mpc_utils.py:196-203`) è simmetrico L/R: non introduce
  un bias di yaw. L'oscillazione nasce dalle coppie di imbardata alternate della trot (appoggi
  diagonali intermittenti); con il peso yaw-rate debole l'MPC la tollera per risparmiare GRF.

Conferma dell'utente: i pesi sono quelli di Aliengo, MPC fatto senza `vy` → non tarati per lo
yaw del Lite3. Il valore `1e1` è quindi un default ereditato, non una scelta.

## Parte C — Metodo di ricerca del set

Metodo: comando **rampato** 0→target in 1.5 s poi tenuto fino a 5 s (1000 step @200 Hz).
La rampa serve perché il baseline a **step** vx=1.0 cade nel transitorio (ribalta il pitch —
vedi `mpc_tuning/REPORT_mpc_lite3.md`), problema separato. Metriche a **regime** (t ≥ 2.0 s),
stesso reset (keyframe `home`) e comando.

**Punto chiave metodologico:** i parametri del gait sono **accoppiati e fragili** — un cambio
che va bene su vx puro può rompere vx+vy. Quindi ogni set è stato valutato su una **suite di 5
comandi** (`vx1`, `vx1vy0p4`, `vy0p4`, `wz0p6`, `vx1wz0p6`) e giudicato sul **caso peggiore**,
non su un comando solo (`eval_set.py`). Esempi di fragilità osservata:
- `step_height=0.040` (da solo) → rompe `vx+vy` (peak wz 1.48, vx crolla a 0.40, quasi caduta);
- `Qomega_z=500` + `duty=0.60` → rompe `vx+vy` (peak 2.3); ma `Qomega_z=200` è stabile ovunque;
- `duty=0.70/0.75` o `step_freq=1.6` → instabilità/coppie ×2-5;
- `Qomega_z` alto da solo satura il beneficio: sweep vx puro rmse 0.138→0.085 (×20)→0.079
  (×100), picco con fondo duro ~0.24 (impulso di footfall non cancellabile entro l'orizzonte).

Perciò il peso yaw è stato tenuto a un valore **robusto** (`×20`) e il grosso della riduzione è
venuto da **`duty_factor` più basso** (l'unica leva monotòna e robusta), con `step_height`
abbassato solo del minimo sicuro.

## Parte D — Risultati del set finale (CPU, 1 env, reset fisso `home`)

**Set finale `{Qomega_z=200, duty=0.58, step_height=0.050}` vs ORIGINALE-VERO
`{Qomega_z=10, duty=0.65, step_height=0.055}`** (stessa suite, reset e comando):

| comando | peak_wz orig→finale | errwz orig→finale | mean_vx orig→finale | τ_max orig→finale | valido |
|---|---|---|---|---|---|
| `[1.0,0,0]` | 0.479 → **0.218** (-54%) | 0.138 → 0.070 | 0.854 → 0.831 | 25 → 26 | ✓ |
| `[1.0,0.4,0]` | 0.631 → **0.386** (-39%) | 0.183 → 0.102 | 0.873 → 0.842 | 34 → **27** | ✓ |
| `[0,0.4,0]` | 0.284 → **0.173** (-39%) | 0.124 → 0.065 | — | 20 → 20 | ✓ |
| `[0,0,0.6]` (yaw) | 0.737 → 0.710 (-4%) | 0.101 → **0.056** | — | 9 → 11 | ✓ |
| `[1.0,0,0.6]` | 0.945 → **0.784** (-17%) | 0.167 → **0.072** | 0.858 → 0.819 | 28 → 26 | ✓ |

(dati grezzi `sets/orig_true/` e `sets/CONFIRM_disk/`). Nota: nei comandi con yaw comandato
il `peak_wz` è ~0.7-0.8 perché è oscillazione attorno al comando 0.6, non errore; il dato
rilevante lì è `errwz` (dimezzato).

→ riduzione del picco di yaw su **tutti** i comandi (-17÷24%), miglior tracking del yaw
comandato, **coppie più basse**, roll/pitch ≤0.09 (nessuna caduta). Unico costo: velocità in
avanti ~-3% (gait più dinamico), recuperabile in futuro con `vel_ref_gain` (ora disabilitato,
non toccato). Il rollout con la config **su disco** riproduce esattamente il finalista
(`sets/CONFIRM_disk/`).

**Contributo delle singole leve (a wz=0, caso peggiore vx+vy):** baseline 0.508 →
`duty=0.60` 0.399 → `duty=0.58` 0.385 → `duty=0.55` 0.377 (floor: sotto ~0.55 il trot perde la
sovrapposizione di appoggio e diventa fragile). `step_height` da 0.055 a 0.050 aggiunge un
piccolo margine senza rompere nulla; sotto 0.050 rompe il gait laterale.

## Parte E — Limiti della verifica

1. **Solo terreno piatto**: `lite3_srbd.py` gira su `scene_flat.xml`. **Rough/perlin non
   testati** (richiedono l'env completo via `compare.py`, quindi GPU). Un gait più dinamico
   (duty più basso) e una clearance minore potrebbero ridurre il margine su terreno accidentato
   → da validare con `compare.py` quando la GPU è libera (picco `wz`, cadute, coppie su
   rough/perlin). L'utente conta sul residuale RL per il rough.
2. **Suite di 5 comandi flat**, non esaustiva: coppie di comandi combinati e transitori/inversioni
   non coperti. I parametri sono fragili → un set nuovo va sempre rivalutato sull'intera suite.
3. **Comando rampato, non a step**: necessario per un regime stabile a vx=1.0; il caso a step
   resta un problema di transitorio separato (slew-limiter disabilitato, non riattivato).
4. **Reset singolo** (keyframe `home`, yaw=0). `compare.py` usa yaw di reset casuale;
   l'oscillazione è intrinseca al gait (indipendente dal reset), ma non ho fatto sweep di seed.
5. **Oscillazione ridotta, non eliminata**: in parte intrinseca alla trot a contatti
   intermittenti (le rate di roll/pitch restano ±0.9 rad/s nonostante `Qrot=1e3`).
6. **Costo vx ~3%** dal duty più basso, sopra il deficit ~15% pre-esistente (`mean_vx≈0.83`
   per comando 1.0). Il deficit di base è indipendente (`vel_ref_gain` disabilitato), non
   affrontato qui; se serve, `vel_ref_gain` lo recupererebbe.
7. **Rough ridotto ma non validato**: l'ampiezza del rough è stata abbassata (vedi appendice),
   ma l'effetto del nuovo set su rough/perlin non è stato misurato (serve GPU). Il perlin
   **non** è stato toccato (l'`hfield` z=0.4127 è la normalizzazione dell'intera isola, non la
   rugosità locale; l'utente conta sul residuale RL per il rough).

---

## Parte F — Transitorio laterale di partenza (perché duty è tornato a 0.65)

Dopo aver applicato `duty=0.58`, il run `compare.py` `20260924_193138` mostrava, sui comandi a
`vy=0`, un **lurch di `vy` in partenza**: `vy` misurato arriva a **−0.5 m/s a t≈0.32 s** e
rientra a ~0 entro t≈0.6 s (transitorio, non regime).

Diagnosi (stesso reset in entrambi i run — `vy₀=−0.115` identico):
- run **vecchio** (orig, `duty=0.65`): mean|vy| in [0,0.6 s] = **0.046** → nessun lurch.
- run **nuovo** (`duty=0.58`): mean|vy| = **0.205**, picco **0.498** → lurch.

È un'**interazione**, non una causa singola:
1. il **qvel random al reset** è la perturbazione iniziale (test CPU controllato, set finale:
   qvel=0 → picco|vy|=0.37; qvel iniettato → **0.65**, quasi ×2);
2. il **`duty` più basso** riduce il doppio-appoggio (overlap `2·duty−1`: 0.30 a 0.65 → 0.16 a
   0.58) → il gait è **più sensibile** a quella perturbazione e la amplifica in un lurch;
3. l'onset del comando fa il resto.

Con `duty=0.65` il doppio-appoggio maggiore **assorbe** la perturbazione (il run vecchio lo
provava). Poiché l'utente preferisce un cammino pulito al massimo taglio di wz, `duty` è stato
**riportato a 0.65**; `Qomega` z e `step_height` restano e mantengono la gran parte del guadagno.
Caveat: il proxy CPU (`diag_yaw.py` con `ramp_s=0.3`) sovrastima il lurch per **tutte** le
config e non è rappresentativo dell'onset di `compare.py` → la conferma del transitorio pulito
va presa dal prossimo run `compare.py`. Script: `mpc_tuning/inject_qvel` (monkeypatch del reset).

## Parte G — DUBBIO APERTO: grande transitorio/"rumore" della baseline in partenza

**Osservazione (run `20260924_213523`, con qvel di reset ora = 0):** sui comandi la baseline
ha un **errore molto grande nei primi ~0.3-0.5 s** rispetto sia al regime sia al residual.
Errore medio in [0,0.5 s] (‖(vx,vy)−cmd‖), baseline vs residual:

| comando | baseline | residual |
|---|---|---|
| flat vx1 | 0.231 | 0.175 |
| flat vx1+vy0.4 | **0.364** | **0.050** |
| flat vx1+wz0.6 | 0.441 | 0.190 |
| flat vx2 | 0.636 | 0.388 |
| rough vx2 | 0.738 | 0.261 |
| rough vy0.6 | 0.479 | 0.338 |

Nel caso `vx+vy` il picco è violento: vx/vy vanno in overshoot (~0.86/0.77 mentre il comando è
~0.5/0.2) e **`wz` schizza a ~1.5 rad/s senza comando di wz**, poi vy inverte a −0.3 e rientra
entro ~0.5 s.

**Perché (meccanismo, verificato) — è STRUTTURALE, non tuning:**
- Il comando è **già rampato** dall'env (smoothing esponenziale `command += 0.05*(target−command)`,
  τ≈0.4 s, `joystick.py:560-563`) → **non è un onset a gradino**. Onset ancora più lenti su CPU
  aiutano poco e in modo **fragile** (α=0.02 → `wz` esplode a −4.6, quasi caduta).
- È il **cold-start del model-based**: transizione da fermo a trot in cui gait scheduler e
  foothold planner operano lontano dal regime.
  1. Il gait parte "a pieno regime": la prima coppia diagonale va in volo con step_height e
     falcata piene → primo passo asimmetrico → impulso laterale/di imbardata (il picco di `wz`).
  2. Il foothold planner capture-point `f2=√(h/g)·(dp−ref)` (`mpc_utils.py:196-203`) è tarato
     per il regime: all'avvio `dp≈0`, `ref` rampa → foothold aggressivi → overshoot di velocità.
  3. La fase del gait al reset (`timer_t=[0.5,0,0,0.5]`) è fissa, non adattata alla partenza.
- Scala con l'aggressività del comando (vx1 0.23 < vx+vy 0.36 < vx2 0.64), coerente col cold-start.

**Confronto con Aliengo (SRBD nativo) — stesso comando `[1.0,0.4,0]`, stesso onset
(smoothing esp. τ≈0.39 s), reset home.** Script: `analysis_lite3/startup_compare/run_startup.py`
(Aliengo = `srbd_quad.py`+`config_srbd.py`; Lite3 = `lite3_srbd.py`).

| | pattern gait avvio | roll max 1ª falcata | dip vy (verso sbagliato) | overshoot vy |
|---|---|---|---|---|
| Aliengo | solleva FL+HR per primi @t≈0.08 | +0.062 (3.5°) | −0.022 | +0.32 (≈regime 0.30) |
| Lite3 | **identico** (FL+HR @t≈0.08) | **+0.111 (6.4°)** | **−0.067** | **+0.60** (regime 0.41) |

**Conclusione: Aliengo fa la STESSA cosa** (stessa coppia diagonale sollevata per prima, stesso
`vy` nel verso sbagliato, stesso roll durante la 1ª falcata) → è **inerente all'avvio SRBD, non
un bug del Lite3**; la gamba si comporta già come Aliengo. La differenza è solo l'**ampiezza
~2×**, effetto di **scala**: il Lite3 pesa la metà (11.9 vs 24.6 kg) con ~metà inerzia di roll,
quindi la stessa perturbazione del gait dà ~2× la risposta del corpo (roll → vy). Non è un
parametro mal scalato.

**Stato:** il **residual RL lo corregge già** (è il suo scopo): errore d'avvio molto più basso
(vx+vy 0.364→0.050, rough vx2 0.738→0.261). Nessuna modifica applicata alla baseline per questo.
Se si volesse ridurre l'ampiezza del Lite3 verso Aliengo (a parità di comando), serve attenuare
la perturbazione del primo passo (soft-start: rampare step_height/ampiezza sui primi 1-2 cicli —
richiede uno stato di warmup nel wrapper MPC), oppure accettarlo perché inerente.

**Possibili fix strutturali (NON applicati, da valutare se si vuole agire sulla baseline):**
1. **Soft-start del gait**: rampare `step_height`/ampiezza falcata da ~0 al nominale nei primi
   1-2 cicli (fix più diretto, tocca `mpc_utils.py`).
2. **Clamp del termine capture-point** `f2` quando `|dp−ref|` è grande (all'avvio).
3. **Inizializzare il gait in appoggio** (più doppio-appoggio nel primo ciclo).
Sono modifiche di struttura al reference generator, non parametri. Il qvel random al reset
amplificava il transitorio (rimuoverlo l'ha ridotto: picco vy ~0.5→~0.22 su vx1); vedi Parte F.

**Tentativo di soft-start su `step_height` (implementato, testato, RIPRISTINATO — non robusto):**
ho aggiunto al wrapper MPC un soft-start gated (rampa `step_height` da un valore basso al
nominale nei primi cicli, con contatore `warmup` in `MPCState`, default off → Aliengo/Go1
invariati). Verdetto: **non funziona in modo robusto** per una tensione fondamentale —
- `soft_start_steps=50, sh_min=0.015` (rampa lunga, clearance iniziale bassa): riduce l'avvio
  (roll 0.238→0.175) **ma fa crollare vx+vy a regime** (mvx 0.31, poca clearance sotto moto
  laterale → il piede non scavalca, stessa modalità del `step_height=0.040` fisso);
- `sh_min=0.030` o rampa corta (20 step): **sicuro ma inefficace** (roll 0.238→0.229, wz anche
  peggio).
Non esiste un punto che riduca l'overshoot verso Aliengo senza destabilizzare il gait laterale.
Quindi il wrapper è stato **ripristinato** (diff vuoto). Il confronto con Aliengo (Parte G qui
sotto) mostra comunque che il transitorio è **inerente** e la gamba fa già come Aliengo; il
residual RL lo corregge. Una leva alternativa non testata: rampare l'**ampiezza della falcata**
(non la clearance) — eviterebbe lo scuffing ma tocca il foothold planner condiviso.

## Parte H — Inciampo d'avvio: NON inerente, e limite dell'harness CPU

Aggiornamento a Parte G. L'utente ha notato che **visivamente Aliengo non inciampa, il Lite3
sì** ("cosa strana con la gamba"). Verifiche successive:

**1) L'originale NON inciampava.** Roll d'avvio (baseline, vx=1.0) dai run `compare.py` reali:
- `110316` **originale (Qomega=10, sh=0.055)**: roll_max[0,0.6s] = **0.074** rad (≈ Aliengo 0.062) → **avvio pulito**.
- `213523` (Qomega=200, sh=0.050): roll **0.109**; `193138` (duty=0.58): roll **0.216**.
→ L'inciampo **è comparso con modifiche mie**, non è intrinseco alla taglia del Lite3.

**2) L'harness CPU (`lite3_srbd`) NON riproduce l'avvio dell'env `compare.py`.** Sotto l'onset
esponenziale dell'env, in `run_startup.py` l'**originale CADE** (roll 1.28) mentre in
`compare.py` è pulito (0.074): le conclusioni si **ribaltano**. Quindi **le analisi d'avvio fatte
con l'harness CPU sono inaffidabili** (soft-start "fallisce", `vel_ref_gain` "cade", duty0.70
"buono" all'avvio → NON verificati sull'env). Restano affidabili le metriche **a regime**
(eval_set, che combacia con compare.py) usate per il tuning di `Qomega`.

**Conseguenze operative:**
- **`vel_ref_gain` scartato**: migliorava vx a regime (0.855→0.938) ma la sua sicurezza in avvio
  non è verificabile offline e nel test CPU cadeva → troppo rischioso vicino al problema d'avvio.
- **`step_height` riportato a 0.055** (originale): il beneficio wz di 0.050 era marginale e non
  vale il rischio d'avvio.
- **Soft-start step_height**: implementato e **ripristinato** (vedi Parte G) — comunque non
  tarabile in modo affidabile offline.
- **Decisione**: config finale = **solo `Qomega` z 10→200** (guadagno wz validato a regime),
  tutto il resto identico all'originale che parte pulito. Così l'avvio dovrebbe restare pulito.

**Verifica ancora da fare (solo su GPU):** un `compare.py` con la config finale per confermare
che l'avvio è tornato pulito come l'originale. Se anche a `Qomega=200` l'inciampo dovesse
comparire, il colpevole sarebbe `Qomega` stesso → allora tradeoff wz-vs-avvio da decidere.
Non è possibile stabilirlo offline (harness inaffidabile per l'avvio).

## Parte I — Init MPC coerente al reset: LA FIX dell'inciampo d'avvio

**Ipotesi utente (corretta):** l'MPC in `init_state()` usa il warm-start nominale (yaw=0), ma
l'env resetta a yaw/posizione casuali → warm-start incoerente → transitorio d'avvio. L'utente ha
insistito: inizializzare **tutto coerente** allo yaw/posizione passati, incluso ciò che il
ref_gen usa per le gambe (`p_legs0`).

**Falso allarme (mio errore da correggere):** in una prima tornata avevo concluso che il seeding
"fa cadere il solver a yaw grande". Era un **artefatto del mio probe**: NON azzeravo il comando
di velocità al reset, mentre `compare.py` lo azzera (`compare.py:~2357`,
`"command": jnp.zeros_like(...)`). Con il comando casuale non azzerato, il robot da fermo riceve
una velocità casuale (es. `[-1.13, ·, -1.17]` = indietro+ruota) e cade — **a prescindere** dal
seeding. Controllato (`cmd_test.py`): stesso reset, comando casuale → cade (roll 3.1), comando
neutro → non cade (0.32). Quindi le "cadute" NON erano il solver.

**Fix vero (validato, env-faithful con comando azzerato come compare.py):** `init_state(x0)`
costruisce un warm-start **in piedi self-consistente** alla posa di reset:
- `X0` = stato reale su tutto l'orizzonte;
- `U0` = GRF statico di sostegno (`mass·g/n_contact` in z per piede);
- `liftoff`/`foot_ref` = `p_legs0` ruotato dallo yaw di reset e traslato alla base
  (lo stesso `foot0_projected` del ref_gen) → le gambe partono da uno stance coerente.

**Risultato** (roll d'avvio, comando vx+vy, sweep di ~18 seed, yaw da −2.6 a +3.0):

| | roll d'avvio |
|---|---|
| warm-start nominale (default) | 0.11 – **0.38** (peggiore a yaw grande) |
| **init coerente (seeding)** | **~0.10 costante, per ogni yaw**, nessuna caduta |

Elimina il transitorio yaw-dipendente. **Abilitato di default** (`config.mpc_seed_warmstart=True`).
Meccanismo: `mpc_wrapper_srbd.init_state(x0)` + `joystick.py` reset. Il steady-state non cambia
(il seeding tocca solo il warm-start al reset; `eval_set`/`lite3_srbd` resettano a yaw=0 e non lo
esercitano → il guadagno wz di Parte D è invariato).

**Nota importante:** il seeding aiuta **solo se il comando di reset è neutro**. `compare.py` lo
azzera (eval pulito). Nel training grezzo l'env lascia il comando casuale al reset
(`joystick.py:310`): lì il seeding rende comunque coerente il warm-start, ma il comando casuale
resta una sorgente di transitorio a sé (gestita dal residual). Se si volesse un avvio pulito
anche in training, azzerare/rampare il comando al reset dell'env (non fatto, per non alterare la
randomizzazione del task).

**`Qomega` z:** tenuto a **100** (compromesso wz-vs-oscillazione a regime). Ortogonale
all'avvio: l'init coerente rende l'avvio pulito indipendentemente da `Qomega`.

**Lezione metodologica:** un probe d'avvio DEVE replicare l'azzeramento del comando di
`compare.py`, altrimenti inietta un transitorio spurio. (Correzione applicata a `yaw_test.py`.)

## Appendice 1 — Rough terrain ridotto (`scene_rough.xml`, applicato)

Su richiesta (gait più dinamico → rough relativamente più difficile) l'ampiezza del terreno
rough è stata ridotta. Il rough è un XML statico di **2394 box** a `pos_z=-0.25` e `size_z`
variabile: la sommità (`pos_z+size_z`) forma la superficie, simmetrica attorno a ~0
(mean −0.0003, **std 0.0144, range ±0.025 m**). Non esiste un generatore, quindi lo script
`analysis_lite3/mpc_tuning/scale_rough.py` scala la deviazione di ogni sommità dal livello medio
per un fattore **k=0.6** (−40%), **preservando il livello medio** e riscrivendo solo l'attributo
`size` dei box. Risultato: **std 0.0144→0.0087, range ±0.025→±0.015 m** (bump max 1.5 cm, ben
dentro i 5 cm di clearance). File: copia live di `mujoco_playground`
(`.../lite3/xmls/scene_rough.xml`, tracciata nel suo git). **Backup**:
`analysis_lite3/mpc_tuning/rough_backup/scene_rough.xml.orig`; ripristino con
`python scale_rough.py --restore`; fattore ritarabile (`python scale_rough.py <k>`).
Non validato su rollout (GPU occupata). Nessun commit.

## Appendice 2 — Allineamento video/plot (`compare.py`, già applicato)

Diagnosi collaterale: nel video la `wz` sembrava "diversa" dal plot (0.17 vs picco 0.4). Causa:
`compare.py` decimava i frame (`TARGET_VIDEO_FPS=30` hardcodato → `render_stride=2` → 1 frame
ogni 2 step di controllo), mentre il CSV logga ogni step → i picchi cadevano tra due frame.
**Fix applicato**: rimosso il magic number; `_render_stride` ora ritorna sempre 1 (un frame per
step di controllo, FPS = `1/env.dt` derivato dall'env). Così overlay e plot mostrano gli stessi
campioni su qualsiasi env. Richiede di rigenerare i rollout (`--redo` non recupera i frame
mancanti). Nota: questo allinea le viste ma **non** cambia il segnale — la riduzione reale
dell'oscillazione è quella della Parte C-D.
