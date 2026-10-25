# Recap — tita_hybrid_plan (18/10 pomeriggio → 19/10 04:10)

Replica di **"Hybrid Balance Control for Wheeled Bipedal Robot via Residual RL Optimization"** (Zhang et al., ACIRS 2025) su Tita, con **MPC al posto dell'LQR**. In corso d'opera l'obiettivo è diventato un **modello robusto (che non cade) migliore di tita_v1** su rough, perlin e pendenza.
Dettagli completi: `MEMORY.md` (diario), `results.md`, `paper_notes.md`, `PLAN.md`.

---

## 1. Architettura finale (HRC su Tita)
**τ_C = τ_ζ + τ_π**, clippata a ±120 Nm.

| Parte | Cosa | Note |
|---|---|---|
| τ_ζ, controllore base | MPC DFCIP + WBC (`tau_ff`) + **PD di postura sulle gambe** verso q0 | thigh kp 85 / kd 12, knee 75 / 15 (guadagni del paper), abduzione 35 / 10; ruote solo `tau_ff` |
| τ_π, rete | coppia diretta su 6 motori (thigh, knee, ruota ×2) | abduzione esclusa come nel paper; τ_max [5, 5, 2] Nm (gambe/ruote) |
| Osservazione | stato del paper s_r, s_g, s_e, s_τ (65 dim) | θ_pd omesso come nel paper; in modalità joystick sono azzerati x_w, x_ref e start_yaw |
| Reward | kernel del paper f(e) = σ(e)σ(−e) − 0.25 su x, v, θ̇_p, z, roll, pitch, yaw + limiti dei giunti + termination −500 | pesi unitari (il paper non li dà) |

**Scoperte chiave:**
1. **Serve `jax_enable_x64`.** In float32 l'MPC/WBC cade in circa 1 s: non è un bug del controllore.
2. **Senza PD sulle gambe** bastano **1–2 Nm costanti** di residuo per far collassare il robot. Con il PD di postura HRC regge anche 8 Nm. Il PD stile `mjx_tita.py` (target ricalcolato dalla q attuale) **non** dà rigidezza.
3. **Il low-pass sull'azione non toglie il bias costante** (guadagno DC = 1).
4. **Riferimento di altezza dell'MPC relativo al contatto** (`mpc_utils.py`: z_com_ref = c_z + 0.4), necessario sulle pendenze.
5. **SAC** (algoritmo del paper) è risultato **instabile** in tutti i run. **PPO** (1024 env, come tita_v1 e il paper del MIT) funziona.

---

## 2. Replica fedele del paper (SAC, task "vai a D m")
| Run | Setup | Esito |
|---|---|---|
| A/B/C | SAC del paper, solo `tau_ff`, τ_max 8 o ridotto | **falliti**: cadono in 2–7 s (mean_action_l2 ≈ 0.45) |
| A3/A4/B4 | + burn-in / τ_max 8 | **falliti**: stessa causa (nessun PD sulle gambe) |
| B5 | pendenze, SAC del paper, PD di postura, τ_max [5, 5, 2] | non cade nell'eval finale, ma non è robusto (sotto) |
| B6 | B5 + burn-in + lr dell'actor 1e-4 | cade ancora a ogni eval, fermato a 900k |

**Validazione di B5** (RMSE x_w [m], ✗ = cade):
| Scenario | solo MPC | B5 |
|---|---|---|
| E1 0° | 0.047 | 0.181 |
| E1 5° | 0.476 | 0.617 ✗ |
| E1 10° | 1.013 | **0.484** (errore finale +1.80 → **−0.13 m**) |
| E2 spinte 10/20/30 N | 0.044 | 0.194 |
| E3 traiettoria poly5 | 0.057 | 1.54 ✗ |
| E3 multi-target | 0.215 | 0.301 |
| E5 rough (6 casi) | 1/6 cadute | 1/6 cadute |

**Lettura:** a 10° la rete corregge la deriva dell'MPC, il risultato chiave del paper, ma peggiora altrove. Cause:
- il kernel del reward è piatto vicino a 0 e contano solo gli errori grandi;
- la rete è cieca alla pendenza;
- nessun incentivo a restare a 0;
- i test sono fuori dalla distribuzione di training.

---

## 3. Modello robusto (PPO, comandi come tita_v1)
- **Comandi:** vx fino a ±2.5 (±2.0 interno), wz ±0.8; sulla pendenza solo lineari \|vx\| ≤ 1.
- **Episodi:** 10 s.
- **Perlin solo in valutazione.**

| Run | Terreno di training | Step | Durata dell'eval di training /1000 (solo MPC → fine) |
|---|---|---|---|
| RA → RA2 → **RA3** | rough sparso (50 box) | 3.3M + 9.8M + 9.8M | 542 → 729 → 856 → 905 → 932 → **969** |
| RS | rampe a corsie 10° (cadute dai bordi) | 3.3M | 665 → 748 |
| RS2 | piano inclinato 10°, comandi fino a 2.5 | 13M | 330 → 809 |
| **RS4** | piano inclinato 10°, solo lineari ≤ 1 | 9.8M (best 6.5M) | 834 → **898** |
| RS5 | RS4 + spawn corretto | 6.5M | 843 → 770 (peggiora) |
| R1 / R2 | scena mista con perlin / senza perlin | – | fermati: troppo lenti (l'heightfield del perlin raddoppia il costo) |

### Protocollo di valutazione (`eval_robust.py`)
- 8 comandi fissi × 5 spawn casuali (posizione e yaw), 10 s → **40 episodi** per terreno.
- Rough (2590 ciottoli, mai visto in training): 16 o 40 episodi.
- Pendenza: piano inclinato 10°, solo comandi lineari, spawn corretto per la pendenza.
- tita_v1 = run `20260918_124502` di `residual_tita`, nel suo codice originale.

### Cadute (metrica principale)
| Terreno | solo MPC | RS2 | RS4 | RA2 | **RA3** | tita_v1 | v1 con azione 0 |
|---|---|---|---|---|---|---|---|
| piano | 0/40 | 0/40 | 0/40 | 0/40 | **0/40** | 0/40 | 8/40 |
| rough (16 ep.) | 12/16 | 5/16 | 5/16 | 3/16 | – | 1/16 | 4/16 |
| **rough (40 ep.)** | **26/40** | – | – | – | **4/40** | **6/40** | – |
| **perlin** | 2/40 | 3/40 | 2/40 | 0/40 | **0/40** | **37/40** | 39/40 |
| **pendenza 10°** (spawn corretto) | **27/40** | – | **9/40** | – | **10/40** | – | – |

**Tempo medio alla caduta (rough):** RA3 4.7 s, solo MPC 3.8 s, tita_v1 6.0 s.
**Disturbo della rete** (‖τ_res‖ / ‖τ_MPC‖): RS2 0.05–0.07, RS4 0.10–0.15, RA2 0.11–0.18, **RA3 0.13–0.18**, tita_v1 0.16–0.20.

**Tracking sul piano** (\|err vx\| m/s / \|err wz\| rad/s):
| solo MPC | RA2 | RA3 | RS4 | tita_v1 |
|---|---|---|---|---|
| 0.032 / 0.025 | 0.080 / 0.099 | 0.085 / 0.096 | 0.137 / 0.103 | 0.051 / 0.030 |

### Conclusioni (fase 1)
- **RA3 è il miglior modello unico.**
  - Zero cadute su piano e perlin.
  - Rough 4/40 contro 26/40 del solo MPC (−85%) e 6/40 di tita_v1, differenza non significativa.
  - Pendenza 10/40 contro 27/40.
- **Perlin:** tita_v1 cade 37/40, noi quasi mai, grazie al riferimento di altezza relativo e alla rete.
- **Prezzo:** il tracking peggiora un po', soprattutto lo **yaw**. È stato corretto nella v2.

---

## 4. Checkpoint (in `checkpoints/` di questa cartella)
- **RA3** (consigliato fase 1): `20261019_003649_RA3_final/params_best.pkl`.
  - Override: `{"terrain_config.scene": "rough_sparse", "randomize_reset": 1.0, "command_config.mode": "joystick", "episode_length": 1000}`.
- **RA2:** `20261018_210643_RA2`. **RA:** `20261018_195719_RA`.
- **RS4:** `20261018_221525_RS4`. **RS2:** `20261018_203901_RS2`.
- **B5** (SAC, replica del paper): `20261018_172405_B5_sac_paper`, con `eval_hybrid/`.

## 5. Codice
- **`save_setup/`** (struttura dei repo):
  - `joystick.py` (branch `tita_hybrid` di mujoco_playground) con le scene `scene_slopes`, `scene_mixed*`, `scene_rough_sparse`, `scene_tilt10`;
  - mpx: `train_srbd.py` (`--algo sac`, `--sac-override`, `--env-override`), `sac_training.py`, `algorithm_initialization.py`, `utils/mpc_utils.py`.
- **`patches/`:** `git diff` rispetto a `residual_tita`. **Nessun commit né push** nei repo.
- **Valutazione:** `python tita_hybrid_plan/eval_robust.py --controller {mpc|hrc --load <run>|v1 ...}`. Risultati in `robust_results/`.

## 6. Limiti
- 1 seed di training.
- 16–40 episodi per terreno (intervalli di confidenza ampi).
- tita_v1 valutato con il suo MPC vecchio (altezze assolute).
- Pendenza su piano inclinato, non su rampe a corsie.
- Le scene di training non contengono ciottoli densi né perlin.
