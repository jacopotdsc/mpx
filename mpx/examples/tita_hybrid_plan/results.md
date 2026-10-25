# Risultati — HRC su Tita (MPC + rete residua in coppia)

_Versione finale del 19/10, ore 04:10._

**In sintesi:** il modello **RA3** (PPO, allenato solo sul rough sparso, 10M step) non cade mai su piano e perlin, cade 4/40 sui ciottoli (tita_v1 6/40, solo MPC 26/40) e 10/40 sulla pendenza da 10° (solo MPC 27/40). Sul perlin tita_v1 cade 37/40.

## 1. Cosa è stato fatto
- **Architettura (dal paper HRC, Zhang et al., ACIRS 2025):** τ = τ_ζ + τ_π.
  - **τ_ζ (controllore base):** MPC DFCIP + WBC (`tau_ff`) + **PD di postura sulle gambe**: thigh kp 85 / kd 12, knee 75 / 15, abduzione 35 / 10; ruote solo `tau_ff`. Corrisponde al "LQR ruote + PD gambe" del paper, con l'MPC al posto dell'LQR.
  - **τ_π (rete):** coppia diretta su 6 motori (thigh, knee, ruota ×2); abduzione esclusa come nel paper. Massimo [5, 5, 2] Nm.
- **Senza il PD sulle gambe** bastano 1–2 Nm costanti di residuo per far collassare il robot. Tutti i primi training (SAC, paper fedele) fallivano per questo motivo.
- **Obiettivo finale (utente): robustezza = non cadere** su rough, perlin e pendenza da 10°, meglio di tita_v1. L'inseguimento è secondario.
- **Training finali:** PPO, 1024 env, comandi come tita_v1 (vx fino a ±2.5, wz ±0.8; sulla pendenza solo lineari, \|vx\| ≤ 1).
  - **RA / RA2 / RA3:** solo rough sparso (50 box).
  - **RS2 / RS4:** solo piano inclinato di 10°.
  - **Perlin solo in valutazione** (generalizzazione).

## 2. Protocollo di valutazione (`eval_robust.py`)
- **Comandi fissi con rampa:**
  - piano, rough, perlin: (0.5,0), (1,0), (1.5,0), (2,0), (1,0.5), (1,−0.8), (2,0.8), (−1,0);
  - pendenza: solo lineari, vx ∈ {±0.25, 0.5, 0.75, ±1, 0} (−0.25, −0.5, −1, 0, 0.25, 0.5, 0.75, 1).
- **Episodi:** 5 spawn per comando (posizione e yaw casuali), 10 s ciascuno. Sul rough 2 spawn (16 episodi) oppure 5 (40 episodi, "rough40").
- **Terreni:**
  - piano;
  - **rough** = campo di ciottoli da 2590 box (`TitaJoystickRoughTerrain`), **mai visto in training**: in training c'erano solo 50 box sparsi;
  - **perlin** (`TitaJoystickPerlinTerrain`), mai visto in training;
  - **pendenza 10°** = piano inclinato infinito, con lo spawn corretto per la pendenza.
- **tita_v1** = run `20260918_124502` di `residual_tita` (PPO 30M, 1024 env), valutato **nel suo codice originale**. "v1 azione 0" non è un MPC puro: il suo blending dà un residuo anche con azione nulla.

## 3. Cadute (metrica principale)
| Terreno | Solo MPC (nostro) | RS4 (pendenza) | RA2 (rough) | **RA3 (rough, finale)** | tita_v1 | v1 azione 0 |
|---|---|---|---|---|---|---|
| piano | 0/40 | 0/40 | 0/40 | **0/40** | 0/40 | 8/40 |
| rough (16 ep.) | 12/16 | 5/16 | 3/16 | – | 1/16 | 4/16 |
| rough (40 ep.) | 26/40 | – | – | **4/40** | 6/40 | – |
| perlin | 2/40 | 2/40 | 0/40 | **0/40** | 37/40 | 39/40 |
| pendenza 10° | 27/40 | 9/40 | 16/40* | **10/40** | – | – |

\* RA2 sulla pendenza valutato prima della correzione dello spawn.

**Tempo medio alla caduta:** RA3 rough 4.7 s, pendenza 1.5 s; solo MPC rough 3.8 s; tita_v1 rough 6.0 s, perlin 3.7 s.

**Disturbo della rete** (‖τ_res‖ / ‖τ_MPC‖, media sugli step vivi): RS2 0.05–0.07; RS4 0.10–0.15; RA2 0.11–0.18; **RA3 0.13–0.18**; tita_v1 0.16–0.20.

**Tracking sul piano** (\|err vx\| m/s, \|err wz\| rad/s): solo MPC 0.032 / 0.025; RA2 0.080 / 0.099; **RA3 0.085 / 0.096**; RS4 0.137 / 0.103; tita_v1 0.051 / 0.030. **Tracking su rough / perlin** (RA3): vx 0.107 / 0.078, wz 0.398 / 0.316; tita_v1 rough vx 0.071, wz 0.149.

## 4. Lettura
- **Perlin:** tita_v1 cade quasi sempre (37/40). Il nostro MPC (riferimento di altezza relativo al contatto) e le nostre reti quasi mai. RA2: 0/40.
- **Pendenza 10°:** la rete specializzata (RS4) riduce le cadute del solo MPC da 27 a **9** su 40 (−67%). Su questa scena tita_v1 non è valutabile, perché il suo env non la supporta.
- **Rough (ciottoli, mai visti in training):** il solo MPC cade 26/40 (65%), RA3 **4/40 (10%)**, tita_v1 6/40 (15%). La rete taglia le cadute dell'MPC di circa l'85%. La differenza RA3 contro tita_v1 (4 contro 6 su 40) **non è statisticamente significativa**: sul rough sono alla pari, con RA3 leggermente avanti.
- **Un solo modello robusto ovunque:** RA3, allenato solo sul rough sparso, generalizza a perlin (0/40) e pendenza (10/40, quasi come RS4 che è specializzato: 9/40).
- **Tracking:** le reti peggiorano un po' il tracking rispetto al solo MPC e a tita_v1 (soprattutto wz sul rough), come accettato dall'utente: il criterio è la robustezza.
- **Cadute residue:** sulla pendenza avvengono quasi tutte nei primi 1.5 s dopo lo spawn (transitorio iniziale su un piano inclinato). Sul rough sono sparse fra i comandi.

## 6. Checkpoint e come riprodurre
- **RA3 (modello consigliato):** `mpx/examples/checkpoints/TitaJoystickFlatTerrain/20261019_003649/params_best.pkl` (PPO, `--name tita`).
  - Env: `--env-override '{"terrain_config.scene": "rough_sparse", "randomize_reset": 1.0, "command_config.mode": "joystick", "episode_length": 1000}'`.
  - Catena: RA (`20261018_195719`) → RA2 (`20261018_210643`) → RA3.
- **RS4 (pendenza):** `20261018_221525`.
- **Valutazione:** `python tita_hybrid_plan/eval_robust.py --controller hrc --load 20261019_003649 --rough-seeds 5`. File grezzi in `tita_hybrid_plan/robust_results/`.
- **Limiti:**
  - 1 seed di training;
  - 40 episodi per terreno (intervalli di confidenza ampi);
  - tita_v1 valutato con la sua versione dell'MPC (altezze assolute);
  - pendenza valutata sul piano inclinato, non sulle rampe a corsie (bordi).

## 5. Replica fedele del paper (SAC, task di distanza) — sintesi
- **SAC con i parametri del paper:** instabile in tutti i run (A/B/C, A3/A4/B4, B5, B6).
- **B5 (pendenze, PD di postura):** corregge la deriva dell'MPC a 10° (errore finale da +1.80 a −0.13 m, RMSE x_w da 1.01 a 0.48), che è il risultato chiave del paper. Però peggiora o cade altrove (5°, traiettorie E3).
- **Motivi:**
  - kernel di reward piatto vicino a 0 (contano solo gli errori grandi);
  - rete cieca alla pendenza;
  - nessun incentivo a restare a 0;
  - training fuori distribuzione rispetto ai test.
- Dettagli in `MEMORY.md` e in `<run B5>/eval_hybrid/`.
