# MEMORY — replica HRC (Hybrid Balance Control via Residual RL) su Tita

File di ripresa per le nuove sessioni. **Va aggiornato a ogni modifica o esperimento.**
Piano completo: [PLAN.md](PLAN.md). Note e numeri del paper: [paper_notes.md](paper_notes.md).

## Stato attuale (19 ottobre 2026, 04:10) — LAVORO CONCLUSO, vedi results.md

## Storico dello stato (18 ottobre 2026, dalle 18:45)
- **Scadenza (aggiornata dall'utente alle 20:00):** tutto finito, valutazione compresa, possibilmente entro le 8:00 e **mai oltre le 9:00 del 19/10**. **RA e RS vanno lasciati arrivare fino alla fine (3M), niente stop anticipati.**
- **Controllore base attuale:** MPC/WBC + **PD di postura HRC** sulle gambe (`nominal_pd.mode = "posture"`), τ_max [5, 5, 2]. Senza PD le gambe collassano con 1–2 Nm di bias.
- **Replica del paper (task di distanza):**
  - **B5** (pendenze, SAC esatto del paper), run `20261018_172405`, finito alle 18:05. Validazione su `final` in `<run>/eval_hybrid/`.
    - RMSE x_w MPC → HRC: 0° 0.047 → 0.181; 5° 0.476 → 0.617 **(cade a 8.7 s)**; 10° **1.013 → 0.484** (errore finale +1.80 → −0.13: corregge la deriva in salita come nel paper).
    - E2 spinte: 0.044 → 0.194, nessuna caduta. E3 poly5: 0.057 → 1.54, **cade**.
    - **Non è robusto.**
  - **B6** (pendenze, PD, burn-in + lr dell'actor 1e-4), run `20261018_180618`: **fermato alle 18:47 a 900k step**, cade ancora a ogni eval (durata circa 1500/2000, \|x\| 0.36–0.82). SAC su questo setup resta instabile.
- **Nuovo obiettivo dell'utente (18:15): modello ROBUSTO, migliore di tita_v1.**
  - tita_v1 = run `20260918_124502` del branch `residual_tita`: PPO 30M, 1024 env, comandi vx ±2.5 (±2.0 interno), wz ±0.8, residuo come offset PD.
  - Priorità: **cadute su rough, rampe e perlin** (dove il solo MPC cade, la rete deve far recuperare), poi tracking, poi quanto la rete disturba l'MPC.
  - La rete deve vedere in training le stesse velocità e gli stessi terreni.
  - **Fatto in `joystick.py`:**
    - `command_config.mode = "joystick"` (campionatore di tita_v1, ramp LPF 0.02). In questo modo errori senza posizione: il termine di yaw diventa errore su wz, e x_w è azzerato nell'obs.
    - `terrain_config.scene = "mixed"` → nuova `xmls/scene_mixed.xml`: piano (−15, 0), 3 rampe, rough sparso (50 box del v2 a x = 25), perlin (hfield alzato a x = 45). Regioni pesate 0.2 / 0.1×3 / 0.25 / 0.25, spawn ±0.5 m, yaw casuale tranne sulle rampe, quota dello spawn calcolata con `mj_ray`.
  - Terreno di training **leggero** (raccomandazione dell'utente: niente ciottoli da 2590 box).
  - **Smoke test della scena mista, solo MPC, 12 seed, 10 s:**
    - nessun errore; il solo MPC **cade** su rough (seed 1) e in curva sulla rampa da 5° (seed 0): sono i casi in cui la rete deve aiutare;
    - bug della quota di spawn sulla corsia a 0° (il raggio colpiva il robot) corretto: il robot viene spostato lontano prima di `mj_ray`.
  - **R1 (18:41): PPO** (`--algo ppo`, PPO_PARAMS: 1024 env, 20M step, lr 3e-4, entropy 1e-3, rete 512-256-128, uscita a zero, std 0.03) sulla scena mista con comandi joystick, episodi da 1000 step.
    - Comando: `--env-override '{"terrain_config.scene": "mixed", "command_config.mode": "joystick", "episode_length": 1000}'`.
    - Log `logs/R1_mixed_joystick_ppo_20261018_1841.log`, PID 2033173.
    - Scelto PPO perché SAC è stato instabile in tutti i run (B5, B6) e tita_v1 e MIT usano PPO.
    - eval#0 (solo MPC sulla scena mista, 18:46): reward −1.69 ± 2.35, durata media 805/1000 (cade spesso).
    - **Lento:** alle 19:36 ancora nessuna eval#1 (2M step). **R1 fermato alle 19:52** (utente: "non puoi fare 8 ore di training, alleggerisci").
- **Benchmark della velocità dell'env** (1024 env, step vmappato, solo env):
  | Scena | ms/step | transizioni/min |
  |---|---|---|
  | piano | 1867 | 33k |
  | mixed_lite (12 box + perlin) | 3317 | 19k |
  | mixed (50 box + perlin) | 3759 | 16k |

  **L'MPC/WBC in x64 è il collo di bottiglia**; l'hfield del perlin quasi raddoppia il costo, i box pesano poco.
- **Decisione dell'utente: perlin SOLO in valutazione** (la rete deve generalizzare).
  - Nuova scena `xmls/scene_mixed_noperlin.xml`: piano + rampe + 12 box rough (anche `scene_mixed_lite.xml` = 12 box + perlin).
  - In modalità joystick sono azzerati nell'obs anche `x_ref` e `start_yaw`: valori del task a distanza del paper, senza significato qui (l'utente aveva chiesto perché ci fossero).
- **R2 (19:56): PPO 6M step, 6 eval.**
  - `--env-override {"terrain_config.scene": "mixed_noperlin", "terrain_config.region_weight": [0.2, 0.1, 0.1, 0.2, 0.4, 0.0], "command_config.mode": "joystick", "episode_length": 1000}`.
  - Log `logs/R2_noperlin_joystick_ppo6M_20261018_1956.log`. **Fermato subito** (utente: "ancora tanto, meglio un training solo rough e uno solo rampa").
- **19:57: due training specializzati in parallelo, PPO 3M step, 3 eval ciascuno, comandi joystick.**
  - **RA, solo rough:** `xmls/scene_rough_sparse.xml` (copia dello `scene_rough_sparse.xml` del v2, 50 box di 1–2.5 cm entro ±5 m), `randomize_reset` 1 (spawn ±0.5 m, yaw casuale come tita_v1). Log `logs/RA_rough_joystick_ppo3M_20261018_1957.log`, PID 2245810.
  - **RS, solo rampa da 10°:** `terrain_config.lanes` [2]; rampe allargate a 2.4 m (`scene_slopes.xml`) per i comandi joystick. Log `logs/RS_slope10_joystick_ppo3M_20261018_1957.log`, PID 2246233.
  - `joystick.py`: scena `rough_sparse` e prefisso `rough_` tra le geometrie di terminazione.
  - **PPO è veloce:** RS ha fatto 3.28M step in 22 minuti (eval ogni 1.6M, circa 11 minuti). La lentezza di R1 dipendeva da perlin (hfield) + GPU condivisa con le mie eval.
  - **RS finito alle 20:22** (run `20261018_195725`): durata episodio 665 → 746 → 748/1000, reward −2.59 → −2.36. Migliora poco perché con comandi in tutte le direzioni il robot **esce di lato dalla rampa** (2.4 m) e cade dal bordo: artefatto della scena.
  - **RA alle 20:25 (1.6M):** durata 542 → 729, reward −3.94 → −2.91.
- **20:38, RS2: piano inclinato di 10° infinito** (`xmls/scene_tilt10.xml`, scena `tilt10`: floor ruotato di −10° su y, passa per l'origine) con yaw casuale → salita, discesa e traverso senza bordi.
  - Solo MPC (4 seed, 5 s): cade in 3 casi su 4 (anche dopo 0.46 s), quindi c'è molto margine.
  - PPO **10M**, 5 eval. Log `logs/RS2_tilt10_joystick_ppo10M_20261018_2038.log`, PID 2364328.
  - In `eval_robust.py` il terreno "slope10" ora è questa scena per tutti i controllori.
  - **RA finito alle 21:00** (run `20261018_195719`): durata 542 → 729 → **856**/1000, reward −3.94 → −2.91 → −1.72 (ancora in salita).
  - **RS2 a 9.8M alle 21:06** (run `20261018_203901`): durata 330 (MPC) → 746 → 655 → 767.
  - **21:07, RA2:** ripresa di RA (`--load 20261018_195719`, best) per altri **7M**, 4 eval, seed 1. Log `logs/RA2_rough_resume_ppo7M_*.log`, PID 2440429.
  - **RS2 finito alle 21:30** (run `20261018_203901`, 13.1M step con l'arrotondamento di brax): durata 330 → 746 → 655 → 767 → **809**/1000, reward −4.50 → −2.18.
  - **RA2** run `20261018_210643`: eval#0 (policy di RA ricaricata) durata 826.
  - **21:37, valutazione robusta round 1** in background (`logs/eval_robust_round1.log`): mpc, hrc RS2 (→ `robust_results/hrc_RS2.json`), v1, v1 zero.
  - **Robust eval, nostro solo MPC** (cadute; 8 comandi × 5 spawn, rough × 2): piano 0/40; **rough 12/16** (tempo alla caduta 5.0 s); perlin 2/40 (6.4 s); **pendenza 10° 35/40** (2.4 s). Le metriche di tracking erano NaN dopo le cadute (NaN × 0), corretto con `where` per i run successivi.
  - **RA2 a 3.3M:** durata 826 → **895**, reward −2.05 → −1.30.
  - **Indicazione dell'utente (22:10):** con la rete il robot deve arrivare a fine episodio (sopravvivere). **Pendenza: SOLO comandi lineari (wz = 0), vx max 1 m/s; rough: fino a 2 m/s, anche angolari.**
    - `eval_robust.py`: sulla pendenza `COMMANDS_SLOPE` con \|vx\| ≤ 1.
    - Il round 1 sulla pendenza (comandi fino a 2 m/s) è da rifare per tutti i controllori.
    - RA2 continua (allenato fino a 2.5, valutato fino a 2).
  - **22:13, RS3 fermato dopo 2 minuti** (aveva wz ±0.8) → **RS4**: come RS3 ma `command_config.a` [1.0, 0.0], `a_learned` [0.8, 0.0]. Log `logs/RS4_tilt10_linear1ms_resume_ppo7M_*.log`.
  - (superato) **22:11, RS3:** ripresa di RS2 (`--load 20261018_203901`) sul piano inclinato con `command_config.a` [1.0, 0.8], `a_learned` [0.8, 0.6], 7M, 4 eval, seed 2. Log `logs/RS3_tilt10_v1ms_resume_ppo7M_*.log`, PID 2611227.
  - **Robust eval di RS2** (allenato solo sul piano inclinato, comandi fino a 2.5 m/s): piano 0/40 (|vx| 0.064, ratio 0.067); **rough 5/16** (contro 12/16 del solo MPC: generalizza); perlin 3/40 (MPC 2/40); ratio del residuo 0.05–0.07.
  - **RS4** run `20261018_221525`: eval#0 (policy di RS2, comandi lineari ≤ 1) durata 834.
  - **Robust eval di tita_v1 (23:06):** piano 0/40 (|vx| 0.051, ratio 0.16); **rough 1/16** (tempo alla caduta 6.5 s, ratio 0.17): **su rough tita_v1 batte RS2 (5/16)**. Il confronto vero è con RA2 (allenato sul rough). Perlin e v1 zero in corso.
  - **23:06:** RA2 a 6.5M, durata 905; RS4 a 3.3M, durata 834 → 802 (lieve calo).
  - **23:36:** **tita_v1 su perlin 37/40 cadute** (tempo alla caduta 3.7 s) contro il nostro MPC 2/40 e RS2 3/40: **sul perlin vinciamo nettamente** (riferimento di altezza relativo + rete). Solo MPC di tita_v1 sul piano: 8/40 cadute. RS4 a 6.5M: durata 898. RA2 a 6.5M: 905, ultima eval in arrivo.
  - **Round 1 completo (23:40), cadute**, file in `robust_results/`:
    | Terreno | nostro MPC | RS2 | tita_v1 | v1 azione 0 |
    |---|---|---|---|---|
    | piano | 0/40 | 0/40 | 0/40 | 8/40 |
    | rough | 12/16 | 5/16 | **1/16** | 4/16 |
    | perlin | **2/40** | 3/40 | 37/40 | 39/40 |
    | pendenza 10° (vecchi comandi fino a 2 m/s) | 35/40 | 21/40 | – | – |

    Il "v1 azione 0" **non** è un MPC puro: il blending di v1 (PD verso la posa di default) dà un residuo ≠ 0 anche con azione 0 (ratio 0.15).
  - **RS4 finito alle 23:39** (run `20261018_221525`): durata 834 → 802 → **898** (6.5M, best) → 809.
  - **23:41, round 2** (`logs/eval_robust_round2.log`): hrc RS4 (best) → `hrc_RS4.json`; solo MPC sulla pendenza con comandi lineari → `mpc_slope_linear.json`. Il vecchio `mpc.json` è rinominato `mpc_round1.json`.
  - **Round 2 (00:05), cadute:** RS4 (best 6.5M): piano 0/40 (|vx| 0.137), rough 5/16 (tempo alla caduta 7.3 s), perlin 2/40, **pendenza 10° lineare 11/40** (tempo alla caduta 1.5 s) contro il solo MPC lineare **33/40** (1.0 s). Le cadute di RS4 sulla pendenza avvengono molto presto: probabilmente spawn difficili, base orizzontale su un piano inclinato.
  - **RA2 finito alle 00:00** (run `20261018_210643`): durata 826 → 895 → 905 → **932**, reward −0.95.
  - **00:07, round 3:** hrc RA2 (best) → `hrc_RA2.json`, log `logs/eval_robust_round3.log`.
  - **Bug dello spawn sul piano inclinato (00:10):** con yaw di traverso le ruote stanno a quote di suolo diverse (±d/2·sin(yaw)·tan 10° ≈ ±5 cm), ma lo spawn assumeva suolo piano → una ruota compenetrata fino a 5 cm → cadute nel primo secondo. Fix in `joystick.py`: `terrain_config.tilt_deg` (10 per tilt10) alza lo spawn perché la ruota più alta tocchi. `eval_robust` sulla pendenza ora passa `tilt_deg` 10. Rivalutazione in corso: `logs/eval_robust_slope_spawnfix.log` → `mpc_slope_spawnfix.json`, `hrc_RS4_slope_spawnfix.json`.
  - **Round 3 + spawn fix (00:36), cadute:**
    | Terreno | nostro MPC | RS4 | **RA2** | tita_v1 |
    |---|---|---|---|---|
    | piano | 0/40 | 0/40 | 0/40 | 0/40 |
    | rough (16 ep.) | 12/16 | 5/16 | **3/16** | 1/16 |
    | perlin | 2/40 | 2/40 | **0/40** | 37/40 |
    | pendenza 10° (lineare, spawn corretto) | 27/40 | **9/40** | 16/40 (senza spawn fix) | – |

    RA2: ratio del residuo 0.11–0.18.
  - **00:37, rilanciati:**
    - **RA3:** ripresa di RA2, +7M, seed 3. `logs/RA3_rough_resume_ppo7M_*.log`.
    - **RS5:** ripresa di RS4 con spawn fix (`tilt_deg` 10), +5M, seed 4. `logs/RS5_tilt10_spawnfix_resume_ppo5M_*.log`.
    - **Rough a 40 episodi** (`--rough-seeds 5`) per v1, mpc, RA2. `logs/eval_robust_rough40.log` → `*_rough40.json`.
  - **01:06:** RA3 run `20261019_003649`, eval#0 (= RA2) durata 964; RS5 run `20261019_003655`, eval#0 (= RS4, spawn fix) 843. Rough40 su v1 in corso.
  - **01:36:** tutto lento (3 processi + rough con 2590 box). RA3 ancora senza eval#1. RS5 eval@3.3M: 843 → **753** (calo; best = migliore eval). Rough40: **catena fermata dopo v1** (killato solo il bash genitore). Il Python di v1 finisce e salva `robust_results/v1.json` (da rinominare in `v1_rough40.json`). Rough40 per il modello finale soltanto.
  - **02:06:**
    - **tita_v1 rough a 40 episodi: 6/40 cadute (15%)**, tempo alla caduta 6.0 s. A 16 episodi era 1/16: campione piccolo.
    - `v1.json` del round 1 era stato sovrascritto: ricostruito dal log in `v1_round1.json`; il rough40 è in `v1_rough40.json`.
    - **RS5 finito: peggiora** (843 → 753 → 770) → come modello per la pendenza resta **RS4** (9/40 con spawn fix).
    - RA3 a 3.3M: 964 → **968**.
    - Lanciato il rough40 del solo MPC → `mpc_rough40.json`.
  - **03:36:** **RA3 finito** (run `20261019_003649`): durata 964 → 968 → 932 → **969** (best a 9.8M, reward −0.59). **Solo MPC rough a 40 episodi: 26/40 cadute (65%)**, tempo alla caduta 3.8 s → `mpc_rough40.json`. Lanciata la **valutazione finale di RA3** (tutti i terreni, rough a 40 episodi, pendenza con spawn fix) → `hrc_RA3_final.json`, log `logs/eval_robust_RA3_final.log`.
  - **Bozza di `results.md` scritta** (03:10). Da completare con RA3 e il rough40 del solo MPC.
  - **04:10, VALUTAZIONE FINALE di RA3** (`hrc_RA3_final.json`): piano 0/40; **rough40 4/40** (tita_v1 6/40, MPC 26/40); **perlin 0/40** (tita_v1 37/40); **pendenza 10° 10/40** (MPC 27/40, RS4 9/40). Ratio del residuo 0.13–0.18. Tracking sul piano vx 0.085, wz 0.096. **RA3 = modello consigliato.** `results.md` finalizzato.
  - **Piano:** valutazione finale di RA3/RS5 (best) verso le 3:00, poi results.md entro le 8:00 (limite 9:00).
  - **Da fare a fine RA2/RS4:** eval_robust di RA2 e RS4 (tutti i terreni), pendenza del solo MPC da rifare con `COMMANDS_SLOPE` lineari, poi results.md.
  - **eval#0 = solo MPC, con i comandi di tita_v1:** RA rough reward −3.94, durata 542/1000; RS rampa 10° −2.59, durata 665/1000. **Il solo MPC cade spesso: c'è margine per la rete.** Run dir: RA `20261018_195719`, RS `20261018_195725`.
- **Valutazione robusta: `tita_hybrid_plan/eval_robust.py`.**
  - **Protocollo:** terreni piano / rough (ciottoli) / perlin, più rampe da 5° e 10° solo per i nostri controllori; 8 comandi fissi con rampa; 5 spawn casuali con yaw casuale; 10 s; batch + `lax.scan`.
  - **Metriche:** cadute, tempo alla caduta, |errore vx|, |errore wz|, ‖τ_res‖/‖τ_nom‖.
  - **Controllori:** `mpc` (nostro), `hrc --load <run>`, `v1` (codice originale nei worktree dello scratchpad `mjpg_orig` e `mpx_v1` = branch `residual_tita`, checkpoint `20260918_124502`) e `v1 --zero` (MPC-only di v1).
  - Risultati in `tita_hybrid_plan/robust_results/<controller>.json`.
  - Le baseline sono partite alle 18:57 (`logs/eval_robust_baselines.log`) e sono state **fermate alle 19:37**.
    - Il rough (2590 box) con 40 env in batch era bloccato da 40 minuti e rubava GPU a R1.
    - Ora `--rough-seeds` vale 2 di default (16 env) e il JSON si salva dopo ogni terreno.
    - Da rilanciare quando R1 avrà finito o quando servirà.
  - Controllo: nostro MPC sul piano, 0/40 cadute, |vx err| 0.032, |wz err| 0.025.
- **Indicazioni dell'utente (19:45):**
  - in valutazione **solo la rampa a 10°** (errori grandi, lì si vede la differenza MPC/rete);
  - **compito principale: robustezza = EQUILIBRIO** (non cadere) su rough, perlin e rampa a 10°. Il tracking è secondario. **Criterio di successo:** meno cadute del solo MPC e di tita_v1 (metrica primaria: % di cadute per terreno e comando, tempo alla caduta);
  - spiegazione data del perché B5 è andato male: kernel piatto vicino a 0 (contano solo gli errori grandi), rete cieca alla pendenza che applica la correzione anti-deriva anche dove non serve, nessun incentivo a restare a 0 (mean_action_l2 ≈ 0.4), fuori distribuzione su E3 e rough.
  - Se R1 disturba dove l'MPC è già buono: kernel più stretto sul tracking (exp come tita_v1) + piccola penalità sul residuo (`residual_torque` del v2, torques del MIT).
- **Valutazione finale prevista:** solo MPC contro tita_v1 contro nuova rete su rough (ciottoli veri, `TitaJoystickRoughTerrain`), perlin, rampe e piano, con comandi fino a vx 2.0, wz 0.8. Per tita_v1 serve il suo codice (worktree `residual_tita`).
- **Solo MPC sul perlin** (dritto, D = 4 m, 10 s): non cade a 0.5–2.0 m/s, ma è indietro di 1–3 m.

## Branch e repo
| Repo | Path | Branch | Base |
|---|---|---|---|
| mujoco_playground | `/home/jacopo/miniconda3/envs/mjpl/lib/python3.11/site-packages/mujoco_playground` | `tita_hybrid` | `residual_tita` @ 6ccb7df |
| mpx | `/home/jacopo/Desktop/repo_rl/tita_rl/test/mpx` | `tita_hybrid` | `residual_tita` @ 4f159d6f |

- Il lavoro lite3 è salvato nello stash `snapshot lite3 18 ottobre` (stash@{0}) su `lite3`, in entrambi i repo.
- `residual_tita_v2` serve solo come riferimento.
- Nessun commit e nessun push ancora: le modifiche sono nel working tree di `tita_hybrid`.

## Regole dell'utente (vincoli fissi)
- **mpx (MPC/WBC) non si tocca:** l'MPC funziona. Controlla tutti gli 8 giunti.
- **La rete agisce su 6 motori** (thigh, knee, ruota ×2). L'abduzione riceve residuo 0, come nel paper (Fig. 2: "hip lateral motors… not considered").
- **τ_ζ = solo `tau_ff` del WBC.** Niente PD nominale (richiesta esplicita).
- **In mujoco_playground si modifica solo `joystick.py`**, mantenendone lo stile. Unico file nuovo previsto: `scene_slopes.xml` (Fase 3a).
- **Niente script/driver nuovi per il training:** si usa `train_srbd.py --algo sac`.
- **Comandi in velocità all'MPC.** CoM fisso a 0.40.
- **Nessuna domain randomization**, mai. Perturbazioni solo in eval, solo sul piano.
- **Setup minimo** (niente randomness al reset, noise 0, deadband 0). Si complica una cosa alla volta, solo dopo che la policy impara.
- **Solo MPC + NN.** Niente DDPG e niente RL puro.

## Lezioni importanti
- **Serve sempre `jax_enable_x64`.** In float32 FDDP/WBC degradano: il robot non segue v_ref e cade in circa 1 s, con `mpc_bad` attivo. `train_srbd.py` e `mjx_tita.py` lo attivano già; gli script di test devono farlo **come prima riga**. (Avevo perso tempo pensando a un MPC rotto: non lo era.)
- **Driver standalone validato:** `mjx_tita.py` (tau + PD 35/10, aggiornato a 500 Hz). Non ha fallback sui NaN.

## Decisioni e parametri non specificati dal paper
| Parametro | Valore | Motivo |
|---|---|---|
| Simulatore | MuJoCo/MJX (paper: Webots) | infrastruttura esistente |
| Controllore base | MPC DFCIP + WBC + **PD di postura sulle gambe** (85/12 thigh, 75/15 knee, 35/10 abduzione), ruote solo tau_ff (paper: LQR ruote + PD gambe) | senza PD le gambe collassano con 1–2 Nm di bias residuo; approvato dall'utente |
| Frequenze | RL/MPC/WBC 100 Hz, fisica 500 Hz | config esistente (mpx: N = 50, orizzonte 0.5 s) |
| Timing dell'MPC nell'env | risolto **a fine step** sullo stato nuovo e applicato nello step dopo | stesso calcolo di prima; così la policy osserva τ_ζ^t prima di agire (paper s_τ) |
| Blending | τ_C = clip(τ_ζ + a ⊙ τ_max, ±120 Nm) | paper: τ_C = τ_ζ + τ_π, senza scala |
| τ_max [thigh, knee, wheel] | **[8, 8, 2]** Nm per lato (prima [5, 12, 2]) | indicazione dell'utente: tetto di 8 Nm; ruota 2 Nm ≈ 22 N a terra. Prima: \|τ_ζ\| MPC-only: thigh 1.4, knee 11.6, wheel 0.07 (picco 0.84). Knee = nominale; thigh e ruota con un minimo di autorità. Da fare sweep. |
| Low-pass sull'azione s | 0 (spento) | da provare nel tuning |
| Task | D ~ U(1, 4) m, profilo v trapezoidale, wz = 0 | paper: target di posizione; utente: velocità |
| v_max, a_max | 0.5 m/s, 0.5 m/s² | non dati |
| Pesi della reward | unitari | non dati |
| Termination | −500 | non nel paper; senza, i kernel ≤ 0 premierebbero la caduta |
| z_c | CoM sopra il punto di contatto (asse ruote + raggio) | coerente con il target 0.4 dell'MPC |
| Errore di yaw | yaw − yaw iniziale (avvolto) | q_ψd = 0 |
| Correzioni SAC (deviazione dal paper) | critic burn-in, lr dell'actor 1e-4 (critic e α 1e-3); in C2 anche zero_action_prior 0.1 | con i parametri del paper l'actor destabilizza l'MPC (A/B/C falliti) |
| Osservazione | 65 dim, componenti di III-A | il paper dichiara 30: incongruente |
| SAC non specificati | 64 update ogni 1024 transizioni, replay 1M, τ 0.005, **256 env** (l'utente vuole tanti env; con 2M ogni env fa circa 1.5 episodi), std iniziale 0.1, ultimo layer a 0, niente burn-in, 10 env di eval | 2M → 2_000_896 (multiplo di 1024) |
| Pendenze | corsie 0/5/10° a y = 0/+3/−3, rampa di 6 m che parte a x = 0.6, plateau dopo; corsia uniforme al reset | paper Group 1 |
| Riferimento di altezza MPC | relativo a c_z misurato (mpx `mpc_utils.py`) | autorizzato dall'utente il 18/10 |
| Spinte (E2) | 10, 20 e 30 N a 12, 20 e 32 s, durata 0.1 s, verso −x | durata non data nel paper |
| Episodio | 5000 step = 50 s | "maximum 5000 steps" |

## File toccati
- **mujoco_playground `_src/locomotion/tita/joystick.py`:**
  - config: episodio 5000, `randomize_reset` 0, noise 0, `residual_config` (motor_ids, tau_max, action_lpf), reward del paper, `command_config` (distance, v_max, a_max); rimossi Kp/Kd;
  - `action_size` = 6; `_nominal_torque` = tau_ff; `_residual_torque`, `_combine_torque`;
  - `_reference_profile`, `_base_rpy`, `_balance_state`, `_task_errors`, `_task_error_kernel`, `sample_distance`;
  - reset: randomness scalata da `randomize_reset`, spawn frame salvato in `info`;
  - step: MPC a fine step; obs a 65 dim; reward del paper; metriche `task/*`;
  - rimossi `_default_pose`, `_residual_joint_targets`, `sample_command` + LPF, `_cost_posture`.
- **mpx `mpx/examples/train_srbd.py`:** `SAC_PARAMS` con i valori del paper; `run_sac_training()` (portata dal v2, eval con `lax.scan`); dispatch `--algo sac`; `from pathlib import Path`.
- **mpx `mpx/examples/sac_training.py` e `algorithm_initialization.py`:** copiati dal v2 senza modifiche.
- **mpx `mpx/examples/tita_hybrid_plan/`:** PLAN.md, MEMORY.md, paper_notes.md.

## Esperimenti
| Data | Cosa | Esito |
|---|---|---|
| 18/10 | Verifica Fase 1 (x64), D = 2.5 m, 20 s | azione 0 = MPC-only (diff qpos 4e-5). **MPC-only:** x_w finale 2.512, RMSE x_w 0.050, v 0.040, θ_p 0.025, z 0.0004, pitch 0.0001, nessuna caduta, 0 fallimenti del solver. Paper: BMC 0.2475, HRC 0.0134. |
| 18/10 | Smoke test SAC, 20k step, 10 env | ciclo completo OK (collect, update, eval, checkpoint). Bug: NaN nelle medie degli errori di eval dopo una caduta, corretto con `where`. |
| 18/10 | Solo MPC sulle rampe (D = 4 m, 20 s, riferimento di altezza relativo) | nessuna caduta, 0 fallimenti. RMSE x_w: 0° 0.063, 5° 0.364, 10° 0.722 (finali +0.67 / +1.33 m). Paper BMC: 0.25 / 0.14 / 0.34. Sulle pendenze c'è margine per l'RL. |
| 18/10 14:40 | Test di `eval_hybrid.py --only E1` sul run fermato (`params_best` = policy a zero) | Script OK, HRC = MPC. **Solo MPC, D = 2.5 m, 25 s:** RMSE x_w 0° 0.045, 5° 0.473, 10° 0.933. Errore finale: 0.012 / 0.85 / 1.69 m. Sulle pendenze l'MPC va a circa 0.58 m/s invece di 0.5 e da fermo **continua a salire a circa 0.075 m/s** (θ_p ≈ 0.05): bias sistematico del modello piano. Nessuna caduta. È il caso che il residuo deve correggere (paper BMC 10°: finale 0.39 m). Output in `<run>/eval_hybrid/`. |
| 18/10 15:06 | A/B/C, SAC del paper (lr 1e-3, nessun burn-in), 256 env, 3 in parallelo (circa 11k step/min ciascuno) | **Falliti**, fermati alle 15:35. Eval a 300k: A R −5.60, durata 211/2000, \|x\| 0.71; B −5.49 / 298 / 0.57; C −5.94 / 670 / 1.09. Solo MPC a 0: A/C −0.009 / 2000 / 0.036; B −0.224 / 2000 / 0.28. Causa: l'actor si allontana da zero subito (mean_action_l2 ≈ 0.45). Run dir `20261018_1506{10,15,20}`. |
| 18/10 15:38 | A2/B2/C2: burn-in + lr dell'actor 1e-4 (+ prior 0.1 in C2), τ_max [5, 12, 2] | fermati dopo 5 minuti per il cambio di τ_max |
| 18/10 15:46 | A3/B3/C3: come A2/B2/C2 ma con τ_max [8, 8, 2] | B3 e C3 fermati alle 15:52. **A3** continua (alle 15:54: 180k step, ancora in burn-in, eval a 100k = solo MPC). |
| 18/10 15:52 | A4 (piano) e B4 (pendenze): SAC **esatto del paper**, τ_max [8, 8, 2], τ_ζ = solo tau_ff | **Falliti**, fermati alle 16:26 (circa 400k step). Cadono in 0.3–3.8 s a ogni eval; mean_action_l2 ≈ 0.5, q_pi ≈ −36. A3 (burn-in) è uguale al solo MPC finché l'actor è congelato (fino a 205k), poi cade in 2–10 s. |
| 18/10 16:30 | Sensibilità a coppie residue costanti (solo MPC + azione fissa, 10 s) | Solo tau_ff: thigh 0.5 Nm ok, 1 Nm cade in 6.7 s; knee 1 Nm ok, 2 Nm cade in 0.66 s; 4 Nm cade in circa 0.5 s; rumore σ 0.1/0.3 ok; ruote 1 Nm: deriva di −0.9 m, nessuna caduta. |
| 18/10 17:00 | Modalità del PD nominale (Task 1, 20 s) | none: cade con 4 Nm. wbc 35/10: cade con 4 Nm (2.5–3.3 s). **posture HRC: non cade nemmeno con 8 Nm** su thigh o knee; solo MPC RMSE x 0.052. Tutto 0.5 (incluse ruote 1 Nm): x finale −1.8 m, nessuna caduta. |
| 18/10 17:19 | **B5:** pendenze, SAC del paper, PD di postura, τ_max [5, 5, 2] | fermato alle 17:22 (GPU occupata via MPS). **Rilanciato alle 17:24**, run `20261018_172405`, circa 47k step/min da solo. Alle 17:54, 1.40M step. Eval (R / durata su 2000 / \|x\|): 0 (MPC) −0.26 / 2000 / 0.30; 100k −3.38 / 1385 / 0.32; 600k −3.00 / 1635 / 0.50; 1.0M −3.39 / 1962 / 0.92; 1.1M −3.20 / 1949 / 0.96; 1.2M −2.59 / 1560 / 0.50; 1.3M −5.41 / 1234 / 0.95. **Molte meno cadute grazie al PD, ma non batte l'MPC su x_err**; mean_action_l2 ≈ 0.42, α 0.002. Prossimo passo: eval_hybrid sul best, poi variante burn-in + lr dell'actor 1e-4 con PD. **Finito alle 18:05.** Eval 1.7M −1.00 / 1872 / 0.33; 1.9M −0.91 / 1938 / 0.31; **2M (final) −1.49 / 2000 (nessuna caduta) / 0.63**. `params_best` = step 0 (MPC, scelto per reward): per la validazione usare **`--suffix final`**. |
| 18/10 18:06 | Validazione completa di B5 final: `eval_hybrid.py --suffix final` (E1 corsie, E2 spinte, E3 traiettorie, **E5 rough**, aggiunto su richiesta dell'utente) | **Finita alle 19:40.** RMSE x_w MPC → HRC: E1 0° 0.047 → 0.181; 5° 0.476 → 0.617 (HRC cade); 10° 1.013 → 0.484. E2: 0.044 → 0.194. E3 poly5: 0.057 → 1.54 (HRC cade); multi: 0.215 → 0.301. E5 rough (ciottoli, 6 casi): **cadute MPC 1/6, HRC 1/6** (entrambi a D 4, v 1.0), inseguimento HRC quasi sempre peggiore. Il SAC fedele al paper **non è robusto**. `logs/eval_B5_final.log`, `<run>/eval_hybrid/`. |
| 18/10 18:06 | **B6:** pendenze, PD, τ_max [5, 5, 2], **critic burn-in + lr dell'actor 1e-4** | fermato alle 18:47 a 900k: eval da 300k a 800k con durata 1122–1568/2000 (cade sempre), \|x\| 0.20–0.82. `logs/B6_slopes_pd_tau5_burnin_20261018_1806.log` |
| 18/10 14:07 | Run SAC completo, 1 seed, 256 env, eval da 5000 step | **fermato** a 56k (eval troppo lente). Eval a 0: −0.009, nessuna caduta. Eval a 50k: −5.23, cade dopo circa 137 step. |

Script di verifica nello scratchpad della sessione: `check_hybrid_env.py`, `check_mpc_only.py`.
