# Recap — tita_hybrid_plan_v2 (19/10, 04:30 → 09:10)

Fine-tune dei modelli della fase 1 (`../tita_hybrid_plan/`, modello di partenza **RA3**).
Obiettivi dell'utente:
- sistemare l'**orientamento** con un piccolo training;
- cercare le metriche migliorabili;
- fare un training veloce sulla **pendenza**, accettando **una rete per terreno**.

Dettagli: `MEMORY.md`, `results.md`, `table.md`.

---

## 1. Cosa non andava in RA3
- **Orientamento = yaw/heading** (roll e pitch erano già piccoli, ≤ 0.075 rad).
  - Sul piano: \|wz err\| 0.096 contro 0.025 del solo MPC e 0.030 di tita_v1.
  - **Deriva della direzione** a fine episodio: 0.89 rad contro 0.24.
- **Causa:** il termine di yaw nel reward del paper è un kernel piatto vicino a 0 (0.3 rad/s → −0.006 per step), irrilevante rispetto alla termination.

## 2. Cosa è stato fatto
| Run | Partenza | Training | Novità |
|---|---|---|---|
| **RA4** | RA3 | 3M step, rough sparso, comandi come tita_v1 | + `tracking_ang_vel` 2.0 e `tracking_lin_vel` 1.0: kernel exp come tita_v1, ripido vicino a 0 (best a 1.6M; durata 949 → 962) |
| **RA5** | RA4 | 3M step, **piano inclinato 10°**, solo comandi lineari \|vx\| ≤ 1, 11 minuti | stessi termini di tracking (durata 851 → 872 → 910) |

- **Codice:** in `joystick.py` i termini `reward_config.scales.tracking_lin_vel` e `tracking_ang_vel` (default 0) e `tracking_sigma` 0.0625. Copia in `save_setup/`.
- **Valutazione:** `eval_robust.py` (stesso protocollo della fase 1) più **\|roll\|, \|pitch\|, deriva della direzione**.
  - 40 episodi per piano, perlin e pendenza; 16 per il rough.
  - RA5 valutato solo sulla pendenza.

## 3. Risultati
| Terreno | Controllore | Cadute | Tempo alla caduta [s] | \|vx err\| | \|wz err\| | \|roll\| | \|pitch\| | Deriva della direzione [rad] | ‖τ_res‖/‖τ_MPC‖ |
|---|---|---|---|---|---|---|---|---|---|
| piano | solo MPC | 0/40 | – | 0.032 | 0.025 | 0.001 | 0.027 | 0.241 | 0.000 |
| piano | RA3 | 0/40 | – | 0.085 | 0.096 | 0.005 | 0.031 | 0.893 | 0.147 |
| piano | **RA4** | **0/40** | – | 0.079 | **0.044** | 0.003 | 0.027 | **0.428** | 0.166 |
| piano | tita_v1 | 0/40 | – | 0.051 | 0.030 | 0.027 | 0.033 | 0.231 | 0.158 |
| rough | solo MPC | 12/16 | 4.5 | 0.263 | 0.820 | 0.028 | 0.162 | 1.238 | 0.000 |
| rough | RA3 | 3/16 | 7.3 | 0.107 | 0.416 | 0.016 | 0.068 | 1.088 | 0.129 |
| rough | **RA4** | **1/16** | 3.8 | 0.109 | 0.395 | 0.014 | 0.071 | **0.862** | 0.129 |
| rough | tita_v1 | 3/16 | 5.4 | 0.152 | 0.269 | 0.032 | 0.046 | 0.357 | 0.160 |
| perlin | solo MPC | 3/40 | 7.2 | 0.074 | 0.522 | 0.031 | 0.051 | 1.358 | 0.000 |
| perlin | RA3 | 0/40 | – | 0.081 | 0.325 | 0.026 | 0.036 | 1.193 | 0.161 |
| perlin | **RA4** | **0/40** | – | 0.077 | **0.239** | 0.028 | 0.038 | 1.113 | 0.173 |
| perlin | tita_v1 | 36/40 | 3.7 | 0.216 | 0.809 | 0.066 | 0.123 | 1.229 | 0.206 |
| pendenza 10° | solo MPC | 26/40 | 1.8 | 0.173 | 0.279 | 0.050 | 0.087 | 0.925 | 0.000 |
| pendenza 10° | RA3 | 12/40 | 2.2 | 0.129 | 0.317 | 0.062 | 0.075 | 1.173 | 0.183 |
| pendenza 10° | RA4 | 15/40 | 3.0 | 0.135 | 0.305 | 0.064 | 0.074 | 1.197 | 0.181 |
| pendenza 10° | **RA5** | **8/40** | **4.5** | **0.090** | **0.230** | 0.049 | 0.068 | 1.239 | 0.193 |

Riferimento dalla fase 1, pendenza: RS4 (specialista) 9/40. Rough a 40 episodi: RA3 4/40, tita_v1 6/40, solo MPC 26/40.

## 4. Lettura
- **Orientamento (RA4 contro RA3):**
  - piano: wz −54% (0.096 → 0.044), deriva della direzione −52% (0.89 → 0.43 rad);
  - perlin: wz −26%;
  - rough: deriva 1.09 → 0.86.
  - Ancora peggiore di tita_v1 sul piano e sul rough, ma il distacco si è dimezzato.
- **Robustezza mantenuta o migliorata:**
  - RA4: piano 0/40, perlin 0/40 (tita_v1 36/40), rough 1/16 (RA3 e tita_v1 3/16).
  - RA5: pendenza **8/40**, il migliore (solo MPC 26/40, RA4 15/40, RS4 9/40), cadendo anche più tardi (4.5 s).
- **Roll e pitch:** invariati e piccoli. **vx:** circa invariato sul piano, migliore in pendenza con RA5.
- **Disturbo della rete:** invariato (0.13–0.19).

## 5. Raccomandazione finale — una rete per terreno
| Terreno | Rete | Checkpoint (in questa cartella) |
|---|---|---|
| piano, rough, perlin | **RA4** | `checkpoints/20261019_065047_RA4/params_best.pkl` |
| pendenza | **RA5** | `checkpoints/20261019_084903_RA5/params_best.pkl` |

**Override per riprodurre:**
- **RA4:**
  ```
  --load 20261019_003649 --env-override '{"terrain_config.scene": "rough_sparse", "randomize_reset": 1.0, "command_config.mode": "joystick", "episode_length": 1000, "reward_config.scales.tracking_lin_vel": 1.0, "reward_config.scales.tracking_ang_vel": 2.0}'
  ```
- **RA5:**
  ```
  --load 20261019_065047 --env-override '{"terrain_config.scene": "tilt10", "terrain_config.tilt_deg": 10.0, "randomize_reset": 1.0, "command_config.mode": "joystick", "command_config.a": [1.0, 0.0], "command_config.a_learned": [0.8, 0.0], "episode_length": 1000, "reward_config.scales.tracking_lin_vel": 1.0, "reward_config.scales.tracking_ang_vel": 2.0}'
  ```

## 6. Limiti e passi successivi
- **Limiti:**
  - rough a soli 16 episodi per RA4 (1 contro 3 cadute: indicativo);
  - RA5 valutato solo sulla pendenza;
  - 1 seed.
- **Possibili passi:**
  - fine-tune su un mix rough + pendenza per avere un'**unica rete**;
  - rough a 40 episodi per RA4;
  - più seed.
