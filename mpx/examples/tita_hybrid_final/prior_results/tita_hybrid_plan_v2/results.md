# Risultati v2 — fine-tune di RA3 (orientamento) e RA5 (pendenza), 19/10

**Raccomandazione finale, una rete per terreno:**
- **piano, rough, perlin → RA4** (`checkpoints/20261019_065047_RA4/params_best.pkl`);
- **pendenza → RA5** (`checkpoints/20261019_084903_RA5/params_best.pkl`).

## Cosa è stato fatto
- **Problema trovato in RA3** (modello di stanotte): la rete disturba l'orientamento, cioè lo **yaw/heading**. Roll e pitch restano piccoli.
  - Sul piano \|wz err\| è 0.096 contro 0.025 del solo MPC.
  - La deriva della direzione a fine episodio è 0.89 rad contro 0.24.
  - Causa: nel reward del paper il termine di yaw è un kernel piatto vicino a 0.
- **RA4 = RA3 + fine-tune breve** (PPO, 3M step, rough sparso, comandi come tita_v1) con due termini di tracking esponenziali come tita_v1: `tracking_ang_vel` 2.0 e `tracking_lin_vel` 1.0. Checkpoint migliore a 1.6M.
- **RA5 = RA4 + 3M step sul piano inclinato 10°** (solo comandi lineari \|vx\| ≤ 1, stessi termini di tracking), 11 minuti di training. In training la durata è salita da 851 a 872 e poi a 910 su 1000. Valutato solo sulla pendenza (utente: una rete per terreno va bene).
- **Valutazione** con `eval_robust.py`, lo stesso protocollo della v1 più roll, pitch e deriva della direzione:
  - 8 comandi × 5 spawn (40 episodi) per piano, perlin e pendenza;
  - rough (ciottoli) con 2 spawn per comando (16 episodi).

## Tabella
| Terreno | Controllore | Cadute | Tempo alla caduta [s] | \|vx err\| | \|wz err\| | \|roll\| [rad] | \|pitch\| [rad] | Deriva heading [rad] | ‖τ_res‖/‖τ_MPC‖ |
|---|---|---|---|---|---|---|---|---|---|
| flat | solo MPC | 0/40 | – | 0.032 | 0.025 | 0.001 | 0.027 | 0.241 | 0.000 |
| flat | RA3 (stanotte) | 0/40 | – | 0.085 | 0.096 | 0.005 | 0.031 | 0.893 | 0.147 |
| flat | RA4 (fine-tune) | 0/40 | – | 0.079 | 0.044 | 0.003 | 0.027 | 0.428 | 0.166 |
| flat | tita_v1 | 0/40 | – | 0.051 | 0.030 | 0.027 | 0.033 | 0.231 | 0.158 |
| rough | solo MPC | 12/16 | 4.5 | 0.263 | 0.820 | 0.028 | 0.162 | 1.238 | 0.000 |
| rough | RA3 (stanotte) | 3/16 | 7.3 | 0.107 | 0.416 | 0.016 | 0.068 | 1.088 | 0.129 |
| rough | RA4 (fine-tune) | 1/16 | 3.8 | 0.109 | 0.395 | 0.014 | 0.071 | 0.862 | 0.129 |
| rough | tita_v1 | 3/16 | 5.4 | 0.152 | 0.269 | 0.032 | 0.046 | 0.357 | 0.160 |
| perlin | solo MPC | 3/40 | 7.2 | 0.074 | 0.522 | 0.031 | 0.051 | 1.358 | 0.000 |
| perlin | RA3 (stanotte) | 0/40 | – | 0.081 | 0.325 | 0.026 | 0.036 | 1.193 | 0.161 |
| perlin | RA4 (fine-tune) | 0/40 | – | 0.077 | 0.239 | 0.028 | 0.038 | 1.113 | 0.173 |
| perlin | tita_v1 | 36/40 | 3.7 | 0.216 | 0.809 | 0.066 | 0.123 | 1.229 | 0.206 |
| slope10 | solo MPC | 26/40 | 1.8 | 0.173 | 0.279 | 0.050 | 0.087 | 0.925 | 0.000 |
| slope10 | RA3 (stanotte) | 12/40 | 2.2 | 0.129 | 0.317 | 0.062 | 0.075 | 1.173 | 0.183 |
| slope10 | RA4 (fine-tune) | 15/40 | 3.0 | 0.135 | 0.305 | 0.064 | 0.074 | 1.197 | 0.181 |
| slope10 | RA5 (pendenza) | 8/40 | 4.5 | 0.090 | 0.230 | 0.049 | 0.068 | 1.239 | 0.193 |

## Pendenza 10° in sintesi
| Controllore | Cadute | Tempo alla caduta | \|vx err\| | \|wz err\| |
|---|---|---|---|---|
| solo MPC | 26/40 | 1.8 s | 0.173 | 0.279 |
| RA3 | 12/40 | 2.2 s | 0.129 | 0.317 |
| RA4 | 15/40 | 3.0 s | 0.135 | 0.305 |
| RS4 (stanotte, specialista) | 9/40 | 3.1 s | – | – |
| **RA5** | **8/40** | **4.5 s** | **0.090** | **0.230** |

## Lettura
- **Orientamento, migliorato su tutti i terreni allenati e su quelli di generalizzazione:**
  - **piano:** \|wz err\| da 0.096 a **0.044** (−54%), deriva della direzione da 0.89 a **0.43 rad** (−52%);
  - **rough:** wz da 0.416 a 0.395, deriva da 1.09 a 0.86;
  - **perlin:** wz da 0.325 a **0.239** (−26%), deriva da 1.19 a 1.11.
  - Resta peggiore di tita_v1 sul piano (0.030 / 0.23) e sul rough (0.27 / 0.36), ma il distacco si è dimezzato.
- **Cadute (metrica principale):**
  - **piano:** 0/40, invariato;
  - **perlin:** 0/40 contro 36/40 di tita_v1;
  - **rough:** 1/16, contro 3/16 di RA3 e 3/16 di tita_v1;
  - **pendenza 10°:** 15/40 contro 12/40 di RA3. Lieve peggioramento, dentro la variabilità tra ripetizioni (RA3 stanotte faceva 10/40); il fine-tune non vedeva la pendenza.
- **Tracking di vx:** praticamente invariato (piano 0.085 → 0.079).
- **Roll e pitch:** invariati e piccoli (≤ 0.075 rad).
- **Disturbo della rete:** ‖τ_res‖/‖τ_MPC‖ circa uguale (0.13–0.18).

## Conclusione
**Conclusione:** RA4 è il modello per piano, rough e perlin; RA5 per la pendenza. Mantiene la robustezza di RA3 (zero cadute su piano e perlin, meno cadute sul rough) e dimezza l'errore di orientamento sul piano.

**Possibili passi successivi**, non fatti per la scadenza delle 9:00:
- fine-tune su un mix rough + pendenza, per avere un'unica rete al posto di RA4 + RA5;
- valutazione del rough a 40 episodi per RA4.

## File
- **`robust_results/`:** JSON (`mpc`, `hrc_RA3`, `hrc_RA4`, `v1`).
- **`table.md`:** la tabella.
- **`make_table.py`:** la rigenera.
- **`logs/`:** log di training e valutazione.
- **`checkpoints/20261019_065047_RA4/`.**
- **Codice:** in `joystick.py` i nuovi termini `reward_config.scales.tracking_lin_vel` e `tracking_ang_vel` (default 0) e `reward_config.tracking_sigma` 0.0625.
- **Override per riprodurre RA4:**
  ```
  --load 20261019_003649 --env-override '{"terrain_config.scene": "rough_sparse", "randomize_reset": 1.0, "command_config.mode": "joystick", "episode_length": 1000, "reward_config.scales.tracking_lin_vel": 1.0, "reward_config.scales.tracking_ang_vel": 2.0}'
  ```
