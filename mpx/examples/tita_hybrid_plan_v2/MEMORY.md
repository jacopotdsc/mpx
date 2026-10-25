# MEMORY v2 — fine-tune di RA3 (orientamento e metriche)

Cartella: `/home/jacopo/Desktop/repo_rl/tita_rl/test/mpx/mpx/examples/tita_hybrid_plan_v2`.
Il lavoro precedente è in `../tita_hybrid_plan/` (results.md, MEMORY.md): l'utente l'ha approvato.
**Tutto il lavoro nuovo va in questa cartella** (richiesta dell'utente, 19/10 mattina).

## Richiesta
- Fine-tune brevi di RA3.
- "Sistemare l'orientamento" durante le eval con un piccolo training.
- Analizzare quali metriche si possono migliorare.

## Analisi di partenza (dai JSON della v1)
| \|errore wz\| (rad/s) | piano | rough | perlin |
|---|---|---|---|
| solo MPC | 0.025 | – | – |
| tita_v1 | 0.030 | 0.149 | 0.589 (cade) |
| **RA3** | **0.096** | **0.398** | 0.316 |

- Anche \|errore vx\|: RA3 0.085 sul piano contro 0.032 (MPC) e 0.051 (tita_v1).
- **Causa:** il termine di yaw è il kernel del paper f(x) ≈ −x²/16, piatto vicino a 0 (0.3 rad/s → −0.006 per step), irrilevante rispetto alla termination.

## Modifiche
- **`joystick.py` (mujoco_playground, branch `tita_hybrid`):** reward opzionali `tracking_lin_vel` e `tracking_ang_vel` (kernel exp come tita_v1, normalizzato con il comando, `tracking_sigma` 0.0625). **Default 0** = comportamento dei run precedenti invariato.
- **`eval_robust.py` (copia in v2):** in più \|roll\|, \|pitch\| medi e deriva della direzione a fine episodio (\|yaw − yaw0 − ∫wz_cmd\|). Risultati in `robust_results/`.

## Scadenza: **finire entro le 9:00 del 19/10** (utente).

## Esperimenti
| Ora | Cosa | Esito |
|---|---|---|
| 19/10 04:35 | **RA4:** fine-tune di RA3 (`--load 20261019_003649`) sul rough sparso, `tracking_lin_vel` 1.0, `tracking_ang_vel` 2.0, PPO 3M, 3 eval, seed 5. Log `logs/RA4_*.log` | in corso |
| 19/10 04:35 | Baseline v2 con metriche di orientamento: RA3, solo MPC, tita_v1. Log `logs/eval_v2_baselines.log` | **RA3:** piano 0/40, rough 3/16, perlin 0/40, pendenza 12/40. **Deriva della direzione** RA3: piano **0.89 rad** (MPC 0.24), rough 1.09, perlin 1.19, pendenza 1.17. Roll e pitch piccoli (≤ 0.075). **Il problema di orientamento è lo yaw/heading.** MPC e v1 in corso. |
| 19/10 07:22 | RA4 a 1.6M: durata in training 949 → 962 (reward con il tracking: 21.0 → 23.5) | **finito alle 07:50**: run `20261019_065047`, eval 949 → **962 (best, 1.6M)** → 940; copiato in `checkpoints/20261019_065047_RA4` |
| 19/10 07:54 | Valutazione v2 di RA4 (best), log `logs/eval_v2_RA4.log` → `robust_results/hrc_RA4.json` | **Fatto (08:23).** Cadute: piano 0/40, rough **1/16** (RA3 3/16, v1 3/16), perlin 0/40 (v1 36/40), pendenza 15/40 (RA3 12/40). Orientamento sul piano: wz 0.096 → **0.044**, deriva 0.89 → **0.43 rad**; perlin wz 0.325 → 0.239. **RA4 = nuovo modello consigliato.** `results.md` e `table.md` scritti; `joystick_v2.py` = copia di joystick.py con i nuovi termini. |
| 19/10 08:25 | **LAVORO v2 CONCLUSO** (prima parte) | – |
| 19/10 08:39 | **RA5 (opzione A, scelta dall'utente):** fine-tune di RA4 (`--load 20261019_065047`) sul piano inclinato 10° (`tilt_deg` 10, solo comandi lineari \|vx\| ≤ 1), stessi termini di tracking, PPO 3M, 3 eval, seed 6. **Obiettivo:** abbassare le cadute in pendenza (RA4 15/40, RS4 specialista 9/40) senza perdere il rough. Parte dopo 10 minuti di GPU < 15% (`launch_RA5.sh`, `logs/launch_RA5.log`). **Eval: solo pendenza** (utente 08:40: va bene una rete per terreno; limite **09:30**). | **Fatto.** Partito alle 08:49, finito alle 09:00 (run `20261019_084903`, durata 851 → 872 → 910). **Pendenza 8/40 cadute** (tempo alla caduta 4.5 s; RA4 15, RS4 9, MPC 26), vx 0.090, wz 0.230. Copiato in `checkpoints/20261019_084903_RA5`. **Finale: RA4 per piano, rough e perlin; RA5 per la pendenza.** |
| 19/10 09:12 | **LAVORO v2 CONCLUSO** (results.md riscritto completo) | – |
| – | `make_table.py`: tabella markdown dai JSON (`python3 make_table.py out.md`) | – |
