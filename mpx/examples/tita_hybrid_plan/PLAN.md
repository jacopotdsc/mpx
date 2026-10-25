# Piano: replica di "Hybrid Balance Control for WBR via Residual RL" (HRC) su Tita: MPC + NN

## Context
Si vuole replicare il paper ACIRS 2025 (Zhang, Lu, Cao, Fang, DOI 10.1109/ACIRS66343.2025.11360885) sul Tita in MuJoCo/MJX, con lo stesso setup di training e gli stessi esperimenti: Task 1 su 0, 5 e 10°, spinte, generalizzazione su traiettorie, parametri diversi del controllore base. L'unica sostituzione è il controllore base: LQR + PD del paper diventano MPC DFCIP + WBC + PD, così come sono.

Si resta **sempre** nel setup MPC + NN residua e si procede **una modifica per volta**: prima la copia fedele più semplice, poi si complica.

Base di partenza: il branch **`residual_tita`**, quello attuale, in entrambi i repo. Si mantiene lo **stile del `joystick.py` attuale**: struttura dei metodi, `default_config` commentata, chiavi di `info`, reward come `_reward_*`/`_cost_*`, `residual_config`.

`residual_tita_v2` serve **solo come riferimento** per la struttura. Da lì si prendono, riscrivendole nello stile attuale:
- il driver SAC `mpx/examples/sac_training.py` e `algorithm_initialization.py`, che su `residual_tita` non esistono;
- il kernel di reward del paper, σ(e)σ(−e) − 0.25 = 1/(eˣ+e⁻ˣ+2) − 0.25;
- il calcolo di θ̇_p del pendolo virtuale (`_balance_posture_errors`);
- la storia delle coppie in `info`.

### Decisioni prese con l'utente
- **mpx non si tocca.** L'MPC e il WBC controllano tutti gli 8 giunti come oggi.
- **La rete agisce solo su 6 motori**, come nel paper: thigh, knee e ruota per lato. L'abduzione (indici 0 e 4) riceve residuo zero, come nella didascalia della Fig. 2: *"the rotation of the hip lateral motors is not considered"*.
- **In mujoco_playground si modifica solo `joystick.py`.** In `base.py` `action_size` vale `nu`, cioè 8. Si aggiunge in `joystick.py` l'override `action_size` → 6, così l'env E2E, che usa anch'esso `base.py`, resta invariato. `_default_pose` non serve più e si toglie. Nessun nuovo env.
- **Comandi in velocità all'MPC fin da subito**, senza loop di posizione. Il "vai a D m" del paper diventa un profilo di velocità con ∫v = D. x_ref(t) = ∫v_ref serve per obs, reward e metriche.
- **CoM fisso a 0.40 m.** Group 2, Task 2 e il test multi-altezza sono esclusi.
- **Nessuna domain randomization**, mai. **Perturbazioni solo in eval, sul piano.**
- **Setup minimo:** niente randomness al reset, noise a 0, deadband a 0, niente push, solo piano. Si complica solo dopo che la policy impara.
- **Niente DDPG e niente RL puro.** Il confronto è MPC-only contro MPC + SAC residuo.

### Note dal paper
- **Blending:** τ_C = τ_ζ + τ_π, dove la NN *"directly generates torques τ_π ∈ ℝ⁶"*.
  - Il paper non dà né scala né Nm massimi, e non dice se l'uscita passa per una tanh.
  - Con τ_π = a ⊙ τ_res_max, a = 0 dà residuo esattamente zero, cioè il controllore base puro.
- **Reward:** r = r_θlim + r_b + r_z + r_ψ, tutti con il kernel f.
  - r_b confronta q_b = [x_w, ẋ_w, θ_p, θ̇_p] con il desiderato, senza θ_pd: limita solo θ̇_p.
  - Pesi non dati.

## Fase 0 — Cartella del piano e memoria (primo passo dopo l'approvazione)
1. Creare il branch locale `tita_hybrid` da `residual_tita` (lo stash `snapshot lite3 18 ottobre` resta dov'è) in `mujoco_playground` (`/home/jacopo/miniconda3/envs/mjpl/lib/python3.11/site-packages/mujoco_playground`) e in `mpx` (`/home/jacopo/Desktop/repo_rl/tita_rl/test/mpx`). In mpx servono solo i nuovi file di piano, eval e training preset. Nessun push se non richiesto.
2. Creare `mpx/examples/tita_hybrid_plan/` con:
   - `PLAN.md`: questo piano.
   - `MEMORY.md`: file di ripresa per le nuove sessioni, **aggiornato a ogni modifica o esperimento**. Contiene stato e fase corrente, branch e commit, file toccati e perché, parametri non specificati dal paper con valore e motivo, esperimenti (comando, checkpoint, esito) e prossimo passo.
   - `paper_notes.md`: estrazione tecnica del paper, con equazioni, stato, azione, reward, iperparametri, scenari e Tab. I/II con i numeri originali.
   - Più avanti: `eval_hybrid.py`, `plots/`, `results.md`.
3. Creare una memoria di progetto in `~/.claude/projects/-home-jacopo-Desktop-repo-rl/memory/` che punta a `MEMORY.md`.

## Fase 1 — Copia fedele, setup minimo, piano (solo `mujoco_playground/_src/locomotion/tita/joystick.py`)
1. **Azione e blending**
   - `action_size` → 6, mappato sui motori 1, 2, 3, 5, 6, 7 (thigh, knee e ruota ×2). Il residuo sui motori 0 e 4 è 0.
   - `_combine_torque` si riscrive nello stile attuale:
     - τ_ζ resta il `tau_nominal` attuale, cioè WBC feedforward più PD esterno;
     - τ_π = a ⊙ τ_res_max, che sostituisce il PD residuo, `scale` e `tau_limit_*` in `residual_config`;
     - τ_C = clip(τ_ζ + τ_π, ±forcerange);
     - `residual_config.enabled` resta, perché con False si ha MPC-only per l'eval.
   - Il blending PD-offset viene sostituito, senza flag di confronto. `_residual_joint_targets` e `_default_pose` si tolgono.
   - **τ_res_max** non è dato dal paper. Si misura |τ_ζ| per motore su rollout MPC-only del task e si parte da quell'ordine di grandezza, clippato al limite. Sweep (×0.5, ×1, ×2) solo nel tuning.
   - **Low-pass opzionale:** `a_applied = s·a_applied(t−1) + (1−s)·a_t`, con s = 0 di default. Si prova nel tuning se compaiono oscillazioni di coppia.
2. **Task (Group 1 sul piano)**
   - A ogni reset: D ~ U(1, 4) m, wz = 0, altezza 0.40.
   - v_ref(t) trapezoidale fino a ∫v = D, poi 0. v_max e a_max non sono nel paper: default 0.5 m/s e 0.5 m/s², in MEMORY.md.
   - `info["command"] = [v_ref, 0]` va all'MPC come oggi. `sample_command` e LPF del v2 vengono disattivati nel task.
   - `info` contiene x_ref, v_ref e x_w, cioè l'avanzamento del punto medio delle ruote dallo spawn.
3. **Setup minimo**
   - Config: `randomize_reset = 0`, `noise_config.level = 0`, `pert_config.enable = False`, `action_deadband = 0`, `terrain = ["flat"]`, `sparse_obstacles.enabled = False`.
   - Va controllato che nel reset non resti altra randomness, a parte D.
4. **Osservazione actor (paper III-A)**
   - **s_r:** q_b = [x_w − x_ref, ẋ_w, θ_p, θ̇_p], p_c e ṗ_c nel frame {C}, q e q̇ dei giunti delle gambe, velocità delle ruote.
   - **s_g:** v_ref, z_cd = 0.4, ż_cd = 0, q_ψd = 0. θ_pd è omesso.
   - **s_e:** e_t ed e_{t−1}, con e = [x_w − x_ref, ẋ_w − v_ref, θ̇_p, z_c − 0.4, roll, pitch, yaw rate].
   - **s_τ:** τ_ζ^t, τ_ζ^{t−1}, τ_π^{t−1} e τ_C^{t−1}, normalizzate.
   - Il paper dichiara 30 dimensioni, ma i componenti elencati sono di più: l'incongruenza va in MEMORY.md.
5. **Reward del paper**
   - **r_b:** f(x_w − x_ref), f(ẋ_w − v_ref), f(θ̇_p).
   - **r_z:** f(z_c − 0.4), con z_c = quota del CoM rispetto al punto medio delle ruote.
   - **r_ψ:** f(roll), f(pitch), f(yaw rate).
   - **r_θlim:** `_cost_joint_pos_limits` su hip forward e knee.
   - Pesi unitari, termination del v2, altri termini v2 a 0.
6. **Verifiche**
   - Con a = 0 la traiettoria deve essere identica a MPC-only.
   - MPC-only deve seguire il profilo fino a 2.5 m.
   - Obs e reward devono essere finite.
   - Smoke test SAC di 20k step.

## Fase 2 — Training della copia fedele (mpx: solo file di training, nessuna modifica a MPC/WBC)
- Portare da v2 `sac_training.py` e `algorithm_initialization.py` in `mpx/examples/`, poi aggiungere a `train_srbd.py` (stile attuale) il dispatch SAC con il preset `HYBRID_PAPER_SAC`, lanciato con `--preset hybrid_paper`.
- **Parametri del paper:**
  - actor e critic 1024-512 tanh;
  - lr 1e-3, γ 0.99, α iniziale 0.1, batch 512, reward_scaling 10;
  - update ogni 1000 transizioni;
  - 2M transizioni = 40 epoche, con eval ogni 50k;
  - episodio massimo 5000 step;
  - 3 seed.
- **Non specificati:** 64 update per blocco, replay 1M, τ 0.005, `num_envs` 10, ultimo layer a zero, nessun critic burn-in. Vanno in MEMORY.md.
- **Criterio per andare avanti:** HRC batte MPC-only sull'RMSE di x_w e θ_p nel Task 1 sul piano.

## Fase 3 — Complicazioni, una alla volta (solo se la Fase 2 impara)
- **3a Slope (Group 1 completo):**
  - Nuova `xmls/scene_slopes.xml` con tre corsie, 0°, 5° e 10°: tratto piano, rampa di circa 6 m, plateau.
  - Ground in `consts.FLOOR_GEOMS` e sensori `FL/FR_<geom>_found`.
  - Spawn per corsia con `terrain_spawn_points`, estendendo il check in `Joystick.__init__`. Corsia uniforme a ogni reset.
  - **Prima verifica MPC-only su rampa:** il reference generator usa altezze assolute (z_com_ref = 0.4). Se l'MPC fallisce in salita lo riporto e decidiamo insieme, senza toccare mpx da solo.
  - Poi retrain.
- **3b Randomness al reset, per gradi:** prima la posizione di spawn, poi piccole velocità iniziali. Mai DR.
- **3c Tuning finale:** sweep su τ_res_max e sul low-pass s.

## Fase 4 — Evaluation (`tita_hybrid_plan/eval_hybrid.py`)
Rollout deterministici, 1 env, setup minimo, MPC-only contro HRC. Si riusa la pipeline rollout → CSV → plot di `mpx/examples/compare.py` e `analysis_tita/`.
- **E1 (Tab. I, Fig. 4, 6b):** Task 1, D = 2.5 m, sul piano e poi a 5° e 10° (dopo la 3a). RMSE di x_w, v, θ_p, z_c e β; errore di x_w a regime.
- **E2 (Fig. 7):** Task 1 sul piano con spinte da 10, 20 e 30 N a 12, 20 e 32 s, opposte al moto.
  - La durata, 0.1 s, va in MEMORY.md.
  - `scheduled_pushes` su `xfrc_applied` del torso, attivo solo in eval.
  - Si misurano x_w, θ_p, spostamento massimo e tempo di recupero.
- **E3 (Fig. 8):** comandi di velocità derivati da una traiettoria polinomiale di 5° ordine tra punti casuali e da una multi-target ad alta dinamica. RMSE di x_w e θ_p, overshoot.
- **E4 (Tab. II, Fig. 10):**
  - Il paper qui cambia i guadagni LQR. Il nostro analogo è cambiare i pesi dell'MPC, ma questo vuol dire toccare `config_dfcip.py`.
  - Lo propongo solo alla fine e chiedo prima di farlo.
- **Curve di training (Fig. 3a):** reward di eval per epoca, media ± std sui 3 seed. MPC-only è una linea costante.
- **Output:** `results.md` con i nostri numeri affiancati a quelli del paper.

## File critici
- `mujoco_playground/_src/locomotion/tita/joystick.py`: `action_size`, `_combine_torque`, `_residual_joint_targets`, `_get_obs`, `_get_reward`, `_balance_posture_errors`, `reset`/`step`, `default_config`. È l'unico file Python modificato nel repo.
- `.../tita/tita_constants.py` e nuovo `.../tita/xmls/scene_slopes.xml`: solo in 3a.
- `mpx/examples/train_srbd.py` (preset), `mpx/examples/sac_training.py`, `mpx/examples/algorithm_initialization.py` (riuso).
- `mpx/examples/tita_hybrid_plan/`: PLAN.md, MEMORY.md, paper_notes.md, eval.
