# Confronto: Tita residuale vs Residual Policy Learning vs RL-augmented MPC

Questo documento confronta tre impostazioni di residual RL:

- **RPL**: Silver, Allen, Tenenbaum, Kaelbling, *Residual Policy Learning* (2018). Fonte: codice `github.com/k-r-allen/residual-policy-learning` (cartelle `tensorflow/experiment` e `rpl_environments`).
- **HECTOR**: Kamohara et al., *RL-augmented Adaptive Model Predictive Control for Bipedal Locomotion over Challenging Terrain* (ICRA 2026). Fonte: codice `github.com/rl-augmented-mpc/IsaacLab4.5` (branch `devel`, task `HECTOR-ManagerBased-RL-SAC-Rough-Blind`) e il fork `github.com/rl-augmented-mpc/rl_games` (`rl_games/algos_torch/sac_agent.py`).
- **Tita**: la configurazione attuale, cioè `TitaJoystickFlatTerrain` + `train_srbd.py` con PPO, lanciata con `--train-cmd 1.0 0.0 --reward-override analysis_tita/tracking_only_rewards.json`. Dove serve è indicata anche la variante SAC (`sac_training.py`).

Dove il paper e il codice non coincidono è riportato il valore del codice e la discrepanza è segnalata. Il contenuto di `tracking_only_rewards.json` e di `algorithm_initialization.py` non era disponibile: per il primo si assume tracking + termination con gli altri termini a zero, per il secondo si assume che il prior agisca sull'azione deterministica tanh(loc), come indica il commento in `train_srbd.py`.

Ogni tabella ha un'ultima colonna, **"Chi coincide / in cosa differiscono"**, che riassume quali impostazioni sono equivalenti su quella riga e qual è la differenza sostanziale.

---

## 1. Robot e task

| | RPL | HECTOR | Tita (PPO) | Chi coincide / in cosa differiscono |
|---|---|---|---|---|
| Robot | braccio Fetch 7-DoF (più Pusher a 7 giunti) | bipede umanoide HECTOR, 10 giunti, circa 13.9 kg | bipede su ruote, 6 giunti gamba + 2 ruote | HECTOR e Tita sono robot da locomozione; RPL è manipolazione |
| Stabilità | statico, nessun rischio di caduta | dinamicamente instabile | dinamicamente instabile (pendolo inverso su ruote) | HECTOR ≈ Tita: un errore del residuo può far cadere il robot; in RPL no |
| Task | push, pick-and-place, hook, complex hook, MPC push | locomozione su scale, stepping stones, superfici scivolose | tracking di velocità su piano | tutti e tre migliorano un controllore esistente; solo HECTOR introduce difficoltà ambientali che il controllore non gestisce |
| Comando | posizione goal dell'oggetto | vx fisso a 0.5 m/s, wz = 0 | vx target fisso a 1.0 m/s, con rampa LPF 0.02 da 0; altezza target 0.40 m | HECTOR ≈ Tita: comando di velocità fisso; Tita aggiunge la rampa |
| Durata episodio | 50 step (default Fetch) | 10 s = 1000 step a 100 Hz | 1000 step = 10 s a 100 Hz | HECTOR = Tita; RPL ha episodi 20 volte più corti |
| Simulatore | MuJoCo (gym robotics, mocap) | IsaacLab / PhysX | MuJoCo MJX | RPL e Tita usano MuJoCo; solo Tita è in JAX/GPU |
| Terreno / curriculum | nessuno | curriculum di terreno | piatto, nessun curriculum | RPL ≈ Tita (nessun curriculum); HECTOR è l'unico con curriculum |
| Randomizzazione / disturbi | varianti "noisy" e "miscalibrated" dei task | terreno e attrito | nessuna: rumore 0, perturbazioni off, nessuna domain randomization | Tita è l'unico senza alcuna fonte di discrepanza rispetto al controllore |
| Reset | posizioni casuali di oggetto e goal | x, y ±0.5 m, yaw ±π | x, y ±0.5 m, yaw fisso, velocità nulle (le righe di randomizzazione sono commentate) | HECTOR ≈ Tita sulla posizione; Tita non randomizza yaw né velocità |

## 2. Controllore base

| | RPL | HECTOR | Tita | Chi coincide / in cosa differiscono |
|---|---|---|---|---|
| Tipo | controllori a mano (macchina a stati con `get_move_action`); MPC solo in `ResidualMPCPush` | MPC convesso SRBD + Raibert + swing con Bézier cubica + low-level (Jacobiano in stance, PD+IK in swing) | MPC DFCIP + WBC + PD esterno | HECTOR ≈ Tita: MPC a modello ridotto + controllore di basso livello; RPL usa per lo più euristiche |
| Frequenza | 1 decisione per step dell'env (20 substep) | MPC e RL a 100 Hz; low-level a 200 Hz in training (`sim.dt=1/200`, `decimation=2`), 400 Hz nel PLAY e nel paper | MPC e RL a 100 Hz; fisica e PD a 500 Hz (5 substep) | HECTOR = Tita a 100 Hz per MPC e RL; Tita ha il low-level più veloce |
| Implementazione | Python, CPU, MPI | C++ su CPU (24 env) oppure batch su GPU (CusADi) | JAX/MJX su GPU, batch | HECTOR (versione GPU) ≈ Tita: MPC batch su GPU |
| A residuo zero | buono ma imperfetto | buono sul piano, fallisce su scale alte e su μ basso | buono fino a circa 1.5 m/s con un gradino, circa 3 m/s con la rampa | in tutti e tre il controllore è già buono; a 1.0 m/s in rampa il Tita è dentro l'inviluppo nominale, quindi c'è poco da correggere |

## 3. Azione e blending

| | RPL | HECTOR | Tita | Chi coincide / in cosa differiscono |
|---|---|---|---|---|
| Architettura | parallela, nello spazio dell'azione | gerarchica: il residuo modifica il modello e i parametri dell'MPC | parallela, in coppia | RPL ≈ Tita (parallela); HECTOR è l'unico gerarchico |
| Dimensione dell'azione | 4 | 15 | 8 (6 gambe + 2 ruote), con `leg_only_actions=False` | dimensioni diverse; Tita ne usa davvero solo 6 |
| Significato | dx, dy, dz della pinza + comando gripper | accelerazioni residue lineari (3) e angolari (3), errori di massa inversa (3) e di inerzia inversa (3), coefficiente del dt dell'MPC (1), Δh dello swing (1), Δ_cp (1) | offset del target di posizione dei giunti gamba (6) e offset di velocità delle ruote (2) | RPL corregge un comando di alto livello, HECTOR il modello del controllore, Tita il basso livello |
| Combinazione | `clip(2·u + controller(s), −1, 1)` | `A + A_res`, `B + B_res`, swing e dt modificati, poi l'MPC risolve | `clip(τ_nominale + τ_rl, ±actuator_forcerange)` | RPL ≈ Tita: somma + clip finale; in HECTOR non c'è somma sull'uscita |
| Azione 0 = controllore? | sì | sì (range simmetrici) | sì | tutti e tre coincidono |
| Vincoli rispettati dopo il residuo | nessuno | sì: attrito, forza massima, line contact | no: il residuo bypassa MPC e WBC, resta solo la saturazione | RPL ≈ Tita (nessun vincolo); solo HECTOR li conserva |
| Asimmetrie | nessuna | Δh clippato a ≥ 0 (`negative_action_clip_idx=[13]`) | — | RPL = Tita (simmetrici); HECTOR ha una dimensione monolaterale |

## 4. Osservazioni

| | RPL | HECTOR | Tita | Chi coincide / in cosa differiscono |
|---|---|---|---|---|
| Actor | stato Fetch (pinza, oggetto, velocità relative, gripper) + goal | gravità proiettata (3), vel lin (3) e ang (3) della base, comando (3), q (10), dq (10), azione precedente (9 nel paper), fase di swing (2), footholds (4), posizione piedi (6), riferimento piedi (6) | vel lin (3), gyro (3), gravità (3), q gamba (6), dq (8), azione (8), comando (2), errore di altezza CoM (1), errori di equilibrio (2), τ_nominale/limite (8), τ_rl/limite (8) | HECTOR ≈ Tita: propriocezione + comando + azione precedente + info del controllore |
| Info dal controllore nell'obs | no | sì: footholds e riferimento dei piedi | sì: coppie nominali e residue | HECTOR ≈ Tita, ma HECTOR dà il piano (riferimenti), Tita l'uscita (coppie) |
| Critic | Q(obs, goal, u/max_u) | Q(obs, azione), doppio Q | V(privileged_state): include actuator_force, contatti, velocità piedi, forza esterna, altezza | RPL ≈ HECTOR (Q simmetrico); Tita è l'unico con un critic asimmetrico privilegiato |
| Normalizzazione | sì: clip_obs 200, poi clip a ±5 dopo la normalizzazione | sì (`normalize_input`), clip obs ±50 | sì: running stats, clip ±10 dopo la normalizzazione | tutti normalizzano; cambiano solo le soglie di clip |

## 5. Reward e terminazione

| | RPL | HECTOR | Tita | Chi coincide / in cosa differiscono |
|---|---|---|---|---|
| Tipo | sparse | densa | densa | HECTOR = Tita; RPL è sparse |
| Termini attivi | −1 per step finché il goal non è raggiunto, poi 0 | tracking lin 0.1, ang 0.1, altezza 0.1; lin_vel_z −0.1; action_rate −0.015; angolo gamba-busto −1.0; distanza piedi −0.5; contatto ginocchio −5.0 | tracking lin 1.0, ang 0.5; termination −500 | HECTOR ≈ Tita sul tracking (stessa forma esponenziale); HECTOR ha più termini di regolarizzazione, Tita solo tracking + caduta |
| Pesi del paper (Tab. III) vs codice | — | il paper riporta tracking lin 1.0, ang 0.5, altezza 0.1 e altri pesi diversi | — | i pesi del paper HECTOR coincidono con quelli del Tita (1.0 / 0.5); quelli del codice HECTOR no |
| Penalità sul residuo nel reward | no | no (solo action_rate) | no (`residual_torque = 0`) | nessuno dei tre penalizza il residuo nel reward: lo fanno tutti nella loss |
| Moltiplicazione per dt | no | sì, × step_dt = 0.01 (RewardManager di IsaacLab) | sì, × dt = 0.01 | HECTOR = Tita |
| Clip del reward | ritorni clippati a [−1/(1−γ), 0] = [−50, 0] | nessuno | ±10000 (di fatto nessuno) | HECTOR ≈ Tita; RPL è l'unico con Q limitato esplicitamente |
| Terminazione | nessuna (episodi a lunghezza fissa) | orientamento > 30°, base troppo bassa, contatto della base, uscita dal terreno | upvector < 0, contatto della base, stato non finito | HECTOR ≈ Tita (entrambi terminano sul contatto della base); HECTOR è più severo sull'orientamento (30° contro 90°) |
| Penalità di caduta | — | nessuna esplicita (solo la perdita dei reward futuri) | −500 × dt = −5 | solo il Tita ha una penalità di terminazione esplicita |
| HER | sì (`future`, k = 4) | no | no | HECTOR = Tita |

## 6. Algoritmo e iperparametri

| | RPL | HECTOR | Tita PPO (attuale) | Tita SAC (alternativa) | Chi coincide / in cosa differiscono |
|---|---|---|---|---|---|
| Algoritmo | DDPG + HER | SAC (rl_games, fork) | PPO (brax 0.14.2) | SAC (`sac_training.py`, losses di brax) | HECTOR = Tita SAC (off-policy, max-entropy); Tita PPO è l'unico on-policy |
| Policy | deterministica, `max_u·tanh(z)` | gaussiana tanh-squashed | gaussiana tanh-squashed | gaussiana tanh-squashed | HECTOR = Tita (entrambe le varianti); RPL è deterministica |
| Reti | 3 × 256, ReLU | (512, 256, 128) ELU, actor e critic separati | (512, 256, 128) ELU | (512, 256, 128) ELU | HECTOR = Tita (identiche) |
| Learning rate | actor 1e-3, critic 1e-3 | actor 1e-4, critic 1e-4, α 1e-4 | 1e-5 condiviso, adaptive KL tra 1e-6 e 1e-5 (desired_kl 0.01) | actor 1e-4, critic 1e-3, α 1e-3 | HECTOR = Tita SAC sull'actor; Tita SAC ha critic e α 10 volte più veloci; Tita PPO è molto più lento |
| γ | 0.98 (= 1 − 1/T) | 0.98 | 0.99 | 0.99 | RPL = HECTOR; il Tita guarda più avanti (orizzonte efficace 100 step contro 50) |
| Batch | 256 | 512 | 256 per minibatch, 32 minibatch, unroll 20 | 512 | HECTOR = Tita SAC |
| Update | 40 batch per ciclo, 50 cicli per epoca | 1 gradient step per ogni step vettoriale dell'env | 2 epoche per batch, clip ε 0.3, GAE λ 0.95, vf coef 0.5, grad norm 1.0 | 64 update ogni 1000 transizioni | rapporti update/dati diversi in tutti e quattro |
| Target / soft update | polyak 0.95 (τ = 0.05) | τ = 5e-3 | — | τ = 5e-3 | HECTOR = Tita SAC; RPL aggiorna il target 10 volte più in fretta |
| Replay | 1e6 | 1e6 | — | 1e6 | RPL = HECTOR = Tita SAC |
| Env paralleli | 2 per thread MPI | 24 (MPC su CPU) o 1024 (GPU) | 1024 | 10 | HECTOR (GPU) = Tita PPO |
| Transizioni totali | dipende dal task | circa 5.8M (10000 epoche × 24 step × 24 env, config del README) | 20M | 20M | Tita usa circa 3.5 volte i dati di HECTOR |
| Mixed precision | no | sì | no (x64 attivo) | no | HECTOR in mezza precisione, Tita in doppia |

## 7. Inizializzazione ed esplorazione

| | RPL | HECTOR | Tita PPO | Tita SAC | Chi coincide / in cosa differiscono |
|---|---|---|---|---|---|
| Ultimo layer dell'actor | pesi a zero (`nn_last_zero`) | pesi e bias a zero (`const_initializer 0`) | zero (`initialize_residual`) | zero (`initialize_residual`) | tutti coincidono: uscita iniziale = 0 |
| Std iniziale (pre-tanh) | — (deterministica) | σ = exp(0) = 1.0 (log_std clamp [−5, 2]) | 0.2 (`INIT_STD`) | 0.2 (`INIT_STD`) | Tita PPO = Tita SAC; HECTOR parte 5 volte più largo |
| Std effettiva dell'azione all'inizio | 0.2 gaussiana + 30% uniforme, circa 0.36 | std di tanh(N(0,1)) ≈ 0.63 | ≈ 0.19 | ≈ 0.19 | il gaussiano di RPL (0.2) coincide con il Tita; RPL aggiunge però le azioni uniformi; HECTOR è circa 3 volte più largo |
| Sorgente dell'esplorazione | rumore aggiunto a mano | campionamento dalla policy | campionamento dalla policy | campionamento dalla policy, più `burn_in_explore_std=0.1` durante il burn-in | HECTOR = Tita PPO; Tita SAC durante il burn-in è un ibrido con RPL |
| Termine di entropia | nessuno | α appreso, iniziale 1.0, target −1.0·dim = −15 | `entropy_cost = 0` | α appreso, iniziale 0.1, target −0.5·dim = −3 | RPL ≈ Tita PPO (nessuna spinta entropica); HECTOR e Tita SAC usano entrambi α appreso, ma HECTOR con α 10 volte più grande e target doppio per dimensione |
| Eval | senza rumore | media | deterministica (media) | deterministica (media) | tutti coincidono |

## 8. Warm-up del critic

| | RPL | HECTOR | Tita PPO | Tita SAC | Chi coincide / in cosa differiscono |
|---|---|---|---|---|---|
| Presente | sì | no (`num_warmup_steps: 0`) | sì (`CRITIC_WARMUP=True`) | sì (`critic_burn_in`) | RPL ≈ Tita; HECTOR non lo fa |
| Meccanismo | `pi_lr = 0` | — | fase PPO separata con stop-gradient sulla policy | actor e α congelati (parametri e stato Adam) | stesso effetto (actor fermo); Tita SAC congela anche α, RPL non ne ha bisogno |
| Durata | finché \|Δ mean(actor_loss)\| tra due epoche è sotto la soglia (0.5 per push e pick-and-place, 0.7 per hook); l'epoch 0 è sempre burn-in | — | fissa: 1M step | adattiva: variazione relativa di Q(s, mode(π)) < 1% tra finestre di 10 blocchi, minimo 20 e massimo 200 blocchi | RPL ≈ Tita SAC (criterio di convergenza su Q della policy); Tita PPO usa una durata fissa |
| Rivalutato dopo lo sblocco | sì, a ogni epoca | — | no | no (sblocco a senso unico) | Tita PPO = Tita SAC; solo RPL può ricongelare l'actor |
| Esplorazione nel warm-up | coin flip: 50% dei rollout senza rumore | — | quella della policy | coin flip + rumore extra 0.1 | RPL ≈ Tita SAC; Tita PPO non fa coin flip |
| Dopo il warm-up | stesso ottimizzatore (i momenti Adam si aggiornano anche con lr 0) | — | nuovo `ppo.train`: Adam ripartito, seed + 1, critic ripristinato | stesso learner | RPL ≈ Tita SAC (continuità); Tita PPO riparte con un ottimizzatore nuovo |

Nota sul codice RPL: in `train_staged.py` le variabili sono scambiate (`losses.append(actor_loss)`), quindi la soglia si applica all'actor loss, cioè a −Q(s, π(s)), e non alla critic loss come scrive il paper.

---

## 9. Catena delle scalature: dall'uscita della rete all'effetto fisico

Ogni riga è uno stadio della catena. La colonna "dove" indica se lo stadio avviene nella rete, nella loss o nell'env; l'ultima colonna indica lo stadio corrispondente negli altri due lavori.

### 9.1 RPL (es. `ResidualFetchPush`)

| Stadio | Formula | Range | Dove | Equivalente negli altri / differenze |
|---|---|---|---|---|
| Uscita grezza dell'ultimo layer | z | ℝ, 0 all'inizio | rete | = μ di HECTOR e loc del Tita; RPL non ha un'uscita di std |
| Azione della policy | u = max_u · tanh(z), con max_u = 1 | [−1, 1] | rete (`actor_critic_*.py`) | = tanh(μ) di HECTOR e tanh(loc) del Tita, ma senza campionamento |
| **Termine `action_l2`** | `action_l2 · mean((u / max_u)²)` = `1.0 · mean(tanh(z)²)` | — | loss: agisce su u, cioè dopo la tanh e prima di ogni scala dell'env | = prior del Tita (stessa quantità, stesso coefficiente, stessa media); HECTOR invece lo applica a μ prima della tanh e lo somma |
| Rumore di esplorazione | u ← clip(u + 0.2·max_u·N(0,1), −1, 1); con p = 0.3, u ← U(−1, 1) | [−1, 1] | rollout (`get_actions`) | Tita SAC lo replica nel burn-in (`burn_in_explore_std`); HECTOR e Tita PPO campionano dalla gaussiana della policy |
| Residuo nell'env | r = 2 · u | [−2, 2] | env (`residual_action = 2. * residual_action`) | = `action_scale_pos` del Tita e la mappa affine di HECTOR: un fattore di scala fisso fuori dalla rete |
| Somma e clip | a = clip(r + controller(s), −1, 1) | [−1, 1] | env | ≈ Tita (`clip(τ_nom + τ_rl)`); in HECTOR non c'è una somma sull'uscita |
| Effetto fisico | Δpos pinza = 0.05 m · a[0:3]; gripper = a[3] | al massimo 5 cm per step | gym `FetchEnv._set_action` | Tita: Kp_res = 20 converte rad in Nm; HECTOR: l'MPC converte i parametri in forze |

In unità fisiche: un'unità di u vale 0.10 m di spostamento richiesto, che la clip finale limita a 0.05 m per step. Il residuo può quindi sovrascrivere del tutto il controllore.

### 9.2 HECTOR (`BlindLocomotionMPCActionResAll`)

| Stadio | Formula | Range | Dove | Equivalente negli altri / differenze |
|---|---|---|---|---|
| Uscita grezza | [μ, log_std] = trunk(obs) | ℝ, 0 all'inizio | rete | = [loc, scale_raw] del Tita; RPL ha solo z |
| Std | σ = exp(clamp(log_std, −5, 2)) | [0.0067, 7.39], 1.0 all'inizio | rete (`DiagGaussianActor`) | Tita: σ = softplus(·) + 0.001, impostata a 0.2; parametrizzazione diversa, stesso ruolo |
| **Termine `reg_loss`** | `0.01 · sum_i μ_i²` | — | loss: agisce su μ **prima** della tanh | RPL e Tita lo applicano dopo la tanh e mediano; HECTOR ha coefficiente 100 volte più piccolo ma somma su 15 dimensioni |
| Azione | a = tanh(μ + σ·ε) in training, tanh(μ) in eval | [−1, 1] | rete (`SquashedNormal`) | = Tita (identico) |
| Clip dell'env wrapper | clip(a, ±`clip_actions` = 1.0) | [−1, 1] | `RlGamesVecEnvWrapper` | = clip in `step()` del Tita; ridondante con la tanh in entrambi |
| Clip delle azioni negative | a_13 ← clamp(a_13, 0, 1) | Δh ∈ [0, 1] | `process_actions` | nessun equivalente in RPL né nel Tita |
| Mappa affine | p = lb + (a + 1)(ub − lb)/2 = a · ub (range simmetrici) | per dimensione | `process_actions` | = `action_scale_pos` del Tita (scala per dimensione); RPL usa un fattore unico (×2) |
| Effetto fisico | va nel modello SRBD, nello swing e nel dt dell'MPC | vedi §11 | MPC C++ | unico: negli altri il residuo arriva diretto all'attuatore o al comando |

### 9.3 Tita (PPO attuale)

| Stadio | Formula | Range | Dove | Equivalente negli altri / differenze |
|---|---|---|---|---|
| Uscita grezza | [loc, scale_raw] (2 × 8) | ℝ; loc = 0 all'inizio | rete | = [μ, log_std] di HECTOR |
| Std | σ = softplus(scale_raw) + 0.001 | 0.2 all'inizio | brax `NormalTanhDistribution` | HECTOR: exp, 1.0 all'inizio; RPL: nessuna |
| **Termine `LOSS_NN_ACTION`** | `1.0 · mean_{batch, dim}(tanh(loc)²)` (assunto, vedi introduzione) | — | loss: agisce dopo la tanh, mediato su 8 dimensioni | = `action_l2` di RPL (stessa quantità, stesso λ = 1, stessa media); diverso da HECTOR (pre-tanh, somma, 0.01) |
| Azione | a = tanh(loc + σ·ε) in training, tanh(loc) in eval | [−1, 1] | rete | = HECTOR |
| Clip nell'env | a ← clip(a, −1, 1) | [−1, 1] | `step()` | = clip del wrapper di HECTOR |
| Offset gamba | Δq = a_gamba · `action_scale_pos` = a · [0.20, 0.35, 0.50] rad (hip, thigh, knee) | ±0.20 / ±0.35 / ±0.50 rad | `_residual_joint_targets` | ≈ mappa affine di HECTOR (scala per dimensione); ≈ ×2 di RPL |
| Offset ruote | Δdq = a_ruota · `action_scale_vel` = a · 0 | 0 | `_residual_joint_targets` | nessun equivalente: dimensioni d'azione senza effetto (né RPL né HECTOR ne hanno) |
| Coppia residua gamba | τ_rl = Kp_res · Δq + Kd_res · (dq_des_rl − dq_des_wbc) = 20 · Δq + 1 · 0 | ±4 / ±7 / ±10 Nm | `_combine_torque` | ≈ ×0.05 m di RPL (conversione in unità fisiche); in HECTOR la conversione la fa l'MPC |
| Coppia residua ruote | τ_rl = Kd_wheel_res · Δdq = 0 · 0 | 0 | `_combine_torque` | — |
| Somma e clip | τ = clip(τ_nominale + τ_rl, ±forcerange) | limiti degli attuatori | `_combine_torque` | ≈ RPL (`clip(2u + controller)`) |

Il termine Kd della gamba è sempre nullo, perché l'offset di velocità viene applicato solo agli indici delle ruote: la coppia residua della gamba è quindi una pura coppia feedforward, `20 · scale · a`.

---

## 10. Scale di reward e di Q

| | RPL | HECTOR | Tita PPO | Tita SAC | Chi coincide / in cosa differiscono |
|---|---|---|---|---|---|
| Reward scaling dell'algoritmo | nessuno | `reward_shaper.scale_value = 1.0` | `reward_scaling = 1.0` | `reward_scaling = 1.0` | tutti a 1 (o assente): nessuno riscala il reward nell'algoritmo |
| Moltiplicazione per dt | no | × 0.01 | × 0.01 | × 0.01 | HECTOR = Tita |
| Reward massimo positivo per step | 0 (reward ∈ {−1, 0}) | (0.1 + 0.1 + 0.1) × 0.01 = 0.003 | (1.0 + 0.5) × 0.01 = 0.015 | 0.015 | Tita ha un reward per step 5 volte più grande di HECTOR |
| Penalità di terminazione | — | nessun termine dedicato | −500 × 0.01 = −5 (pari a circa 333 step di reward massimo) | −5 | solo Tita |
| Range di Q (o V) | [−50, 0] (clip esplicita) | Q del task ≲ 0.003/(1 − 0.98) = 0.15 | V ≲ 0.015/(1 − 0.99) = 1.5, fino a −5 alla caduta | come PPO | scale molto diverse: RPL ~50, Tita ~1.5-5, HECTOR ~0.15 |
| Contributo dell'entropia al target | — | α · (−log π) per step: con α = 1 e −log π ≈ +10 (vedi sotto) vale circa 10 per step, quindi una soft-Q iniziale di circa 500 | — | α = 0.1, −log π ≈ +1.4 per 6 dimensioni con σ = 0.2: circa 0.14 per step | HECTOR e Tita SAC hanno entrambi entropia nel target, ma in HECTOR domina il task di 3000 volte, nel Tita SAC è dello stesso ordine del task |
| Advantage | — | — | normalizzato (`normalize_advantage=True`): scala circa 1 qualunque sia il reward | — | solo PPO è indipendente dalla scala del reward |

L'entropia iniziale di HECTOR è stata calcolata numericamente: per tanh(N(0,1)) vale circa 0.67 nat per dimensione, quindi circa +10 su 15 dimensioni, molto sopra il target di −15. Con `alpha_lr = 1e-4` e Adam, log α cala di circa 1e-4 per update: per passare da α = 1 a α = 0.1 servono circa 23.000 update. All'inizio del training il segnale del task (circa 0.15) è quindi circa lo 0.03% della soft-Q (circa 500). Questo è coerente con quanto scrive il paper, cioè che la policy finiva per produrre azioni saturate.

Per Tita SAC con σ = 0.2, l'entropia iniziale per dimensione è circa −0.23, quindi −1.4 su 6 dimensioni, sopra il target di −3: α **cala**. L'equilibrio a −0.5 per dimensione corrisponde a σ ≈ 0.15 pre-tanh.

---

## 11. Quanto può modificare la rete (in unità fisiche)

### 11.1 RPL (Fetch)

| Dimensione | Residuo massimo (u = 1) | Esplorazione iniziale | Confronto con gli altri |
|---|---|---|---|
| dx, dy, dz pinza | 2 in unità env → 0.10 m richiesti, limitati a 0.05 m per step dalla clip | rumore gaussiano di 0.2 → 0.4 in unità env → circa 2 cm per step; nel 30% dei passi un residuo uniforme su tutto il range | autorità piena (può cancellare il controllore); HECTOR e Tita hanno autorità parziale |
| gripper | 2 in unità env (il comando finale è clippato a ±1) | come sopra | nessun equivalente |

### 11.2 HECTOR (config rough, `HECTORBlindLocomotionActionsCfg`)

| Dimensione | ub (residuo massimo) | Riferimento fisico | Esplorazione iniziale (0.63 · ub) | Confronto con gli altri |
|---|---|---|---|---|
| acc. lineare x, y | ±2.0 m/s² | circa 20% di g | ±1.26 m/s² | nessun equivalente diretto; nel Tita una coppia sui giunti produce indirettamente un'accelerazione del CoM |
| acc. lineare z | ±4.0 m/s² | circa 41% di g | ±2.5 m/s² | come sopra |
| acc. angolare x, y, z | ±1.0 rad/s² | — | ±0.63 rad/s² | come sopra |
| errore di massa inversa (3) | ±0.2/13.856 1/kg | ±20% di 1/m | ±12.6% | unico: correzione del modello |
| errore di inerzia inversa (3) | ±0.2/I (I = 0.5413, 0.52, 0.0691) | ±20% di 1/I | ±12.6% | unico: correzione del modello |
| coefficiente del dt dell'MPC s | ±0.25 (il paper riporta 0.3) | durata del passo ±25% | ±16% | unico: il Tita non ha passi |
| Δh swing | [0, 0.15] m (clippato a ≥ 0) | apex nominale 0.1 m | metà dei campioni a 0, gli altri fino a circa 0.15 m | unico |
| Δ_cp | ±0.66 | spostamento dei punti di controllo della Bézier | ±0.41 | unico |

Per il terreno scivoloso (`HECTORSlipActionsCfg`) cambiano le ultime tre dimensioni: s ±0.3, Δh ±0.05 m (senza clip a ≥ 0), Δ_cp ±0.33.

### 11.3 Tita (PPO attuale)

| Giunto | Offset massimo (a = 1) | Coppia residua massima | Esplorazione iniziale: offset (0.19 · scala) | Esplorazione iniziale: coppia | Confronto con gli altri |
|---|---|---|---|---|---|
| hip (sx, dx) | ±0.20 rad | ±4 Nm | ±0.039 rad | ±0.77 Nm | come RPL agisce a valle del controllore, ma con autorità limitata (come HECTOR) |
| thigh (sx, dx) | ±0.35 rad | ±7 Nm | ±0.068 rad | ±1.35 Nm | come sopra |
| knee (sx, dx) | ±0.50 rad | ±10 Nm | ±0.097 rad | ±1.93 Nm | come sopra |
| ruote (sx, dx) | 0 rad/s | 0 Nm | 0 | 0 | nessuno dei due lavori ha dimensioni d'azione senza effetto |

Per capire quanto conta la coppia residua va confrontata con `actuator_forcerange` del modello Tita e con le coppie tipiche di τ_nominale, che sono già loggate in `info["tau_nominal"]` e `info["tau_residual"]`.

---

## 12. Pesi delle loss e quantità su cui agiscono

### 12.1 Termini che tirano l'azione verso zero (o la allargano)

| | RPL | HECTOR | Tita PPO | Tita SAC | Chi coincide / in cosa differiscono |
|---|---|---|---|---|---|
| Nome | `action_l2` | `reg_loss_coef` | `LOSS_NN_ACTION_COST` | `zero_action_prior` | stesso scopo in tutti e quattro |
| Coefficiente λ | 1.0 | 0.01 | 1.0 | 1.0 | RPL = Tita (PPO e SAC); HECTOR 100 volte più piccolo |
| Quantità penalizzata | u/max_u = tanh(z): uscita della rete **dopo** la tanh, prima di ×2 e ×0.05 | μ **prima** della tanh | tanh(loc), dopo la tanh, prima di `action_scale_pos` e Kp | tanh(loc) (`mode`) | RPL = Tita: azione normalizzata in [−1, 1], prima delle scale fisiche; HECTOR penalizza la variabile non saturata |
| Riduzione sulle dimensioni | media su batch e dimensioni (d = 4) | **somma** sulle dimensioni (d = 15), media sul batch | media su batch e dimensioni (d = 8, incluse le 2 ruote morte) | media su batch e dimensioni (d = 6) | RPL = Tita (media); HECTOR somma |
| Peso effettivo per dimensione | λ/d = 0.25 | λ = 0.01 | λ/d = 0.125 | λ/d ≈ 0.167 | RPL ≈ Tita (stesso ordine di grandezza); HECTOR circa 15-25 volte più piccolo |
| Gradiente per dimensione | 2(λ/d)·u = 0.5·u rispetto a u | 2λ·μ = 0.02·μ rispetto a μ | 2(λ/d)·a·(1 − a²) = 0.25·a·(1 − a²) rispetto a loc | 0.33·a·(1 − a²) rispetto a loc | RPL ≈ Tita (stessa forma con il fattore (1 − a²) rispetto alla variabile pre-tanh); HECTOR è lineare in μ |
| Comportamento in saturazione | il gradiente rispetto a z svanisce come (1 − u²) | continua a crescere con μ: è questo che impedisce la saturazione | svanisce come (1 − a²) | svanisce come (1 − a²) | RPL = Tita: non contrastano la saturazione; HECTOR sì |
| Termine di entropia | nessuno | α · log π, α iniziale 1.0 | `entropy_cost = 0` | α · log π, α iniziale 0.1 | RPL = Tita PPO (nessuno); HECTOR ≈ Tita SAC (α appreso), con α 10 volte diverso |
| Altri termini | — | `bounds_loss` disponibile ma spenta | — | — | — |

### 12.2 Equilibrio tra il termine del task e il prior (al primo ordine)

Le formule seguenti si ottengono annullando la derivata dell'actor loss rispetto alla componente deterministica dell'azione, trascurando l'entropia e la clip.

| | Actor loss (per campione) | Condizione di stazionarietà | Azione residua di equilibrio | Chi coincide / in cosa differiscono |
|---|---|---|---|---|
| RPL | −Q(s, u) + (λ/d) Σ_i u_i² | −∂Q/∂u_i + 0.5·u_i = 0 | u_i = 2 · ∂Q/∂u_i | stessa forma di Tita SAC, con fattore 2 invece di 3 |
| HECTOR | α log π − Q(s, tanh μ) + λ Σ_i μ_i² | −(1 − a_i²) ∂Q/∂a_i + 0.02·μ_i = 0 | μ_i = 50 · (1 − a_i²) · ∂Q/∂a_i | prior molto più debole (fattore 50), ma Q del task molto più piccolo |
| Tita SAC | α log π − Q(s, a) + (λ/d) Σ_i a_i² | −∂Q/∂a_i + 0.33·a_i = 0 (fattore (1 − a²) comune) | a_i = 3 · ∂Q/∂a_i | ≈ RPL |
| Tita PPO | L_clip(Â) + 0.5·L_V + (λ/d) Σ_i tanh(loc_i)² | non c'è un ∂Q/∂a: il gradiente del task è −Â·ρ·(x_i − μ_i)/σ², di ampiezza circa 0.8·\|Â\|/σ = 4 per campione con σ = 0.2 e Â normalizzato | non esprimibile in forma chiusa: il prior (≤ 0.1 per dimensione) è sistematico, mentre il termine PPO è grande per campione ma in gran parte rumore che si media sul batch | unico: il segnale del task non passa per un Q differenziabile in a |

### 12.3 Peso relativo del prior rispetto alla scala del task (ordine di grandezza)

| | Peso del prior per dimensione | Scala tipica del segnale del task | Rapporto | Chi coincide / in cosa differiscono |
|---|---|---|---|---|
| RPL | 0.25 | range di Q circa 50 | circa 0.005 | il più debole in proporzione |
| HECTOR | 0.01 (su μ²) | Q del task circa 0.15 (soft-Q iniziale circa 500, dominata dall'entropia) | circa 0.07 rispetto al Q del task | intermedio, ma all'inizio domina l'entropia |
| Tita SAC | 0.167 | V tra circa −5 e 1.5 | circa 0.1 | il più forte in proporzione: circa 20 volte RPL |
| Tita PPO | 0.125 | vantaggio normalizzato, circa 1 | non confrontabile direttamente: il vantaggio è normalizzato per batch | non confrontabile |

In proporzione alla scala del task, quindi, RPL ha il prior più debole e Tita il più forte. HECTOR sta in mezzo per il task, ma all'inizio è dominato dall'entropia. Il coefficiente 1.0 del Tita coincide nominalmente con quello di RPL, ma la scala di Q è circa 30 volte più piccola: per avere lo stesso peso relativo di RPL servirebbe un coefficiente intorno a 0.03-0.05.

---

## 13. Differenze più rilevanti

1. **Architettura.** HECTOR è gerarchico: il residuo modifica il modello e i parametri dell'MPC, che continua a rispettare i vincoli di contatto. RPL e Tita sono paralleli. Tita interviene in coppia, bypassando MPC e WBC, ed è l'impostazione più delicata delle tre su un robot instabile.
2. **Esplorazione.** HECTOR parte con un'esplorazione enorme (σ = 1, α = 1, std dell'azione circa 0.63). RPL esplora molto, ma con rumore fisso esterno (circa 0.36). Tita parte stretto (circa 0.19) e con entropia nulla in PPO.
3. **Il prior verso zero è definito in modo diverso nei tre casi.** RPL lo applica dopo la tanh e lo media; HECTOR prima della tanh e lo somma, con un coefficiente 100 volte più piccolo; Tita dopo la tanh, mediato su 8 dimensioni di cui 2 inutili. Tita e RPL coincidono nella definizione, ma non nel peso relativo, perché la scala di Q è diversa.
4. **Warm-up del critic.** Lo fanno solo RPL e Tita. RPL usa un criterio di convergenza rivalutato a ogni epoca; Tita PPO usa una durata fissa di 1M step; Tita SAC usa un criterio simile a RPL ma a senso unico.
5. **Autorità del residuo.** RPL ha autorità piena; HECTOR ha range fisici limitati (±20% sulla dinamica, ±25% sul dt); Tita ha 4-10 Nm sui giunti della gamba e nessuna autorità sulle ruote.

## 14. Punti da verificare nel codice Tita

1. **Azioni delle ruote morte.** Con `leg_only_actions=False` la policy produce 8 azioni, ma `action_scale_vel = 0` e `residual_config.Kd_wheel = 0`, quindi le due azioni delle ruote non generano coppia. Rientrano comunque nel prior (diluendolo da 1/6 a 1/8 per dimensione), nell'osservazione e nel rumore di esplorazione. Con PPO conviene `leg_only_actions=True`, come già succede in SAC.
2. **Guadagni del PD nominale.** `_combine_torque` calcola anche il PD *nominale* con `residual_config.Kp`, `Kd` e `Kd_wheel` (20, 1, 0), mentre il commento in `default_config` descrive un PD nominale con Kp = 35, Kd = 10, Kd_wheel = 10, e attribuisce a quel PD la stabilità del controllore puro. Con `Kd_wheel = 0` le ruote ricevono solo τ_ff, senza tracking di velocità. Se è voluto va bene, altrimenti il PD nominale dovrebbe usare `self._config.Kp`, `Kd` e `Kd_wheel`.
3. **Il termine Kd della gamba nel residuo è sempre nullo.** `dq_des_rl − dq_des_wbc` è diverso da zero solo sugli indici delle ruote, quindi `Kd_res = 1` sulla gamba non ha effetto.

## 15. Note sulle fonti

- **RPL.** Il blending è in `rpl_environments/envs/*_env.py`, l'esplorazione in `ddpg_controller.py` (`get_actions`), l'actor loss con `action_l2` in `ddpg_controller.py`, l'inizializzazione a zero in `models/utils.py` (`nn_last_zero`), il burn-in in `train_staged.py` e gli iperparametri in `configs/config_standard_*.py`.
- **HECTOR.** Le azioni sono in `config/hector/mdp/actions/mpc_actions.py` e `env_cfg/action_cfg.py`, i reward in `env_cfg/reward_cfg.py`, le terminazioni in `env_cfg/termination_cfg.py`, l'ambiente in `rough_env_sac_cfg.py`, gli iperparametri in `agents/rl_games_sac_st_blind.yaml` e la loss in `rl_games/algos_torch/sac_agent.py` (`update_actor_and_alpha`, `reg_loss`).
- **Tita.** L'env è `locomotion/tita/joystick.py` (`_residual_joint_targets`, `_combine_torque`, `step`, `_get_obs`), il training `train_srbd.py` (`PPO_PARAMS`, `SAC_PARAMS`, flag in testa al file) e `sac_training.py`. I default di brax sono in `brax/training/agents/ppo/train.py` e `brax/training/distribution.py`.
