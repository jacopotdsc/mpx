# Recap — `mjx_policy_tita.py` come frontend dell'environment

## Cosa era la discrepanza (cause reali, dal codice eseguibile)

1. **Comando diverso.** `train_srbd.py --eval` (senza `--cmd`) NON usa nessun comando fisso:
   `env.reset()` campiona un `info["command"]` **random**, e `env.step()` lo **ricampiona a
   caso** a intervalli casuali. Il vecchio `mjx_policy_tita.py` partiva invece da comando **zero**.
   Traiettorie di comando diverse ⇒ comportamenti diversi. Questa da sola spiega gran parte
   della differenza.
2. **Stato iniziale diverso.** L'env in `reset()` forza `z = 0.4435` (keyframe `home` a `0.44`,
   alzato di 3.5 mm per non far compenetrare le ruote). Il vecchio tester resettava dal keyframe
   (`mj_resetDataKeyframe`, `z = 0.44`) ⇒ contatto/impulso al passo 0 e divergenza immediata.
3. **Motore fisico diverso.** L'env avanza con `mjx_env.step` (MJX); il vecchio tester con
   `mujoco.mj_step` (MuJoCo C) su un `MjData` ricostruito con `mjx.put_data`. Vicini ma non identici.
4. **Batching/RNG.** L'eval fa `jax.vmap(reset)` con `PRNGKey(42)` splittato in `EVAL_BATCH+1`
   (evita il crash cuSolver su MJX single-instance). Il vecchio tester non era batched.

Non erano invece cause: **policy/checkpoint/network** (costruzione PPO identica: stessi
`observation_size`/`action_size`, hidden `(512,256,128)`, `tanh_normal`, `running_statistics.normalize`,
`make_inference_fn(...)(params, deterministic=True)`), e **la mappa azione→attuatori** (identica:
gambe `default_pose + a*0.5`, ruote `a*25.0`, con gli stessi gain PD overriddati in `base.py`).

## Correzione di rotta importante sul modello mentale

Nel **codice attuale dell'environment (`joystickE2E.py`) NON esistono** `info["target_command"]`
né lo smoothing `command += 0.02*(target-command)`. Esiste solo `info["command"]`, ricampionato in
modo discreto/random dentro `step()`. Il `target_command` compare **solo** in `train_srbd.py`/
`plot_eval.py` come chiave di logging: nel ramo `--cmd` viene scritto `target_command` (mai letto
dall'env) e la riga `"command"` è commentata ⇒ quel ramo è **codice morto** (e chiamerebbe
`_get_obs` con 2 argomenti mentre l'env ne vuole 3).

Conseguenza pratica: la tastiera deve scrivere **`info["command"]`** (non `target_command`, che
sarebbe un no-op silenzioso). Non essendoci smoothing, **Command == Target** nella HUD; l'ho
comunque mostrato in due riquadri separati. Se un domani aggiungerai lo smoothing all'env, la HUD
divergerà da sola senza toccare il tester.

## Com'è ora il tester

- `state = env.reset(rng)` — unica sorgente di verità (stesso path vmap/seed dell'eval).
- Loop: leggi tastiera → `set_keyboard_command(state)` → `policy(state.obs[0])` →
  `env.step(state, action)` → `set_keyboard_command(state)` (di nuovo, per battere il resampler
  random) → `sync_viewer_data(state)` → HUD → `state.done`.
- `MjData` nativo = **solo mirror** (`qpos/qvel` ← `state.data`, `mj_forward`, `sync`). Nessun
  secondo `mj_step`.
- Timing: 1 iterazione = 1 inference + 1 `env.step` (i 5 substep sono dentro `env.step`),
  pacing `sleep(env.dt - elapsed)`.

## Funzioni/variabili eliminate

`_reset_to_initial_state`, `_base_touches_floor`, `_build_env_get_obs_fn`, `apply_policy_action`,
e le variabili parallele `policy_info`, `previous_action`, `current_action`, `current_command`,
`policy_period`, più il duplicato `load_params`. Nomenclatura ora allineata all'env:
`state / state.data / state.obs / state.info / action / command`.

## Differenze residue rispetto a `train_srbd.py --eval`

- Il comando è dato dalla tastiera (voluto) invece che random.
- C'è una latenza di **1 control step (10 ms)** tra pressione tasto e comparsa in `state.obs`
  (il comando iniettato prima di `step()` entra nell'obs ricostruito da `step()`, consumato
  all'inferenza successiva) — coerente con la semantica dell'env.
- Rendering/pacing wall-clock del viewer (irrilevante per la fisica).

## Aggiornamenti successivi (2 patch)

### PATCH 1 — durata illimitata del test
- Costante `EPISODE_LENGTH = 10_000_000`.
- `registry.load(env_name, config_overrides={"episode_length": EPISODE_LENGTH})`: override **locale**
  al tester, `joystickE2E.default_config().episode_length` invariato.
- `--steps` e `main(..., steps=...)` ora hanno default `EPISODE_LENGTH`; il loop headless usa quel cap,
  il loop interattivo è `while viewer.is_running()`. In entrambi i casi la sola terminazione anticipata
  è `state.done` (caduta/contatto base). Nota: l'env raw non tronca a episode_length dentro `step()`
  (solo termination fisica), quindi l'override è soprattutto una salvaguardia + coerenza semantica.

### PATCH 2 — HUD `Command` + `Target command` dallo stato reale
- `set_keyboard_command` ora scrive **sia** `info["command"]` **sia** `info["target_command"]` (il tester
  popola la chiave, senza toccare `joystickE2E.py`). Così la HUD legge il target dallo **stato reale**
  `state.info["target_command"]`, non da una variabile esterna.
- `build_hud` mostra un blocco unico a **TOPRIGHT** (help tastiera resta a TOPLEFT, nessuna sovrapposizione):
  ```
  Command
  vx: +0.23
  wz: -0.08

  Target command
  vx: +0.50
  wz: +0.00
  ```
- Poiché l'env non ha smoothing, **`command` e `target_command` coincidono**. Se un domani aggiungi lo
  smoothing all'env, la HUD divergerà da sola senza modifiche al tester.

## Correzione (env reale con target_command + smoothing)

**Scoperta:** l'environment realmente caricato da `registry.load` NON è il `joystickE2E.py` presente nel
progetto (copia stale). Quello eseguito ha, dentro `step()`:
```python
steps_until_next_cmd -= 1
target_command = where(steps_until_next_cmd <= 0, sample_command(...), target_command)
command = command + 0.02 * (target_command - command)
```
cioè `target_command` + smoothing `0.02` + resampling automatico.

**Bug reale della versione precedente:** `set_keyboard_command` scriveva direttamente
`state.info["command"] = keyboard` (oltre a `target_command`) **prima e dopo** `env.step()`. Così:
1. lo smoothing `0.02` veniva annullato (command saltava istantaneamente invece di convergere);
2. il command "vero" non era più prodotto dalla legge dell'env;
3. il resampler non era disabilitato, quindi `sample_command` poteva sovrascrivere `target_command`
   a metà step.

**Fix (solo `mjx_policy_tita.py`):**
- `set_keyboard_command` → **`set_target_command`**: scrive **solo** `target_command = keyboard` e tiene
  `steps_until_next_cmd = 10_000_000` (`NO_RESAMPLE_STEPS`), **senza mai toccare `command`**. Lo smoothing
  dell'env muove `command` verso `target` da solo; il resampler non scatta mai.
- Reset: `command` inizializzato a 0, `target_command = keyboard`, resampler disabilitato, obs ricostruito.
- `_step_once`: `set_target_command` **prima** di `env.step()`; niente reiniezione di `command`.
- Diagnostica `--debug-cmd`: stampa per ogni step `kbd | target before->after | command before->after |
  steps_until_next_cmd before->after`.
- HUD `build_hud`: unico overlay a TOPRIGHT (help tastiera a TOPLEFT), ricostruito **ogni frame** in
  **una sola** `viewer.set_texts(...)`, con `Command` (da `state.info["command"]`, che si muove) e
  `Target command` (da `state.info["target_command"]`, fisso al valore tastiera).

**Ordine temporale:** dentro `step()` l'`_get_obs()` gira **prima** dell'update finale del command,
quindi `state_{t+1}.obs` contiene il `command` **precedente** all'aggiornamento di quello step
(offset di 1 step, semantica dell'env, non modificata).

**Cosa devi vedere** con tastiera a `vx=0.5, wz=0.0`:
```
Target command : vx +0.50  wz +0.00     (resta 0.50 finché non lo cambi tu)
Command        : vx +0.01  wz +0.00
Command        : vx +0.0198 ...
Command        : vx +0.0294 ...
...            (converge verso 0.50 col filtro 0.02, ~2 s)
```
Nessun target random deve più comparire.