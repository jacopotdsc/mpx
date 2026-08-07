# Generalizzazione N-dimensionale dei comandi (viewer + eval)

File modificato: `mpx/examples/train_srbd.py`

## Problema di partenza

Nel viewer di rollout (`run_viewer_rollout`), l'HUD mostrava sempre 3 comandi
(`vx`, `vy`, `wz`) anche per TITA, che non ha un grado di libertà laterale
(`vy`). La dimensionalità del comando era hardcoded a 3 in più punti dello
script di valutazione, indipendentemente da cosa dichiarasse effettivamente
`command_config` dell'environment caricato.

## Causa

- `_cmd_text` (HUD) formattava sempre `c[0]`, `c[1]`, `c[2]` con le label fisse
  `vx=`, `vy=`, `wz=`.
- Il blocco di injection di `--cmd` (comando fisso da CLI) forzava sempre una
  shape `(EVAL_BATCH, 3)`.
- Il patch dell'osservazione per `--cmd` scriveva a mano negli indici
  `[..., 45:48]`, un offset valido solo per il layout osservativo di Go1
  (quadrupede). Per TITA il comando si trova altrove nel vettore obs
  (es. indice 57 in `joystick.py`, indice 31 in `joystickE2E.py`), quindi
  quel patch scriveva nel posto sbagliato.
- Il flag CLI `--cmd` accettava `nargs=3` fisso (`VX VY WZ`).

## Cosa genera davvero i comandi

- **Sampling casuale (default, nessun `--cmd`)**: già gestito internamente
  dall'env (`sample_command()` in `joystick.py` / `joystickE2E.py`), che legge
  `self._cmd_a` / `self._cmd_b` da `self._config.command_config.a/b`. Lo
  script di valutazione si limita a leggere `state.info["command"]` dopo lo
  step: questo percorso era già config-driven.
- **Comando fisso (`--cmd ...`)**: era invece hardcoded a 3 componenti e a un
  offset fisso nell'obs — il percorso corretto da questo intervento.

## Modifiche applicate

1. **`--cmd`** (parser CLI): `nargs=3` → `nargs="+"`, metavar generico
   `CMD_I`. Il numero di valori richiesti non è più fissato a priori.

2. **`cmd_dim`**: subito dopo il reset, letto dinamicamente da
   `state.info["command"].shape[-1]` — riflette esattamente la
   dimensionalità dichiarata da `command_config` dell'env caricato (2 per
   TITA `[vx, wz]`, 3 per un quadrupede `[vx, vy, wz]`, N per qualunque env
   futuro).

3. **Injection post-reset di `--cmd`**:
   - validazione esplicita: se il numero di valori passati da CLI non
     combacia con `cmd_dim`, viene sollevato un `ValueError` con messaggio
     chiaro invece di un comportamento indefinito/crash silenzioso.
   - rimosso il patch manuale `.at[..., 45:48]`: l'osservazione iniziale
     viene invece rigenerata chiamando `eval_env._get_obs(state.data,
     state.info)` (vmapped) — è l'env stesso a sapere dove/come collocare il
     comando nel vettore di osservazione, niente più offset hardcoded.

4. **Re-injection di `--cmd` nel loop** (sia headless che viewer): la shape
   `(EVAL_BATCH, 3)` è diventata `(EVAL_BATCH, cmd_dim)`.

5. **`_cmd_text` (HUD nel viewer)**: non stampa più `vx=`, `vy=`, `wz=` fissi;
   itera su tutte le componenti del comando corrente e le formatta come
   numeri (`+0.50  +0.00  ...`), adattandosi automaticamente a qualunque N.

## Flusso risultante

```
CLI --cmd (N valori)
        │  (validato contro cmd_dim)
        ▼
state.info["target_command"] / "command"   ← dimensione = command_config dell'env
        │
        ▼
eval_env._get_obs(...)  → obs coerente con il layout dell'env
        │
        ▼
HUD (_cmd_text)  → stampa dinamicamente N valori, nessuna label fissa
```

Nessun punto della catena assume più 3 componenti: la dimensionalità è
sempre derivata da `state.info["command"]`, che a sua volta discende da
`command_config` dell'environment attivo.
