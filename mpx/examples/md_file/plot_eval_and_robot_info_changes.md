# Modifiche a `plot_eval.py` e `joystickE2E.py` (TITA)

Riepilogo delle modifiche applicate durante questa sessione. Due file toccati:

- `test/mpx/mpx/examples/plot_eval.py` (funzione `plot_reward_terms_separate`)
- `mujoco_playground/_src/locomotion/tita/joystickE2E.py` (`info["robot"]`, in `reset()` e `step()`)

---

## 1. `plot_eval.py` — `plot_reward_terms_separate`

Per tre reward term (`tracking_lin_vel`, `base_height`, `wheel_track`) il grafico del
reward nel tempo e il grafico di tracking corrispondente sono stati **uniti nella
stessa figura** (stesso file PNG), invece di essere due PNG separati. In tutti e tre i
casi il subplot in alto è il reward term grezzo (stesso identico contenuto che prima
finiva nel PNG generico `<k>.png`), e sotto ci sono uno o più subplot di tracking. Dopo
aver salvato la figura combinata, il blocco esegue `continue` per saltare la
generazione del PNG generico duplicato per quella chiave.

### 1.1 `tracking_lin_vel` → `tracking_lin_vel.png`

File: `plots/<prefix>/tracking_lin_vel.png` — 3 subplot:

1. `tracking_lin_vel (reward term)` — valore del reward nel tempo (mean/min/max/sum in overlay).
2. **Linear velocity tracking** — comando (`command_0`) vs velocità lineare misurata (`robot/local_linvel_0`).
3. **Angular velocity tracking** — comando (ultimo `command_i` disponibile) vs velocità angolare misurata (`robot/gyro_2`).

Fix applicato: il numero di componenti del comando non è più hardcoded a
`("command_0", "command_2")` (che per TITA — comando a 2 componenti `[vx, wz]` — non
esisteva mai, quindi il blocco non veniva mai eseguito). Ora le chiavi `command_*`
presenti in `terms` vengono rilevate dinamicamente con una regex
(`re.fullmatch(r"command_\d+", ck)`), e la componente angolare è sempre **l'ultima**
disponibile (coerente con `commands[-1]` usato in `joystickE2E.py` per il reward
angolare). Funziona sia per TITA (2 componenti) sia per un quadrupede (3 componenti).

### 1.2 `base_height` → `base_height.png`

File: `plots/<prefix>/base_height.png` — 2 subplot:

1. `base_height (reward term)` — valore del reward nel tempo.
2. **Base height tracking** — `robot/base_height_target` (linea tratteggiata) vs `robot/com_height` misurata.

### 1.3 `wheel_track` → `wheel_track.png` (nuovo blocco)

File: `plots/<prefix>/wheel_track.png` — 2 subplot:

1. `wheel_track (reward term)` — valore del reward nel tempo.
2. **Wheel track tracking** — distanza reale tra le due ruote vs target.

La distanza viene calcolata come `norm(feet_pos_sinistra - feet_pos_destra)` a partire
dalle chiavi `robot/feet_pos_0..5` (appiattimento di `info["robot"]["feet_pos"]`,
shape `(2, 3)`: indici 0-2 = ruota sinistra, 3-5 = ruota destra). Il target è
`config.d` (0.567 m), importato da `mpx.config.config_dfcip` — la stessa costante
usata dal reward `_cost_wheel_track` in `joystickE2E.py`.

Prima di questa modifica non esisteva alcun grafico dedicato al tracking della
larghezza del passo: il reward `wheel_track` veniva plottato solo come singola curva
generica, senza confronto con la distanza reale misurata.

---

## 2. `joystickE2E.py` — arricchimento di `info["robot"]`

`info["robot"]` (popolato in `reset()` e aggiornato in `step()`) conteneva solo:

```python
"robot": {
    "qpos": data.qpos,
    "qvel": data.qvel,
    "com_height": data.sensordata[self._base_com_adr][2],
    "base_height_target": self._config.reward_config.base_height_target,
    "local_linvel": self.get_local_linvel(data),
    "gyro": self.get_gyro(data),
}
```

Sono state aggiunte tutte le grandezze **direttamente disponibili** (letture di
sensori o campi grezzi di `data`, nessun calcolo derivato aggiuntivo):

| Chiave | Sorgente |
|---|---|
| `global_linvel` | `self.get_global_linvel(data)` |
| `global_angvel` | `self.get_global_angvel(data)` |
| `gravity` | `self.get_gravity(data)` |
| `upvector` | `self.get_upvector(data)` |
| `accelerometer` | `self.get_accelerometer(data)` |
| `joint_pos` | `data.qpos[7:]` |
| `joint_vel` | `data.qvel[6:]` |
| `actuator_force` | `data.actuator_force` |
| `ctrl` | `data.ctrl` |
| `feet_pos` | `data.site_xpos[self._feet_site_id]` |
| `feet_vel` | `data.sensordata[self._foot_linvel_sensor_adr]` |
| `ext_force` | `data.xfrc_applied[self._torso_body_id, :3]` |
| `joint_pos_3d` | `data.xanchor` (posizione 3D world-frame di ogni giunto, campo diretto di `mjx.Data`, verificato disponibile) |

Questo è ciò che ha reso possibile il nuovo blocco `wheel_track` in `plot_eval.py`
(che si appoggia su `robot/feet_pos_0..5`), e rende disponibili per futuri plot anche
le altre grandezze (es. `robot/actuator_force_*`, `robot/ctrl_*`, `robot/joint_pos_*`).

Non sono state aggiunte grandezze **calcolate** (es. la distanza ruota-ruota stessa,
o boolean di contatto) dentro `info["robot"]`: quei calcoli restano nello script di
plotting, che li deriva al volo dai dati grezzi loggati.
