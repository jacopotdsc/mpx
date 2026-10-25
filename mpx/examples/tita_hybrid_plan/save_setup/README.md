# save_setup — snapshot dei file del setup (tita_hybrid_plan)

Copie dei file usati per training e valutazione, con la **stessa struttura di cartelle dei repo**:

- `mujoco_playground/mujoco_playground/_src/locomotion/tita/`: env Tita completo (branch `tita_hybrid`): `joystick.py`, `base.py`, `tita_constants.py`, `randomize.py`, `joystickE2E.py` e `xmls/` con le scene nuove (`scene_slopes`, `scene_mixed*`, `scene_rough_sparse`, `scene_tilt10`).
- `mujoco_playground/mujoco_playground/_src/locomotion/__init__.py`: registry (invariato).
- `mpx/mpx/examples/`: `train_srbd.py` (`--algo sac`, `--sac-override`, `--env-override`), `sac_training.py`, `algorithm_initialization.py`.
- `mpx/mpx/utils/`: `mpc_utils.py` (**modificato**: riferimento di altezza dell'MPC relativo al contatto) e `mpc_wrapper_dfcip.py` (invariato).
- `mpx/mpx/config/config_dfcip.py`: config dell'MPC (invariata).

Nota: **joystick.py com'era a fine v1** (19/10 04:15): PD di postura, scene, comandi joystick, correzione dello spawn in pendenza; **senza** i termini di tracking della v2. Le altre parti (scene, `base.py`, mpx) sono identiche alla v2.

Per ripristinare: copiare i file sopra nei rispettivi percorsi dei repo, cioè mujoco_playground in site-packages del conda env `mjpl` e mpx in `~/Desktop/repo_rl/tita_rl/test/mpx`.
