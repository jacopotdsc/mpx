# tita_hybrid_plan — HRC su Tita (MPC + rete residua)

- **results.md**: risultati finali (leggere per primo).
- **MEMORY.md**: diario completo di decisioni ed esperimenti.
- **PLAN.md**, **paper_notes.md**: piano e note sul paper.
- **checkpoints/**: run PPO/SAC.
  - Modello consigliato: `20261019_003649_RA3_final/params_best.pkl`.
  - Altri: RA2, RA, RS4 (pendenza), RS2, B5 (SAC, replica del paper, con `eval_hybrid/`).
- **save_setup/**: copie dei file modificati e nuovi, con la struttura dei repo (vedi save_setup/README.md).
- **patches/**: `git diff` dei due repo rispetto a `residual_tita`. Gli XML nuovi sono non tracciati e quindi solo in `save_setup/`.
- **eval_robust.py**, **eval_hybrid.py**, **launch_when_free.sh**: script di valutazione e lancio.
- **robust_results/**: JSON delle valutazioni di robustezza.
- **logs/**: log dei training e delle eval.
