# `train_srbd.py` Cleanup Summary

Reduced from **1570** to **669** lines. The stack is unchanged: JAX + Brax PPO + MuJoCo Playground, existing environments, `plot_eval.py` utilities, and the MuJoCo viewer. Training and evaluation behavior has been preserved.

---

## 1. Essential Code Kept

- PPO training through `ppo.train`.
- MuJoCo Playground environment loading and registration of the custom `QuadrupedMPCEnv`.
- Progress callback with reward tracking and training curve generation.
- `best` and `final` checkpoints, timestamped run directories, and checkpoint saving on Ctrl+C.
- Full evaluation pipeline:
  - viewer rollout
  - headless rollout
  - rollout CSV export
  - all existing `plot_eval.py` calls:
    - LLC plots
    - command tracking
    - reward terms
    - MPC output
    - network actions
    - video recording

---

## 2. Meaningful Experimental Controls Kept

- `POLICY_HIDDEN_LAYER_SIZES`
- `DISTRIBUTION_TYPE`
- `INIT_STD`
- `ZERO_INIT_OUTPUT_LAYER`
- Deterministic evaluation
- All PPO hyperparameters, now collected in a **single** `PPO_PARAMS` dictionary at the top of the file.

---

## 3. Debug / Diagnostic Code Removed

- The `self_test` block inside the network builder (~120 lines), including:
  - kernel norm printing
  - raw output sampling
  - per-leaf shape inspection
- Signature introspection inside `make_train_fn` (`"Available options in ppo.train..."`).
- Parameter shape / activation closure / output-kernel norm inspection during fresh training.
- `_make_fresh_params` diagnostics.
- Import-time printing of `PPO_PARAMS` / `SAC_PARAMS`.

---

## 4. Redundant / Duplicated Code Removed

- **Two** definitions of `PPO_PARAMS`.
  - The second one was the active definition, so its values were preserved:
    - `discounting=0.97`
    - `learning_rate=3e-4`
    - `entropy_cost=1e-2`
    - `max_grad_norm=1.0`
    - `num_resets_per_eval=10`
- Duplicate imports:
  - `os` ×3
  - `jnp` ×2
  - duplicated `dataclass` imports
- Duplicated reset / step / JIT inference setup in the rollout code.
- The two almost-identical rollout loops for headless and viewer evaluation were merged into a shared `advance()` step.
- The `accepted` / `dropped` kwarg filtering block, which computed a filtered dictionary that was never used.
- `.npz` checkpoint saving, since those files were never loaded; only `.pkl` checkpoints are used.

---

## 5. Functionality Simplified

- `--load` now accepts an **explicit run directory** instead of relying on the previous auto-discovery machinery (`_resolve_load`, `_list_run_dirs`, regex matching).

  Loading rule:

  1. Prefer `params_best.pkl`.
  2. Otherwise use `params_final.pkl`.

  This logic is handled by `load_checkpoint`.

- The network builder no longer introspects Brax's default initializer and instead directly uses `lecun_uniform`.

- Console output was rewritten to describe the actual experiment rather than implementation details. It now reports information such as:
  - mode
  - environment
  - checkpoint
  - network architecture
  - distribution type
  - initial standard deviation
  - zero-output initialization
  - PPO parameters
  - run directory
  - training progress

---

## 6. Functionality Removed

### SAC / `--algo`

SAC support and the `--algo` option were removed.

The file is centered around Brax **PPO**, and the intended CLI does not require `--algo`. SAC introduced additional global state such as `ALGO` / `ALGO_PARAMS` and required branching throughout the codebase, which was exactly the kind of accidental complexity this cleanup was intended to remove.

SAC can still be reintroduced later if it becomes necessary.

### Training Resume Through `--load`

Training resume through `--load` was removed.

In the original file, `restore_params` was computed, but the corresponding argument passed to `ppo.train` was commented out:

```python
# restore_params=...
```

Therefore, resume functionality was already effectively a no-op.

`--load` is now used only for evaluation, while training always starts from scratch.

### Other Dead Code

Also removed:

- `gpu_python_process_cleanup`, which was only referenced by a commented-out line.
- The module-level `wrap_for_brax_training`, which was unused because the environment's own wrapper is used instead.

---

## How `INIT_STD` and `ZERO_INIT_OUTPUT_LAYER` Are Preserved

In the original file, both options were **commented out in the active PPO path**, meaning they had no effect. They were only connected in the SAC path.

They are now functional inside `make_policy_networks`, following the convention already used by the previous SAC implementation.

### `INIT_STD`

`INIT_STD` is passed to `make_ppo_networks` as:

```python
init_noise_std=INIT_STD
```

### `ZERO_INIT_OUTPUT_LAYER`

When `ZERO_INIT_OUTPUT_LAYER=True`, a small custom kernel initializer sets the policy output layer to zero.

For `tanh_normal`, this corresponds to the layer with output shape:

```text
(*, 2 * action_size)
```

Hidden layers continue using Brax's default initialization.

When:

```python
ZERO_INIT_OUTPUT_LAYER=False
```

no initializer override is applied, so Brax's default initialization behavior is preserved.

Because these parameters are kwargs exposed by the network API in the custom Brax fork, network construction is wrapped in a `try/except TypeError`.

If the current Brax version does not support these hooks, the code falls back to the standard network builder and prints a warning instead of crashing.

The same `make_policy_networks` function is used both during training and when rebuilding the network for checkpoint evaluation, ensuring that the network definition cannot silently diverge between the two paths.

---

## Resulting CLI and Execution Flow

```text
--name      Environment name or shortcut
            (go1 / tita / aliengo / titae2e).
            Training is the default mode.

--eval      Switch to evaluation mode.

--load      Run directory to evaluate.
            Loads best checkpoint first, then final.
            Default: most recent run.

--headless  Run evaluation without the MuJoCo viewer.

--cmd       Fixed joystick command.
            Example: --cmd 0.5 0.0

--random    Use an untrained/random policy baseline.

--zero      Use the zero-action / MPC-only baseline.
```

### Training

```text
CLI
→ environment
→ make_policy_networks
→ PPO_PARAMS
→ ppo.train
→ checkpoints/<env>/<timestamp>/
```

### Evaluation

```text
CLI
→ environment
→ --load run directory
→ load_checkpoint
→ same network definition
→ inference
→ run_rollout
→ CSV / plots / video
```

---

## Point to Verify

The previously active configuration left `policy_obs_key` and `value_obs_key` at the Brax defaults.

The `network_factory` dictionary stored inside `PPO_PARAMS` was silently overridden by the `network_factory` argument supplied directly at the `ppo.train` call site, so those values were never actually used.

The cleanup therefore preserves the **effective behavior of the old code**, rather than preserving dead configuration.

If the intended behavior is for the value network to use `privileged_state`, this should be added explicitly inside `make_policy_networks`.

That would be a new functional change rather than something the previous active implementation was already doing.