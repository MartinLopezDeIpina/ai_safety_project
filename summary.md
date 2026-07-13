# Conversation Summary

**Date:** 2026-07-13
**Project:** AI Safety — Replication of "LLMs Encode Harmfulness and Refusal Separately" (Zhao et al., 2025)
**Model:** Qwen 7B (Qwen2-7B-Instruct)

## Key Script Created: `get_intervene_vectors.py`

**Path:** `src/experiments_replication/get_intervene_vectors.py`

### Purpose
Load cached `.pt` activation tensors, compute harmfulness ("hf") and refusal steering directions, and save them.

### Activation Tensors
- Shape: `(L, N, T, H) = (29, samples, 6, 3584)` — float16
- **L=29** layers, **H=3584** hidden dim (Qwen 7B), **T=6** token positions
- `tinst` = index 1 (last instruction token — harmfulness encoding)
- `tpost` = index -1 (last prompt token — refusal encoding)

### Datasets
- **Harmful:** advbench, jbb, sorrybench (files have `_gentinst`/`_gentpost` suffix)
- **Harmless:** alpaca, xstest (no position suffix; tensor holds both positions)

### Computation
- `hf direction = mean(harmful_tinst) - mean(harmless_tinst)`
- `refusal direction = mean(harmful_tpost) - mean(harmless_tpost)`
- Both output shape: `(29, 3584)` float16

### Key Functions
| Function | Description |
|---|---|
| `collect_acts(act_dir, datasets, position)` | Unified loader for harmful/harmless datasets |
| `compute_intervene_vectors(act_dir)` | Computes both hf and refusal directions |
| `save_intervene_vectors(vectors, out_dir)` | Saves each vector as `<key>.pt` |

### CLI Usage
```
python get_intervene_vectors.py <act_dir> [--out-dir <dir>]
```


## Supporting File: `run_get_intervene_vectors.sh`

**Path:** `src/experiments_replication/run_get_intervene_vectors.sh`

Bash wrapper that:
- Auto-detects repo-root `.venv` Python
- Runs `get_intervene_vectors.py` with `activations_qwen` as input
- Saves output to `steering_vectors/qwen-7b/`

## Verification Results

Both runs completed successfully (exit code 0):

| File | Size | Shape | dtype |
|---|---|---|---|
| `steering_vectors/qwen-7b/hf.pt` | 205 KB | (29, 3584) | float16 |
| `steering_vectors/qwen-7b/refusal.pt` | 205 KB | (29, 3584) | float16 |


## Activation File Inventory (`activations_qwen/`)

16 `.pt` files total:
- **Harmful datasets (12 files):** advbench, jbb, sorrybench — each with `_gentinst`/`_gentpost` variants × accepted/refused
- **Harmless datasets (4 files):** alpaca, xstest — each with accepted/refused (no position suffix)


## Technical Details

- **Model Architecture:** Qwen2-7B-Instruct, 29 layers, 3584 hidden dim
- **Token Conventions:** `tinst` = position 1 (last instruction token), `tpost` = position -1 (last prompt token)
- **Activation shape:** `(L, N, T, H)` where T=6: [2nd-last-inst, tinst, suffix1, suffix2, suffix3, tpost]
- **File I/O:** PyTorch with `weights_only=True`


## Next Steps

The steering vectors are now ready. The immediate next steps in the replication workflow are:
1. **Run intervention experiments** — Use `hf.pt` and `refusal.pt` with the intervention pipeline (`intervention.py`, `complete_intervene.sh`, or the new `modal_intervene.py` — see below)
2. **Replicate figures** — Run the figure-plotting stages (Figures 2 & 3 from paper)
3. **Other models** — The repo supports Llama-2/3 as well; can extend the same pipeline


## Modal Script Created: `modal_intervene.py`

**Path:** `src/experiments_replication/modal_intervene.py`

### Purpose
Modal equivalent of `slurm/intervene.slurm`. Runs `src/intervention.py` on Modal A100 containers, then copies the produced output JSON file(s) onto a results volume. The local entrypoint pulls every finished file back into `src/experiments_replication/intervention_outputs/`, auto-indexing (`key`, `key_2`, `key_3`, …) so it never clobbers a previous local run.

### Design
- **Image / layout** — Reproduces the repo under `/root` so `intervention.py`'s `from utils import ...` and `from template_inversion import ...` (both in `src/`) resolve. Sets `PYTHONPATH=/root/src` and `cwd=/root/src`.
- **Steering vectors** — Unlike `modal_run.py`, this script does **not** ignore `*.pt` under `experiments_replication/`; the steering vectors (`steering_vectors/qwen-7b/hf.pt`, `refusal.pt`) are mounted at runtime.
- **Volume** — Uses a dedicated `intervention-results` volume (separate from `behavior-results`) so intervention outputs don't collide with pipeline outputs.
- **Defaults** — Mirror `intervene.slurm` exactly: `qwen:7b`, `advbench`, `hf` vector, `reverse_intervention=1` (less-harm), `layer_s=0`, `layer_e=28`, `coeff_select=1`, `max_token_generate=100`, `use_inversion=1`, `inversion_prompt_idx=1`, `right=50`.
- **Output collection** — `intervention.py` writes per-layer JSONs (`<stem>-intervene<layer>.json`), so the `finally` block copies **all** JSONs from the run's output dir to the volume.

### Key Functions
| Function | Description |
|---|---|
| `run_intervention(...)` | Modal remote function: runs `intervention.py` for one config, persists output JSON(s) to `/results/<key>` |
| `_key(model, size, dataset, vector, left, right, reverse)` | Deterministic results-volume key (includes direction: `lessharm`/`moreharm`) |
| `_parse_runs(runs)` | Parses `model:size[:left[:right]]` tokens (right defaults 50) |
| `_pull(name, key)` | Fetches `/<key>/` from the results volume into a fresh indexed local dir |
| `entry(...)` | Modal local entrypoint — launches runs in parallel, waits, pulls results |

### CLI Usage

```bash
# Default — replicates intervene.slurm (advbench + hf, less-harm)
modal run src/experiments_replication/modal_intervene.py --runs "qwen:7b:0:50"

# Multiple datasets/vectors in parallel (zipped with --runs)
modal run src/experiments_replication/modal_intervene.py \
    --runs "qwen:7b:0:50,qwen:7b:0:50" \
    --datasets "advbench,jbb" --vectors "hf,refusal"

# Flip to more-harm (amplify harmfulness)
modal run src/experiments_replication/modal_intervene.py \
    --runs "qwen:7b:0:50" --reverse-intervention 0

# Detached — safe to close laptop, collect later
modal run --detach src/experiments_replication/modal_intervene.py --runs "qwen:7b:0:50" --no-wait
modal run src/experiments_replication/modal_intervene.py --runs "qwen:7b:0:50" --collect-only
```

Experiment:

```bash
modal run --detach src/experiments_replication/modal_intervene.py --runs "qwen:7b:0:50" --no-wait
modal run src/experiments_replication/modal_intervene.py --runs "qwen:7b:0:50" --collect-only
```

Look for outputs in `src/experiments_replication/intervention_outputs/qwen7b-advbench-hf-lessharm-0-50/`.

### Reference
Modeled after `src/experiments_replication/modal_run.py` (colleague's pipeline-runner), adapted for the intervention use case.


## Cleanup: `Zone.Identifier` Files

All Windows metadata files (`*Zone.Identifier`) in `activations_qwen/` were removed with:
```
find activations_qwen -name '*Zone.Identifier' -delete
```


## File Paths Reference

| Path | Description |
|---|---|
| `src/experiments_replication/main.py` | Pipeline orchestrator |
| `src/experiments_replication/get_intervene_vectors.py` | Steering vector computation |
| `src/experiments_replication/run_get_intervene_vectors.sh` | Bash wrapper |
| `src/experiments_replication/activations_qwen/` | Cached activation tensors (.pt) |
| `src/experiments_replication/steering_vectors/qwen-7b/` | Output steering vectors |
| `src/intervention.py` | Intervention experiments (run by Modal) |
| `src/experiments_replication/complete_intervene.sh` | Intervention batch script (local) |
| `src/experiments_replication/modal_intervene.py` | Modal runner for intervention experiments |
| `src/experiments_replication/modal_run.py` | Modal runner for the full pipeline (reference) |
| `slurm/intervene.slurm` | Slurm script (original, now ported to Modal) |
