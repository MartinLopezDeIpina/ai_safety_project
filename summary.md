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
- Saves output to `steering_vectors/qwen7b/`

## Verification Results

Both runs completed successfully (exit code 0):

| File | Size | Shape | dtype |
|---|---|---|---|
| `steering_vectors/qwen7b/hf.pt` | 205 KB | (29, 3584) | float16 |
| `steering_vectors/qwen7b/refusal.pt` | 205 KB | (29, 3584) | float16 |


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
1. ~~**Run intervention experiments**~~ — ✅ Done (see "Intervention Experiments" below)
2. **Replicate figures** — Run the figure-plotting stages (Figures 2 & 3 from paper) (NOT ON THIS BRANCH)
3. **Other models** — The repo supports Llama-2/3 as well; can extend the same pipeline


## Modal Script Created: `modal_intervene.py`

**Path:** `src/experiments_replication/modal_intervene.py`

### Purpose
Modal equivalent of `slurm/intervene.slurm`. Runs `src/intervention.py` on Modal A100 containers, then copies the produced output JSON file(s) onto a results volume. The local entrypoint pulls every finished file back into `src/experiments_replication/intervention_outputs/`, auto-indexing (`key`, `key_2`, `key_3`, …) so it never clobbers a previous local run.

### Design
- **Image / layout** — Reproduces the repo under `/root` so `intervention.py`'s `from utils import ...` and `from template_inversion import ...` (both in `src/`) resolve. Sets `PYTHONPATH=/root/src` and `cwd=/root/src`.
- **Steering vectors** — Unlike `modal_run.py`, this script does **not** ignore `*.pt` under `experiments_replication/`; the steering vectors (`steering_vectors/qwen7b/hf.pt`, `refusal.pt`) are mounted at runtime.
  IMPORTANT TODO: allow the user to specify the steering vector location.
- **Volume** — Uses a dedicated `intervention-results` volume (separate from `behavior-results`) so intervention outputs don't collide with pipeline outputs.
- **Defaults** — Mirror `intervene.slurm` exactly: `qwen:7b`, `advbench`, `hf` vector, `reverse_intervention=1` (less-harm), `layer_s=0`, `layer_e=28`, `coeff_select=1`, `max_token_generate=100`, `use_inversion=1`, `inversion_prompt_idx=1`, `right=50`.
- **Output collection** — `intervention.py` writes per-layer JSONs (`<stem>-intervene<layer>.json`), so the `finally` block copies **all** JSONs from the run's output dir to the volume.

### CLI Usage

DON'T RUN THIS FILE. Use `_045_interventions.py` instead.


## Local Environment Setup (uv)

### `pyproject.toml` — updated for local Blackwell GPU

| Field | Before (cluster) | After (local) |
|---|---|---|
| `requires-python` | `>=3.11` | `>=3.12,<3.13` |
| `torch` | unpinned, `pytorch-cu121` index | `torch==2.12.1`, `pytorch-cu130` index |
| Other deps | unpinned | pinned to match `requirements.txt` versions |

- **Why CUDA 13.0 torch** — Local RTX 5060 Ti is Blackwell (sm_120); the cluster's `cu121` torch only ships up to sm_90 and has no kernels for sm_120.
- **`requirements.txt` was NOT touched** — it remains the cluster reference. The local override lives entirely in `pyproject.toml`.
- **Verified**: `torch 2.12.1+cu130` detects RTX 5060 Ti, `sm_120` in arch list; all other deps match `requirements.txt` versions exactly.


## Intervention Experiments (`_045_interventions.py`)

### Test Run (500 examples) — scripts ready

The scripts have been bumped to 500 examples (`qwen:7b:0:500`) and expanded from 6 to 14
experiments across four datasets (advbench, jbb, alpaca, xstest-harmless), plus two
no-inversion runs on alpaca.

### Python Scripts

| Script | Purpose |
|---|---|
| `_045_interventions.py` | Run (`--mode run`) or collect (`--mode collect`) all experiments |

The experiment table is generated programmatically from dataset/vector definitions.
A 5-second delay between `modal run` calls avoids the rate limit.

```bash
# Launch all experiments detached
uv run python src/experiments_replication/_045_interventions.py --mode run --runs "qwen:7b:0:500"

# Collect results
uv run python src/experiments_replication/_045_interventions.py --mode collect --runs "qwen:7b:0:500"
```

### Full Experiment Table (14 experiments)

| # | Dataset | Vector | Reverse | Prompt key | `use_inversion` | Effect |
|---|---|---|---|---|---|---|
| 1 | advbench | hf | 1 | `bad_q` | 1 | Less-harm steering |
| 2 | advbench | refusal | 1 | `bad_q` | 1 | Less-refusal steering |
| 3 | advbench | refusal | 0 | `bad_q` | 1 | More-refusal steering |
| 4 | alpaca_data_instruction | hf | 0 | `instruction` | 1 | More-harm steering |
| 5 | alpaca_data_instruction | refusal | 0 | `instruction` | 1 | More-refusal steering |
| 6 | alpaca_data_instruction | refusal | 1 | `instruction` | 1 | Less-refusal steering |
| 7 | jbb | hf | 1 | `bad_q` | 1 | Less-harm steering |
| 8 | jbb | refusal | 1 | `bad_q` | 1 | Less-refusal steering |
| 9 | jbb | refusal | 0 | `bad_q` | 1 | More-refusal steering |
| 10 | xstest-harmless | hf | 0 | `bad_q` | 1 | More-harm steering |
| 11 | xstest-harmless | refusal | 0 | `bad_q` | 1 | More-refusal steering |
| 12 | xstest-harmless | refusal | 1 | `bad_q` | 1 | Less-refusal steering |
| 13 | alpaca_data_instruction | hf | 0 | `instruction` | 0 | More-harm (no inversion) |
| 14 | alpaca_data_instruction | refusal | 0 | `instruction` | 0 | More-refusal (no inversion) |

Experiments 1–12 use the inversion prompt (`use_inversion=1`); experiments 13–14 steer
without the inversion question so the model responds directly to the instruction.

### Intervention Token Scope (Appendix E.1)

Per the paper's Appendix E.1, the two steering directions require different token scopes in the reply inversion task:

| Vector | `--intervene_context_only` | `--intervene_all` | Tokens intervened |
|---|---|---|---|
| `hf` (harmfulness) | 1 | 0 | Context only (before the inversion question) |
| `refusal` | 0 | 1 | All input tokens (including post-instruction tokens) |

The paper finds that steering with the refusal direction only works effectively when applied to all input tokens, because refusal is processed after seeing post-instruction tokens (Section 3.1). The harmfulness direction, however, only needs to be applied to the context tokens before the inversion question.

In `modal_intervene.py`, both `--intervene-context-only` and `--intervene-all` default to `-1` (auto), which picks the correct values based on the vector: `refusal` → `intervene_all=1, intervene_context_only=0`; `hf` → `intervene_all=0, intervene_context_only=1`. An explicit `0`/`1` overrides the auto default.

### Multi-Run Argument Cycling

IMPORTANT: This is not ready. Don't use this feature. (And we don't need it.)

When passing fewer `--datasets` or `--vectors` values than `--runs` entries, the last value is reused for the remaining runs. For example:

```bash
modal run modal_intervene.py --runs "qwen:7b:0:50,qwen:7b:0:50,qwen:7b:0:50" \
    --datasets "advbench,jbb" --vectors "hf,refusal"
```

| Run | Dataset | Vector |
|---|---|---|
| 1 | advbench | hf |
| 2 | jbb | refusal |
| 3 | jbb (last repeated) | refusal (last repeated) |


## Figure 5: Refusal Rate Across Intervention Layers

**Path:** `src/experiments_replication/_05_figure5.py`

Replicates Figure 5 from the paper. Loads per-layer intervention JSONs, classifies responses
with `easy_eval` (mode=`'inversion'`), and plots refusal rate vs. layer.

### Features
- Two panels: (a) harmless instructions (pools alpaca + xstest-harmless), (b) harmful instructions (pools advbench + jbb)
- Refusal rate computed by pooling all examples across datasets per layer
- CLI args: `--model`, `--right`, `--harmless`, `--harmful`, `--output-dir`. By default, the script tries to load all harmful and harmless instructions.
- The script assumes that the generations can be found in `intervention_outputs/`.

### Usage
```bash
# Default: all four datasets, 50 examples
uv run python src/experiments_replication/_05_figure5.py

# 500-example run
uv run python src/experiments_replication/_05_figure5.py --right 500

# Only specific datasets
uv run python src/experiments_replication/_05_figure5.py --harmless "alpaca_data_instruction" --harmful "advbench"
```

Output: `output/<model>/figure5_refusal_rate.png`

## File Paths Reference

| Path | Description |
|---|---|
| `src/experiments_replication/main.py` | Pipeline orchestrator |
| `src/experiments_replication/get_intervene_vectors.py` | Steering vector computation |
| `src/experiments_replication/run_get_intervene_vectors.sh` | Bash wrapper (steering vectors) |
| `src/experiments_replication/activations_qwen/` | Cached activation tensors (.pt) |
| `src/experiments_replication/steering_vectors/qwen7b/` | Output steering vectors (`hf.pt`, `refusal.pt`) |
| `src/intervention.py` | Intervention experiments (run by Modal) |
| `src/experiments_replication/complete_intervene.sh` | Intervention batch script (local) |
| `src/experiments_replication/modal_intervene.py` | Modal runner for intervention experiments |
| `src/experiments_replication/modal_run.py` | Modal runner for the full pipeline (reference) |
| `src/experiments_replication/intervention_outputs/` | Collected intervention results (6 experiments × 28 layers) |
| `src/experiments_replication/_045_interventions.py` | Run/collect all 14 intervention experiments on Modal |
| `src/experiments_replication/_05_figure5.py` | Figure 5: refusal rate across intervention layers |
| `slurm/intervene.slurm` | Slurm script (original, now ported to Modal) |
| `run_interventions_modal.sh` | Launches intervention experiments on Modal (legacy bash, replaced by _045_interventions.py) |
| `collect_interventions_modal.sh` | Collects intervention results (legacy bash, replaced by _045_interventions.py) |
| `pyproject.toml` | Local uv environment (CUDA 13.0 torch for Blackwell GPU) |


## TODO

- ~~Check if `_05_figure5.py` has been correctly adapted to the new file naming scheme (with `inv1` suffix).~~ Done.
- Add `_04_figure4.py`.
- Get steering vectors for the other models and let the user pick which one to load for the intervention experiments.
- Run intervention experiments on all data and thinking model.
