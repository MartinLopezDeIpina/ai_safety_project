# Conversation Summary

**Date:** 2026-07-15
**Project:** AI Safety — Replication of "LLMs Encode Harmfulness and Refusal Separately" (Zhao et al., 2025)

# TL;DR

- Go to `run_get_intervene_vectors.sh`. Change `ACT_DIR` and `OUT_DIR`. In the `OUT_DIR` you will get the steering vectors for the harmfulness direction (`hf.pt`) and the refusal direction (`refusal`).
- Go to `modal_intervene.py`. Change `vector_pth` in line 108 (or somewhere near if you have changed this file). Make sure that `vector_pth` matches your `OUT_DIR` in the previous step.
- To run the intervention experiments for figure 5, run:
  ```bash
  uv run python src/experiments_replication/_045_interventions.py --mode run --runs "qwen:7b:0:500"
  ```
  After 2h or so, run
  ```bash
  uv run python src/experiments_replication/_045_interventions.py --mode collect --runs "qwen:7b:0:500"
  ```
  to collect the results.
  Then run `uv run python src/experiments_replication/_05_figure5.py --right 500` to produce figure 5.
- For figure 4, run:
  ```bash
  modal run --detach src/experiments_replication/modal_intervene.py \
    --runs qwen:7b:0:100 \
    --datasets alpaca_data_instruction \
    --vectors hf \
    --reverse-intervention 0 \
    --arg-key-prompt instruction \
    --intervene-context-only 0 \
    --intervene-all 1 \
    --use-inversion 0 \
    --no-wait
  # after some while
  modal run --detach src/experiments_replication/modal_intervene.py \
    --runs qwen:7b:0:100 \
    --datasets alpaca_data_instruction \
    --vectors refusal \
    --reverse-intervention 0 \
    --arg-key-prompt instruction \
    --intervene-context-only 0 \
    --intervene-all 1 \
    --use-inversion 0 \
    --no-wait
  ```
  After 3h or so, run the same comments but with `--no-wait` replaced by `--collect-only` to get the results.
  I have not finished the experiment yet, so I just produced figure 4 from the inversion prompting results.
  But changing the code to no-inversion results should be easy.

Be careful that increasing the data size could lead to time-outs.

One more thing to try is to set "coeff_select" to 2 or something else. This is the scaling factor applied to the steering vectors.

---

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

### Intervention Token Scope (Appendix E.1)

Per the paper's Appendix E.1, the two steering directions require different token scopes in the reply inversion task:

| Vector | `--intervene_context_only` | `--intervene_all` | Tokens intervened |
|---|---|---|---|
| `hf` (harmfulness) | 1 | 0 | Context only (before the inversion question) |
| `refusal` | 0 | 1 | All input tokens (including post-instruction tokens) |

The paper finds that steering with the refusal direction only works effectively when applied to all input tokens, because refusal is processed after seeing post-instruction tokens (Section 3.1). The harmfulness direction, however, only needs to be applied to the context tokens before the inversion question.

In `modal_intervene.py`, both `--intervene-context-only` and `--intervene-all` default to `-1` (auto), which picks the correct values based on the vector: `refusal` → `intervene_all=1, intervene_context_only=0`; `hf` → `intervene_all=0, intervene_context_only=1`. An explicit `0`/`1` overrides the auto default.


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

## TODO

- ~~Check if `_05_figure5.py` has been correctly adapted to the new file naming scheme (with `inv1` suffix).~~ Done.
- Add `_04_figure4.py`.
- Get steering vectors for the other models and let the user pick which one to load for the intervention experiments.
- Run intervention experiments on all data and thinking model.
