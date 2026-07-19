# LLMs Encode Harmfulness and Refusal Separately — Replication

This repository contains all the code for our AI Safety project replicating the paper
***LLMs Encode Harmfulness and Refusal Separately*** (Zhao et al., NeurIPS 2025). It is a fork of
the paper's original repository:
<https://github.com/CHATS-lab/Llms_Encode_Harmfulness_Refusal_Separately>.

The original codebase provided the utility functions for inference and activation extraction; we
implemented all the dataset handling and experiment logic on top of it. **All of our work lives in
`src/experiments_replication/`.**

We run the pipeline on two tracks:

- **Instruct track** — the paper's original replication (Qwen2, Llama-2, Llama-3).
- **Thinking track** — our extension to a reasoning model (Qwen3.5), with its own generation modes
  (`genthink`, `gennothink`, `gennothink_stripped`).

---

## Repository structure

```
src/experiments_replication/
  configs/
    bucketing/    how activations are composed into behavioural buckets
    eval/         judge-LLM evaluation configs
    intervene/    steering / intervention experiment configs
    sampling/     generation sampling params
  output/         results, one directory per model (e.g. qwen359b)
  *.py            the pipeline (see below)
data/             the datasets, already downloaded
scripts/          runnable smoke-test + full-run shell scripts (see "Running the experiments")
```

**Configs.** `bucketing/bucket_config_clean.json` is the configuration used for the Qwen2
replication; `bucketing/bucket_config_qwen35_*.json` are the thinking-track configs (one per
generation mode). Because each source's activations are stored in a separate file per bucket,
re-bucketing is a cheap CPU re-index — you can try different dataset compositions and regenerate the
plots without touching the GPU.

**Output.** Under each model directory (e.g. `output/qwen359b/`) we store the raw generations, then
the classified buckets under `classified_generations/` or `judge_classifications/` (depending on
whether the judge LLM was used), and the activations under `activations/` or `judge_activations/`.
The activation tensors are **removed from the repository** for storage reasons — they are shared
out-of-band in a Drive folder: **#todo (add link)**.

---

## The pipeline

The main pipeline is driven by `main.py`, which runs a sequence of **stages** — set the `stages`
tuple and each runs in order, composing results:

| stage | what it does |
|---|---|
| `infer` | run inference on the datasets (via the original `inference.py`) |
| `eval` | classify each generation as accepted / refused, splitting into buckets |
| `acts` | extract activations (via the original `extract_hidden.py`) |
| `gen_buckets` | compose the activation files into train/test buckets per the bucket config |
| `fig` | Figure 2 — the per-layer activation-clustering plot |
| `fig3` | Figure 3 — the harmfulness-vs-refusal belief scatter (instruct track only) |

Because activations are stored per bucket, `gen_buckets`/`fig`/`fig3` are pure CPU and let you
iterate on the bucketing configuration quickly without re-running inference.

**Steering & intervention experiments** (Figures 4 and 5) live in `_04_intervention.py`, with four
stages of their own: `vectors` (build the steering directions), `intervene` (steered generation),
`judge4` (label the outputs), `fig4` (plot). `launch_intervention.py` orchestrates the whole
intervention suite; `dynamic_bucket_formation.py` handles bucket composition; `judge_llm.py` is the
model-as-judge.

**GCG jailbreak.** `qwen35_jailbreak/gcg_attack.py` runs the nanoGCG adversarial-suffix attack (the
jailbreak prompt template lives alongside it); `gcg_to_generations.py` feeds the results back into
the pipeline. These are run on Modal via `modal_gcg.py`.

---

## Running the experiments

Everything runs on **Modal**. Each experiment ships **two runnable scripts** in `scripts/` — a
smoke test (smallest useful run, blocking) and the full run (detached):

| experiment | smoke | full |
|---|---|---|
| thinking pipeline | `scripts/thinking_smoke.sh` | `scripts/thinking_full.sh` |
| Qwen2 replication | `scripts/qwen2_smoke.sh` | `scripts/qwen2_full.sh` |
| interventions (Fig 4/5) | `scripts/intervention_smoke.sh` | `scripts/intervention_full.sh` |
| GCG attack | `scripts/gcg_smoke.sh` | `scripts/gcg_full.sh` |

They `cd` to the repo root themselves, so run from anywhere:

```bash
./scripts/intervention_smoke.sh            # quick end-to-end check
./scripts/intervention_full.sh             # launch the real run (detached)
./scripts/intervention_full.sh collect     # pull the results afterwards
```

Each `*_full.sh` takes an optional `collect` argument for the pull phase. Env overrides: `MODAL`
(default `modal`; set `MODAL="python -m modal"` if `modal` isn't on PATH) and `PY` (default
`python`).

The full flag reference — how to change datasets, token slots, coefficients, row slices, etc. — is
in **[`scripts.md`](scripts.md)**.

The underlying Modal runners are:

- `modal_run.py` — the `infer/eval/acts/fig/fig3` pipeline (both tracks; `--thinking` for Qwen3.5).
- `modal_intervene.py` — a single steered-generation or judge run (driven by `launch_intervention.py`).
- `modal_gcg.py` — the GCG attack.

> The GPU stages for `qwen7b` and `llama38b` are already committed. Normal iteration is CPU-only
> (`stages=("fig","fig3")` locally through `main.py`) — only re-run `infer/eval/acts` when
> deliberately regenerating from scratch.
