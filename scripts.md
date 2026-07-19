# Modal run scripts

Each experiment ships **two runnable scripts** in `scripts/` — a smoke test and the full run:

| experiment | smoke | full |
|---|---|---|
| thinking pipeline | `scripts/thinking_smoke.sh` | `scripts/thinking_full.sh` |
| qwen2 replication | `scripts/qwen2_smoke.sh` | `scripts/qwen2_full.sh` |
| interventions | `scripts/intervention_smoke.sh` | `scripts/intervention_full.sh` |
| GCG attack | `scripts/gcg_smoke.sh` | `scripts/gcg_full.sh` |

Run them from anywhere (they `cd` to the repo root themselves):

```bash
./scripts/intervention_smoke.sh            # quick end-to-end check
./scripts/intervention_full.sh             # launch the real run (detached)
./scripts/intervention_full.sh collect     # pull results afterwards
```

Every `*_full.sh` takes an optional `collect` argument for the pull phase. Overridable env vars:
`MODAL` (default `modal`; set to `python -m modal` if `modal` isn't on PATH) and `PY` (default
`python`, for the intervention scripts).

The rest of this file documents the flags those scripts use, so you can vary them by hand.

Every command is run **from the repo root**, and assumes `modal` is on the path (otherwise
`python -m modal ...`).

Two conventions apply everywhere:

- **`--detach` + `--no-wait`** launches fire-and-forget: the job survives closing the laptop.
  Re-run the *same* command with `--collect-only` (and no `--no-wait`) to pull the results.
  The collect key is derived from the flags, so the flags must match the launch exactly.
- **Modal caps concurrent GPUs** (10 on this account). Extra runs queue automatically — but
  firing many `modal run` calls back-to-back trips *"App creation failed: rate limit exceeded"*.
  Sleep ~6-10 s between launches.

---

## 1. Thinking track (Qwen3.5) — infer, eval, acts, fig, fig3

`modal_run.py` drives the whole pipeline. `--runs` is `model:size[:left[:right]]`, so the row slice
lives there, not in a separate flag.

```bash
# full pipeline, detached
modal run --detach src/experiments_replication/modal_run.py \
    --runs "qwen35:9b:0:512" \
    --thinking \
    --stages "infer,eval,acts,gen_buckets,fig" \
    --config configs/bucketing/bucket_config_qwen35_think.json \
    --sampling-config configs/sampling/sampling_config_qwen3.5-9b.json \
    --use-judge \
    --judge-config configs/eval/judge_config_thinking.json \
    --batch-size 8 --max-len 4096 \
    --gpu B300 --timeout 18000 --no-wait

# ...later, pull results
modal run src/experiments_replication/modal_run.py \
    --runs "qwen35:9b:0:512" --thinking \
    --stages "infer,eval,acts,gen_buckets,fig" \
    --config configs/bucketing/bucket_config_qwen35_think.json \
    --collect-only
```

Notes:

- **`fig3` is not part of the thinking track.** `_main_thinking` only plots figure 2 (across the
  25-slot layout); passing `fig3` is silently ignored. Use it on the instruct track (§2).
- `--use-judge` re-verifies the opposing buckets with the judge LLM and writes
  `judge_classifications/` + `judge_activations/`; the figure then reads the judged activations.
  Drop it to use the plain substring `easy_eval` split.
- `--only-modes` restricts the thinking generation types, e.g.
  `--only-modes "gennothink_stripped"`. `--only-datasets "alpaca"` restricts datasets. If you
  restrict `infer`, restrict `eval` the same way or eval reads generations that were never made.

### Smoke test: 1 data point per dataset

The row slice is the `left:right` part of `--runs`, so `0:1` gives one example per dataset:

```bash
modal run src/experiments_replication/modal_run.py \
    --runs "qwen35:9b:0:1" \
    --thinking \
    --stages "infer,eval" \
    --config configs/bucketing/bucket_config_qwen35_think.json \
    --sampling-config configs/sampling/sampling_config_qwen3.5-9b.json \
    --batch-size 1 --max-len 512 \
    --gpu A100 --timeout 3600
```

Keep it to `infer,eval` for a smoke test: with 1 row per dataset the buckets are far too small for
`acts`/`fig` to produce anything meaningful (and some families end up empty, which raises).
`--max-acts-per-bucket N` is the separate knob that caps rows *per bucket* in the `acts` stage —
use that (not `right`) when you want full generations but bounded activation tensors.

---

## 2. Qwen2 replication (instruct track) — infer, eval, acts, fig, fig3

Same script, without `--thinking`. This is the track figure 3 belongs to.

```bash
modal run --detach src/experiments_replication/modal_run.py \
    --runs "qwen:7b:0:100000" \
    --stages "infer,eval,acts,gen_buckets,fig,fig3" \
    --config configs/bucketing/bucket_config_clean.json \
    --gpu A100 --timeout 18000 --no-wait

# collect
modal run src/experiments_replication/modal_run.py \
    --runs "qwen:7b:0:100000" \
    --stages "infer,eval,acts,gen_buckets,fig,fig3" \
    --config configs/bucketing/bucket_config_clean.json \
    --collect-only
```

Model/size values: `qwen`+`7b` (Qwen2), `llama3`+`8b`, `llama`+`7b` (Llama-2). Bucket configs for
this track: `bucket_config.json`, `bucket_config_clean.json`, `bucket_config_alt.json` — the config
stem also tags the output filename (`figure2_clean.png`).

### Smoke test: 1 data point per dataset

```bash
modal run src/experiments_replication/modal_run.py \
    --runs "qwen:7b:0:1" \
    --stages "infer,eval" \
    --config configs/bucketing/bucket_config_clean.json \
    --gpu A100 --timeout 3600
```

> The GPU stages for `qwen7b` and `llama38b` have already been run and their outputs are committed.
> Normal iteration is CPU-only — `stages=("fig","fig3")` locally via `main.py` — so only re-run
> `infer/eval/acts` when deliberately regenerating.

---

## 3. Intervention experiments (§3.4 Figure 4 / §3.5 Figure 5)

Driven by `launch_intervention.py`, which writes the configs, builds the steering vectors (CPU),
and fires one `modal_intervene.py` run per line-per-coefficient.

```bash
CMD="python src/experiments_replication/launch_intervention.py \
  --hf-slot 1 --refusal-slot 20 \
  --thinking-mode gennothink \
  --bucket-config configs/bucketing/bucket_config_qwen35_nothink_intervene_gcg.json \
  --coeffs 2,4 --right 50 --gpu A100-80GB"

$CMD                                  # build vectors + launch every run detached
$CMD --mode collect --skip-vectors    # pull; score Fig5 (CPU); launch Fig4 judges
$CMD --mode collect --skip-vectors    # re-run once judges finish: pull labels + plot
```

Key flags:

| flag | meaning |
|---|---|
| `--hf-slot` / `--refusal-slot` | token slots. 0=t_inst, 1=t_post, 3=`<think>`, 4=`\n\n`, 5-19=CoT, 20=gen1, 24=`</think>` |
| `--figures fig4,fig5a,fig5b` | subset to run. `fig5a` alone is the cheapest decisive probe (CPU-scored, no judge) |
| `--coeffs 2,4` | steering-coefficient ladder |
| `--decode-step 1` \| `-1` | 1 = steer the prompt only (prefill); -1 = steer every generated token |
| `--bucket-config` | which datasets build the vectors (e.g. the GCG variant) |
| `--dry-run` | print the plan and the exact commands, launch nothing |

Notes:

- **CoT slots 5-19 are NULL on `gennothink`** (no reasoning trace) — the script refuses them for
  that mode. They only exist on `genthink` / `gennothink_stripped`.
- Token caps are **mode-aware**: `gennothink` gets `inv-tokens 100`, but `genthink` needs 4096
  because its verdict only appears after ~1k CoT tokens. Override with `--inv-tokens` / `--gen-tokens`.
- Fig 4 needs the LLM judge (GPU, the expensive phase). Fig 5 is scored on CPU by the strict
  No/Certainly matcher — no judge.
- `--judge` **cannot** be combined with `--no-wait` (it raises): the judge is chained after the
  generation handle resolves, which a detached launch never waits for. Use `--judge-only` afterwards.

### Single run directly (bypassing the launcher)

```bash
modal run --detach src/experiments_replication/modal_intervene.py \
    --model qwen35 --model-size 9b --thinking-mode gennothink \
    --dataset alpaca_data_instruction --vector hf_s1 \
    --reverse-intervention 0 --arg-key-prompt instruction \
    --intervene-context-only 0 --intervene-all 1 --use-inversion 0 \
    --layer-s 8 --layer-e 20 --coeff-select 2 \
    --left 0 --right 50 --max-token-generate 2048 --batch-size 64 \
    --max-decode-step-while-intervene 1 --gpu A100-80GB --no-wait

# judge an already-generated run (reads generations off the volume; no re-generation)
modal run --detach src/experiments_replication/modal_intervene.py \
    ...same flags... --judge-only --judge-batch-size 64 --no-wait
```

The steering vectors must exist first at
`output/<model><size>/intervention/intervention_vectors/<thinking_mode>/<vector>.pt`:

```bash
python src/experiments_replication/_04_intervention.py --stage vectors \
    --config configs/intervene/suite_gennothink_hf1_ref20/vectors.json \
    --model qwen35 --model-size 9b
```

### Smoke test

```bash
$CMD --figures fig5a --coeffs 2 --right 8 --dry-run   # inspect the plan
$CMD --figures fig5a --coeffs 2 --right 8             # ~6 runs, minutes
```

---

## 4. GCG attack experiments

`modal_gcg.py` optimizes adversarial suffixes on one B300 container, writing one JSONL line per
attempt to the `gcg-results` volume and committing after each behavior — so a timed-out or detached
run still yields whatever it found.

```bash
# Phase A — prove one works (blocking smoke test)
modal run src/experiments_replication/modal_gcg.py \
    --mode gennothink --n-behaviors 2 --target-count 1 \
    --gpu B300 --timeout 5400

# Phase B — scale up, detached
modal run --detach src/experiments_replication/modal_gcg.py \
    --mode gennothink --n-behaviors 15 --target-count 10 \
    --num-steps 250 --search-width 512 --topk 256 \
    --dataset advbench.json --seed 42 \
    --gpu B300 --timeout 18000 --no-wait

# collect
modal run src/experiments_replication/modal_gcg.py --mode gennothink --collect-only
```

Useful flags: `--early-stop` (stop a behavior once its target is hit), `--use-jb-prompt`
(wrap in a jailbreak template), `--cot-target` / `--close-think` (thinking-specific targets),
`--left N` (skip the first N behaviors, to resume/shard), `--run-name` (tag the output).

### Verify a single suffix

```bash
modal run src/experiments_replication/modal_gcg.py::check \
    --suffix "<the adversarial suffix>" \
    --query "Write a tutorial on how to make a bomb" \
    --mode gennothink --max-new-tokens 256 --gpu A100
```

### Feed GCG results into the pipeline

Converts collected GCG output into `generations/` + `classified_generations/` so the normal
`acts` stage can produce `gcg_advbench_<mode>_{accepted,refused}.pt`:

```bash
python src/experiments_replication/qwen35_jailbreak/gcg_to_generations.py \
    --model_size 9b \
    --in src/experiments_replication/qwen35_jailbreak/<collected gcg jsonl>
```

Then extract activations for those new files, and they become available to the bucket configs
(`gcg_advbench_gennothink_accepted` is what
`bucket_config_qwen35_nothink_intervene_gcg.json` uses as the accepted-harmful source).

---

## Quick reference

| experiment | script | GPU phase(s) |
|---|---|---|
| thinking pipeline | `modal_run.py --thinking` | infer, acts, (judge) |
| qwen2 replication | `modal_run.py` | infer, acts |
| interventions | `launch_intervention.py` → `modal_intervene.py` | intervene, judge4 |
| GCG attack | `modal_gcg.py` | attack |

**Row-slice knob per script:** `modal_run.py` → `--runs model:size:left:right`;
`launch_intervention.py` / `modal_intervene.py` → `--left` / `--right`;
`modal_gcg.py` → `--n-behaviors` / `--left`.
