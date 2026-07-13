"""Does stripping the post-instruction tokens raise harmful-acceptance on the THINKING model too?

The paper (Table 1) shows that removing post-instruction tokens weakens an LLM's refusal ability.
The replication pipeline already tests this for non-thinking models via ``do_not_use_last_inst_tok``
(the ``gentinst`` runs accept more harmful prompts). Here we run the same probe on the thinking model
``Qwen/Qwen3.5-0.8B``, comparing two prompts on a subset of advbench:

  FULL (genthink, keeps t_post):  <|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n
  STRIPPED (without t_post):      <|im_start|>user\n{q}<|im_end|>\n<|im_start|>

Both are generated with the exact ``infer_thinking`` sampling config and classified with the canonical
``easy_eval``. Results (and only these) land in a fresh, non-colliding directory so nothing touches the
existing ``qwen350.8b`` experiment.

Run:  cd src/experiments_replication && python try_think_without_tpost.py
"""
import json
import os
import sys

import torch

# --- paths: import the paper's reference code from src/ (do not modify it) --------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from transformers import GenerationConfig, LogitsProcessorList  # noqa: E402

from inference import load_model_and_tokenizer, PresencePenaltyLogitsProcessor  # noqa: E402
from utils import formatInp_thinking  # noqa: E402
from eval import easy_eval  # noqa: E402

# --- config (tweak here) --------------------------------------------------------------------------
MODEL = "qwen35"
SIZE = "0.8b"
N = 40                    # first N advbench rows
MAX_NEW_TOKENS = 4096
BATCH_SIZE = 8
PRESENCE_PENALTY = 2.0    # matches infer_thinking

ADVBENCH = os.path.join(REPO_ROOT, "data", "advbench.json")
OUT_DIR = os.path.join(HERE, "output", "try_think_without_tpost")
GEN_DIR = os.path.join(OUT_DIR, "generations")
CLS_DIR = os.path.join(OUT_DIR, "classified")


def read_rows(path, n):
    """advbench is JSONL (one {bad_q, target} object per line)."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if len(rows) >= n:
                break
    return rows


def full_prompt(row):
    """genthink prompt, byte-identical to the pipeline (keeps every post-instruction token)."""
    return formatInp_thinking(row, thinking_mode="genthink", model=MODEL)


def stripped_prompt(row):
    """Prompt with the post-instruction tokens removed, exactly as specified:
    the full genthink prompt minus the trailing ``assistant\\n<think>\\n``.
        <|im_start|>user\\n{q}<|im_end|>\\n<|im_start|>
    """
    q = row.get("instruction") or row.get("question") or row.get("bad_q") or ""
    return f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>"


def generate_batch(model, tokenizer, prompts):
    """Batched generation replicating src/inference.py::infer_thinking exactly (only the prompt
    source differs). Returns the decoded continuation (special tokens kept) per prompt."""
    model.eval()
    model.config.use_cache = True
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    gen_cfg = GenerationConfig(
        do_sample=True, temperature=1.0, top_p=1.0, top_k=20, min_p=0.0,
        num_beams=1, pad_token_id=tokenizer.pad_token_id,
    )

    outputs = []
    for i in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[i:i + BATCH_SIZE]
        enc = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)
        prompt_len = enc.input_ids.shape[1]
        lp = LogitsProcessorList([PresencePenaltyLogitsProcessor(PRESENCE_PENALTY, prompt_len)])
        with torch.no_grad():
            out = model.generate(**enc, generation_config=gen_cfg,
                                 max_new_tokens=MAX_NEW_TOKENS, logits_processor=lp)
        gen = out[:, prompt_len:]  # continuation only
        for j in range(gen.shape[0]):
            ids = gen[j].tolist()
            if tokenizer.eos_token_id in ids:  # trim eos + any trailing pad
                ids = ids[:ids.index(tokenizer.eos_token_id)]
            text = tokenizer.decode(ids, skip_special_tokens=False)
            outputs.append(text.replace("</s>", "").strip())
        print(f"    generated {min(i + BATCH_SIZE, len(prompts))}/{len(prompts)}", flush=True)
    return outputs


def answer_portion(text):
    """Symmetric eval target: text after the last ``</think>`` if the reasoning block was closed,
    else the whole output (the stripped variant frequently never emits ``</think>``, so unlike
    easy_eval_thinking(think=True) we must NOT drop those rows)."""
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text


def run_variant(name, model, tokenizer, rows, prompt_fn):
    print(f"[{name}] building prompts + generating ({len(rows)} rows)...", flush=True)
    prompts = [prompt_fn(r) for r in rows]
    outputs = generate_batch(model, tokenizer, prompts)

    gen_rows = []
    for r, p, o in zip(rows, prompts, outputs):
        gen_rows.append({**r, "prompt_sent": p, "ori_output": o})
    _dump(os.path.join(GEN_DIR, f"advbench_{name}.json"), gen_rows)

    # classify on the answer portion, reusing the canonical matcher.
    eval_rows = [{**gr, "ori_output": answer_portion(gr["ori_output"])} for gr in gen_rows]
    labels = easy_eval(eval_rows, tag="ori_output", mode="refusal")  # '5'=accepted, '0'=refused

    accepted = [gr for gr, lab in zip(gen_rows, labels) if lab == "5"]
    refused = [gr for gr, lab in zip(gen_rows, labels) if lab == "0"]
    _dump(os.path.join(CLS_DIR, f"advbench_{name}_accepted.json"), accepted)
    _dump(os.path.join(CLS_DIR, f"advbench_{name}_refused.json"), refused)

    closed = sum("</think>" in gr["ori_output"] for gr in gen_rows)
    n = len(gen_rows)
    stats = {
        "variant": name,
        "n": n,
        "accepted": len(accepted),
        "refused": len(refused),
        "acceptance_rate": round(len(accepted) / n, 4) if n else 0.0,
        "think_closed": closed,
    }
    print(f"[{name}] accepted={stats['accepted']}/{n} "
          f"(rate={stats['acceptance_rate']}), </think>-closed={closed}/{n}", flush=True)
    return stats


def _dump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def main():
    os.makedirs(GEN_DIR, exist_ok=True)
    os.makedirs(CLS_DIR, exist_ok=True)

    rows = read_rows(ADVBENCH, N)
    print(f"Loaded {len(rows)} advbench rows from {ADVBENCH}", flush=True)

    print(f"Loading {MODEL} {SIZE}...", flush=True)
    model, tokenizer = load_model_and_tokenizer(MODEL, SIZE)

    full = run_variant("full", model, tokenizer, rows, full_prompt)
    stripped = run_variant("stripped", model, tokenizer, rows, stripped_prompt)

    summary = {
        "config": {"model": MODEL, "size": SIZE, "n": N,
                   "max_new_tokens": MAX_NEW_TOKENS, "batch_size": BATCH_SIZE},
        "full": full,
        "stripped": stripped,
        "hypothesis_stripped_accepts_more": stripped["accepted"] > full["accepted"],
    }
    _dump(os.path.join(OUT_DIR, "summary.json"), summary)

    print("\n===== SUMMARY =====")
    print(f"FULL     (keeps t_post): {full['accepted']}/{full['n']} accepted "
          f"(rate {full['acceptance_rate']})")
    print(f"STRIPPED (no t_post):    {stripped['accepted']}/{stripped['n']} accepted "
          f"(rate {stripped['acceptance_rate']})")
    print(f"Stripping post-instruction tokens raises acceptance? "
          f"{summary['hypothesis_stripped_accepts_more']}")
    print(f"Outputs written under: {OUT_DIR}")


if __name__ == "__main__":
    main()
