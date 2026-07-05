"""
All sections (prerequisite)  —  Collect Model Behaviors on All Datasets
========================================================================

Paper requirement:
    Most experiments in §3.2–§4 require four categories of instructions defined
    by the CROSS of (harmful vs harmless) × (refused vs accepted):

      refused_harmful   : harmful instructions the model refuses  (advbench)
      accepted_harmful  : harmful instructions the model accepts  (sorry-badq / human-seed)
      refused_harmless  : harmless instructions the model refuses (xstest)
      accepted_harmless : harmless instructions the model accepts (alpaca / xstest)

    Figure 6 also needs per-jailbreak-type categories:
      jailbreak_template   : GPTFuzzer template-based jailbreaks (accepted)
      jailbreak_adversarial: human-seed adversarial prompts (accepted)
      jailbreak_persuasion : sorry-badq persuasion-style prompts (accepted)

What this script does:
    Runs the model in greedy-decode mode on each dataset, labels each sample
    by whether it was refused (contains a refusal phrase), and saves a unified
    behaviors.json file that all subsequent scripts depend on.

Outputs (in results/):
    behaviors.json — {category: [{text, response, dataset}], ...}
"""

import gc
import json
import os
import sys

import torch
from transformers import GenerationConfig

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from config import (
    HARMFUL_DATA, HARMLESS_DATA, XSTEST_DATA, SORRY_DATA,
    GPTZFUZZER_DATA, HUMAN_SEED_DATA, CATQA_DATA, RESULTS_DIR,
    N_HARMFUL, N_HARMLESS, XSTEST_N, N_JAILBREAK,
    JUDGE, JUDGE_MODEL_NAME, JUDGE_THINKING, JUDGE_MAX_NEW_TOKENS, JUDGE_BATCH_SIZE,
)
from model_utils import (
    load_model, load_judge_model, load_data, tokenize_fn, is_refusal,
    judge_accepted_batch,
)


MAX_NEW_TOKENS = 80

# Accumulates every sample the judge reclassified from accepted->refused, across all
# datasets, so main() can write a report (results/judge_report.json).
JUDGE_MOVED = []


def run_inference(model, tokenizer, data_dicts, max_new_tokens=MAX_NEW_TOKENS):
    """Generate a response for each prompt; returns list of response strings."""
    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    responses = []
    for d in data_dicts:
        inputs = tokenize_fn(tokenizer, [d])
        with torch.no_grad():
            out = model.generate(
                input_ids=inputs.input_ids.to(model.device),
                attention_mask=inputs.attention_mask.to(model.device),
                generation_config=gen_cfg,
            )
        generated = out[0, inputs.input_ids.shape[1]:]
        responses.append(tokenizer.decode(generated, skip_special_tokens=True))
    return responses


def collect(model, tokenizer, path, key, n, dataset_name):
    """
    Run inference on `n` examples from `path` (key=`key`) and split by the substring
    refusal classifier. Returns (refused, accepted) lists of {text, response, dataset}.

    NO judging happens here. To avoid holding the inference model AND the (larger)
    judge model on the GPU at once, judging is deferred: main() unloads the inference
    model after ALL inference, then runs the judge in one batched pass over every
    accepted-harmful sample (see _judge_reclassify).
    """
    data = load_data(path, key=key, n=n)
    responses = run_inference(model, tokenizer, data)
    refused, accepted = [], []
    for d, resp in zip(data, responses):
        text = d.get(key, d.get("instruction", d.get("prompt", "")))
        entry = {"text": text, "response": resp, "dataset": dataset_name}
        if not resp.strip():
            continue
        (refused if is_refusal(resp) else accepted).append(entry)
    print(f"  {dataset_name}: {len(refused)} refused / {len(accepted)} accepted "
          f"out of {len(data)} ({len(data) - len(refused) - len(accepted)} empty skipped)")
    return refused, accepted


def _judge_reclassify(behaviors):
    """
    Load the judge model and re-verify every accepted sample from a HARMFUL source
    in ONE batched pass, moving false-accepts to refused. Called by main() AFTER the
    inference model has been freed, so only one model is on the GPU at a time.

    A candidate the judge deems a refusal is removed from accepted_harmful and the
    jailbreak categories; if it came from the harmful pool (accepted_harmful) it is
    appended to refused_harmful, whereas jailbreak-only refusals are dropped (matching
    the paper's handling of refused jailbreaks). Populates JUDGE_MOVED for the report.
    """
    # Candidates = every accepted sample on a harmful source. persuasion entries are
    # the same objects already in accepted_harmful; the adversarial/template jailbreak
    # accepts are the only extra ones. harmful_ids marks the harmful-pool members so
    # their refusals route to refused_harmful (jailbreak-only ones are just dropped).
    harmful_ids = {id(e) for e in behaviors["accepted_harmful"]}
    candidates, seen = [], set()
    for e in (behaviors["accepted_harmful"]
              + behaviors["jailbreak_adversarial"]
              + behaviors["jailbreak_template"]):
        if id(e) not in seen:
            seen.add(id(e))
            candidates.append(e)
    if not candidates:
        return

    print(f"\nLoading judge model … {JUDGE_MODEL_NAME} (thinking={JUDGE_THINKING})")
    judge_model, judge_tok = load_judge_model()
    print(f"Judging {len(candidates)} accepted-harmful responses "
          f"(batch_size={JUDGE_BATCH_SIZE}) …")
    verdicts = judge_accepted_batch(
        judge_model, judge_tok,
        [(e["text"], e["response"]) for e in candidates],
        max_new_tokens=JUDGE_MAX_NEW_TOKENS,
        batch_size=JUDGE_BATCH_SIZE,
        thinking=JUDGE_THINKING,
    )

    moved_ids = {id(e) for e, ok in zip(candidates, verdicts) if not ok}
    if not moved_ids:
        print("  judge moved 0 accepted->refused")
        return
    moved = [e for e in candidates if id(e) in moved_ids]
    for cat in ("accepted_harmful", "jailbreak_persuasion",
                "jailbreak_adversarial", "jailbreak_template"):
        behaviors[cat] = [x for x in behaviors[cat] if id(x) not in moved_ids]
    n_pool = 0
    for e in moved:
        if id(e) in harmful_ids:
            behaviors["refused_harmful"].append(e)
            n_pool += 1
    JUDGE_MOVED.extend(moved)
    print(f"  judge moved {len(moved)} accepted->refused "
          f"({n_pool} to refused_harmful, {len(moved) - n_pool} jailbreak-only dropped)")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading inference model …")
    model, tokenizer = load_model()

    behaviors = {
        "refused_harmful": [],
        "accepted_harmful": [],
        "refused_harmless": [],
        "accepted_harmless": [],
        "jailbreak_template": [],
        "jailbreak_adversarial": [],
        "jailbreak_persuasion": [],
    }

    # ------------------------------------------------------------------
    # advbench — harmful prompts (full set)
    # ------------------------------------------------------------------
    print("\n[1/7] advbench (harmful) …")
    refused, accepted = collect(model, tokenizer, HARMFUL_DATA, "bad_q",
                                n=N_HARMFUL, dataset_name="advbench")
    behaviors["refused_harmful"].extend(refused)
    behaviors["accepted_harmful"].extend(accepted)   # model complied with harmful

    # ------------------------------------------------------------------
    # alpaca — harmless prompts.
    # alpaca is the accepted_harmless source ONLY. We deliberately do NOT route alpaca
    # refusals into refused_harmless: on this instruction-only file, most "refusals" are
    # artifacts — capability disclaimers ("As an AI I don't have…") and missing-input
    # apologies ("I'm sorry, you haven't provided any options") from prompts whose input
    # field was stripped. The paper's over-refusal (refused_harmless) source is xstest.
    # ------------------------------------------------------------------
    print("[2/7] alpaca (harmless) …")
    refused, accepted = collect(model, tokenizer, HARMLESS_DATA, "instruction",
                                n=N_HARMLESS, dataset_name="alpaca")
    behaviors["accepted_harmless"].extend(accepted)
    print(f"  (dropped {len(refused)} alpaca 'refusals' — artifacts, not routed to refused_harmless)")

    # ------------------------------------------------------------------
    # xstest — the sole over-refusal (refused_harmless) source
    # ------------------------------------------------------------------
    print("[3/7] xstest (harmless, over-refusal candidates) …")
    refused, accepted = collect(model, tokenizer, XSTEST_DATA, "bad_q",
                                n=XSTEST_N, dataset_name="xstest")
    behaviors["refused_harmless"].extend(refused)
    behaviors["accepted_harmless"].extend(accepted)

    # ------------------------------------------------------------------
    # sorry-badq — persuasion-style harmful prompts (full set)
    # ------------------------------------------------------------------
    print("[4/7] sorry-badq (persuasion jailbreak) …")
    refused, accepted = collect(model, tokenizer, SORRY_DATA, "bad_q",
                                n=N_HARMFUL, dataset_name="sorry-badq")
    behaviors["refused_harmful"].extend(refused)
    behaviors["accepted_harmful"].extend(accepted)  # jailbreak succeeded
    behaviors["jailbreak_persuasion"].extend(accepted)

    # ------------------------------------------------------------------
    # catqa — blatantly harmful direct questions (12 category files, full sets).
    # These are naturally harmful (not disguised), so accepted ones carry a genuine
    # harmful signature at t_inst — they go into accepted_harmful (NOT a jailbreak cat).
    # ------------------------------------------------------------------
    print(f"[4b/7] catqa ({len(CATQA_DATA)} category files, harmful) …")
    for path in CATQA_DATA:
        name = "catqa-" + os.path.splitext(os.path.basename(path))[0].replace("catqa_", "")
        refused, accepted = collect(model, tokenizer, path, "bad_q",
                                    n=N_HARMFUL, dataset_name=name)
        behaviors["refused_harmful"].extend(refused)
        behaviors["accepted_harmful"].extend(accepted)

    # ------------------------------------------------------------------
    # human-seed — adversarial jailbreak prompts.
    # Jailbreaks are NOT part of the paper's harmful pool (advbench/sorry/catqa) —
    # they are a separate Figure-6 category. So the ACCEPTED (successful) jailbreaks
    # go to jailbreak_adversarial, and the REFUSED ones are dropped (the paper does
    # not fold refused jailbreaks into refused_harmful; their disguised wrappers also
    # look harmless at tinst and would not belong in the harmful cluster).
    # ------------------------------------------------------------------
    print("[5/7] human-seed-50 (adversarial jailbreak) …")
    _refused, accepted = collect(model, tokenizer, HUMAN_SEED_DATA, "bad_q",
                                 n=N_JAILBREAK, dataset_name="human-seed")
    behaviors["jailbreak_adversarial"].extend(accepted)

    # ------------------------------------------------------------------
    # GPTFuzzer — template-based jailbreak prompts (same handling as human-seed).
    # ------------------------------------------------------------------
    print("[6/7] GPTFuzzer-50 (template jailbreak) …")
    _refused, accepted = collect(model, tokenizer, GPTZFUZZER_DATA, "bad_q",
                                 n=N_JAILBREAK, dataset_name="gptzfuzzer")
    behaviors["jailbreak_template"].extend(accepted)

    # ------------------------------------------------------------------
    # All inference done. Free the inference model BEFORE loading the judge so the
    # two never occupy the GPU at the same time (1.5B + 4B would OOM a 16GB card),
    # then judge every accepted-harmful sample in one batched pass.
    # ------------------------------------------------------------------
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    if JUDGE:
        _judge_reclassify(behaviors)

    # ------------------------------------------------------------------
    # Deterministically shuffle accepted_harmful so a first-N slice (01 uses the first
    # N_TRAIN samples) is representative of ALL sources. Its natural order is
    # advbench → sorry-badq (191) → catqa, so an unshuffled first-100 would be entirely
    # mild sorry-badq and exclude every catqa acceptance — dropping exactly the
    # genuinely-harmful examples that give the Figure 2 red line a positive signature.
    # ------------------------------------------------------------------
    import random
    random.seed(0)
    random.shuffle(behaviors["accepted_harmful"])

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n=== Behavior summary ===")
    for cat, items in behaviors.items():
        print(f"  {cat:25s}: {len(items):4d} examples")

    # Warn if misbehaving categories are too small
    for misbeh in ("accepted_harmful", "refused_harmless"):
        if len(behaviors[misbeh]) < 3:
            print(f"\n  WARNING: '{misbeh}' has only {len(behaviors[misbeh])} examples.")
            print("  Figures 2/3/5 will still plot but lines may be noisy.")

    out_path = os.path.join(RESULTS_DIR, "behaviors.json")
    with open(out_path, "w") as f:
        json.dump(behaviors, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")

    # ------------------------------------------------------------------
    # Judge report — every response the judge reclassified accepted->refused,
    # so the reclassifications can be audited by hand (spot-check for over-moves).
    # ------------------------------------------------------------------
    if JUDGE:
        from collections import Counter
        by_ds = Counter(e["dataset"] for e in JUDGE_MOVED)
        report_path = os.path.join(RESULTS_DIR, "judge_report.json")
        with open(report_path, "w") as f:
            json.dump({
                "total_moved": len(JUDGE_MOVED),
                "moved_by_dataset": dict(sorted(by_ds.items())),
                "moved": JUDGE_MOVED,
            }, f, indent=2, ensure_ascii=False)

        print(f"\n=== Judge report: {len(JUDGE_MOVED)} responses moved accepted->refused ===")
        for ds, c in sorted(by_ds.items()):
            print(f"  {ds:32s}: {c}")
        for e in JUDGE_MOVED[:5]:
            resp = " ".join(e["response"].split())
            print(f"\n  [{e['dataset']}] Q: {e['text'][:90]}")
            print(f"      R: {resp[:180]}{'…' if len(resp) > 180 else ''}")
        if len(JUDGE_MOVED) > 5:
            print(f"\n  … +{len(JUDGE_MOVED) - 5} more in {report_path}")
        print(f"\nSaved judge report to {report_path}")


if __name__ == "__main__":
    main()
