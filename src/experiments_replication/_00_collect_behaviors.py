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
    RESULTS_DIR, DATASET_ROUTING, JUDGE_BUCKET, JUDGE_MOVE_TO,
    JUDGE, JUDGE_MODEL_NAME, JUDGE_THINKING, JUDGE_MAX_NEW_TOKENS, JUDGE_BATCH_SIZE,
    TINST_ACCEPTED_NO_POSTINST, TINST_ACCEPTED_SOURCES, POLE_HARMFUL_SOURCES,
)
from model_utils import (
    load_model, load_judge_model, load_data, tokenize_fn, is_refusal,
    judge_accepted_batch, format_prompt_no_postinst,
)


MAX_NEW_TOKENS = 80


def _interleave_by_dataset(items, priority_sources):
    """Reorder `items` so the priority-source datasets are round-robin interleaved at the
    front (balanced across sources), followed by every other dataset in original order.

    Used so a first-N_TRAIN slice is balanced across the priority corpora rather than
    dominated by whichever one is most numerous (e.g. keeps the μ_harmful pole a mix of
    advbench+jbb instead of all-advbench, and holds sorry-badq out of the pole slice).
    """
    from collections import deque
    groups = {s: deque(it for it in items if it["dataset"] == s)
              for s in sorted(priority_sources)}
    queues = [q for q in groups.values() if q]
    out = []
    while any(queues):
        for q in queues:
            if q:
                out.append(q.popleft())
    out.extend(it for it in items if it["dataset"] not in priority_sources)
    return out

# Accumulates every sample the judge reclassified from accepted->refused, across all
# datasets, so main() can write a report (results/judge_report.json).
JUDGE_MOVED = []
# Accumulates EVERY judged sample (not just moved) with the judge's 3-way label
# (ACCEPT / REJECT / NULL) and full raw response (thinking trace + verdict), so the
# report can show what the judge actually did — including NULLs where it ran out of
# thinking budget before emitting a verdict.
JUDGE_ALL = []


def run_inference(model, tokenizer, data_dicts, max_new_tokens=MAX_NEW_TOKENS,
                  no_postinst=False):
    """Generate a response for each prompt; returns list of response strings.

    no_postinst=True formats prompts WITHOUT the post-instruction tokens
    (format_prompt_no_postinst) — used for the §3.1 / Table 1 regime where refusal
    collapses, to harvest the accepted-harmful set for the Figure 2 t_inst panel.
    """
    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    responses = []
    for d in data_dicts:
        if no_postinst:
            inputs = tokenizer([format_prompt_no_postinst(d)],
                               padding=True, return_tensors="pt")
        else:
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


def collect(model, tokenizer, path, key, n, dataset_name, no_postinst=False):
    """
    Run inference on `n` examples from `path` (key=`key`) and split by the substring
    refusal classifier. Returns (refused, accepted) lists of {text, response, dataset}.

    no_postinst=True prompts without the post-instruction tokens (Figure 2 t_inst set).

    NO judging happens here. To avoid holding the inference model AND the (larger)
    judge model on the GPU at once, judging is deferred: main() unloads the inference
    model after ALL inference, then runs the judge in one batched pass over every
    accepted-harmful sample (see _judge_reclassify).
    """
    data = load_data(path, key=key, n=n)
    responses = run_inference(model, tokenizer, data, no_postinst=no_postinst)
    refused, accepted = [], []
    for d, resp in zip(data, responses):
        text = d.get(key, d.get("instruction", d.get("prompt", "")))
        entry = {"text": text, "response": resp, "dataset": dataset_name}
        if not resp.strip() or resp.strip() == ".":
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
    harmful_ids = {id(e) for e in behaviors[JUDGE_BUCKET]}
    candidates, seen = [], set()
    for e in (behaviors[JUDGE_BUCKET]
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
    details = judge_accepted_batch(
        judge_model, judge_tok,
        [(e["text"], e["response"]) for e in candidates],
        max_new_tokens=JUDGE_MAX_NEW_TOKENS,
        batch_size=JUDGE_BATCH_SIZE,
        thinking=JUDGE_THINKING,
        return_details=True,
    )

    # Log every judged sample (label + full raw trace) for the report.
    for e, d in zip(candidates, details):
        JUDGE_ALL.append({
            "dataset": e["dataset"],
            "text": e["text"],
            "response": e["response"],
            "judge_label": d["label"],   # ACCEPT / REJECT / NULL
            "judge_raw": d["raw"],       # full judge output incl. <think>…</think>
        })

    verdicts = [d["accepted"] for d in details]
    moved_ids = {id(e) for e, ok in zip(candidates, verdicts) if not ok}
    if not moved_ids:
        print("  judge moved 0 accepted->refused")
        return
    moved = [e for e in candidates if id(e) in moved_ids]
    for cat in (JUDGE_BUCKET, "jailbreak_persuasion",
                "jailbreak_adversarial", "jailbreak_template"):
        behaviors[cat] = [x for x in behaviors[cat] if id(x) not in moved_ids]
    n_pool = 0
    for e in moved:
        if id(e) in harmful_ids:
            behaviors[JUDGE_MOVE_TO].append(e)
            n_pool += 1
    JUDGE_MOVED.extend(moved)
    print(f"  judge moved {len(moved)} accepted->refused "
          f"({n_pool} to {JUDGE_MOVE_TO}, {len(moved) - n_pool} jailbreak-only dropped)")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading inference model …")
    model, tokenizer = load_model()

    # Buckets = every bucket named in DATASET_ROUTING, plus the standard ones the judge
    # and downstream stages expect (so they always exist even if empty).
    behaviors = {b: [] for _n, _p, _k, _nn, rb, ab in DATASET_ROUTING for b in (*rb, *ab)}
    for b in ("refused_harmful", "accepted_harmful", "refused_harmless", "accepted_harmless",
              "jailbreak_template", "jailbreak_adversarial", "jailbreak_persuasion",
              "accepted_harmful_tinst"):
        behaviors.setdefault(b, [])

    # ------------------------------------------------------------------
    # Collect every dataset per config.DATASET_ROUTING (edit routing in config.py, not here).
    # ------------------------------------------------------------------
    for i, (name, path, key, n, refused_buckets, accepted_buckets) in enumerate(DATASET_ROUTING, 1):
        print(f"\n[{i}/{len(DATASET_ROUTING)}] {name} → refused:{refused_buckets or '(drop)'} "
              f"accepted:{accepted_buckets or '(drop)'} …")
        refused, accepted = collect(model, tokenizer, path, key, n=n, dataset_name=name)
        for b in refused_buckets:
            behaviors[b].extend(refused)
        for b in accepted_buckets:
            behaviors[b].extend(accepted)
        if not refused_buckets and refused:
            print(f"  (dropped {len(refused)} '{name}' refusals — not routed to any bucket)")
        if not accepted_buckets and accepted:
            print(f"  (dropped {len(accepted)} '{name}' accepts — not routed to any bucket)")

    # ------------------------------------------------------------------
    # Position-specific accepted-harmful set for the Figure 2 t_inst panel (Appendix B).
    # Re-decode the harmful accepted-sources WITHOUT post-instruction tokens (§3.1 /
    # Table 1): refusal collapses, so strongly-harmful Advbench/JBB prompts get accepted
    # and enter the t_inst red line. Substring-labelled; NOT judged (paper convention —
    # a false-accept still has a genuinely-harmful t_inst vector, so judging is moot here).
    # ------------------------------------------------------------------
    if TINST_ACCEPTED_NO_POSTINST:
        tinst_sources = [(name, path, key, n)
                         for name, path, key, n, _rb, ab in DATASET_ROUTING
                         if "accepted_harmful" in ab
                         and (TINST_ACCEPTED_SOURCES is None or name in TINST_ACCEPTED_SOURCES)]
        print(f"\n[t_inst] no-post-inst accepted-harmful pass over "
              f"{[s[0] for s in tinst_sources]} …")
        for name, path, key, n in tinst_sources:
            _refused, accepted = collect(model, tokenizer, path, key, n=n,
                                         dataset_name=name, no_postinst=True)
            behaviors["accepted_harmful_tinst"].extend(accepted)

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
    # Same rationale for the t_inst set: its natural order is advbench → jbb → sorry, so an
    # unshuffled first-100 (what 01 extracts) could exclude sources. Shuffle for a mix.
    random.shuffle(behaviors["accepted_harmful_tinst"])

    # Lead refused_harmful with the pole sources (advbench+jbb) so the first-N_TRAIN slice
    # 01 averages into μ_harmful / μ_refuse is advbench+jbb only — Sorry-Bench and any
    # judge-moved samples are held out of the centroids (Appendix B). Runs AFTER the judge
    # so judge-moved accepts (appended to refused_harmful) are ordered too.
    if POLE_HARMFUL_SOURCES is not None:
        behaviors["refused_harmful"] = _interleave_by_dataset(
            behaviors["refused_harmful"], POLE_HARMFUL_SOURCES)

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
        label_counts = Counter(e["judge_label"] for e in JUDGE_ALL)   # ACCEPT/REJECT/NULL
        by_ds = Counter(e["dataset"] for e in JUDGE_MOVED)
        report_path = os.path.join(RESULTS_DIR, "judge_report.json")
        with open(report_path, "w") as f:
            json.dump({
                "total_judged": len(JUDGE_ALL),
                "label_counts": dict(label_counts),
                "total_moved": len(JUDGE_MOVED),
                "moved_by_dataset": dict(sorted(by_ds.items())),
                # Every judged sample, with its 3-way label and full judge response
                # (thinking trace + verdict). REJECT ones are the moved ones; NULL =
                # judge emitted no verdict (e.g. ran out of thinking budget).
                "judged": JUDGE_ALL,
            }, f, indent=2, ensure_ascii=False)

        print(f"\n=== Judge report: {len(JUDGE_ALL)} judged "
              f"({dict(label_counts)}), {len(JUDGE_MOVED)} moved accepted->refused ===")
        for ds, c in sorted(by_ds.items()):
            print(f"  moved {ds:28s}: {c}")
        for e in JUDGE_ALL[:5]:
            resp = " ".join(e["response"].split())
            raw = " ".join(e["judge_raw"].split())
            print(f"\n  [{e['judge_label']:6s}] Q: {e['text'][:80]}")
            print(f"      R:     {resp[:120]}{'…' if len(resp) > 120 else ''}")
            print(f"      judge: {raw[:160]}{'…' if len(raw) > 160 else ''}")
        print(f"\nSaved full judge report ({len(JUDGE_ALL)} examples) to {report_path}")


if __name__ == "__main__":
    main()
