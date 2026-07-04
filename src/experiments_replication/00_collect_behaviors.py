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

import json
import os
import sys

import torch
from transformers import GenerationConfig

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from config import (
    HARMFUL_DATA, HARMLESS_DATA, XSTEST_DATA, SORRY_DATA,
    GPTZFUZZER_DATA, HUMAN_SEED_DATA, RESULTS_DIR,
    N_BEHAVIOR, XSTEST_N, N_JAILBREAK,
)
from model_utils import load_model, load_data, tokenize_fn, is_refusal


MAX_NEW_TOKENS = 80


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


def collect(model, tokenizer, path, key, n, true_label, dataset_name):
    """
    Run inference on `n` examples from `path` (key=`key`).
    Returns (refused_list, accepted_list) — each a list of {text, response, dataset} dicts.
    """
    data = load_data(path, key=key, n=n)
    responses = run_inference(model, tokenizer, data)
    refused, accepted = [], []
    for d, resp in zip(data, responses):
        text = d.get(key, d.get("instruction", d.get("prompt", "")))
        entry = {"text": text, "response": resp, "dataset": dataset_name}
        if is_refusal(resp):
            refused.append(entry)
        else:
            accepted.append(entry)
    print(f"  {dataset_name}: {len(refused)} refused / {len(accepted)} accepted "
          f"out of {len(data)}")
    return refused, accepted


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading model …")
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
    # advbench — harmful prompts
    # ------------------------------------------------------------------
    print("\n[1/6] advbench (harmful) …")
    refused, accepted = collect(model, tokenizer, HARMFUL_DATA, "bad_q",
                                n=N_BEHAVIOR, true_label="harmful", dataset_name="advbench")
    behaviors["refused_harmful"].extend(refused)
    behaviors["accepted_harmful"].extend(accepted)   # rare: model complied with harmful

    # ------------------------------------------------------------------
    # alpaca — harmless prompts
    # ------------------------------------------------------------------
    print("[2/6] alpaca (harmless) …")
    refused, accepted = collect(model, tokenizer, HARMLESS_DATA, "instruction",
                                n=N_BEHAVIOR, true_label="harmless", dataset_name="alpaca")
    behaviors["refused_harmless"].extend(refused)   # rare: over-refusal
    behaviors["accepted_harmless"].extend(accepted)

    # ------------------------------------------------------------------
    # xstest — harmless prompts that may trigger over-refusal
    # ------------------------------------------------------------------
    print("[3/6] xstest (harmless, over-refusal candidates) …")
    refused, accepted = collect(model, tokenizer, XSTEST_DATA, "bad_q",
                                n=XSTEST_N, true_label="harmless", dataset_name="xstest")
    behaviors["refused_harmless"].extend(refused)
    behaviors["accepted_harmless"].extend(accepted)

    # ------------------------------------------------------------------
    # sorry-badq — persuasion-style harmful prompts
    # ------------------------------------------------------------------
    print("[4/6] sorry-badq (persuasion jailbreak) …")
    refused, accepted = collect(model, tokenizer, SORRY_DATA, "bad_q",
                                n=N_BEHAVIOR, true_label="harmful", dataset_name="sorry-badq")
    behaviors["refused_harmful"].extend(refused)
    behaviors["accepted_harmful"].extend(accepted)  # jailbreak succeeded
    behaviors["jailbreak_persuasion"].extend(accepted)

    # ------------------------------------------------------------------
    # human-seed — adversarial prompts
    # Template/adversarial jailbreaks go ONLY into their jailbreak category,
    # NOT into accepted_harmful: their text is disguised so they look harmless
    # at tinst, which would corrupt the Figure 2/3 clustering analysis.
    # ------------------------------------------------------------------
    print("[5/6] human-seed-50 (adversarial jailbreak) …")
    refused, accepted = collect(model, tokenizer, HUMAN_SEED_DATA, "bad_q",
                                n=N_JAILBREAK, true_label="harmful", dataset_name="human-seed")
    behaviors["refused_harmful"].extend(refused)
    behaviors["jailbreak_adversarial"].extend(accepted)

    # ------------------------------------------------------------------
    # GPTFuzzer — template-based jailbreak prompts
    # Same reasoning: template jailbreaks disguise harmful intent as benign
    # text, so at tinst they look harmless. Figure 6 only.
    # ------------------------------------------------------------------
    print("[6/6] GPTFuzzer-50 (template jailbreak) …")
    refused, accepted = collect(model, tokenizer, GPTZFUZZER_DATA, "bad_q",
                                n=N_JAILBREAK, true_label="harmful", dataset_name="gptzfuzzer")
    behaviors["refused_harmful"].extend(refused)
    behaviors["jailbreak_template"].extend(accepted)

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


if __name__ == "__main__":
    main()
