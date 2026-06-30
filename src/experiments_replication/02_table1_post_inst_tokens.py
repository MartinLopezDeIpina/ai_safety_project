"""
§3.1  |  Table 1  —  Removing Post-Instruction Tokens Weakens Refusal
======================================================================

Paper claim (§3.1):
    LLMs refuse harmful instructions at tpost-inst (after the post-instruction
    special tokens like <|im_end|>\\n<|im_start|>assistant\\n) but will
    sometimes accept them when prompted from tinst alone (without those tokens).
    Removing the post-instruction tokens reduces the refusal rate.

What this script measures:
    For the same set of harmful prompts, compares the refusal rate when:
    1. FULL template:     <|im_start|>user\\n{instruction}<|im_end|>\\n<|im_start|>assistant\\n
    2. TRUNCATED template:<|im_start|>user\\n{instruction}
    (The truncated form omits the post-instruction tokens, forcing the model to
    generate from right after the user's message ends.)

Outputs (in results/):
    table1.json   — {full_rate, truncated_rate, samples}
"""

import json
import os
import sys

import torch
from transformers import GenerationConfig

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from config import HARMFUL_DATA, RESULTS_DIR, SRC_DIR
from model_utils import (
    load_model, load_data, tokenize_fn,
    format_prompt, format_prompt_no_postinst, is_refusal,
)

sys.path.insert(0, SRC_DIR)

N_SAMPLES = 50
MAX_NEW_TOKENS = 100


def generate_with_template(model, tokenizer, data_dicts, use_full_template):
    """
    Generate responses using either the full or truncated Qwen2 chat template.
    Returns list of response strings.
    """
    gen_cfg = GenerationConfig(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    fmt_fn = format_prompt if use_full_template else format_prompt_no_postinst
    responses = []
    for d in data_dicts:
        prompt = fmt_fn(d)
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        with torch.no_grad():
            out = model.generate(
                input_ids=inputs.input_ids.to(model.device),
                attention_mask=inputs.attention_mask.to(model.device),
                generation_config=gen_cfg,
            )
        gen = out[0, inputs.input_ids.shape[1]:]
        responses.append(tokenizer.decode(gen, skip_special_tokens=True))
    return responses


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading model …")
    model, tokenizer = load_model()

    harmful = load_data(HARMFUL_DATA, key="bad_q", n=N_SAMPLES)

    print(f"\n--- Full template (with post-instruction tokens) ---")
    full_responses = generate_with_template(model, tokenizer, harmful, use_full_template=True)
    n_refuse_full = sum(1 for r in full_responses if is_refusal(r))
    rate_full = 100 * n_refuse_full / N_SAMPLES
    print(f"  Refusal rate: {n_refuse_full}/{N_SAMPLES} = {rate_full:.1f}%")

    print(f"\n--- Truncated template (WITHOUT post-instruction tokens) ---")
    trunc_responses = generate_with_template(model, tokenizer, harmful, use_full_template=False)
    n_refuse_trunc = sum(1 for r in trunc_responses if is_refusal(r))
    rate_trunc = 100 * n_refuse_trunc / N_SAMPLES
    print(f"  Refusal rate: {n_refuse_trunc}/{N_SAMPLES} = {rate_trunc:.1f}%")

    print(f"\n=== Table 1 ===")
    print(f"  {'Template':<40} {'Refusal Rate':>15}")
    print(f"  {'-'*56}")
    print(f"  {'Full (with post-inst tokens)':<40} {rate_full:>14.1f}%")
    print(f"  {'Truncated (without post-inst tokens)':<40} {rate_trunc:>14.1f}%")
    print(f"\n  Paper prediction: full > truncated (removing post-inst tokens weakens refusal)")

    result = {
        "n_samples": N_SAMPLES,
        "full_template_refusal_rate_pct": round(rate_full,  1),
        "truncated_template_refusal_rate_pct": round(rate_trunc, 1),
        "difference_pct": round(rate_full - rate_trunc, 1),
        "samples": [
            {"text": d.get("bad_q",""), "full": r_f, "truncated": r_t,
             "full_refused": is_refusal(r_f), "trunc_refused": is_refusal(r_t)}
            for d, r_f, r_t in zip(harmful, full_responses, trunc_responses)
        ]
    }
    with open(os.path.join(RESULTS_DIR, "table1.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("\nSaved to results/table1.json")


if __name__ == "__main__":
    main()
