"""
§3.4  |  Figure 4  —  Per-Layer Steering Induces Refusal on Harmless Inputs
============================================================================

Paper claim (§3.4):
    Applying the harmfulness direction (+dir_hf) at layer L causes the model to refuse
    harmless instructions. Applying the refusal direction (+dir_refuse) does the same.
    Crucially, they have different layer profiles — proving the two concepts are encoded
    at different abstraction levels.

What this script does (x-axis = layer L, NOT coefficient):
    For each layer L ∈ {0, …, n_layers-1}:
      1. Use intervention.complete_with_intervention() (pre-hook, paper's method) to apply
         +STEERING_COEFF × unit(dir_hf[L]) at only layer L on harmless inputs.
      2. Score with eval.easy_eval() — '0' = refused.
      3. Record refusal rate → refusal_rate_hf[L].
      4. Repeat with dir_refuse[L] → refusal_rate_refuse[L].

    Directions are normalized to unit norm so the coefficient is comparable across
    dir_hf and dir_refuse (which have different raw norms).

Depends on: results/ from 01_extract_directions.py + results/behaviors.json
Saves:      results/figure4-data.json, results/figure4.png
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from config import RESULTS_DIR, STEERING_COEFF, N_TEST, SRC_DIR, MODEL_TYPE
from model_utils import load_model, tokenize_fn, is_refusal
from transformers import GenerationConfig

sys.path.insert(0, SRC_DIR)
import intervention as iv
import eval as eval_mod


MAX_NEW_TOKENS = 40


def load_behaviors_cat(cat, n=None):
    path = os.path.join(RESULTS_DIR, "behaviors.json")
    with open(path) as f:
        behaviors = json.load(f)
    items = behaviors.get(cat, [])
    dicts = [{"instruction": it["text"]} for it in items]
    if n is not None:
        dicts = dicts[:n]
    return dicts


def baseline_refusal_rate(model, tokenizer, data_dicts):
    """Measure refusal rate with no steering applied."""
    gen_cfg = GenerationConfig(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    n_refused = 0
    for d in data_dicts:
        inputs = tokenize_fn(tokenizer, [d])
        with torch.no_grad():
            out = model.generate(
                input_ids=inputs.input_ids.to(model.device),
                attention_mask=inputs.attention_mask.to(model.device),
                generation_config=gen_cfg,
            )
        gen = out[0, inputs.input_ids.shape[1]:]
        resp = tokenizer.decode(gen, skip_special_tokens=True)
        if is_refusal(resp):
            n_refused += 1
    return 100 * n_refused / len(data_dicts)


def sweep_layers(model, tokenizer, data_dicts, direction_norm, coeff, desc=""):
    """
    For each layer L, call complete_with_intervention() at that layer only.
    Returns per-layer refusal rates (%).
    direction_norm: [1, n_layers, hidden] — already unit-normalized.
    """
    n_layers = len(model.model.layers)

    args = {
        'add_coef_intervene': coeff,
        'max_token_generate': MAX_NEW_TOKENS,
        'arg_key_prompt': 'instruction',
        'intervene_all': True,
        'intervene_context_only': False,
        'inversion_prompt_idx': 0,
        'record_probs': False,
    }

    def tok_fn(batch):
        return tokenize_fn(tokenizer, batch)

    # Drop output_attentions from generate() to avoid OOM on 7B model
    # (intervention.py hardcodes output_attentions=True which uses ~2.4 GB extra)
    _orig_generate = model.generate
    model.generate = lambda *a, **kw: _orig_generate(
        *a, **{k: v for k, v in kw.items() if k != 'output_attentions'}
    )

    rates = []
    try:
        for l_idx in range(n_layers):
            completions = iv.complete_with_intervention(
                model, tokenizer, data_dicts, tok_fn,
                intervene_layers=[l_idx], batch_size=1,
                intervention_vector_ori=direction_norm, args=args,
            )
            scores = eval_mod.easy_eval(completions, tag='response', mode='refusal')
            n_refused = sum(1 for s in scores if s == '0')
            rate = 100 * n_refused / len(scores)
            rates.append(rate)
            print(f"    {desc} layer {l_idx:2d}: {rate:.1f}%")
    finally:
        model.generate = _orig_generate  # always restore

    return rates


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Set intervention globals before calling complete_with_intervention.
    # MODEL must match the model family: intervention.py uses it to size its per-layer cache
    # (28 for qwen, 32 otherwise), so hardcoding 'qwen' breaks Llama-3-8B (32 layers).
    iv.MODEL = MODEL_TYPE
    iv.DECODING_STEP = -1  # intervene on all decode steps

    print("Loading model …")
    model, tokenizer = load_model()

    print("Loading directions …")
    dir_hf     = torch.load(os.path.join(RESULTS_DIR, "dir-hf.pt"),     weights_only=True).float()
    dir_refuse = torch.load(os.path.join(RESULTS_DIR, "dir-refuse.pt"), weights_only=True).float()

    # Normalize to unit norm so coefficient is comparable across the two directions
    dir_hf_norm  = F.normalize(dir_hf,     dim=-1).unsqueeze(0)  # [1, n_layers, hidden]
    dir_ref_norm = F.normalize(dir_refuse, dim=-1).unsqueeze(0)  # [1, n_layers, hidden]

    print("Loading harmless instructions …")
    harmless = load_behaviors_cat("accepted_harmless", n=N_TEST)
    if len(harmless) == 0:
        raise ValueError("No accepted_harmless samples — run 00_collect_behaviors.py first.")

    print(f"\nBaseline (no steering) on {len(harmless)} harmless prompts …")
    baseline = baseline_refusal_rate(model, tokenizer, harmless)
    print(f"  Baseline refusal rate: {baseline:.1f}%")

    print(f"\nSweeping layers with +dir_hf (coeff={STEERING_COEFF}) …")
    rates_hf = sweep_layers(model, tokenizer, harmless, dir_hf_norm, STEERING_COEFF, "+hf")

    print(f"\nSweeping layers with +dir_refuse (coeff={STEERING_COEFF}) …")
    rates_refuse = sweep_layers(model, tokenizer, harmless, dir_ref_norm, STEERING_COEFF, "+refuse")

    # ------------------------------------------------------------------
    # Save data
    # ------------------------------------------------------------------
    n_layers = len(model.model.layers)
    result = {
        "n_layers": n_layers,
        "steering_coeff": STEERING_COEFF,
        "n_samples": len(harmless),
        "baseline_pct": baseline,
        "rates_hf_pct": rates_hf,
        "rates_refuse_pct": rates_refuse,
    }
    with open(os.path.join(RESULTS_DIR, "figure4-data.json"), "w") as f:
        json.dump(result, f, indent=2)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    layers = list(range(n_layers))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(layers, rates_hf,     label="Harmfulness direction (+dir_hf)", color="#e41a1c", linewidth=2)
    ax.plot(layers, rates_refuse, label="Refusal direction (+dir_refuse)",  color="#377eb8", linewidth=2, linestyle="--")
    ax.axhline(baseline, color="gray", linewidth=1, linestyle=":", label=f"Baseline ({baseline:.1f}%)")
    ax.set_xlabel("Layer L (hook applied at this layer only)")
    ax.set_ylabel("Refusal rate (%)")
    ax.set_title(f"Figure 4  |  Per-Layer Steering on Harmless Inputs\n"
                 f"(coeff = +{STEERING_COEFF}, N = {len(harmless)})")
    ax.legend(fontsize=9)
    ax.set_ylim(-5, 105)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "figure4.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
