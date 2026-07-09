"""
§3.4 extension  |  Per-layer steering WITH saved responses (both directions)
============================================================================

Like 05_figure4 (steer harmless prompts UP the harmfulness / refusal directions and record
the per-layer refusal rate), but:
  (1) ALSO does the inverse — steer HARMFUL prompts DOWN the harmfulness / refusal directions
      (coeff = −STEERING_COEFF), which should REDUCE the refusal rate (the model accepts them).
  (2) saves the actual generated RESPONSES per layer/direction/prompt to JSON (behaviors.json
      style), not just the aggregate rate.

Conditions (each per layer, both dir_hf and dir_refuse, directions unit-normalized):
  harmless_up   : accepted_harmless prompts, +STEERING_COEFF   (baseline refusal ~0% → rises)
  harmful_down  : refused_harmful  prompts, −STEERING_COEFF    (baseline refusal ~100% → falls)

Inputs (from RESULTS_DIR): dir-hf.pt, dir-refuse.pt, behaviors.json.
Outputs (to RESULTS_DIR):
  steering_responses.json  — {meta, harmless_up:{baseline_refusal_rate, dir_hf:{rates,responses}, dir_refuse:{...}}, harmful_down:{...}}
  steering_responses.png   — two subplots (up-on-harmless, down-on-harmful)
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import GenerationConfig

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from config import RESULTS_DIR, STEERING_COEFF, N_TEST, SRC_DIR, MODEL_TYPE, MODEL_NAME
from model_utils import load_model, tokenize_fn, is_refusal

sys.path.insert(0, SRC_DIR)
import intervention as iv

# This sweeps every layer × 2 directions × 2 conditions × N_QUERIES generations, so keep the
# query count / length modest to fit a Modal preemption window (override via env if wanted).
N_QUERIES      = int(os.environ.get("STEER_N", "12"))
MAX_NEW_TOKENS = int(os.environ.get("STEER_TOKENS", "48"))
# Steering uses the RAW diff-in-means direction (NOT unit-normalized): its per-layer norm is
# ~0.5× the residual-stream norm for both Qwen and Llama, so a small coeff gives a comparable,
# well-scaled perturbation on either model. (Unit-norm × 20 was Qwen-calibrated and destroys
# Llama, whose residual norms are ~6× smaller.) STEER_COEFF ~2-4 ≈ 1-2× the residual norm.
COEFF        = float(os.environ.get("STEER_COEFF", "4"))
# The harmfulness direction is far more disruptive than the refusal direction (on Llama it
# degenerates into gibberish well before eliciting a clean behavioural flip), so allow a
# SEPARATE coefficient per direction. Default each to COEFF.
COEFF_HF     = float(os.environ.get("STEER_COEFF_HF",  str(COEFF)))
COEFF_REF    = float(os.environ.get("STEER_COEFF_REF", str(COEFF)))
# Optional comma-separated layer subset for a cheap coefficient probe (default = all layers).
_LAYERS_ENV  = os.environ.get("STEER_LAYERS", "").strip()


def load_cat(cat, n):
    with open(os.path.join(RESULTS_DIR, "behaviors.json")) as f:
        b = json.load(f)
    items = b.get(cat, [])[:n]
    return [{"instruction": it["text"]} for it in items]


def baseline(model, tokenizer, data):
    """No-steering responses + refusal rate."""
    cfg = GenerationConfig(max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                           pad_token_id=tokenizer.pad_token_id)
    out = []
    for d in data:
        inp = tokenize_fn(tokenizer, [d])
        with torch.no_grad():
            g = model.generate(input_ids=inp.input_ids.to(model.device),
                               attention_mask=inp.attention_mask.to(model.device),
                               generation_config=cfg)
        resp = tokenizer.decode(g[0, inp.input_ids.shape[1]:], skip_special_tokens=True).strip()
        out.append({"query": d["instruction"], "response": resp, "refused": bool(is_refusal(resp))})
    rate = 100 * sum(r["refused"] for r in out) / len(out)
    return rate, out


def sweep(model, tokenizer, data, dir_vec, coeff, desc, layers):
    """Per-layer steering over `layers`; returns (rates[list aligned to layers],
    responses{layer:[{query,response,refused}]})."""
    args = {'add_coef_intervene': coeff, 'max_token_generate': MAX_NEW_TOKENS,
            'arg_key_prompt': 'instruction', 'intervene_all': True,
            'intervene_context_only': False, 'inversion_prompt_idx': 0, 'record_probs': False}

    def tok_fn(batch):
        return tokenize_fn(tokenizer, batch)

    # Drop output_attentions to avoid OOM (intervention.py hardcodes it True).
    _orig = model.generate
    model.generate = lambda *a, **kw: _orig(*a, **{k: v for k, v in kw.items() if k != 'output_attentions'})

    rates, responses = [], {}
    try:
        for L in layers:
            comps = iv.complete_with_intervention(
                model, tokenizer, data, tok_fn, intervene_layers=[L], batch_size=1,
                intervention_vector_ori=dir_vec, args=args)
            rows = []
            for d, c in zip(data, comps):
                resp = c["response"]
                rows.append({"query": d["instruction"], "response": resp,
                             "refused": bool(is_refusal(resp))})
            rate = 100 * sum(r["refused"] for r in rows) / len(rows)
            rates.append(rate)
            responses[str(L)] = rows
            print(f"    {desc} layer {L:2d}: refusal {rate:.1f}%")
    finally:
        model.generate = _orig
    return rates, responses


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    iv.MODEL = MODEL_TYPE
    iv.DECODING_STEP = -1

    print("Loading model …", MODEL_NAME)
    model, tokenizer = load_model()

    # RAW diff-in-means directions (naturally per-layer/model scaled) — do NOT unit-normalize.
    dir_hf     = torch.load(os.path.join(RESULTS_DIR, "dir-hf.pt"),     weights_only=True).float()
    dir_refuse = torch.load(os.path.join(RESULTS_DIR, "dir-refuse.pt"), weights_only=True).float()
    dir_hf_n  = dir_hf.unsqueeze(0)
    dir_ref_n = dir_refuse.unsqueeze(0)
    # Combined direction = both at once, with each direction's own coefficient baked in, so it
    # is applied with coeff ±1 (up = +hf +ref, down = −hf −ref).
    dir_both_n = (COEFF_HF * dir_hf + COEFF_REF * dir_refuse).unsqueeze(0)

    n_layers = len(model.model.layers)
    layers = [int(x) for x in _LAYERS_ENV.split(",")] if _LAYERS_ENV else list(range(n_layers))

    harmless = load_cat("accepted_harmless", N_QUERIES)
    harmful  = load_cat("refused_harmful",   N_QUERIES)
    print(f"harmless={len(harmless)} harmful={len(harmful)}  RAW dir × coeff_hf=±{COEFF_HF} "
          f"coeff_ref=±{COEFF_REF}  layers={layers}")

    result = {"meta": {"model": MODEL_NAME, "model_type": MODEL_TYPE,
                       "coeff_hf": COEFF_HF, "coeff_refuse": COEFF_REF,
                       "direction": "raw_diff_in_means", "layers": layers,
                       "n_queries": {"harmless": len(harmless), "harmful": len(harmful)},
                       "max_new_tokens": MAX_NEW_TOKENS, "n_layers": n_layers}}

    # (1) harmless + steer UP  (original Figure 4)
    print("\n=== harmless_up (+coeff) ===")
    b_hl, b_hl_resp = baseline(model, tokenizer, harmless)
    print(f"  baseline refusal {b_hl:.1f}%")
    up_hf, up_hf_r   = sweep(model, tokenizer, harmless, dir_hf_n,   +COEFF_HF,  "+hf ", layers)
    up_rf, up_rf_r   = sweep(model, tokenizer, harmless, dir_ref_n,  +COEFF_REF, "+ref", layers)
    up_bo, up_bo_r   = sweep(model, tokenizer, harmless, dir_both_n, +1.0,       "+both", layers)
    result["harmless_up"] = {
        "baseline_refusal_rate": b_hl, "baseline_responses": b_hl_resp,
        "dir_hf":     {"rates": up_hf, "responses": up_hf_r},
        "dir_refuse": {"rates": up_rf, "responses": up_rf_r},
        "dir_both":   {"rates": up_bo, "responses": up_bo_r},   # +hf and +refuse together
    }

    # (2) harmful + steer DOWN  (inverse)
    print("\n=== harmful_down (−coeff) ===")
    b_hf, b_hf_resp = baseline(model, tokenizer, harmful)
    print(f"  baseline refusal {b_hf:.1f}%")
    dn_hf, dn_hf_r   = sweep(model, tokenizer, harmful, dir_hf_n,   -COEFF_HF,  "-hf ", layers)
    dn_rf, dn_rf_r   = sweep(model, tokenizer, harmful, dir_ref_n,  -COEFF_REF, "-ref", layers)
    dn_bo, dn_bo_r   = sweep(model, tokenizer, harmful, dir_both_n, -1.0,       "-both", layers)
    result["harmful_down"] = {
        "baseline_refusal_rate": b_hf, "baseline_responses": b_hf_resp,
        "dir_hf":     {"rates": dn_hf, "responses": dn_hf_r},
        "dir_refuse": {"rates": dn_rf, "responses": dn_rf_r},
        "dir_both":   {"rates": dn_bo, "responses": dn_bo_r},   # −hf and −refuse together
    }

    with open(os.path.join(RESULTS_DIR, "steering_responses.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("\nSaved steering_responses.json")

    # Plot (x-axis = the swept layers)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    ax1.plot(layers, up_hf, color="#e41a1c", lw=2, label="+harmfulness dir")
    ax1.plot(layers, up_rf, color="#377eb8", lw=2, ls="--", label="+refusal dir")
    ax1.plot(layers, up_bo, color="#4daf4a", lw=2, ls="-.", label="+both dirs")
    ax1.axhline(b_hl, color="gray", ls=":", lw=1, label=f"baseline ({b_hl:.0f}%)")
    ax1.set_title("Harmless prompts + steer UP\n(refusal rate should RISE)")
    ax1.set_xlabel("Layer"); ax1.set_ylabel("Refusal rate (%)"); ax1.set_ylim(-5, 105); ax1.legend(fontsize=8)
    ax2.plot(layers, dn_hf, color="#e41a1c", lw=2, label="−harmfulness dir")
    ax2.plot(layers, dn_rf, color="#377eb8", lw=2, ls="--", label="−refusal dir")
    ax2.plot(layers, dn_bo, color="#4daf4a", lw=2, ls="-.", label="−both dirs")
    ax2.axhline(b_hf, color="gray", ls=":", lw=1, label=f"baseline ({b_hf:.0f}%)")
    ax2.set_title("Harmful prompts + steer DOWN\n(refusal rate should FALL)")
    ax2.set_xlabel("Layer"); ax2.set_ylim(-5, 105); ax2.legend(fontsize=8)
    fig.suptitle(f"Per-layer steering (both directions) — {MODEL_NAME}", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "steering_responses.png"), dpi=150, bbox_inches="tight")
    print("Saved steering_responses.png")


if __name__ == "__main__":
    main()
