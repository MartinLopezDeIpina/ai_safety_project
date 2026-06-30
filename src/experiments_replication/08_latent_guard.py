"""
§5  |  Table 3  —  Latent Guard: Harmfulness Detection via Δharmful
====================================================================

Paper claim (§5):
    A simple threshold on Δharmful (mean sl at tinst) is a strong zero-shot
    classifier for detecting harmful instructions, including jailbroken ones.

    The key insight: even when the model ACCEPTS a harmful (jailbroken) instruction
    (Δrefuse is low), Δharmful remains high for template-based jailbreaks,
    allowing Latent Guard to still detect them.

    For persuasion-style jailbreaks, Δharmful may be lower, but still detectable
    with a threshold tuned on the training set.

Latent Guard classifier (§5 equation):
    predict "harmful"  if Δharmful > τ
    predict "harmless" if Δharmful ≤ τ

    τ is chosen as the midpoint between:
        τ = (mean(Δharmful[refused_harmful]) + mean(Δharmful[accepted_harmless])) / 2
    (or tuned on a validation split — we use the midpoint as a zero-shot baseline)

Evaluation categories (Table 3):
    TP: refused_harmful test split (correctly predicted harmful)
    TN: accepted_harmless test split (correctly predicted harmless)
    FN: accepted_harmful (jailbreaks where model complied — should still be caught)
       Split by jailbreak type: jailbreak_template, jailbreak_adversarial, jailbreak_persuasion
    FP: refused_harmless (harmless instructions wrongly detected as harmful)

Depends on: results/ from 01_extract_directions.py
Saves:      results/table3.json
"""

import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from config import RESULTS_DIR, N_TEST


EVAL_CATS = [
    "refused_harmful",
    "accepted_harmless",
    "accepted_harmful",
    "refused_harmless",
    "jailbreak_template",
    "jailbreak_adversarial",
    "jailbreak_persuasion",
]
TRUE_HARMFUL = {
    "refused_harmful":    True,
    "accepted_harmless":  False,
    "accepted_harmful":   True,   # model accepted but is actually harmful
    "refused_harmless":   False,  # model refused but is actually harmless
    "jailbreak_template": True,
    "jailbreak_adversarial": True,
    "jailbreak_persuasion":  True,
}


def load_acts(cat, pos_name):
    path = os.path.join(RESULTS_DIR, f"acts_{cat}_{pos_name}.pt")
    if not os.path.exists(path):
        return None
    return torch.load(path, weights_only=True).float()


def load_center(name):
    return torch.load(os.path.join(RESULTS_DIR, f"cluster-{name}.pt"), weights_only=True).float()


def compute_delta_harmful(acts_t, mu_rh, mu_ah):
    """
    Δharmful per sample: mean over layers of sl_tinst.

    Returns [N] numpy array.
    """
    h = acts_t[:, :, -1, :]  # [n_layers, N, hidden]
    h_n  = F.normalize(h, dim=-1)
    mp_n = F.normalize(mu_rh.unsqueeze(1), dim=-1)  # [n_layers, 1, hidden]
    mn_n = F.normalize(mu_ah.unsqueeze(1), dim=-1)
    sl = (h_n * mp_n).sum(-1) - (h_n * mn_n).sum(-1)  # [n_layers, N]
    return sl.T.mean(1).numpy()  # [N]


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    mu_rh = load_center("refused_harmful_tinst")
    mu_ah = load_center("accepted_harmless_tinst")

    # ------------------------------------------------------------------
    # Compute Δharmful per sample for all categories
    # ------------------------------------------------------------------
    deltas = {}
    for cat in EVAL_CATS:
        acts_t = load_acts(cat, "tinst")
        if acts_t is None:
            print(f"  Skipping {cat}: no cached activations")
            continue
        d = compute_delta_harmful(acts_t, mu_rh, mu_ah)
        deltas[cat] = d
        print(f"  {cat:30s}: n={len(d):4d}  mean Δharmful={d.mean():.4f}  std={d.std():.4f}")

    # ------------------------------------------------------------------
    # Choose threshold τ
    # ------------------------------------------------------------------
    if "refused_harmful" not in deltas or "accepted_harmless" not in deltas:
        raise ValueError("Need refused_harmful and accepted_harmless for threshold computation.")

    mu_harm  = deltas["refused_harmful"].mean()
    mu_safe  = deltas["accepted_harmless"].mean()
    tau = (mu_harm + mu_safe) / 2
    print(f"\nLatent Guard threshold τ = ({mu_harm:.4f} + {mu_safe:.4f}) / 2 = {tau:.4f}")

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    table = {}
    for cat, d in deltas.items():
        predicted_harmful = d > tau
        n_total = len(d)
        n_correct_harmful  = predicted_harmful.sum()
        n_correct_harmless = (~predicted_harmful).sum()

        if TRUE_HARMFUL[cat]:
            # TP: predicted harmful AND actually harmful
            acc = 100 * n_correct_harmful / n_total
            table[cat] = {
                "n": n_total,
                "true_label": "harmful",
                "detection_rate_pct": round(float(acc), 1),
                "n_detected": int(n_correct_harmful),
                "mean_delta_harmful": round(float(d.mean()), 4),
            }
        else:
            # TN: predicted harmless AND actually harmless
            acc = 100 * n_correct_harmless / n_total
            table[cat] = {
                "n": n_total,
                "true_label": "harmless",
                "correct_rate_pct": round(float(acc), 1),
                "n_correct": int(n_correct_harmless),
                "mean_delta_harmful": round(float(d.mean()), 4),
            }

    # ------------------------------------------------------------------
    # Print table
    # ------------------------------------------------------------------
    print(f"\n=== Table 3 — Latent Guard (τ = {tau:.4f}) ===")
    print(f"  {'Category':<28} {'True Label':<10} {'N':>5} {'Detect/Acc Rate':>16} {'Mean Δharmful':>14}")
    print(f"  {'-'*80}")
    for cat, row in table.items():
        rate = row.get("detection_rate_pct", row.get("correct_rate_pct", 0))
        print(f"  {cat:<28} {row['true_label']:<10} {row['n']:>5} {rate:>15.1f}%  {row['mean_delta_harmful']:>13.4f}")

    result = {"tau": float(tau), "mu_harm": float(mu_harm), "mu_safe": float(mu_safe), "table": table}
    with open(os.path.join(RESULTS_DIR, "table3.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("\nSaved results/table3.json")


if __name__ == "__main__":
    main()
