"""
§4  |  Figure 6  —  Jailbreak Analysis Scatter (Δharmful vs Δrefuse)
=====================================================================

Paper claim (§4):
    Template-based jailbreaks (GPTFuzzer) suppress the REFUSAL signal (low Δrefuse)
    but do NOT change the harmfulness encoding (high Δharmful).

    Persuasion-style jailbreaks (sorry-badq) additionally FLIP the harmfulness
    encoding itself (lower Δharmful) — a deeper attack.

    Adversarial human-written jailbreaks (human-seed) fall in between.

    Normal accepted_harmful (the few that bypassed without jailbreaking) also appear.

What this script does:
    1. Loads cluster centers from results/ (from 01_extract_directions.py).
    2. Extracts hidden states for each jailbreak category at tinst and tpostinst.
       (These are stored in behaviors.json under jailbreak_template,
        jailbreak_adversarial, jailbreak_persuasion.)
    3. Computes Δharmful and Δrefuse per sample (same formula as Figure 3).
    4. Plots a scatter overlaid on the Figure 3 background.

Depends on: results/ from 01_extract_directions.py + results/behaviors.json
Saves:      results/figure6.png, results/figure6-data.json
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from config import RESULTS_DIR, N_JAILBREAK
from model_utils import load_model, extract_hidden_states
from config import POS_TINST, POS_TPOSTINST


JB_CATS = ["jailbreak_template", "jailbreak_adversarial", "jailbreak_persuasion"]
JB_LABELS = {
    "jailbreak_template":    "Template (GPTFuzzer)",
    "jailbreak_adversarial": "Adversarial (human-seed)",
    "jailbreak_persuasion":  "Persuasion (sorry-badq)",
}
JB_COLORS = {
    "jailbreak_template":    "#984ea3",
    "jailbreak_adversarial": "#a65628",
    "jailbreak_persuasion":  "#f781bf",
}
JB_MARKERS = {
    "jailbreak_template":    "*",
    "jailbreak_adversarial": "P",
    "jailbreak_persuasion":  "X",
}

# Background categories from Figure 3 (small, lighter)
BG_CATS = ["refused_harmful", "accepted_harmless"]
BG_LABELS = {"refused_harmful": "Refused Harmful", "accepted_harmless": "Accepted Harmless"}
BG_COLORS = {"refused_harmful": "#e41a1c", "accepted_harmless": "#4daf4a"}


def load_behaviors_cat(cat, n=None):
    path = os.path.join(RESULTS_DIR, "behaviors.json")
    with open(path) as f:
        behaviors = json.load(f)
    items = behaviors.get(cat, [])
    dicts = [{"instruction": it["text"]} for it in items]
    if n is not None:
        dicts = dicts[:n]
    return dicts


def load_acts(cat, pos_name):
    path = os.path.join(RESULTS_DIR, f"acts_{cat}_{pos_name}.pt")
    if os.path.exists(path):
        return torch.load(path, weights_only=True).float()
    return None


def load_center(name):
    path = os.path.join(RESULTS_DIR, f"cluster-{name}.pt")
    return torch.load(path, weights_only=True).float()


def compute_deltas(acts_t, acts_p, mu_rh_tinst, mu_ah_tinst, mu_ref_tpost, mu_acc_tpost):
    """
    Returns (delta_harmful [N], delta_refuse [N]).
    """
    def _sl_samples(acts, mu_pos, mu_neg):
        h = acts[:, :, -1, :]
        h_n = F.normalize(h, dim=-1)
        mp = F.normalize(mu_pos.unsqueeze(1), dim=-1)
        mn = F.normalize(mu_neg.unsqueeze(1), dim=-1)
        sl = ((h_n * mp).sum(-1) - (h_n * mn).sum(-1))  # [n_layers, N]
        return sl.T  # [N, n_layers]

    sl_t = _sl_samples(acts_t, mu_rh_tinst, mu_ah_tinst)
    sl_p = _sl_samples(acts_p, mu_ref_tpost, mu_acc_tpost)
    return sl_t.mean(1).numpy(), sl_p.mean(1).numpy()


def extract_jb_acts(model, tokenizer, cat_name, data_dicts, pos_name, positions):
    save_path = os.path.join(RESULTS_DIR, f"acts_{cat_name}_{pos_name}.pt")
    if os.path.exists(save_path):
        print(f"  Loaded cached {save_path}")
        return torch.load(save_path, weights_only=True).float()
    print(f"  Extracting {cat_name} at {pos_name} ({len(data_dicts)} samples) …")
    _, all_acts = extract_hidden_states(model, tokenizer, data_dicts, positions=positions)
    # extract_hidden.py uses .squeeze() which removes N=1 dim; restore it so shape
    # is always [n_layers, N, pos_window, hidden] even for single-sample categories.
    if all_acts.dim() == 3:
        all_acts = all_acts.unsqueeze(1)
    torch.save(all_acts, save_path)
    return all_acts.float()


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    mu_rh_tinst  = load_center("refused_harmful_tinst")
    mu_ah_tinst  = load_center("accepted_harmless_tinst")
    mu_ref_tpost = load_center("refused_tpostinst")
    mu_acc_tpost = load_center("accepted_tpostinst")

    # ------------------------------------------------------------------
    # Check if we need to run the model for jailbreak activations
    # ------------------------------------------------------------------
    needs_model = any(
        not os.path.exists(os.path.join(RESULTS_DIR, f"acts_{cat}_tinst.pt"))
        for cat in JB_CATS
    )

    model = tokenizer = None
    if needs_model:
        print("Loading model for jailbreak extraction …")
        from model_utils import load_model as _load
        model, tokenizer = _load()

    # ------------------------------------------------------------------
    # Jailbreak categories
    # ------------------------------------------------------------------
    scatter_data = {}

    for cat in JB_CATS:
        dicts = load_behaviors_cat(cat, n=N_JAILBREAK)
        if len(dicts) < 3:
            print(f"  Skipping {cat}: only {len(dicts)} samples (need ≥3 for scatter)")
            continue

        if model is not None:
            acts_t = extract_jb_acts(model, tokenizer, cat, dicts, "tinst",     POS_TINST)
            acts_p = extract_jb_acts(model, tokenizer, cat, dicts, "tpostinst", POS_TPOSTINST)
        else:
            acts_t = load_acts(cat, "tinst")
            acts_p = load_acts(cat, "tpostinst")
            if acts_t is None or acts_p is None:
                print(f"  Skipping {cat}: cached activations not found")
                continue

        dh, dr = compute_deltas(acts_t, acts_p, mu_rh_tinst, mu_ah_tinst, mu_ref_tpost, mu_acc_tpost)
        scatter_data[cat] = {"delta_harmful": dh.tolist(), "delta_refuse": dr.tolist()}
        print(f"  {cat:30s}: Δharmful={dh.mean():.3f}  Δrefuse={dr.mean():.3f}")

    # ------------------------------------------------------------------
    # Background scatter (Figure 3 data)
    # ------------------------------------------------------------------
    bg_data = {}
    for cat in BG_CATS:
        acts_t = load_acts(cat, "tinst")
        acts_p = load_acts(cat, "tpostinst")
        if acts_t is None or acts_p is None:
            continue
        dh, dr = compute_deltas(acts_t, acts_p, mu_rh_tinst, mu_ah_tinst, mu_ref_tpost, mu_acc_tpost)
        bg_data[cat] = {"delta_harmful": dh.tolist(), "delta_refuse": dr.tolist()}

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))

    for cat, data in bg_data.items():
        ax.scatter(data["delta_harmful"], data["delta_refuse"],
                   label=BG_LABELS[cat], color=BG_COLORS[cat],
                   alpha=0.2, s=20, marker="o", edgecolors="none")

    for cat, data in scatter_data.items():
        ax.scatter(data["delta_harmful"], data["delta_refuse"],
                   label=JB_LABELS[cat], color=JB_COLORS[cat],
                   marker=JB_MARKERS[cat], alpha=0.8, s=80, edgecolors="black", linewidths=0.5)

    ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.7, linestyle="--")
    ax.set_xlabel("Δharmful  (mean sl at tinst)")
    ax.set_ylabel("Δrefuse   (mean sl at tpost-inst)")
    ax.set_title("Figure 6  |  Jailbreak Types in Δharmful × Δrefuse Space\n"
                 "Template jailbreaks suppress refusal; persuasion jailbreaks flip harmfulness")
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout()

    out = os.path.join(RESULTS_DIR, "figure6.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved {out}")

    combined = {"jailbreaks": scatter_data, "background": bg_data}
    with open(os.path.join(RESULTS_DIR, "figure6-data.json"), "w") as f:
        json.dump(combined, f, indent=2)
    print("Saved figure6-data.json")


if __name__ == "__main__":
    main()
