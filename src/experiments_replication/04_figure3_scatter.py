"""
§3.3  |  Figure 3  —  Scatter of Δharmful vs Δrefuse
======================================================

Paper claim (§3.3):
    Harmfulness and refusal are encoded SEPARATELY.
    The scatter of (Δharmful, Δrefuse) reveals four distinct quadrants:
      Q1 (high harm, high refuse) = refused_harmful    — well-aligned
      Q2 (low harm, low refuse)   = accepted_harmless  — well-aligned
      Q3 (high harm, low refuse)  = accepted_harmful   — safety failure: model did NOT refuse
      Q4 (low harm, high refuse)  = refused_harmless   — over-refusal: model over-refused

    The two dimensions are NOT perfectly correlated, proving they are separable.

Quantities:
    Δharmful = mean_over_layers(sl_tinst(hˡ))
    Δrefuse  = mean_over_layers(sl_tpostinst(hˡ))

Depends on: results/ from 01_extract_directions.py
Saves:      results/figure3.png, results/figure3-data.json
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
from config import RESULTS_DIR


CATS = ["refused_harmful", "accepted_harmful", "refused_harmless", "accepted_harmless"]
CAT_LABELS = {
    "refused_harmful":   "Refused Harmful",
    "accepted_harmful":  "Accepted Harmful",
    "refused_harmless":  "Refused Harmless",
    "accepted_harmless": "Accepted Harmless",
}
CAT_COLORS = {
    "refused_harmful":   "#e41a1c",
    "accepted_harmful":  "#ff7f00",
    "refused_harmless":  "#377eb8",
    "accepted_harmless": "#4daf4a",
}
CAT_MARKERS = {
    "refused_harmful":   "o",
    "accepted_harmful":  "^",
    "refused_harmless":  "s",
    "accepted_harmless": "D",
}


def load_acts(cat, pos_name):
    path = os.path.join(RESULTS_DIR, f"acts_{cat}_{pos_name}.pt")
    if not os.path.exists(path):
        return None
    return torch.load(path, weights_only=True).float()  # [n_layers, N, pos_window, hidden]


def load_center(name):
    path = os.path.join(RESULTS_DIR, f"cluster-{name}.pt")
    return torch.load(path, weights_only=True).float()  # [n_layers, hidden]


def compute_sl_samples(acts, mu_pos, mu_neg):
    """
    Compute sl(hˡ) per sample per layer.

    acts:   [n_layers, N, pos_window, hidden]
    mu_pos: [n_layers, hidden]
    mu_neg: [n_layers, hidden]

    Returns [N, n_layers]
    """
    h = acts[:, :, -1, :]           # [n_layers, N, hidden]
    mu_p = mu_pos.unsqueeze(1)       # [n_layers, 1, hidden]
    mu_n = mu_neg.unsqueeze(1)

    h_norm   = F.normalize(h,   dim=-1)
    mup_norm = F.normalize(mu_p, dim=-1)
    mun_norm = F.normalize(mu_n, dim=-1)

    cos_p = (h_norm * mup_norm).sum(dim=-1)  # [n_layers, N]
    cos_n = (h_norm * mun_norm).sum(dim=-1)

    sl = (cos_p - cos_n)  # [n_layers, N]
    return sl.T.numpy()   # [N, n_layers]


def main():
    mu_rh_tinst  = load_center("refused_harmful_tinst")
    mu_ah_tinst  = load_center("accepted_harmless_tinst")
    mu_ref_tpost = load_center("refused_tpostinst")
    mu_acc_tpost = load_center("accepted_tpostinst")

    scatter_data = {}  # cat -> {"delta_harmful": [...], "delta_refuse": [...]}

    for cat in CATS:
        acts_t = load_acts(cat, "tinst")
        acts_p = load_acts(cat, "tpostinst")
        if acts_t is None or acts_p is None:
            print(f"  Skipping {cat}: missing activations")
            continue

        sl_t = compute_sl_samples(acts_t, mu_rh_tinst, mu_ah_tinst)   # [N, L]
        sl_p = compute_sl_samples(acts_p, mu_ref_tpost, mu_acc_tpost)  # [N, L]

        delta_harmful = sl_t.mean(axis=1).tolist()
        delta_refuse  = sl_p.mean(axis=1).tolist()
        scatter_data[cat] = {"delta_harmful": delta_harmful, "delta_refuse": delta_refuse}
        print(f"  {cat:25s}: {len(delta_harmful)} samples  "
              f"Δharmful={np.mean(delta_harmful):.3f}  Δrefuse={np.mean(delta_refuse):.3f}")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))

    for cat, data in scatter_data.items():
        ax.scatter(
            data["delta_harmful"], data["delta_refuse"],
            label=CAT_LABELS[cat],
            color=CAT_COLORS[cat],
            marker=CAT_MARKERS[cat],
            alpha=0.6, s=30, edgecolors="none",
        )

    ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.7, linestyle="--")
    ax.set_xlabel("Δharmful  (mean sl at tinst)")
    ax.set_ylabel("Δrefuse   (mean sl at tpost-inst)")
    ax.set_title("Figure 3  |  Δharmful vs Δrefuse per sample\n"
                 "Harmfulness and Refusal are Encoded Separately")
    ax.legend(fontsize=9, loc="best")

    # Annotate quadrants
    xlim = ax.get_xlim(); ylim = ax.get_ylim()
    mid_x = (xlim[0] + xlim[1]) / 2; mid_y = (ylim[0] + ylim[1]) / 2
    ax.text(xlim[1]*0.7, ylim[1]*0.85, "Q1\nrefused harmful",  fontsize=7, color="gray", ha="center")
    ax.text(xlim[0]*0.7, ylim[0]*0.85, "Q2\naccepted harmless", fontsize=7, color="gray", ha="center")
    ax.text(xlim[1]*0.7, ylim[0]*0.85, "Q3\naccepted harmful",  fontsize=7, color="gray", ha="center")
    ax.text(xlim[0]*0.7, ylim[1]*0.85, "Q4\nrefused harmless",  fontsize=7, color="gray", ha="center")

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "figure3.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved {out}")

    with open(os.path.join(RESULTS_DIR, "figure3-data.json"), "w") as f:
        json.dump(scatter_data, f, indent=2)
    print("Saved figure3-data.json")


if __name__ == "__main__":
    main()
