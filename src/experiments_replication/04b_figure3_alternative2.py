"""
§3.3  |  Figure 3 (Alternative 2)  —  Scatter of Δharmful vs Δrefuse
      projected onto the STORED difference-in-means directions
=====================================================================

Same idea as 04_figure3_scatter.py (each sample placed by its harmfulness belief on x
and refusal belief on y), but the two beliefs are measured differently:

  04_figure3_scatter.py (original, paper Eq. 3/4):
      Δharmful = mean_l [ cos(h_tinst, μ_refused_harmful) − cos(h_tinst, μ_accepted_harmless) ]
      Δrefuse  = mean_l [ cos(h_tpost, μ_refused)         − cos(h_tpost, μ_accepted)          ]

  THIS script (alternative 2):
      project each hidden state directly onto the stored difference-in-means directions
      (01_extract_directions.py):
          dir_hf     = μ_refused_harmful_tinst − μ_accepted_harmless_tinst   (dir-hf.pt)
          dir_refuse = μ_refused_tpostinst     − μ_accepted_tpostinst        (dir-refuse.pt)
      Δharmful = mean_l cos(h_tinst^l, dir_hf^l)
      Δrefuse  = mean_l cos(h_tpost^l, dir_refuse^l)

  Note dir_hf == (μ_pos − μ_neg) of the harmful axis and dir_refuse == (μ_pos − μ_neg) of the
  refusal axis, so this is the single-axis projection variant of the original two-center
  difference. Cosine similarity keeps the axes on a comparable, scale-invariant range.

Depends on: dir-hf.pt, dir-refuse.pt, acts_*.pt from 01_extract_directions.py
Saves:      results/figure3_alternative_2.png, results/figure3_alternative_2-data.json

CPU-only re-analysis of saved tensors — no model needed. Point RESULTS_DIR at a run dir via
the REPL_RESULTS_DIR env var (e.g. REPL_RESULTS_DIR=.../llama_8b_results python 04b_...py).
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


def load_direction(name):
    path = os.path.join(RESULTS_DIR, f"{name}.pt")
    return torch.load(path, weights_only=True).float()  # [n_layers, hidden]


def proj_samples(acts, direction):
    """Per-sample, per-layer cosine similarity between the hidden state and the stored
    difference-in-means direction, returned as [N, n_layers].

    acts:      [n_layers, N, pos_window, hidden]  (uses the target token = index -1)
    direction: [n_layers, hidden]
    """
    h = acts[:, :, -1, :]                       # [n_layers, N, hidden]
    h_norm = F.normalize(h, dim=-1)
    d_norm = F.normalize(direction.unsqueeze(1), dim=-1)  # [n_layers, 1, hidden]
    cos = (h_norm * d_norm).sum(dim=-1)         # [n_layers, N]
    return cos.T.numpy()                        # [N, n_layers]


def main():
    dir_hf     = load_direction("dir-hf")       # harmful direction @ t_inst  [L, hidden]
    dir_refuse = load_direction("dir-refuse")   # refusal direction @ t_post  [L, hidden]

    scatter_data = {}  # cat -> {"delta_harmful": [...], "delta_refuse": [...]}

    for cat in CATS:
        acts_t = load_acts(cat, "tinst")
        acts_p = load_acts(cat, "tpostinst")
        if acts_t is None or acts_p is None:
            print(f"  Skipping {cat}: missing activations")
            continue

        delta_harmful = proj_samples(acts_t, dir_hf).mean(axis=1).tolist()      # onto dir-hf
        delta_refuse  = proj_samples(acts_p, dir_refuse).mean(axis=1).tolist()  # onto dir-refuse
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
    ax.set_xlabel("Δharmful  (mean cos(h_tinst, dir-hf))")
    ax.set_ylabel("Δrefuse   (mean cos(h_tpost, dir-refuse))")
    ax.set_title("Figure 3 (alt 2)  |  projection onto stored dir-hf / dir-refuse\n"
                 "Harmfulness and Refusal are Encoded Separately")
    ax.legend(fontsize=9, loc="best")

    # Annotate quadrants
    xlim = ax.get_xlim(); ylim = ax.get_ylim()
    ax.text(xlim[1]*0.7, ylim[1]*0.85, "Q1\nrefused harmful",   fontsize=7, color="gray", ha="center")
    ax.text(xlim[0]*0.7, ylim[0]*0.85, "Q2\naccepted harmless", fontsize=7, color="gray", ha="center")
    ax.text(xlim[1]*0.7, ylim[0]*0.85, "Q3\naccepted harmful",  fontsize=7, color="gray", ha="center")
    ax.text(xlim[0]*0.7, ylim[1]*0.85, "Q4\nrefused harmless",  fontsize=7, color="gray", ha="center")

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "figure3_alternative_2.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved {out}")

    with open(os.path.join(RESULTS_DIR, "figure3_alternative_2-data.json"), "w") as f:
        json.dump(scatter_data, f, indent=2)
    print("Saved figure3_alternative_2-data.json")


if __name__ == "__main__":
    main()
