"""
§3.2  |  Figure 2  —  Layer-wise Cluster Distance sl(hˡ)
=========================================================

Paper claim (§3.2):
    At tinst (the <|im_end|> token), the hidden state hˡ clusters by HARMFULNESS:
      • refused_harmful and accepted_harmful both score high sl (close to μ_rh)
      • refused_harmless and accepted_harmless both score low sl
    At tpost-inst (the last \\n), the clustering REVERSES to refusal behaviour:
      • refused_harmful and refused_harmless both score high sl
      • accepted_harmful and accepted_harmless both score low sl

Quantity (§3.2 eq.):
    sl(hˡ) = cos_sim(hˡ, μ_refused_harmful[l]) − cos_sim(hˡ, μ_accepted_harmless[l])
           (subscript omitted; both cluster centers chosen to be refused_harmful vs
            accepted_harmless at the same token position)

Plot format:
    2 rows, 1 column.
    Top:    tinst  — 4 coloured lines (one per category), x = layer, y = sl
    Bottom: tpostinst — same.

Depends on: results/ from 01_extract_directions.py
Saves:      results/figure2.png
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
    "refused_harmful":   "#e41a1c",   # red
    "accepted_harmful":  "#ff7f00",   # orange
    "refused_harmless":  "#377eb8",   # blue
    "accepted_harmless": "#4daf4a",   # green
}
CAT_STYLE = {
    "refused_harmful":   "-",
    "accepted_harmful":  "--",
    "refused_harmless":  "-.",
    "accepted_harmless": ":",
}


def load_acts(cat, pos_name):
    path = os.path.join(RESULTS_DIR, f"acts_{cat}_{pos_name}.pt")
    if not os.path.exists(path):
        return None
    return torch.load(path, weights_only=True).float()  # [n_layers, N, pos_window, hidden]


def load_center(name):
    path = os.path.join(RESULTS_DIR, f"cluster-{name}.pt")
    return torch.load(path, weights_only=True).float()  # [n_layers, hidden]


def compute_sl_per_sample(acts, mu_pos, mu_neg):
    """
    Compute per-sample, per-layer sl(hˡ) = cos_sim(hˡ, mu_pos[l]) - cos_sim(hˡ, mu_neg[l]).

    acts:   [n_layers, N, pos_window, hidden]
    mu_pos: [n_layers, hidden]
    mu_neg: [n_layers, hidden]

    Returns mean_sl: [n_layers]  (averaged over samples)
    """
    n_layers = acts.shape[0]
    h = acts[:, :, -1, :]  # [n_layers, N, hidden]

    mu_p = mu_pos.unsqueeze(1)  # [n_layers, 1, hidden]
    mu_n = mu_neg.unsqueeze(1)  # [n_layers, 1, hidden]

    # Normalise
    h_norm   = F.normalize(h,   dim=-1)
    mup_norm = F.normalize(mu_p, dim=-1)
    mun_norm = F.normalize(mu_n, dim=-1)

    cos_p = (h_norm * mup_norm).sum(dim=-1)  # [n_layers, N]
    cos_n = (h_norm * mun_norm).sum(dim=-1)

    sl = (cos_p - cos_n).mean(dim=1)  # [n_layers]
    return sl.numpy()


def plot_sl(ax, layers, sl_dict, title):
    for cat in CATS:
        if cat not in sl_dict:
            continue
        sl = sl_dict[cat]
        ax.plot(layers, sl,
                label=CAT_LABELS[cat],
                color=CAT_COLORS[cat],
                linestyle=CAT_STYLE[cat],
                linewidth=1.8)
    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Layer")
    ax.set_ylabel("sl(hˡ)")
    ax.legend(fontsize=8, loc="upper left")


def main():
    # Load cluster centers
    mu_rh_tinst   = load_center("refused_harmful_tinst")    # [n_layers, hidden]
    mu_ah_tinst   = load_center("accepted_harmless_tinst")
    mu_ref_tpost  = load_center("refused_tpostinst")
    mu_acc_tpost  = load_center("accepted_tpostinst")

    n_layers = mu_rh_tinst.shape[0]
    layers   = list(range(n_layers))

    sl_tinst  = {}
    sl_tpost  = {}

    for cat in CATS:
        acts_t = load_acts(cat, "tinst")
        acts_p = load_acts(cat, "tpostinst")

        if acts_t is not None:
            sl_tinst[cat] = compute_sl_per_sample(acts_t, mu_rh_tinst, mu_ah_tinst)
        if acts_p is not None:
            sl_tpost[cat] = compute_sl_per_sample(acts_p, mu_ref_tpost, mu_acc_tpost)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    plot_sl(ax1, layers, sl_tinst,
            "tinst — Clusters by Harmfulness\n"
            "(wrong-behaviour samples still fall in harmfulness cluster)")
    plot_sl(ax2, layers, sl_tpost,
            "tpost-inst — Clusters by Refusal Behaviour\n"
            "(harmfulness clusters dissolve; refusal clusters emerge)")

    ax1.set_xlabel("")   # only bottom axis needs x label
    fig.suptitle("Figure 2  |  Layer-wise sl(hˡ) at tinst and tpost-inst", fontsize=13, y=1.01)
    plt.tight_layout()

    out = os.path.join(RESULTS_DIR, "figure2.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")

    # Save data as JSON for inspection
    data = {
        "layers": layers,
        "tinst":  {k: v.tolist() for k, v in sl_tinst.items()},
        "tpostinst": {k: v.tolist() for k, v in sl_tpost.items()},
    }
    with open(os.path.join(RESULTS_DIR, "figure2-data.json"), "w") as f:
        json.dump(data, f, indent=2)
    print("Saved figure2-data.json")


if __name__ == "__main__":
    main()
