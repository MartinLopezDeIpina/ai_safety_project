"""
§3.2  |  Figure 2  —  Layer-wise Cluster Distance sl(hˡ)
=========================================================

Paper claim (§3.2):
    At tinst (the last instruction token), the hidden state hˡ clusters by HARMFULNESS:
      • refused_harmful and accepted_harmful both score high sl (close to μ_rh)
      • refused_harmless and accepted_harmless both score low sl
    At tpost-inst (the last post-instruction token), the clustering REVERSES to refusal:
      • refused_harmful and refused_harmless both score high sl
      • accepted_harmful and accepted_harmless both score low sl

Quantity (§3.2 eq. 2):
    sl(hˡ) = cos_sim(hˡ, μ_refused_harmful[l]) − cos_sim(hˡ, μ_accepted_harmless[l])
           The SAME two cluster anchors (refused_harmful, accepted_harmless) are used
           at BOTH token positions; only the position at which hˡ and the anchors are
           taken changes between the two panels. (Do not confuse these with the
           all-refused / all-accepted Δrefuse anchors from §3.3 eq. 4 used in Fig 3/6/8.)

Plot format:
    2 rows, 1 column. Each panel shows the two MISBEHAVING categories
    (accepted_harmful, refused_harmless) over the harmful (red) / harmless (green) regions.
    Top:    tinst      — x = layer, y = sl
    Bottom: tpostinst  — same anchors, taken at tpost-inst.

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


CATS = ["accepted_harmful", "refused_harmless"]
CAT_LABELS = {
    "accepted_harmful":  "Accepted Harmful",
    "refused_harmless":  "Refused Harmless",
}
CAT_COLORS = {
    "accepted_harmful":  "#e41a1c",   # red
    "refused_harmless":  "#4daf4a",   # green
}
CAT_STYLE = {
    "accepted_harmful":  "-",
    "refused_harmless":  "-",
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
    # Shade above and below y=0
    ax.axhspan(0, 1e3,  facecolor="#e41a1c", alpha=0.08, zorder=0)  # red above 0
    ax.axhspan(-1e3, 0, facecolor="#4daf4a", alpha=0.08, zorder=0)  # green below 0

    for cat in CATS:
        if cat not in sl_dict:
            continue
        sl = sl_dict[cat]
        ax.plot(layers, sl,
                label=CAT_LABELS[cat],
                color=CAT_COLORS[cat],
                linestyle=CAT_STYLE[cat],
                linewidth=2.0)

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Layer")
    ax.set_ylabel("sl(hˡ)")
    ax.legend(fontsize=9, loc="upper left")


def acts_center(cat, pos_name):
    """Mean cluster center over samples from a cached per-sample acts tensor.

    Mirrors cluster_center() in 01_extract_directions.py: take the target-position
    slice ([:, :, -1, :]) and average over the sample axis.  Returns [n_layers, hidden]
    or None if the acts file is missing.
    """
    acts = load_acts(cat, pos_name)
    if acts is None:
        return None
    return acts[:, :, -1, :].mean(dim=1)


def main():
    # Paper formula (§3.2 eq. 2): the SAME two cluster anchors are used at BOTH token
    # positions — Cl_refused_harmful and Cl_accepted_harmless, each computed at that
    # position:
    #   sl(hˡ) = cos_sim(hˡ, μ_refused_harmful) − cos_sim(hˡ, μ_accepted_harmless)
    #
    # NOTE: do NOT use the all-refused / all-accepted centers here — those are the
    # Δrefuse anchors from §3.3 eq. (4), used for Figure 3/6/8, not Figure 2.

    # Tinst poles: refused_harmful vs accepted_harmless at tinst.
    mu_harmful_tinst  = load_center("refused_harmful_tinst")
    mu_harmless_tinst = load_center("accepted_harmless_tinst")

    # Tpostinst poles: the SAME two categories, but their centers at tpost-inst.
    # 01_extract_directions.py does not save these as cluster-*.pt files, so recompute
    # them from the cached per-sample activations.
    mu_refused_tpost  = acts_center("refused_harmful",   "tpostinst")
    mu_accepted_tpost = acts_center("accepted_harmless", "tpostinst")
    if mu_refused_tpost is None or mu_accepted_tpost is None:
        raise FileNotFoundError(
            "Missing acts_refused_harmful_tpostinst.pt / acts_accepted_harmless_tpostinst.pt "
            "— run 01_extract_directions.py first."
        )

    n_layers = mu_harmful_tinst.shape[0]
    layers   = list(range(n_layers))

    sl_tinst  = {}
    sl_tpost  = {}

    for cat in CATS:
        acts_t = load_acts(cat, "tinst")
        acts_p = load_acts(cat, "tpostinst")

        if acts_t is not None:
            sl_tinst[cat] = compute_sl_per_sample(acts_t, mu_harmful_tinst, mu_harmless_tinst)
        if acts_p is not None:
            sl_tpost[cat] = compute_sl_per_sample(acts_p, mu_refused_tpost, mu_accepted_tpost)

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

    # Restore tight y-limits after axhspan used ±1e3 as sentinel
    for ax, sl_dict in [(ax1, sl_tinst), (ax2, sl_tpost)]:
        vals = np.concatenate([v for v in sl_dict.values()]) if sl_dict else np.array([0.0])
        pad = max(abs(vals).max() * 0.15, 0.02)
        ax.set_ylim(vals.min() - pad, vals.max() + pad)

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
