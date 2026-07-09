"""
§3.3  |  Figure 3 (ALTERNATIVE)  —  exact paper Eq. 3 / Eq. 4 poles
===================================================================

The shipped `04_figure3_scatter.py` reuses the §3.2 *diagonal* poles for the belief scores:
  Δharmful : refused_harmful      vs accepted_harmless   (t_inst)
  Δrefuse  : refused (pool)       vs accepted (pool)      (t_post)   [via cluster-*.pt]

But §3.3 Eq. 3/4 define the poles as the PURE harmfulness / refusal clusters, pooled across
the other axis:
  Δharmful = 1/L Σ_l [ cos(h_tinst^l, μ_harmful^l)  − cos(h_tinst^l, μ_harmless^l) ]
  Δrefuse  = 1/L Σ_l [ cos(h_tpost^l,  μ_refuse^l)   − cos(h_tpost^l,  μ_accept^l)  ]
with
  μ_harmful  = mean over t_inst of  (refused_harmful ∪ accepted_harmful)
  μ_harmless = mean over t_inst of  (refused_harmless ∪ accepted_harmless)
  μ_refuse   = mean over t_post  of (refused_harmful ∪ refused_harmless)
  μ_accept   = mean over t_post  of (accepted_harmful ∪ accepted_harmless)

This script rebuilds the poles that way straight from the cached acts_*.pt (CPU only) and
plots the four categories in Δharmful × Δrefuse space.

Usage:
  python 04b_figure3_alternative.py <acts_dir> <out_png> ["<title>"]
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
CAT_MARKERS = {"refused_harmful": "o", "accepted_harmful": "^",
               "refused_harmless": "s", "accepted_harmless": "D"}

# Pure Eq.3/4 poles (pooled across the OTHER axis).
HARMFUL_POOL  = ["refused_harmful", "accepted_harmful"]     # @ t_inst  → μ_harmful
HARMLESS_POOL = ["refused_harmless", "accepted_harmless"]   # @ t_inst  → μ_harmless
REFUSE_POOL   = ["refused_harmful", "refused_harmless"]     # @ t_post  → μ_refuse
ACCEPT_POOL   = ["accepted_harmful", "accepted_harmless"]   # @ t_post  → μ_accept


def load_acts(d, cat, pos):
    p = os.path.join(d, f"acts_{cat}_{pos}.pt")
    if not os.path.exists(p):
        return None
    return torch.load(p, weights_only=True).float()   # [n_layers, N, window, hidden]


def last_pos(acts):
    return acts[:, :, -1, :]                           # [n_layers, N, hidden]


def pooled_center(d, cats, pos):
    """μ = mean over the pooled samples of `cats` at token position `pos`, per layer → [L, H]."""
    vs = []
    ns = {}
    for c in cats:
        a = load_acts(d, c, pos)
        if a is not None:
            v = last_pos(a)                            # [L, Nc, H]
            vs.append(v)
            ns[c] = v.shape[1]
    if not vs:
        raise ValueError(f"No acts for pool {cats} at {pos} in {d}")
    allv = torch.cat(vs, dim=1)                        # [L, sumN, H]
    return allv.mean(dim=1), ns                        # [L, H], counts


def delta(acts, mu_pos, mu_neg):
    """Per-sample Δ = 1/L Σ_l [cos(h^l, μ_pos^l) − cos(h^l, μ_neg^l)]  → [N]."""
    h  = last_pos(acts)                                # [L, N, H]
    hn = F.normalize(h, dim=-1)
    mp = F.normalize(mu_pos.unsqueeze(1), dim=-1)      # [L, 1, H]
    mn = F.normalize(mu_neg.unsqueeze(1), dim=-1)
    sl = (hn * mp).sum(-1) - (hn * mn).sum(-1)         # [L, N]
    return sl.mean(0).numpy()                          # average over layers → [N]


def main(acts_dir, out_png, title):
    mu_harmful,  n_hf = pooled_center(acts_dir, HARMFUL_POOL,  "tinst")
    mu_harmless, n_hl = pooled_center(acts_dir, HARMLESS_POOL, "tinst")
    mu_refuse,   n_rf = pooled_center(acts_dir, REFUSE_POOL,   "tpostinst")
    mu_accept,   n_ac = pooled_center(acts_dir, ACCEPT_POOL,   "tpostinst")
    print(f"[{title}] pole sizes: μ_harmful{n_hf} μ_harmless{n_hl} μ_refuse{n_rf} μ_accept{n_ac}")

    data = {}
    for cat in CATS:
        at = load_acts(acts_dir, cat, "tinst")
        ap = load_acts(acts_dir, cat, "tpostinst")
        if at is None or ap is None:
            print(f"  skip {cat}: missing acts")
            continue
        dh = delta(at, mu_harmful, mu_harmless)
        dr = delta(ap, mu_refuse, mu_accept)
        data[cat] = {"delta_harmful": dh.tolist(), "delta_refuse": dr.tolist()}
        print(f"  {cat:20s} n={len(dh):3d}  Δharmful={dh.mean():+.3f}  Δrefuse={dr.mean():+.3f}")

    fig, ax = plt.subplots(figsize=(7, 6))
    for cat, v in data.items():
        ax.scatter(v["delta_harmful"], v["delta_refuse"], label=CAT_LABELS[cat],
                   color=CAT_COLORS[cat], marker=CAT_MARKERS[cat], alpha=0.6, s=32,
                   edgecolors="none")
    ax.axhline(0, color="gray", lw=0.7, ls="--"); ax.axvline(0, color="gray", lw=0.7, ls="--")
    ax.set_xlabel("Δharmful  (μ_harmful − μ_harmless, t_inst)")
    ax.set_ylabel("Δrefuse   (μ_refuse − μ_accept, t_post-inst)")
    ax.set_title(f"Figure 3 (Eq. 3/4 pure poles) — {title}")
    ax.legend(fontsize=9, loc="best")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    with open(out_png.replace(".png", "-data.json"), "w") as f:
        json.dump(data, f, indent=2)
    print(f"  saved {out_png}")


if __name__ == "__main__":
    acts_dir = sys.argv[1]
    out_png  = sys.argv[2]
    title    = sys.argv[3] if len(sys.argv) > 3 else os.path.basename(acts_dir)
    main(acts_dir, out_png, title)
