"""Figure 3: per-instruction correlation between the harmfulness score (Delta_harmful, x)
and the refusal score (Delta_refuse, y). Each point is one instruction, coloured/marked by
its behavioral category.

    Delta_harmful(x) = (1/L) sum_l [ cos(h^l_tinst, mu^l_harmful)  - cos(h^l_tinst, mu^l_harmless) ]
    Delta_refuse(x)  = (1/L) sum_l [ cos(h^l_tpost, mu^l_refused)  - cos(h^l_tpost, mu^l_accepted) ]

Delta_harmful uses each point's t_inst activation (token idx 1); Delta_refuse uses t_post
(idx -1). Both positions live in every bucket tensor (L, N, T, H).
"""

import os

import numpy as np
import torch
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

# token index in the (layers, N, tokens, hidden) activation tensor
POSITION_INDEX = {"tinst": 1, "tpost": -1}

# anchors: name -> (buckets to combine, token position)
ANCHORS = {
    "harmful": (["tinst_accepted_harmful", "tinst_refused_harmful"], "tinst"),
    "harmless": (["accepted_harmless", "refused_harmless"], "tinst"),
    "refused": (["tpost_refused_harmful", "refused_harmless"], "tpost"),
    "accepted": (["tpost_accepted_harmful", "accepted_harmless"], "tpost"),
}

# scatter points: legend category -> bucket (harmful uses the tinst-labelled buckets)
POINT_BUCKETS = {
    "accepted harmful": "tinst_accepted_harmful",
    "refused harmful": "tinst_refused_harmful",
    "accepted harmless": "accepted_harmless",
    "refused harmless": "refused_harmless",
}

# ---- per-bucket color + marker; colors estimated from the target figure ----
STYLE = {
    "accepted harmful": dict(color="#B0423B", marker="+", s=75, linewidths=1.7),
    "accepted harmless": dict(color="#8FC63D", marker="s", s=55, alpha=0.90,
                              edgecolors="none"),
    "refused harmless": dict(color="#1B9A9A", marker="x", s=62, linewidths=1.9),
    "refused harmful": dict(color="#EF7C6E", marker="o", s=72, alpha=0.90,
                            edgecolors="none"),
}
# legend/plot order matches the target (row1, row2 across 2 columns)
ORDER = ["accepted harmful", "accepted harmless", "refused harmless", "refused harmful"]


def _acts_dir(model, model_size):
    return os.path.join(HERE, "output", f"{model}{model_size}", "buckets_activations")


def _load(acts_dir, name):
    """Load a bucket's activation tensor (L, N, T, H) as float32, or None if absent."""
    path = os.path.join(acts_dir, name + ".pt")
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location="cpu").float()


def _cosine(hidden, center):
    """Cosine per layer between hidden (L, N, H) and center (L, H) -> (L, N)."""
    center = center[:, None, :]
    numerator = (hidden * center).sum(-1)
    denominator = hidden.norm(dim=-1) * center.norm(dim=-1)
    return numerator / denominator


def _anchor_center(acts_dir, bucket_names, position):
    """Combined per-layer mean over several buckets at one token position -> (L, H)."""
    idx = POSITION_INDEX[position]
    slices = [_load(acts_dir, n)[:, :, idx, :] for n in bucket_names]  # each (L, Ni, H)
    return torch.cat(slices, dim=1).mean(1)  # (L, H)


def _delta(point_tensor, position, mu_pos, mu_neg):
    """Layer-averaged score s^l per instruction at one token position -> (N,)."""
    hidden = point_tensor[:, :, POSITION_INDEX[position], :]  # (L, N, H)
    score = _cosine(hidden, mu_pos) - _cosine(hidden, mu_neg)  # (L, N)
    return score.mean(0).numpy()  # average over layers -> (N,)


def plot_belief_correlation(
    buckets,                       # dict: name -> (delta_harmful_array, delta_refuse_array)
    xlim=(-0.2, 0.2),
    ylim=(-0.35, 0.38),
    save_path=None,
):
    fig, ax = plt.subplots(figsize=(6.2, 5.6))

    for name in ORDER:
        if name not in buckets:
            continue
        dx, dy = buckets[name]
        ax.scatter(np.asarray(dx, dtype=float), np.asarray(dy, dtype=float),
                   label=name, zorder=3, **STYLE[name])

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r"$\Delta$ harmful", fontsize=20)
    ax.set_ylabel(r"$\Delta$ refuse", fontsize=20)

    ax.grid(True, linestyle="-", linewidth=0.7, alpha=0.35, zorder=0)
    ax.tick_params(labelsize=13)

    # legend above the axes, two columns
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2,
              fontsize=13, framealpha=0.0, handletextpad=0.3,
              columnspacing=1.4, borderaxespad=0.0)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return fig, ax


def plot_figure3(model, model_size):
    """Build and save Figure 3 (Delta_harmful vs Delta_refuse scatter)."""
    acts_dir = _acts_dir(model, model_size)
    out_dir = os.path.join(HERE, "output", f"{model}{model_size}")

    mu = {name: _anchor_center(acts_dir, names, pos)
          for name, (names, pos) in ANCHORS.items()}

    buckets = {}
    for category, bucket in POINT_BUCKETS.items():
        tensor = _load(acts_dir, bucket)
        if tensor is None:
            print(f"{category}: bucket '{bucket}' missing, skipping")
            continue
        delta_harmful = _delta(tensor, "tinst", mu["harmful"], mu["harmless"])
        delta_refuse = _delta(tensor, "tpost", mu["refused"], mu["accepted"])
        buckets[category] = (delta_harmful, delta_refuse)

    # data-driven limits (0.5B magnitudes differ from the paper's fixed defaults)
    all_x = np.concatenate([dx for dx, _ in buckets.values()])
    all_y = np.concatenate([dy for _, dy in buckets.values()])
    xlim = (all_x.min() - 0.02, all_x.max() + 0.02)
    ylim = (all_y.min() - 0.03, all_y.max() + 0.06)  # extra top room for the legend

    save_path = os.path.join(out_dir, "figure3.png")
    plot_belief_correlation(buckets, xlim=xlim, ylim=ylim, save_path=save_path)
    print("figure3.png saved")
