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

# anchors (computed on the TRAIN split): name -> single-token clusters to combine. The clusters
# already carry their token position (tinst for the harmful axis, tpost for the refusal axis).
ANCHORS = {
    "harmful": ["accepted_harmful_tinst", "refused_harmful_tinst"],
    "harmless": ["accepted_harmless_tinst", "refused_harmless_tinst"],
    "refused": ["refused_harmful_tpost", "refused_harmless_tpost"],
    "accepted": ["accepted_harmful_tpost", "accepted_harmless_tpost"],
}

# scatter points (from the TEST split): legend category -> cluster base name. Delta_harmful reads
# <base>_tinst, Delta_refuse reads <base>_tpost.
POINT_BASES = {
    "accepted harmful": "accepted_harmful",
    "refused harmful": "refused_harmful",
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


def _cosine(hidden, center):
    """Cosine per layer between hidden (L, N, H) and center (L, H) -> (L, N)."""
    center = center[:, None, :]
    numerator = (hidden * center).sum(-1)
    denominator = hidden.norm(dim=-1) * center.norm(dim=-1)
    return numerator / denominator


def _anchor_center(acts, cluster_names):
    """Combined per-layer mean over several single-token clusters -> (L, H).

    acts: a bucket dict cluster_name -> (L, N, H). Missing/empty clusters are skipped.
    """
    slices = [acts[n] for n in cluster_names if n in acts and acts[n].shape[1] > 0]
    return torch.cat(slices, dim=1).mean(1)  # (L, H)


def _delta(point_tensor, mu_pos, mu_neg):
    """Layer-averaged score s^l per instruction of a single-token cluster (L, N, H) -> (N,)."""
    score = _cosine(point_tensor, mu_pos) - _cosine(point_tensor, mu_neg)  # (L, N)
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


def plot_figure3(model, model_size, buckets, save_path=None):
    """Build and save Figure 3 (Delta_harmful vs Delta_refuse scatter).

    buckets: {"train": {...}, "test": {...}} from gen_buckets. The 4 mu anchors come from the train
        clusters; the scatter points from the test clusters (Delta_harmful from <base>_tinst,
        Delta_refuse from <base>_tpost, paired by instruction index).
    """
    train, test = buckets["train"], buckets["test"]
    out_dir = os.path.join(HERE, "output", f"{model}{model_size}")

    mu = {name: _anchor_center(train, names) for name, names in ANCHORS.items()}

    points = {}
    for category, base in POINT_BASES.items():
        tinst, tpost = test.get(f"{base}_tinst"), test.get(f"{base}_tpost")
        if tinst is None or tpost is None or tinst.shape[1] == 0 or tpost.shape[1] == 0:
            print(f"{category}: test cluster '{base}_tinst/_tpost' missing or empty, skipping")
            continue
        delta_harmful = _delta(tinst, mu["harmful"], mu["harmless"])
        delta_refuse = _delta(tpost, mu["refused"], mu["accepted"])
        n = min(len(delta_harmful), len(delta_refuse))
        if len(delta_harmful) != len(delta_refuse):
            print(f"{category}: tinst/tpost counts differ "
                  f"({len(delta_harmful)} vs {len(delta_refuse)}); pairing first {n}")
        points[category] = (delta_harmful[:n], delta_refuse[:n])

    # data-driven limits (0.5B magnitudes differ from the paper's fixed defaults)
    all_x = np.concatenate([dx for dx, _ in points.values()])
    all_y = np.concatenate([dy for _, dy in points.values()])
    xlim = (all_x.min() - 0.02, all_x.max() + 0.02)
    ylim = (all_y.min() - 0.03, all_y.max() + 0.06)  # extra top room for the legend

    save_path = save_path or os.path.join(out_dir, "figure3.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plot_belief_correlation(points, xlim=xlim, ylim=ylim, save_path=save_path)
    print(f"saved {save_path}")


# ---------------------------------------------------------------------------
# Qwen3.5 thinking track. gen_buckets_thinking keeps whole (L, N, 25, H) families keyed by the
# UNSUFFIXED name — which happens to be exactly POINT_BASES' four values — so slicing two slots and
# re-suffixing rebuilds the dict plot_figure3 already consumes. The score itself is not reimplemented.
# ---------------------------------------------------------------------------
def _flatten_thinking(buckets, slot_harmful, slot_refuse):
    """Thinking buckets {family: (L, N, 25, H)} -> the {<base>_tinst, <base>_tpost} -> (L, n, H)
    dict plot_figure3 expects.

    The null-row mask is taken jointly over BOTH slots so the two stay row-aligned: Delta_harmful
    and Delta_refuse are paired per instruction, and a row dropped at only one slot would shift
    every later pairing. fp16 -> fp32 per slice, as in _02_3.1_figure2._slice_pos.
    """
    out = {}
    for side in ("train", "test"):
        flat = {}
        for base, tensor in buckets[side].items():
            inst = tensor[:, :, slot_harmful, :].float()
            post = tensor[:, :, slot_refuse, :].float()
            keep = ((inst.norm(dim=-1) > 0) & (post.norm(dim=-1) > 0)).all(dim=0)
            if bool(keep.any()):
                flat[f"{base}_tinst"], flat[f"{base}_tpost"] = inst[:, keep], post[:, keep]
        out[side] = flat
    return out


def plot_figure3_thinking(model, model_size, buckets, slot_harmful=0, slot_refuse=1,
                          save_path=None):
    """Figure 3 on the 25-slot thinking layout; the score itself is plot_figure3's.

    buckets: {"train"/"test": {family -> (L, N, 25, H)}} from gen_buckets_thinking. Slots default
    to t_inst/t_post; callers pass whatever the bucket config names (dynamic_bucket_formation.
    fig3_slots).
    """
    plot_figure3(model, model_size, _flatten_thinking(buckets, slot_harmful, slot_refuse),
                 save_path=save_path)
