"""Figure 2: per-layer score s^l at t_inst and t_post.

    s^l(h) = cos_sim(h, mu_refused_harmful^l) - cos_sim(h, mu_accepted_harmless^l)

Both token positions are drawn as two panels of a single figure. The two anchors
(mu_refused_harmful, mu_accepted_harmless) are the per-layer bucket means at that position;
the two plotted lines are the misbehaving buckets accepted-harmful (coral, solid) and
refused-harmless (teal, dash-dot), each as mean +/- std over its examples per layer.
"""

import os

import numpy as np
import torch
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

# The two panels: each token position uses that position's single-token clusters. Anchors
# (mu) come from the train split; the plotted lines from the test split.
POSITIONS = ["tinst", "tpost"]

TITLES = {
    "tinst": r"token position $t_{\mathrm{inst}}$",
    "tpost": r"token position $t_{\mathrm{post\text{-}inst}}$",
}

CORAL = "#E8766C"   # accepted harmful  (solid line, upper region)
TEAL = "#158A8A"    # refused harmless  (dash-dot line, lower region)


def _cosine(hidden, center):
    """Cosine per layer between hidden (L, N, H) and center (L, H) -> (L, N)."""
    center = center[:, None, :]
    numerator = (hidden * center).sum(-1)
    denominator = hidden.norm(dim=-1) * center.norm(dim=-1)
    return numerator / denominator


def _score_stats(line_tensor, mu_refused_harmful, mu_accepted_harmless):
    """Per-layer mean/std of s^l over a cluster's examples, or (None, None) if absent/empty.

    line_tensor is a single-token cluster (L, N, H).
    """
    if line_tensor is None or line_tensor.shape[1] == 0:
        return None, None
    score = _cosine(line_tensor, mu_refused_harmful) - _cosine(line_tensor, mu_accepted_harmless)
    return score.mean(1).numpy(), score.std(1).numpy()


def _draw_score(
    ax,
    layers,
    accepted_harmful_mean, accepted_harmful_std,
    refused_harmless_mean, refused_harmless_std,
    title,
    ymin, ymax,
):
    layers = np.asarray(layers, dtype=float)
    accepted_harmful_mean = np.asarray(accepted_harmful_mean, dtype=float)
    accepted_harmful_std = np.asarray(accepted_harmful_std, dtype=float)

    # tinted half-planes split at 0 (faint background)
    ax.axhspan(0, ymax, color=CORAL, alpha=0.08, zorder=0)
    ax.axhspan(ymin, 0, color=TEAL, alpha=0.08, zorder=0)

    # zero reference line
    ax.axhline(0, color="0.5", linestyle="--", linewidth=1.2, zorder=1)

    # accepted harmful: solid coral + std band
    ax.fill_between(layers, accepted_harmful_mean - accepted_harmful_std,
                    accepted_harmful_mean + accepted_harmful_std,
                    color=CORAL, alpha=0.22, linewidth=0, zorder=2)
    ax.plot(layers, accepted_harmful_mean, color=CORAL, linewidth=2.6, linestyle="-",
            label="accepted harmful", zorder=3)

    # refused harmless: dash-dot teal + std band (skipped when the bucket is empty)
    if refused_harmless_mean is not None:
        refused_harmless_mean = np.asarray(refused_harmless_mean, dtype=float)
        refused_harmless_std = np.asarray(refused_harmless_std, dtype=float)
        ax.fill_between(layers, refused_harmless_mean - refused_harmless_std,
                        refused_harmless_mean + refused_harmless_std,
                        color=TEAL, alpha=0.20, linewidth=0, zorder=2)
        ax.plot(layers, refused_harmless_mean, color=TEAL, linewidth=2.6, linestyle="-.",
                label="refused harmless", zorder=3)

    # cluster labels in the two corners
    ax.text(0.02, 0.975, r"$\mathcal{C}_{\mathrm{refused\ harmful}}$",
            transform=ax.transAxes, ha="left", va="top", fontsize=13, color="0.30")
    ax.text(0.02, 0.025, r"$\mathcal{C}_{\mathrm{accepted\ harmless}}$",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=13, color="0.30")

    ax.set_xlim(layers.min(), layers.max() + 1)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Layers", fontsize=22)
    ax.set_ylabel(r"$s^l(h^l)$", fontsize=22)
    ax.set_title(title, fontsize=18, pad=12)

    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35, zorder=0)
    ax.tick_params(labelsize=13)
    ax.legend(loc="lower right", fontsize=15, framealpha=0.9, edgecolor="0.8")


def plot_figure2(model, model_size, buckets, out_path=None):
    """Build and save Figure 2 (t_inst and t_post as two panels of one PNG).

    buckets: {"train": {...}, "test": {...}} from dynamic_bucket_formation.gen_buckets. Anchors
        (mu) come from the train clusters; the plotted lines from the test clusters.
    out_path: override the output PNG path (default: output/<model><size>/figure2.png).
    """
    train, test = buckets["train"], buckets["test"]
    out_dir = os.path.join(HERE, "output", f"{model}{model_size}")

    fig, axes = plt.subplots(1, len(POSITIONS), figsize=(7.2 * len(POSITIONS), 5.4))

    for ax, position in zip(axes, POSITIONS):
        mu_refused_harmful = train[f"refused_harmful_{position}"].mean(1)     # (L, H)
        mu_accepted_harmless = train[f"accepted_harmless_{position}"].mean(1)

        accepted_harmful_mean, accepted_harmful_std = _score_stats(
            test.get(f"accepted_harmful_{position}"),
            mu_refused_harmful, mu_accepted_harmless)
        refused_harmless_mean, refused_harmless_std = _score_stats(
            test.get(f"refused_harmless_{position}"),
            mu_refused_harmful, mu_accepted_harmless)

        layers = np.arange(mu_refused_harmful.shape[0])
        # symmetric y-limits keyed to the mean lines (std bands may clip, as in the paper),
        # so one noisy layer's wide band doesn't squash the informative range.
        span = np.abs(accepted_harmful_mean)
        if refused_harmless_mean is not None:
            span = np.concatenate([span, np.abs(refused_harmless_mean)])
        ymax = float(np.max(span)) * 1.4 or 0.08

        _draw_score(
            ax, layers,
            accepted_harmful_mean, accepted_harmful_std,
            refused_harmless_mean, refused_harmless_std,
            title=TITLES[position], ymin=-ymax, ymax=ymax)

        if refused_harmless_mean is None:
            print(f"{position}: refused_harmless empty -> green line omitted")

    fig.tight_layout()
    save_path = out_path or os.path.join(out_dir, "figure2.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {save_path}")
