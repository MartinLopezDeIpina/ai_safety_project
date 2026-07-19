"""Figure 2: per-layer score s^l at t_inst and t_post.

    s^l(h) = cos_sim(h, mu_refused_harmful^l) - cos_sim(h, mu_accepted_harmless^l)

Both token positions are drawn as two panels of a single figure. The two anchors
(mu_refused_harmful, mu_accepted_harmless) are the per-layer bucket means at that position;
the two plotted lines are the misbehaving buckets accepted-harmful (coral, solid) and
refused-harmless (teal, dash-dot), each as mean +/- std over its examples per layer.
"""

import glob
import os
import re

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


RUN_DIR_PREFIX = "fig2_run_"


def _next_run_dir(base_dir, prefix=RUN_DIR_PREFIX):
    """Create and return base_dir/<prefix><N>, N being the next free integer.

    Each one_image_per_figure call gets its own folder so re-running never overwrites an earlier
    run's panels. Existing dirs whose suffix isn't an integer are ignored.
    """
    used = []
    for path in glob.glob(os.path.join(base_dir, f"{prefix}*")):
        match = re.fullmatch(rf"{re.escape(prefix)}(\d+)", os.path.basename(path))
        if match and os.path.isdir(path):
            used.append(int(match.group(1)))
    run_dir = os.path.join(base_dir, f"{prefix}{max(used, default=0) + 1}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


class _PanelSink:
    """Where the per-panel axes come from, and when each panel is written.

    Two modes behind one interface, so the plotting loops stay single-copy:
      grid  (run_dir=None) - one shared figure; `axis()` hands out the next cell of a pre-built
                             grid and `close()` saves the whole thing to `out_path`.
      split (run_dir set)  - one figure per panel; `axis()` opens a fresh figure and the matching
                             `finish()` saves it under `run_dir`.
    """

    def __init__(self, out_path, run_dir, nrows=1, ncols=1, figsize=(7.2, 5.4), dpi=200,
                 split_figsize=(7.2, 5.4), split_dpi=200):
        self.out_path = out_path
        self.run_dir = run_dir
        # a standalone panel gets the full-size canvas even when the grid packs small cells:
        # _draw_score's font sizes are absolute, so a 4.2x3.6 cell alone reads as oversized text
        self.figsize = split_figsize
        self.dpi = split_dpi if run_dir is not None else dpi
        self._fig = None
        self._axes = None
        self._index = 0
        if run_dir is None:
            self._fig, axes = plt.subplots(nrows, ncols,
                                           figsize=(figsize[0] * ncols, figsize[1] * nrows),
                                           squeeze=False)
            self._axes = axes.ravel()

    def axis(self):
        """The axes for the next panel."""
        if self.run_dir is None:
            ax = self._axes[self._index]
        else:
            self._fig, ax = plt.subplots(figsize=self.figsize)
        self._index += 1
        return ax

    def finish(self, filename):
        """End the panel opened by the last `axis()`; saves it in split mode, no-op in grid mode."""
        if self.run_dir is None:
            return
        path = os.path.join(self.run_dir, filename)
        self._fig.tight_layout()
        self._fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(self._fig)
        self._fig = None
        print(f"saved {path}")

    def close(self):
        """Save the grid figure (grid mode); in split mode every panel is already on disk."""
        if self.run_dir is not None:
            return
        for ax in self._axes[self._index:]:
            ax.axis("off")
        self._fig.tight_layout()
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        self._fig.savefig(self.out_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(self._fig)
        print(f"saved {self.out_path}")


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


# ---- Qwen3.5 thinking Figure 2: the same score s^l, drawn across the fixed 25-slot layout ----
# Slots 1-4, 20 and 24 hold different tokens per generation mode, so their labels are per-mode.
# A "(gen)"/"(prompt)" tag marks where a slot's origin differs between modes; untagged slots 1-4 are
# prompt tokens in every mode that has them. Verified against the Qwen3.5-9B tokenizer:
#
#   slot 4   genthink   '\n'   (prompt) - the template ends "<think>\n".
#            gennothink '\n\n' (prompt) - Qwen's canonical empty block is "<think>\n\n</think>\n\n",
#                       and that newline pair is ONE token (id 271), not two.
#   slot 20  genthink   '\n\n' GENERATED after the model's own </think> (100/100 rows).
#            gennothink '\n\n' supplied by the PROMPT (the template's trailing pair).
#   slot 24  generated in genthink/stripped (the model closes its own block); a prompt token in
#            gennothink/v2, where the template hands </think> to the model.
#
# So genthink and gennothink are token-aligned across 20-23 and differ only in where the '\n\n'
# came from; in both, the first ANSWER token is slot 21. Hence "gen N" = the Nth answer token, and
# the '\n\n' is named for what it is instead of being counted as generated output.
_SLOT_LABELS_COMMON = {
    0: r"$t_{\mathrm{inst}}$",
    5: "CoT 1", 6: "CoT 2", 7: "CoT 3", 8: "CoT 4", 9: "CoT 5",
    10: "mid 1", 11: "mid 2", 12: "mid 3", 13: "mid 4", 14: "mid 5",
    15: "last 1", 16: "last 2", 17: "last 3", 18: "last 4", 19: "last 5",
    21: "gen 1", 22: "gen 2", 23: "gen 3",
}
_SLOT_LABELS_BY_MODE = {
    "genthink": {
        1: r"$t_{\mathrm{post}}$", 2: r"\n", 3: "<think>", 4: r"\n",
        24: "</think> (gen)", 20: r"\n\n (gen)",
    },
    "gennothink": {
        1: r"$t_{\mathrm{post}}$", 2: r"\n", 3: "<think>", 4: r"\n\n",
        24: "</think> (prompt)", 20: r"\n\n (prompt)",
    },
    # gennothink_stripped cuts the prompt at <|im_start|>, so EVERY slot but 0 is the model's own
    # output - it re-emits the template itself. Slots 1/2/4 are mixtures across rows, so their
    # labels name the majority token (of 200 advbench rows):
    #   slot 1  'assistant' x118, null x56, '<|im_start|>' x23 (rows where the model emitted a second
    #           <|im_start|> before <think>, so tho-2 lands back on the prompt's own)
    #   slot 2  '\n' x119, null x56, '<|im_start|>' x23
    #   slot 4  '\n\n' x111 (empty think block), '\n' x32 (a real CoT follows), null x56
    # Slot 4 predicts the CoT exactly: '\n\n' -> no CoT (111/111), '\n' -> CoT (32/32). Only ~22% of
    # the rows that open a <think> go on to reason; the rest reproduce "<think>\n\n</think>".
    "gennothink_stripped": {
        1: r"$t_{\mathrm{post}}$ (gen)", 2: r"\n (gen)", 3: "<think> (gen)", 4: r"\n / \n\n (gen)",
        24: "</think> (gen)", 20: r"\n\n (gen)",
    },
    # v2 is not plotted (its \n\n-stripping question was answered behaviourally: the model still
    # refuses, since refusal rides on <think>). Kept so main.py's grid loop resolves. Its slot 20
    # would hold the first answer token, not '\n\n', because inference.py:291 .strip()s the model's
    # leading whitespace before storing - see the pipeline notes.
    "gennothink_stripped_v2": {
        1: r"$t_{\mathrm{post}}$", 2: r"\n", 3: "<think>", 4: r"\n\n",
        24: "</think> (prompt)", 20: "gen 1 (\\n\\n stripped)",
    },
}


def slot_labels(mode):
    """Slot -> x-axis label for one generation mode (see _SLOT_LABELS_BY_MODE)."""
    if mode not in _SLOT_LABELS_BY_MODE:
        raise KeyError(f"unknown thinking mode {mode!r} (have {sorted(_SLOT_LABELS_BY_MODE)})")
    return {**_SLOT_LABELS_COMMON, **_SLOT_LABELS_BY_MODE[mode]}


# Mode-neutral default, for callers that only name a slot in a diagnostic (_04_intervention).
SLOT_LABELS = slot_labels("genthink")
# Panels are drawn in list order, so these list the slots in TOKEN order, which is not slot order:
# extract_hidden puts </think> in slot 24 but the 4 tokens after it in slots 20-23
# (slots[24] = thc, slots[20..23] = thc+1..thc+4, extract_hidden.py:396-397), so 24 always precedes
# 20-23 in the sequence, in every mode. Reading a grid left-to-right therefore walks the prompt.
THINK_POSITIONS = [0, 1, 2, 3, 4] + list(range(5, 20)) + [24, 20, 21, 22, 23]
# meaningful slots without a reasoning trace: the CoT slots (5-19) are null there, so they are dropped
NOTHINK_POSITIONS = [0, 1, 2, 3, 4, 24, 20, 21, 22, 23]
EXPECTED_SLOTS = 25  # must match extract_hidden.THINK_SLOTS


def _check_layout(buckets):
    """Reject .pt written by the superseded 22-slot extraction.

    The 22-slot layout is not a prefix of this one: it had no </think> slot, and its slots 20-21 were
    the first 2 NON-whitespace answer tokens, whereas 20-23 are now taken contiguously from </think>
    (so slot 20 is the \\n\\n itself). Slots 0-19 still agree, but 20-21 would be plotted under
    labels that no longer describe them — silently wrong rather than merely truncated, so fail here.
    """
    for side in ("train", "test"):
        for name, tensor in buckets[side].items():
            got = tensor.shape[2]
            if got != EXPECTED_SLOTS:
                raise ValueError(
                    f"{side}/{name}: activations have {got} token slots, expected {EXPECTED_SLOTS}. "
                    f"These .pt predate the </think> slot change and their slots 20-21 mean something "
                    f"different, so they cannot be plotted on the current layout. Re-run the `acts` "
                    f"stage to regenerate them, or read the 25-slot judge_activations/ instead "
                    f"(use_judged_classifications=True)."
                )


def _slice_pos(tensor, p):
    """(L, N, 25, H) -> (L, n, H) at slot p, dropping null (zero-norm) rows.

    Buckets are held in fp16 (their on-disk dtype) to bound memory; upcast this one small
    (L, n, H) slice to fp32 here so the downstream mean/cosine reductions stay fp32-accurate.
    """
    sl = tensor[:, :, p, :].float()
    keep = (sl.norm(dim=-1) > 0).all(dim=0)
    return sl[:, keep]


def plot_figure2_thinking(model, model_size, buckets, positions, out_path, mode, ncols=6,
                          one_image_per_figure=False):
    """Figure 2 across the 25-slot thinking layout (one panel per slot in `positions`).

    Same score as plot_figure2: anchors mu_refused_harmful / mu_accepted_harmless from the train
    families, coral/teal lines (accepted_harmful / refused_harmless) from test. Families are whole
    (L, N, 25, H) tensors; each panel slices its own slot and drops null rows.

    `mode` is the generation mode these buckets came from ("genthink", "gennothink", ...); it picks
    the panel labels, which are not the same across modes (see _SLOT_LABELS_BY_MODE).

    one_image_per_figure: write one PNG per slot into a fresh <out_path dir>/fig2_run_<N>/ folder
    instead of the single grid PNG (see _next_run_dir).
    """
    _check_layout(buckets)
    labels = slot_labels(mode)
    train, test = buckets["train"], buckets["test"]
    n = len(positions)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    run_dir = _next_run_dir(os.path.dirname(out_path)) if one_image_per_figure else None
    if run_dir:
        print(f"one image per figure -> {run_dir}")
    sink = _PanelSink(out_path, run_dir, nrows=nrows, ncols=ncols, figsize=(4.2, 3.6), dpi=150)

    for order, p in enumerate(positions):
        # slot order is not token order (THINK_POSITIONS), so prefix with the panel's position in
        # the sequence to keep the files sorted the way the grid reads
        filename = f"{order:02d}_slot{p:02d}.png"
        ax = sink.axis()
        title = labels.get(p, str(p))
        rh_tr = _slice_pos(train["refused_harmful"], p)
        ah_tr = _slice_pos(train["accepted_harmless"], p)
        if rh_tr.shape[1] == 0 or ah_tr.shape[1] == 0:
            ax.set_title(title, fontsize=12)
            ax.text(0.5, 0.5, "no anchors", ha="center", va="center", transform=ax.transAxes)
            sink.finish(filename)
            continue
        mu_refused_harmful, mu_accepted_harmless = rh_tr.mean(1), ah_tr.mean(1)

        ah_line = _slice_pos(test["accepted_harmful"], p) if "accepted_harmful" in test else None
        rh_line = _slice_pos(test["refused_harmless"], p) if "refused_harmless" in test else None
        accepted_harmful_mean, accepted_harmful_std = _score_stats(
            ah_line, mu_refused_harmful, mu_accepted_harmless)
        refused_harmless_mean, refused_harmless_std = _score_stats(
            rh_line, mu_refused_harmful, mu_accepted_harmless)

        if accepted_harmful_mean is None:
            ax.set_title(title, fontsize=12)
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            sink.finish(filename)
            continue

        layers = np.arange(mu_refused_harmful.shape[0])
        span = np.abs(accepted_harmful_mean)
        if refused_harmless_mean is not None:
            span = np.concatenate([span, np.abs(refused_harmless_mean)])
        ymax = float(np.max(span)) * 1.4 or 0.08

        _draw_score(
            ax, layers,
            accepted_harmful_mean, accepted_harmful_std,
            refused_harmless_mean, refused_harmless_std,
            title=title, ymin=-ymax, ymax=ymax)
        sink.finish(filename)

    sink.close()


def plot_figure2(model, model_size, buckets, out_path=None, one_image_per_figure=False):
    """Build and save Figure 2 (t_inst and t_post as two panels of one PNG).

    buckets: {"train": {...}, "test": {...}} from dynamic_bucket_formation.gen_buckets. Anchors
        (mu) come from the train clusters; the plotted lines from the test clusters.
    out_path: override the output PNG path (default: output/<model><size>/figure2.png).
    one_image_per_figure: write one PNG per token position (tinst.png, tpost.png) into a fresh
        <out dir>/fig2_run_<N>/ folder instead of the single two-panel PNG (see _next_run_dir).
    """
    train, test = buckets["train"], buckets["test"]
    out_dir = os.path.join(HERE, "output", f"{model}{model_size}")
    save_path = out_path or os.path.join(out_dir, "figure2.png")

    run_dir = _next_run_dir(os.path.dirname(save_path)) if one_image_per_figure else None
    if run_dir:
        print(f"one image per figure -> {run_dir}")
    sink = _PanelSink(save_path, run_dir, nrows=1, ncols=len(POSITIONS))

    for position in POSITIONS:
        ax = sink.axis()
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
        sink.finish(f"{position}.png")

    sink.close()
