"""Figure 5: refusal rate across layers for three intervention strategies on each dataset.

Replicates Figure 5 from Zhao et al. (2025). Loads the per-layer intervention JSONs from
intervention_outputs/, classifies each response with easy_eval (mode='inversion'), and
plots refusal rate vs. layer for the three steering variants per dataset.

Harmless panels (alpaca, xstest-harmless):
    harmfulness dir       = hf-more      (steer along hf → model flips to "Certainly" → refusal drops)
    refusal dir           = refusal-more (steer along refusal → model keeps "No")
    reverse refusal dir   = refusal-less (reverse refusal → model flips to "Certainly")
Harmful panels (advbench, jbb):
    reverse harmfulness dir = hf-less     (reverse hf → model replies "No" → refusal rises)
    refusal dir             = refusal-more (steer along refusal → model replies "No")
    reverse refusal dir     = refusal-less (reverse refusal → fails to elicit refusal)

Output: output/<model>/figure5_refusal_rate.png (model inferred from --model)

Usage:
    # Default: all four datasets, 50-example run
    uv run python _05_figure5.py

    # 500-example run
    uv run python _05_figure5.py --right 500

    # Only specific datasets
    uv run python _05_figure5.py --harmless "alpaca_data_instruction" --harmful "advbench"

    # Different model
    uv run python _05_figure5.py --model llama2-7b --right 500
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt

# src/ must be on the path for `from eval import easy_eval` and `from utils import read_row`.
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from eval import easy_eval
from utils import read_row

OUTPUT_DIR = os.path.join(HERE, "intervention_outputs")
PIPELINE_OUTPUT = os.path.join(HERE, "output")
NUM_LAYERS = 28

# Harmless panel order: harmfulness, refusal, reverse refusal → orange, blue, teal.
HARMLESS_COLORS = ["#E8766C", "#4A6FA5", "#158A8A"]
# Harmful panel order: reverse harmfulness, refusal, reverse refusal → orange, teal, blue.
HARMFUL_COLORS = ["#E8766C", "#158A8A", "#4A6FA5"]


def _harmless_strategies(datasets: list[str], model: str, right: int) -> list[dict]:
    """Build the three harmless-instruction strategies, each pooling the given datasets."""
    return [
        {"label": "harmfulness dir",
         "dirs": [f"{model}-{ds}-hf-more-0-{right}" for ds in datasets],
         "file_stems": [f"{ds}-more" for ds in datasets]},
        {"label": "refusal dir",
         "dirs": [f"{model}-{ds}-refusal-more-0-{right}" for ds in datasets],
         "file_stems": [f"{ds}-more" for ds in datasets]},
        {"label": "reverse refusal dir",
         "dirs": [f"{model}-{ds}-refusal-less-0-{right}" for ds in datasets],
         "file_stems": [f"{ds}-less" for ds in datasets]},
    ]


def _harmful_strategies(datasets: list[str], model: str, right: int) -> list[dict]:
    """Build the three harmful-instruction strategies, each pooling the given datasets."""
    return [
        {"label": "reverse harmfulness dir",
         "dirs": [f"{model}-{ds}-hf-less-0-{right}" for ds in datasets],
         "file_stems": [f"{ds}-less" for ds in datasets]},
        {"label": "refusal dir",
         "dirs": [f"{model}-{ds}-refusal-more-0-{right}" for ds in datasets],
         "file_stems": [f"{ds}-more" for ds in datasets]},
        {"label": "reverse refusal dir",
         "dirs": [f"{model}-{ds}-refusal-less-0-{right}" for ds in datasets],
         "file_stems": [f"{ds}-less" for ds in datasets]},
    ]


def compute_refusal_rate(dir_names: list[str], file_stems: list[str]) -> list[float]:
    """Compute per-layer refusal rate (%) for one intervention strategy, pooling across datasets.

    dir_names and file_stems are parallel lists — one entry per dataset. For each layer,
    all examples from all datasets are combined before computing the rate.

    Returns a list of length NUM_LAYERS; NaN for missing layers.
    """
    rates = []
    for layer in range(NUM_LAYERS):
        all_valid = []
        for dir_name, file_stem in zip(dir_names, file_stems):
            fname = f"{file_stem}-intervene{layer}.json"
            fpath = os.path.join(OUTPUT_DIR, dir_name, fname)
            if not os.path.exists(fpath):
                print(f"  [warn] missing: {fpath}")
                continue

            eval_data = read_row(fpath)
            if not eval_data:
                continue

            scores = easy_eval(eval_data, tag="response", mode="inversion")
            all_valid.extend(s for s in scores if s != "-1")

        if not all_valid:
            rates.append(float("nan"))
            continue
        refused = sum(1 for s in all_valid if s == "0")
        rates.append(refused / len(all_valid) * 100)

    return rates


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot Figure 5: refusal rate across intervention layers")
    parser.add_argument("--model", default="qwen7b", type=str,
                        help="Model name used in intervention directory names (e.g. qwen7b, llama2-7b)")
    parser.add_argument("--right", default=50, type=int,
                        help="Number of examples (right slice) used in the experiments (e.g. 50, 500)")
    parser.add_argument("--harmless", default="alpaca_data_instruction,xstest-harmless", type=str,
                        help="Comma-separated harmless datasets to plot")
    parser.add_argument("--harmful", default="advbench,jbb", type=str,
                        help="Comma-separated harmful datasets to plot")
    parser.add_argument("--output-dir", default=None, type=str,
                        help="Override output directory (default: output/<model>/)")
    args = parser.parse_args()

    harmless_list = [d.strip() for d in args.harmless.split(",") if d.strip()]
    harmful_list = [d.strip() for d in args.harmful.split(",") if d.strip()]

    # Build two panels: harmless (pooling all harmless datasets) and harmful (pooling all harmful).
    panels = []
    if harmless_list:
        panels.append({
            "title": "(a) Harmless instructions",
            "colors": HARMLESS_COLORS,
            "strategies": _harmless_strategies(harmless_list, args.model, args.right),
        })
    if harmful_list:
        panels.append({
            "title": "(b) Harmful instructions",
            "colors": HARMFUL_COLORS,
            "strategies": _harmful_strategies(harmful_list, args.model, args.right),
        })

    if not panels:
        print("No valid datasets selected. Exiting.")
        return

    figure_dir = args.output_dir or os.path.join(PIPELINE_OUTPUT, args.model)
    os.makedirs(figure_dir, exist_ok=True)

    n = len(panels)
    n_cols = 2
    n_rows = (n + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows),
                             sharey=True, squeeze=False)
    layers = list(range(NUM_LAYERS))

    for idx, panel in enumerate(panels):
        ax = axes[idx // n_cols][idx % n_cols]
        for i, strat in enumerate(panel["strategies"]):
            print(f"Processing {panel['title']} — {strat['label']}...")
            rates = compute_refusal_rate(strat["dirs"], strat["file_stems"])
            ax.plot(layers, rates, color=panel["colors"][i], label=strat["label"],
                    linewidth=2, marker="o", markersize=3)

        ax.set_title(panel["title"], fontsize=13)
        ax.set_xlabel("Intervention layer", fontsize=12)
        ax.set_xlim(-0.5, NUM_LAYERS - 0.5)
        ax.set_xticks(range(0, NUM_LAYERS, 4))
        ax.set_ylim(-5, 105)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    # Hide unused subplots.
    for idx in range(n, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    axes[0][0].set_ylabel("Refusal rate (%)", fontsize=12)
    fig.tight_layout()

    out_path = os.path.join(figure_dir, f"figure5_{args.model}_{args.right}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
