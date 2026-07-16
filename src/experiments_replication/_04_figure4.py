"""Figure 4: refusal rate per layer for harmfulness and refusal steering on alpaca.
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

COLORS = ["#E8766C", "#4A6FA5"]  # harmfulness = orange, refusal = blue


def compute_refusal_rate(dir_name: str, file_stem: str) -> list[float]:
    """Compute per-layer refusal rate (%) for one intervention strategy.

    Returns a list of length NUM_LAYERS; NaN for missing layers.
    """
    rates = []
    for layer in range(NUM_LAYERS):
        fname = f"{file_stem}-intervene{layer}.json"
        fpath = os.path.join(OUTPUT_DIR, dir_name, fname)
        if not os.path.exists(fpath):
            print(f"  [warn] missing: {fpath}")
            rates.append(float("nan"))
            continue

        eval_data = read_row(fpath)
        if not eval_data:
            rates.append(float("nan"))
            continue

        scores = easy_eval(eval_data, tag="response")
        valid = [s for s in scores if s != "-1"]
        if not valid:
            rates.append(float("nan"))
            continue
        refused = sum(1 for s in valid if s == "0")
        rates.append(refused / len(valid) * 100)

    return rates


def main():
    parser = argparse.ArgumentParser(
        description="Plot Figure 4: refusal rate per layer for harmfulness and refusal steering on alpaca"
    )
    parser.add_argument("--model", default="qwen7b", type=str,
                        help="Model name used in intervention directory names (e.g. qwen7b, llama2-7b)")
    parser.add_argument("--right", default=100, type=int,
                        help="Number of examples (right slice) used in the experiments (e.g. 50, 100)")
    parser.add_argument("--dataset", default="alpaca_data_instruction", type=str,
                        help="Dataset name (default: alpaca_data_instruction)")
    parser.add_argument("--output-dir", default=None, type=str,
                        help="Override output directory (default: output/<model>/)")
    args = parser.parse_args()

    ds = args.dataset

    strategies = [
        {"label": "harmfulness dir",
         "dir": f"{args.model}-{ds}-hf-more-0-{args.right}-inv0",
         "file_stem": f"{ds}-more"},
        {"label": "refusal dir",
         "dir": f"{args.model}-{ds}-refusal-more-0-{args.right}-inv0",
         "file_stem": f"{ds}-more"},
    ]

    layers = list(range(NUM_LAYERS))
    fig, ax = plt.subplots(figsize=(7, 5))

    for i, strat in enumerate(strategies):
        print(f"Processing {strat['label']}...")
        rates = compute_refusal_rate(strat["dir"], strat["file_stem"])
        ax.plot(layers, rates, color=COLORS[i], label=strat["label"],
                linewidth=2, marker="o", markersize=3)

    ax.set_title("Alpaca — harmfulness vs. refusal steering", fontsize=13)
    ax.set_xlabel("Intervention layer", fontsize=12)
    ax.set_ylabel("Refusal rate (%)", fontsize=12)
    ax.set_xlim(-0.5, NUM_LAYERS - 0.5)
    ax.set_xticks(range(0, NUM_LAYERS, 4))
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    fig.tight_layout()

    figure_dir = args.output_dir or os.path.join(PIPELINE_OUTPUT, args.model)
    os.makedirs(figure_dir, exist_ok=True)
    out_path = os.path.join(figure_dir, f"figure4_{args.model}_{args.right}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()