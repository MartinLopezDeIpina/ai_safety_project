"""Figure 5: refusal rate across layers for three intervention strategies on each dataset.

Replicates Figure 5 from Zhao et al. (2025). Loads the per-layer intervention JSONs from
intervention_outputs/, classifies each response with easy_eval (mode='inversion'), and
plots refusal rate vs. layer for the three steering variants per dataset.

Panel (a): Harmless instructions (alpaca):
    harmfulness dir       = hf-more      (steer along hf → model flips to "Certainly" → refusal drops)
    refusal dir           = refusal-more (steer along refusal → model keeps "No")
    reverse refusal dir   = refusal-less (reverse refusal → model flips to "Certainly")
Panel (b): Harmful instructions (advbench):
    reverse harmfulness dir = hf-less     (reverse hf → model replies "No" → refusal rises)
    refusal dir             = refusal-more (steer along refusal → model replies "No")
    reverse refusal dir     = refusal-less (reverse refusal → fails to elicit refusal)

Output: output/<model>/figure5_refusal_rate.png (model inferred from intervention dir names)
"""

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

# Paper Figure 5 layout:
#   (a) Harmless instructions (alpaca):
#       harmfulness dir       = hf-more      (steer along hf → model flips to "Certainly" → refusal drops)
#       refusal dir           = refusal-more (steer along refusal → model keeps "No")
#       reverse refusal dir   = refusal-less (reverse refusal → model flips to "Certainly")
#   (b) Harmful instructions (advbench):
#       reverse harmfulness dir = hf-less     (reverse hf → model replies "No" → refusal rises)
#       refusal dir             = refusal-more (steer along refusal → model replies "No")
#       reverse refusal dir     = refusal-less (reverse refusal → fails to elicit refusal)
PANELS = [
    {
        "title": "(a) Harmless instructions",
        "strategies": [
            {"label": "harmfulness dir",     "dir": "qwen7b-alpaca_data_instruction-hf-more-0-50",      "file_stem": "alpaca_data_instruction-more"},
            {"label": "refusal dir",         "dir": "qwen7b-alpaca_data_instruction-refusal-more-0-50", "file_stem": "alpaca_data_instruction-more"},
            {"label": "reverse refusal dir", "dir": "qwen7b-alpaca_data_instruction-refusal-less-0-50", "file_stem": "alpaca_data_instruction-less"},
        ],
    },
    {
        "title": "(b) Harmful instructions",
        "strategies": [
            {"label": "reverse harmfulness dir", "dir": "qwen7b-advbench-hf-less-0-50",      "file_stem": "advbench-less"},
            {"label": "refusal dir",             "dir": "qwen7b-advbench-refusal-more-0-50",  "file_stem": "advbench-more"},
            {"label": "reverse refusal dir",     "dir": "qwen7b-advbench-refusal-less-0-50",  "file_stem": "advbench-less"},
        ],
    },
]

# Paper colors: orange for harmfulness, blue for refusal, teal for reverse refusal.
COLORS = ["#E8766C", "#4A6FA5", "#158A8A"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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

        scores = easy_eval(eval_data, tag="response", mode="inversion")

        # In inversion mode: '5' = accepted (jailbreak), '0' = refused, '-1' = ambiguous.
        # Refusal rate = fraction classified as refused ('0').
        valid = [s for s in scores if s != "-1"]
        if not valid:
            rates.append(float("nan"))
            continue
        refused = sum(1 for s in valid if s == "0")
        rates.append(refused / len(valid) * 100)

    return rates


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def main():
    # Infer the model name (e.g. "qwen7b") from the first intervention directory,
    # so the figure lands in output/<model>/ alongside the generation results.
    first_dir = PANELS[0]["strategies"][0]["dir"]
    model_name = first_dir.split("-")[0]
    figure_dir = os.path.join(PIPELINE_OUTPUT, model_name)
    os.makedirs(figure_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    layers = list(range(NUM_LAYERS))

    for ax_idx, panel in enumerate(PANELS):
        ax = axes[ax_idx]
        for i, strat in enumerate(panel["strategies"]):
            print(f"Processing {panel['title']} — {strat['label']}...")
            rates = compute_refusal_rate(strat["dir"], strat["file_stem"])
            ax.plot(layers, rates, color=COLORS[i], label=strat["label"],
                    linewidth=2, marker="o", markersize=3)

        ax.set_title(panel["title"], fontsize=13)
        ax.set_xlabel("Intervention layer", fontsize=12)
        ax.set_xlim(-0.5, NUM_LAYERS - 0.5)
        ax.set_xticks(range(0, NUM_LAYERS, 4))
        ax.set_ylim(-5, 105)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    axes[0].set_ylabel("Refusal rate (%)", fontsize=12)
    fig.tight_layout()

    out_path = os.path.join(figure_dir, "figure5_refusal_rate.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
