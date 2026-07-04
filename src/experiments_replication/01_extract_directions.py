"""
§3.2 (foundation)  —  Extract Hidden States and Directions at tinst and tpost-inst
====================================================================================

Paper setup:
    The paper studies the hidden states at two token positions:
      tinst     = <|im_end|> (position -5 in the full Qwen2 template)
                  Marks the END of the user instruction.
      tpost-inst = \\n after <|im_start|>assistant (position -1)
                  The last input token before the model starts generating.

    From these positions, it computes:
      dir_hf     = μ_refused_harmful[tinst]   − μ_accepted_harmless[tinst]
      dir_refuse = μ_refused[tpost-inst]      − μ_accepted[tpost-inst]

    where μ denotes the mean hidden state (cluster center) over training samples.

What this script does:
    For each of the 4 behavior categories (loaded from behaviors.json):
    1. Runs a forward pass extracting hidden states at POS_TINST and POS_TPOSTINST.
    2. Saves per-sample tensors and cluster centers.
    3. Computes and saves dir_hf and dir_refuse.

Depends on: results/behaviors.json (from 00_collect_behaviors.py)

Outputs (in results/):
    dir-hf.pt              [n_layers, hidden_dim] — harmfulness direction
    dir-refuse.pt          [n_layers, hidden_dim] — refusal direction
    acts_{cat}_{pos}.pt    per-sample [n_layers, N, 3, hidden_dim] for each category × position
    cluster-{name}.pt      [n_layers, hidden_dim] cluster centers
"""

import json
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from config import (
    RESULTS_DIR, N_TRAIN, POS_TINST, POS_TPOSTINST, SRC_DIR
)
from model_utils import load_model, extract_hidden_states

sys.path.insert(0, SRC_DIR)


def load_behaviors(n_per_cat=None):
    """Load behaviors.json and return text dicts for each category."""
    path = os.path.join(RESULTS_DIR, "behaviors.json")
    if not os.path.exists(path):
        raise FileNotFoundError("Run 00_collect_behaviors.py first.")
    with open(path) as f:
        behaviors = json.load(f)

    # Convert text strings back to dicts (format_prompt expects dicts)
    result = {}
    for cat, items in behaviors.items():
        dicts = [{"instruction": it["text"]} for it in items]
        if n_per_cat is not None:
            dicts = dicts[:n_per_cat]
        result[cat] = dicts
    return result


def extract_for_category(model, tokenizer, data_dicts, cat_name, pos_name, positions):
    """
    Extract hidden states for a list of dicts at the specified positions.
    Returns all_acts [n_layers, N, pos_window, hidden_dim].
    Saves to results/acts_{cat_name}_{pos_name}.pt and returns the tensor.
    """
    save_path = os.path.join(RESULTS_DIR, f"acts_{cat_name}_{pos_name}.pt")
    if os.path.exists(save_path):
        print(f"  Loaded cached {save_path}")
        return torch.load(save_path, weights_only=True)

    print(f"  Extracting {cat_name} at {pos_name} ({len(data_dicts)} samples) …")
    _, all_acts = extract_hidden_states(model, tokenizer, data_dicts, positions=positions)
    torch.save(all_acts, save_path)
    return all_acts


def cluster_center(all_acts):
    """
    Compute mean over samples of the target-position activation.

    all_acts: [n_layers, N, pos_window, hidden_dim]
    Returns: [n_layers, hidden_dim]
    """
    # pos_window[-1] = the specified token position (see model_utils.extract_hidden_states)
    return all_acts[:, :, -1, :].float().mean(dim=1)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading behaviors …")
    behaviors = load_behaviors(n_per_cat=N_TRAIN)

    # Merge refused and accepted across categories for the refusal-axis clusters
    refused_all  = behaviors["refused_harmful"]  + behaviors["refused_harmless"]
    accepted_all = behaviors["accepted_harmful"] + behaviors["accepted_harmless"]

    print("\nLoading model …")
    model, tokenizer = load_model()

    # ------------------------------------------------------------------
    # Extract at tinst (POS_TINST = [-5] = <|im_end|>)
    # ------------------------------------------------------------------
    print("\n--- Extracting at tinst ---")
    acts_refused_harmful_tinst   = extract_for_category(model, tokenizer, behaviors["refused_harmful"],  "refused_harmful",  "tinst", POS_TINST)
    acts_accepted_harmless_tinst = extract_for_category(model, tokenizer, behaviors["accepted_harmless"], "accepted_harmless","tinst", POS_TINST)
    acts_accepted_harmful_tinst  = extract_for_category(model, tokenizer, behaviors["accepted_harmful"],  "accepted_harmful", "tinst", POS_TINST)
    acts_refused_harmless_tinst  = extract_for_category(model, tokenizer, behaviors["refused_harmless"],  "refused_harmless", "tinst", POS_TINST)

    # ------------------------------------------------------------------
    # Extract at tpost-inst (POS_TPOSTINST = [-1])
    # ------------------------------------------------------------------
    print("\n--- Extracting at tpost-inst ---")
    acts_refused_harmful_tpostinst   = extract_for_category(model, tokenizer, behaviors["refused_harmful"],  "refused_harmful",  "tpostinst", POS_TPOSTINST)
    acts_accepted_harmless_tpostinst = extract_for_category(model, tokenizer, behaviors["accepted_harmless"], "accepted_harmless","tpostinst", POS_TPOSTINST)
    acts_accepted_harmful_tpostinst  = extract_for_category(model, tokenizer, behaviors["accepted_harmful"],  "accepted_harmful", "tpostinst", POS_TPOSTINST)
    acts_refused_harmless_tpostinst  = extract_for_category(model, tokenizer, behaviors["refused_harmless"],  "refused_harmless", "tpostinst", POS_TPOSTINST)

    # ------------------------------------------------------------------
    # Cluster centers
    # ------------------------------------------------------------------
    print("\n--- Computing cluster centers ---")
    μ_refused_harmful_tinst   = cluster_center(acts_refused_harmful_tinst)   # [n_layers, hidden_dim]
    μ_accepted_harmless_tinst = cluster_center(acts_accepted_harmless_tinst)

    # Refusal-axis clusters use BOTH refused categories at tpost-inst
    if len(behaviors["refused_harmless"]) > 0:
        refused_tpost_all = torch.cat([acts_refused_harmful_tpostinst, acts_refused_harmless_tpostinst], dim=1)
    else:
        refused_tpost_all = acts_refused_harmful_tpostinst
    if len(behaviors["accepted_harmful"]) > 0:
        accepted_tpost_all = torch.cat([acts_accepted_harmless_tpostinst, acts_accepted_harmful_tpostinst], dim=1)
    else:
        accepted_tpost_all = acts_accepted_harmless_tpostinst

    μ_refused_tpostinst  = cluster_center(refused_tpost_all)
    μ_accepted_tpostinst = cluster_center(accepted_tpost_all)

    # Save all four cluster centers
    for name, tensor in [
        ("refused_harmful_tinst",   μ_refused_harmful_tinst),
        ("accepted_harmless_tinst", μ_accepted_harmless_tinst),
        ("refused_tpostinst",       μ_refused_tpostinst),
        ("accepted_tpostinst",      μ_accepted_tpostinst),
    ]:
        p = os.path.join(RESULTS_DIR, f"cluster-{name}.pt")
        torch.save(tensor, p)
        print(f"  cluster-{name}: {tensor.shape}")

    # ------------------------------------------------------------------
    # Directions
    # ------------------------------------------------------------------
    dir_hf     = μ_refused_harmful_tinst - μ_accepted_harmless_tinst
    dir_refuse = μ_refused_tpostinst       - μ_accepted_tpostinst

    torch.save(dir_hf,     os.path.join(RESULTS_DIR, "dir-hf.pt"))
    torch.save(dir_refuse, os.path.join(RESULTS_DIR, "dir-refuse.pt"))
    print(f"\ndir_hf shape:     {dir_hf.shape}")
    print(f"dir_refuse shape: {dir_refuse.shape}")
    print("\nAll outputs saved to results/")


if __name__ == "__main__":
    main()
