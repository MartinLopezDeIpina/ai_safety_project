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
    RESULTS_DIR, N_TRAIN, POS_TINST, POS_TPOSTINST, SRC_DIR, TPOST_FIRST_GEN_TOKEN,
    EXTRACT_CATEGORIES, EXTRACT_TINST_ONLY,
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
    if pos_name == "tpostinst" and TPOST_FIRST_GEN_TOKEN:
        from model_utils import extract_first_gen_token_states
        print("    (t_post = FIRST GENERATED token)")
        all_acts = extract_first_gen_token_states(model, tokenizer, data_dicts)
    else:
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

    print("\nLoading model …")
    model, tokenizer = load_model()

    # ------------------------------------------------------------------
    # Extract every configured bucket (config.EXTRACT_CATEGORIES) at BOTH positions,
    # capped at N_TRAIN each. acts[(cat, pos)] -> [n_layers, N, window, hidden].
    # ------------------------------------------------------------------
    acts = {}
    for pos_name, positions in (("tinst", POS_TINST), ("tpostinst", POS_TPOSTINST)):
        print(f"\n--- Extracting at {pos_name} ---")
        for cat in EXTRACT_CATEGORIES:
            # t_inst-only buckets (e.g. the no-post-inst accepted-harmful set) have no
            # post-instruction tokens, so t_post is meaningless for them — skip it.
            if pos_name == "tpostinst" and cat in EXTRACT_TINST_ONLY:
                continue
            data = behaviors.get(cat, [])
            if not data:
                print(f"  {cat}: empty — skipped")
                continue
            acts[(cat, pos_name)] = extract_for_category(
                model, tokenizer, data, cat, pos_name, positions)

    # ------------------------------------------------------------------
    # Standard cluster centers + directions (consumed by 03-08). Guarded so a
    # re-routed config that omits a standard bucket doesn't crash 01.
    # ------------------------------------------------------------------
    print("\n--- Computing cluster centers ---")

    def center(cat, pos):
        t = acts.get((cat, pos))
        return cluster_center(t) if t is not None else None

    def pooled_center(cats, pos):
        ts = [acts[(c, pos)] for c in cats if (c, pos) in acts]
        return cluster_center(torch.cat(ts, dim=1)) if ts else None

    μ_refused_harmful_tinst   = center("refused_harmful",   "tinst")
    μ_accepted_harmless_tinst = center("accepted_harmless", "tinst")
    # Refusal-axis (pooled) clusters use BOTH refused / BOTH accepted at tpost-inst.
    μ_refused_tpostinst  = pooled_center(["refused_harmful", "refused_harmless"],   "tpostinst")
    μ_accepted_tpostinst = pooled_center(["accepted_harmless", "accepted_harmful"], "tpostinst")

    for name, tensor in [
        ("refused_harmful_tinst",   μ_refused_harmful_tinst),
        ("accepted_harmless_tinst", μ_accepted_harmless_tinst),
        ("refused_tpostinst",       μ_refused_tpostinst),
        ("accepted_tpostinst",      μ_accepted_tpostinst),
    ]:
        if tensor is None:
            print(f"  cluster-{name}: SKIPPED (source bucket empty)")
            continue
        torch.save(tensor, os.path.join(RESULTS_DIR, f"cluster-{name}.pt"))
        print(f"  cluster-{name}: {tensor.shape}")

    # ------------------------------------------------------------------
    # Directions
    # ------------------------------------------------------------------
    if μ_refused_harmful_tinst is not None and μ_accepted_harmless_tinst is not None:
        dir_hf = μ_refused_harmful_tinst - μ_accepted_harmless_tinst
        torch.save(dir_hf, os.path.join(RESULTS_DIR, "dir-hf.pt"))
        print(f"\ndir_hf shape:     {dir_hf.shape}")
    if μ_refused_tpostinst is not None and μ_accepted_tpostinst is not None:
        dir_refuse = μ_refused_tpostinst - μ_accepted_tpostinst
        torch.save(dir_refuse, os.path.join(RESULTS_DIR, "dir-refuse.pt"))
        print(f"dir_refuse shape: {dir_refuse.shape}")
    print("\nAll outputs saved to results/")


if __name__ == "__main__":
    main()
