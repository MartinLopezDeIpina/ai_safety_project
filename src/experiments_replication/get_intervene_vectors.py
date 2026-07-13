"""Load cached activations from a directory of .pt files and group them for intervention vectors.

Each .pt file is a tensor of shape (L, N, T, H) = (layers, samples, token_positions, hidden_dim).
Token positions:
    tinst  -> index 1   (last token of the instruction)
    tpost  -> index -1  (last token of the whole prompt)

File naming:
    harmful datasets : advbench, jbb, sorrybench  (with _gentinst/_gentpost suffixes)
    harmless datasets: alpaca, xstest             (single config; tensor holds both positions)

This module exposes load_activations(), which returns a dict with four stacked tensors:
    "harmful_tinst"  : (L, N_harmful, H)  -- harmful instructions at t_inst
    "harmful_tpost"  : (L, N_harmful, H)  -- harmful instructions at t_post
    "harmless_tinst" : (L, N_harmless, H) -- harmless instructions at t_inst
    "harmless_tpost" : (L, N_harmless, H) -- harmless instructions at t_post

Usage:
    from get_intervene_vectors import load_activations
    groups = load_activations("path/to/activations_qwen")

    # or as a script:
    python get_intervene_vectors.py <act_dir> [--out-dir <dir>]
"""

import torch
import argparse
import os

HARMFUL_DATASETS = ("advbench", "jbb", "sorrybench")
HARMLESS_DATASETS = ("alpaca", "xstest")


def collect_acts(act_dir, datasets, position):
    """Stack activations (accepted + refused) for a list of datasets at one token position.

    For harmful datasets the filename carries a _gentinst/_gentpost config suffix matching the
    requested position; harmless datasets have no such suffix (each tensor holds both positions).
    All accepted + refused tensors are concatenated along the sample axis.
    """
    config_suffix = "gentinst" if position == "tinst" else "gentpost"
    token_position = 1 if position == "tinst" else -1
    parts = []
    for dataset in datasets:
        for label in ("accepted", "refused"):
            if dataset in HARMFUL_DATASETS:
                fname = f"{dataset}_{config_suffix}_{label}.pt"
            else:
                fname = f"{dataset}_{label}.pt"
            path = os.path.join(act_dir, fname)
            # loaded tensor has shape (layers, samples, token_positions, hidden_dim)
            t = torch.load(path, map_location="cpu", weights_only=True)
            # slice only the requested token position
            parts.append(t[:, :, token_position, :]) # (layers, samples, hidden_dim)
    if not parts:
        raise ValueError(f"No activation files found in {act_dir} for datasets {datasets} at position {position}")
    # stack along the sample axis
    acts = torch.cat(parts, dim=1) # (layers, total_samples, hidden_dim)
    return acts


def compute_intervene_vectors(act_dir):
    """Compute the harmfulness and refusal intervention directions.

    Harmful direction ("hf"):     mean(harmful_tinst) - mean(harmless_tinst)
    Refusal direction ("refusal"): mean(harmful_tpost) - mean(harmless_tpost)

    Args:
        act_dir: directory containing the .pt activation files.

    Returns:
        dict with keys "hf" and "refusal", each a tensor of shape (n_layers, hidden_dim).
    """
    # acts = load_activations(act_dir)
    acts_harmful_tinst = collect_acts(act_dir, HARMFUL_DATASETS, "tinst")
    acts_harmful_tpost = collect_acts(act_dir, HARMFUL_DATASETS, "tpost")
    acts_harmless_tinst = collect_acts(act_dir, HARMLESS_DATASETS, "tinst")
    acts_harmless_tpost = collect_acts(act_dir, HARMLESS_DATASETS, "tpost")

    hf_dir = acts_harmful_tinst.mean(dim=1) - acts_harmless_tinst.mean(dim=1)
    refusal_dir = acts_harmful_tpost.mean(dim=1) - acts_harmless_tpost.mean(dim=1)

    return {"hf": hf_dir, "refusal": refusal_dir}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load cached activations and group them for intervention vectors.")
    parser.add_argument(
        "--act-dir", help="Directory containing the .pt activation files."
    )
    parser.add_argument(
        "--out-dir", default='steering_vectors/qwen-7b',
        help="Directory to save the intervention vectors as .pt files."
    )
    args = parser.parse_args()
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    vectors = compute_intervene_vectors(args.act_dir)
    for name, tensor in vectors.items():
        print(f"{name}: shape {tuple(tensor.shape)}")
        path = os.path.join(out_dir, f"{name}.pt")
        torch.save(tensor, path)
        print(f"Saved {name} -> {path}")
