"""Diagnostics for Figure 2 replication (no GPU).

Loads a model's buckets_activations/*.pt and reports, per token position:
  - bucket sizes (N)
  - anchor self-scores (sanity): refused_harmful vs accepted_harmless
  - the two plotted lines' s^l at a few layers, and per-layer sign fractions
  - contamination check: does refused_harmless sit closer to refused_harmful at tpost?
Also peeks at bucket JSONs for obvious mislabels.

Usage: python diagnose_fig2.py qwen 7b
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
POSITION_INDEX = {"tinst": 1, "tpost": -1}

BUCKETS = {
    "tinst": {
        "anchor_refused_harmful": "tinst_refused_harmful",
        "anchor_accepted_harmless": "accepted_harmless",
        "line_accepted_harmful": "tinst_accepted_harmful",
        "line_refused_harmless": "refused_harmless",
    },
    "tpost": {
        "anchor_refused_harmful": "tpost_refused_harmful",
        "anchor_accepted_harmless": "accepted_harmless",
        "line_accepted_harmful": "tpost_accepted_harmful",
        "line_refused_harmless": "refused_harmless",
    },
}


def _dirs(model, size):
    base = os.path.join(HERE, "output", f"{model}{size}")
    return os.path.join(base, "buckets_activations"), os.path.join(base, "buckets")


def _load(acts_dir, name):
    path = os.path.join(acts_dir, name + ".pt")
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location="cpu").float()


def _cosine(hidden, center):
    center = center[:, None, :]
    num = (hidden * center).sum(-1)
    den = hidden.norm(dim=-1) * center.norm(dim=-1)
    return num / den


def _scores(tensor, mu_rh, mu_ah, pidx):
    """Return per-example, per-layer s^l -> (L, N)."""
    h = tensor[:, :, pidx, :]
    return _cosine(h, mu_rh) - _cosine(h, mu_ah)


def main(model, size):
    acts_dir, buckets_dir = _dirs(model, size)
    print(f"=== {model}{size} ===  acts_dir={acts_dir}\n")

    tensors = {}
    for name in ["tinst_refused_harmful", "tinst_accepted_harmful",
                 "tpost_refused_harmful", "tpost_accepted_harmful",
                 "accepted_harmless", "refused_harmless"]:
        t = _load(acts_dir, name)
        tensors[name] = t
        if t is None:
            print(f"  {name:26s}: MISSING")
        else:
            print(f"  {name:26s}: shape={tuple(t.shape)}  N={t.shape[1]}")
    print()

    for position in ["tinst", "tpost"]:
        names = BUCKETS[position]
        pidx = POSITION_INDEX[position]
        rh = tensors[names["anchor_refused_harmful"]]
        ah = tensors[names["anchor_accepted_harmless"]]
        mu_rh = rh[:, :, pidx, :].mean(1)   # (L,H)
        mu_ah = ah[:, :, pidx, :].mean(1)
        L = mu_rh.shape[0]
        layers_probe = [L // 4, L // 2, (3 * L) // 4, L - 1]

        print(f"----- position {position} (pidx={pidx}, L={L}) -----")

        # anchor self-scores (sanity): refused_harmful should be >0, accepted_harmless <0
        for label, tname in [("refused_harmful(anchor)", names["anchor_refused_harmful"]),
                             ("accepted_harmless(anchor)", names["anchor_accepted_harmless"]),
                             ("accepted_harmful(line)", names["line_accepted_harmful"]),
                             ("refused_harmless(line)", names["line_refused_harmless"])]:
            t = tensors[tname]
            if t is None:
                print(f"  {label:26s}: MISSING")
                continue
            s = _scores(t, mu_rh, mu_ah, pidx)          # (L,N)
            mean_l = s.mean(1)
            frac_pos = (s > 0).float().mean(1)          # fraction of examples with s>0 per layer
            probe = "  ".join(f"L{l}:{mean_l[l]:+.3f}(p+{frac_pos[l]:.2f})" for l in layers_probe)
            print(f"  {label:26s}: {probe}")
        print()

    # peek at bucket JSONs for contamination
    print("----- bucket JSON peek -----")
    for name in ["tinst_accepted_harmful", "refused_harmless"]:
        path = os.path.join(buckets_dir, name + ".json")
        if not os.path.exists(path):
            print(f"  {name}: MISSING json")
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        print(f"\n  {name}: {len(data)} examples. First 5 ori_output (repr, truncated):")
        for d in data[:5]:
            out = repr(d.get("ori_output", ""))[:160]
            print(f"    - {out}")


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen"
    size = sys.argv[2] if len(sys.argv) > 2 else "7b"
    main(model, size)
