"""Deeper diagnostics: source breakdown + non-circular anchor test (no GPU).

- split-half test: split refused_harmful, use half as anchor, project the other half
  (removes the circularity of scoring the anchor against its own mean).
- per-source breakdown of tinst_accepted_harmful (advbench / jbb / sorrybench), using the
  fact that the .pt rows are in the same order as the bucket JSON, and each source has
  distinguishing JSON keys.

Usage: python diagnose_fig2b.py qwen 7b
"""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PIDX = {"tinst": 1, "tpost": -1}


def _dirs(model, size):
    base = os.path.join(HERE, "output", f"{model}{size}")
    return os.path.join(base, "buckets_activations"), os.path.join(base, "buckets")


def _load(acts_dir, name):
    p = os.path.join(acts_dir, name + ".pt")
    return torch.load(p, map_location="cpu").float() if os.path.exists(p) else None


def _cos(hidden, center):
    center = center[:, None, :]
    return (hidden * center).sum(-1) / (hidden.norm(dim=-1) * center.norm(dim=-1))


def _score(tensor, mu_rh, mu_ah, pidx):
    h = tensor[:, :, pidx, :]
    return _cos(h, mu_rh) - _cos(h, mu_ah)  # (L,N)


def _source_of(rec):
    if "question_id" in rec or "turns" in rec or "prompt_style" in rec:
        return "sorrybench"
    if "source" in rec or "category" in rec:
        return "jbb"
    if "instruction" in rec:
        return "alpaca"
    if "prompt" in rec and "focus" in rec:
        return "xstest"
    return "advbench"


def main(model, size):
    acts_dir, buckets_dir = _dirs(model, size)
    rh = _load(acts_dir, "tinst_refused_harmful")
    ah = _load(acts_dir, "accepted_harmless")
    acc = _load(acts_dir, "tinst_accepted_harmful")
    L = rh.shape[0]
    probe = [L // 4, L // 2, (3 * L) // 4, L - 1]

    # ---- non-circular split-half test at tinst ----
    print("=== split-half anchor test (tinst) ===")
    N = rh.shape[1]
    half = N // 2
    mu_rh_a = rh[:, :half, PIDX["tinst"], :].mean(1)
    mu_ah = ah[:, :, PIDX["tinst"], :].mean(1)
    held = rh[:, half:, :, :]  # held-out refused_harmful (definitely harmful)
    s = _score(held, mu_rh_a, mu_ah, PIDX["tinst"])
    print("  held-out refused_harmful (should be >0):",
          "  ".join(f"L{l}:{s.mean(1)[l]:+.3f}(p+{(s>0).float().mean(1)[l]:.2f})" for l in probe))

    # ---- per-source breakdown of tinst_accepted_harmful ----
    with open(os.path.join(buckets_dir, "tinst_accepted_harmful.json"), encoding="utf-8") as f:
        recs = json.load(f)
    assert len(recs) == acc.shape[1], f"json {len(recs)} != acts {acc.shape[1]}"
    srcs = [_source_of(r) for r in recs]
    mu_rh = rh[:, :, PIDX["tinst"], :].mean(1)
    print("\n=== tinst_accepted_harmful per-source score (tinst, should be >0 if harmful) ===")
    from collections import Counter
    print("  source counts:", dict(Counter(srcs)))
    for src in ["advbench", "jbb", "sorrybench"]:
        idx = [i for i, s_ in enumerate(srcs) if s_ == src]
        if not idx:
            continue
        sub = acc[:, idx, :, :]
        s = _score(sub, mu_rh, mu_ah, PIDX["tinst"])
        print(f"  {src:10s} (n={len(idx):4d}):",
              "  ".join(f"L{l}:{s.mean(1)[l]:+.3f}(p+{(s>0).float().mean(1)[l]:.2f})" for l in probe))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "qwen",
         sys.argv[2] if len(sys.argv) > 2 else "7b")
