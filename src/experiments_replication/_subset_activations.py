"""Subset existing activation tensors to a fixed cap per bucket.

Each activation .pt (one per classified/judged bucket, e.g. advbench_gennothink_stripped_accepted.pt)
is a (L, N, ...) tensor whose dim-1 N is the number of rows (activations) collected for that bucket.
Loading every bucket at full N (up to ~2048) can OOM figure/bucket building — a single bucket can be
>12 GB. This writes a PARALLEL directory keeping only the first N rows of each bucket. The cap is
strictly per file, so buckets are never mixed.

Reads with mmap so a 12 GB tensor is never fully materialised — only the kept slice is copied out.

Going forward the `acts` stage can cap at extraction time via main(..., max_acts_per_bucket=N); this
script is for activations that were already extracted at full size.

Edit the __main__ config block and run:  python _subset_activations.py
"""

import glob
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def _acts_dir(model, model_size, subdir):
    """output/<model><size>/datasets_outputs/<subdir>."""
    return os.path.join(HERE, "output", f"{model}{model_size}", "datasets_outputs", subdir)


def subset_activations(model, model_size, src_subdir="judge_activations",
                       dst_subdir="judged_activations_subset", cap=256):
    """Copy each .pt in src_subdir to dst_subdir keeping only the first `cap` rows (dim 1).

    Buckets are processed one file at a time and read with mmap, so peak memory is one bucket's kept
    slice, not the whole tensor. Files with fewer than `cap` rows are copied whole.
    """
    src = _acts_dir(model, model_size, src_subdir)
    dst = _acts_dir(model, model_size, dst_subdir)
    if not os.path.isdir(src):
        raise FileNotFoundError(f"no source activations dir: {src}")
    os.makedirs(dst, exist_ok=True)

    for p in sorted(glob.glob(os.path.join(src, "*.pt"))):
        name = os.path.basename(p)
        try:
            t = torch.load(p, map_location="cpu", mmap=True)
        except Exception:  # non-mmappable (e.g. legacy format) -> plain load
            t = torch.load(p, map_location="cpu")
        if t.ndim >= 2:                       # (L, N, ...) — N is dim 1
            n = t.shape[1]
            sub = t[:, :cap].clone().contiguous()   # clone: detach from the mmap before saving
        else:                                 # 1-D fallback: treat dim 0 as rows
            n = t.shape[0]
            sub = t[:cap].clone().contiguous()
        torch.save(sub, os.path.join(dst, name))
        print(f"{name}: {tuple(t.shape)} -> {tuple(sub.shape)} (kept {min(cap, n)}/{n})")
        del t, sub

    print(f"\nsubset (cap={cap}) -> {dst}")


if __name__ == "__main__":
    # ---- configure here, then run: python _subset_activations.py ----
    MODEL = "qwen35"
    MODEL_SIZE = "9b"
    SRC_SUBDIR = "judge_activations"          # the already-extracted (judged) activations
    DST_SUBDIR = "judge_activations_subset"  # parallel dir this writes
    CAP = 512                                 # max activations kept per bucket
    # -----------------------------------------------------------------

    subset_activations(MODEL, MODEL_SIZE, SRC_SUBDIR, DST_SUBDIR, CAP)
