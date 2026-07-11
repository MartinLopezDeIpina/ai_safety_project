"""WS0: CPU diagnostic on the NEW activation store, consuming dynamic_bucket_formation.gen_buckets.

Reports, per token position and for a given bucket config:
  - bucket sizes (train / test) for every cluster,
  - anchor self-scores (sanity: refused_harmful should be >0, accepted_harmless <0),
  - the two Figure-2 lines' per-layer score s^l = cos(h, mu_refused_harmful) - cos(h, mu_accepted_harmless)
    at probe layers, with the positive-fraction p+ = mean(s>0) across examples.

This is the instrument used to EXPLAIN a config rather than eyeballing the PNG. The paper pattern:
  tinst  -> coral (accepted_harmful) p+ ~1.0, teal (refused_harmless) p+ ~0.0
  tpost  -> flip: coral p+ -> 0.0, teal p+ -> 1.0 (esp. late layers)

Usage: python diagnose_buckets.py qwen 7b [bucket_config.json]
"""

import os
import sys

import torch

from dynamic_bucket_formation import gen_buckets

POSITIONS = ["tinst", "tpost"]


def _cos(hidden, center):
    center = center[:, None, :]
    return (hidden * center).sum(-1) / (hidden.norm(dim=-1) * center.norm(dim=-1))


def _score(line, mu_rh, mu_ah):
    return _cos(line, mu_rh) - _cos(line, mu_ah)  # (L, N)


def _fmt(score, probe):
    return "  ".join(
        f"L{l}:{score.mean(1)[l]:+.3f}(p+{(score > 0).float().mean(1)[l]:.2f})" for l in probe
    )


def main(model, size, bucket_config=None):
    buckets = gen_buckets(model, size, bucket_config)
    train, test = buckets["train"], buckets["test"]

    print(f"=== {model}{size}  config={bucket_config or 'MODULE-DEFAULT'} ===\n")
    print("cluster sizes (train / test):")
    for name in sorted(set(train) | set(test)):
        ntr = train[name].shape[1] if name in train else 0
        nte = test[name].shape[1] if name in test else 0
        print(f"  {name:26s}: {ntr:5d} / {nte:5d}")

    for pos in POSITIONS:
        rh = train.get(f"refused_harmful_{pos}")
        ah = train.get(f"accepted_harmless_{pos}")
        if rh is None or ah is None:
            print(f"\n[{pos}] missing anchors, skipping")
            continue
        mu_rh, mu_ah = rh.mean(1), ah.mean(1)
        L = mu_rh.shape[0]
        probe = [L // 4, L // 2, (3 * L) // 4, L - 1]
        print(f"\n[{pos}] probe layers {probe} (L={L})")

        # anchor self-scores (sanity)
        print("  anchor self-score refused_harmful (want >0):", _fmt(_score(rh, mu_rh, mu_ah), probe))
        print("  anchor self-score accepted_harmless(want <0):", _fmt(_score(ah, mu_rh, mu_ah), probe))

        # the two plotted lines (TEST split)
        coral = test.get(f"accepted_harmful_{pos}")
        teal = test.get(f"refused_harmless_{pos}")
        if coral is not None and coral.shape[1]:
            print(f"  coral accepted_harmful (n={coral.shape[1]:4d}):", _fmt(_score(coral, mu_rh, mu_ah), probe))
        if teal is not None and teal.shape[1]:
            print(f"  teal  refused_harmless (n={teal.shape[1]:4d}):", _fmt(_score(teal, mu_rh, mu_ah), probe))


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen"
    size = sys.argv[2] if len(sys.argv) > 2 else "7b"
    cfg = sys.argv[3] if len(sys.argv) > 3 else None
    main(model, size, cfg)
