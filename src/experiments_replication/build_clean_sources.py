"""WS2: build cleaned activation sources on CPU (no GPU, paper scripts untouched).

Activations depend only on the instruction+template, so pooling a dataset's _accepted.pt and
_refused.pt reconstitutes its full per-instruction set (rows are in JSON order, verified by
shape). This module re-labels that pool with improved_classifier.classify and writes new source
tensors <dataset>_<label>_clean.pt into the activations/ dir, which existing configs can then
reference without any change to dynamic_bucket_formation.build_splits.

Primary use: xstest_refused_clean = genuine over-refusals only (drops the ~8 empty artifacts and
~8 informational/empathetic "sorry" false-refuses that contaminate the refused_harmless bucket).

Usage: python build_clean_sources.py qwen 7b
"""

import json
import os
import sys
from collections import Counter, defaultdict

import torch

from improved_classifier import classify

HERE = os.path.dirname(os.path.abspath(__file__))

# dataset stem -> (list of source files to pool). Each file is <name>.{json,pt} present in
# classified_generations/ and activations/. Pooling accepted+refused reconstitutes the dataset.
POOL = {
    "xstest": ["xstest_accepted", "xstest_refused"],
}


def _dirs(model, model_size):
    base = os.path.join(HERE, "output", f"{model}{model_size}", "datasets_outputs")
    return (os.path.join(base, "classified_generations"),
            os.path.join(base, "activations"))


def build(model, model_size, pool=POOL):
    cls_dir, acts_dir = _dirs(model, model_size)
    for dataset, files in pool.items():
        rows_by_label = defaultdict(list)   # label -> list of (L,T,H) tensors
        counts = Counter()
        for name in files:
            jpath = os.path.join(cls_dir, name + ".json")
            ppath = os.path.join(acts_dir, name + ".pt")
            if not (os.path.exists(jpath) and os.path.exists(ppath)):
                print(f"[skip] {name}: missing json or pt")
                continue
            recs = json.load(open(jpath, encoding="utf-8"))
            acts = torch.load(ppath, map_location="cpu")  # (L,N,T,H), keep dtype
            assert acts.shape[1] == len(recs), f"{name}: {acts.shape[1]} != {len(recs)}"
            for i, rec in enumerate(recs):
                label = classify(rec.get("ori_output", ""))
                rows_by_label[label].append(acts[:, i])  # (L,T,H)
                counts[label] += 1

        for label in ("refused", "accepted"):
            rows = rows_by_label.get(label, [])
            if not rows:
                print(f"[warn] {dataset}_{label}_clean: 0 rows")
                continue
            tensor = torch.stack(rows, dim=1)  # (L, n, T, H)
            out = os.path.join(acts_dir, f"{dataset}_{label}_clean.pt")
            torch.save(tensor, out)
            print(f"saved {os.path.basename(out)}: {tuple(tensor.shape)}")
        print(f"  {dataset} reclassified: {dict(counts)} (empty dropped from clean sources)")


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen"
    model_size = sys.argv[2] if len(sys.argv) > 2 else "7b"
    build(model, model_size)
