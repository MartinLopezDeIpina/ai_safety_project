"""
Evaluate the judge (the config model) against a hand-labeled ground-truth set.
=============================================================================

Runs the EXACT judge used in 00_collect_behaviors.py — model_utils.judge_accepted_batch,
loaded from config.MODEL_NAME via load_model() — over a set of (text, response) pairs
that a human has labeled ACCEPT / REFUSE, and reports how well the judge agrees.

The judge is always the model configured in config.py (currently Qwen2.5-1.5B), NOT
whatever model generated the responses. The ground-truth file's folder just names the
source of the responses (e.g. qwen7b/); we are measuring how well the config judge
labels those responses.

Usage (from src/experiments_replication/):
    python responses_judging/eval_judge.py                       # default qwen7b set
    python responses_judging/eval_judge.py path/to/ground_truth.json
    python responses_judging/eval_judge.py --batch-size 16

Outputs (next to this script, in responses_judging/):
    judge-<size>_on_<set>_results.json    confusion matrix + metrics + misclassified rows
    judge-<size>_on_<set>_confusion.png   2x2 confusion-matrix heatmap
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)          # src/experiments_replication
sys.path.insert(0, _PARENT)

from config import (  # noqa: E402
    JUDGE_MODEL_NAME, JUDGE_THINKING, JUDGE_MAX_NEW_TOKENS, JUDGE_BATCH_SIZE,
)
from model_utils import load_judge_model, judge_accepted_batch            # noqa: E402

LABELS = ["ACCEPT", "REFUSE"]   # positive class for judge-quality metrics = REFUSE
DEFAULT_GT = os.path.join(_HERE, "qwen7b", "ground_truth.json")


def confusion(gt, pred):
    """2x2 counts, rows = ground truth, cols = judge prediction, order = LABELS."""
    idx = {lab: i for i, lab in enumerate(LABELS)}
    cm = np.zeros((2, 2), dtype=int)
    for g, p in zip(gt, pred):
        cm[idx[g], idx[p]] += 1
    return cm


def print_confusion(cm):
    total = cm.sum()
    acc = np.trace(cm) / total if total else 0.0
    print("\n=== Confusion matrix (rows = human label, cols = judge) ===")
    print(f"{'':>14}{'judge ACCEPT':>15}{'judge REFUSE':>15}")
    for i, lab in enumerate(LABELS):
        print(f"{'human ' + lab:>14}{cm[i, 0]:>15}{cm[i, 1]:>15}")
    print(f"\nAccuracy: {acc:.3f}  ({int(np.trace(cm))}/{int(total)})")

    # Per-class precision / recall / F1.
    print(f"\n{'class':>8}{'precision':>12}{'recall':>10}{'f1':>8}{'support':>10}")
    metrics = {}
    for i, lab in enumerate(LABELS):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        metrics[lab] = {"precision": prec, "recall": rec, "f1": f1,
                        "support": int(cm[i, :].sum())}
        print(f"{lab:>8}{prec:>12.3f}{rec:>10.3f}{f1:>8.3f}{int(cm[i, :].sum()):>10}")
    return acc, metrics


def save_png(cm, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=[f"judge\n{l}" for l in LABELS])
    ax.set_yticks([0, 1], labels=[f"human\n{l}" for l in LABELS])
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=14)
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ground_truth", nargs="?", default=DEFAULT_GT)
    ap.add_argument("--batch-size", type=int, default=JUDGE_BATCH_SIZE)
    args = ap.parse_args()

    with open(args.ground_truth) as f:
        data = json.load(f)
    data = [d for d in data if d.get("label") in LABELS]
    gt = [d["label"] for d in data]

    set_name = os.path.basename(os.path.dirname(os.path.abspath(args.ground_truth)))
    judge_tag = JUDGE_MODEL_NAME.split("/")[-1]
    print(f"Judge model (from config): {JUDGE_MODEL_NAME}  (thinking={JUDGE_THINKING})")
    print(f"Ground truth: {args.ground_truth}")
    print(f"Evaluating on {len(data)} labeled responses "
          f"({gt.count('ACCEPT')} ACCEPT / {gt.count('REFUSE')} REFUSE)")

    model, tok = load_judge_model()
    verdicts = judge_accepted_batch(
        model, tok,
        [(d["text"], d["response"]) for d in data],
        max_new_tokens=JUDGE_MAX_NEW_TOKENS,
        batch_size=args.batch_size,
        thinking=JUDGE_THINKING,
    )
    pred = ["ACCEPT" if v else "REFUSE" for v in verdicts]

    cm = confusion(gt, pred)
    acc, metrics = print_confusion(cm)

    # Misclassified rows, for hand auditing.
    wrong = [
        {"id": d.get("id"), "dataset": d.get("dataset"), "human": g, "judge": p,
         "text": d["text"], "response": d["response"]}
        for d, g, p in zip(data, gt, pred) if g != p
    ]
    print(f"\nMisclassified: {len(wrong)} / {len(data)}")
    for w in wrong[:5]:
        resp = " ".join(w["response"].split())
        print(f"  [{w['dataset']}] human={w['human']} judge={w['judge']} | "
              f"{w['text'][:70]}")
        print(f"      R: {resp[:150]}{'…' if len(resp) > 150 else ''}")
    if len(wrong) > 5:
        print(f"  … +{len(wrong) - 5} more (see results json)")

    base = os.path.join(_HERE, f"judge-{judge_tag}_on_{set_name}")
    with open(base + "_results.json", "w") as f:
        json.dump({
            "judge_model": JUDGE_MODEL_NAME,
            "thinking": JUDGE_THINKING,
            "ground_truth": os.path.abspath(args.ground_truth),
            "n": len(data),
            "labels": LABELS,
            "confusion_matrix": {"rows": "human", "cols": "judge",
                                 "order": LABELS, "counts": cm.tolist()},
            "accuracy": acc,
            "per_class": metrics,
            "misclassified": wrong,
        }, f, indent=2, ensure_ascii=False)
    save_png(cm, base + "_confusion.png",
             f"Judge {judge_tag} on {set_name}  (acc={acc:.2f}, n={len(data)})")
    print(f"\nSaved results to {base}_results.json")
    print(f"Saved confusion matrix to {base}_confusion.png")


if __name__ == "__main__":
    main()
