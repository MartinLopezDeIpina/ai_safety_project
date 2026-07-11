"""WS1: audit the paper's substring accept/refuse labels against improved_classifier (CPU).

Non-destructive: reads datasets_outputs/classified_generations/ (the paper's easy_eval split) and
re-labels every response with improved_classifier.classify. Reports the disagreements that matter
for the buckets:

  - FALSE-ACCEPT : paper said accepted, improved says refused  (a missed refusal leaking into an
                   accepted-harmful bucket). This is the ~10-15% the user asked about.
  - FALSE-REFUSE : paper said refused, improved says accepted  (an informational/empathetic reply
                   that merely contains "sorry", polluting the refused_harmless bucket).
  - EMPTY-AS-REFUSED : paper said refused but the response is empty (a generation artifact).

Writes output/<model>/audit_report.json (counts + example lists) and prints a summary table. The
paper's eval.py is never modified.

Usage: python audit_classification.py qwen 7b
"""

import glob
import json
import os
import sys
from collections import defaultdict

from src.experiments_replication.diagnose.improved_classifier import classify, strip_special

HERE = os.path.dirname(os.path.abspath(__file__))


def _paper_label(stem):
    """The easy_eval label carried by the filename: *_accepted -> 'accepted', *_refused -> 'refused'."""
    if stem.endswith("_accepted"):
        return "accepted"
    if stem.endswith("_refused"):
        return "refused"
    return None


def audit(model, model_size, max_examples=8):
    cls_dir = os.path.join(HERE, "output", f"{model}{model_size}",
                           "datasets_outputs", "classified_generations")
    report = {"model": f"{model}{model_size}", "per_file": {}, "totals": defaultdict(int)}

    for path in sorted(glob.glob(os.path.join(cls_dir, "*.json"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        paper = _paper_label(stem)
        if paper is None:
            continue
        recs = json.load(open(path, encoding="utf-8"))
        false_accept, false_refuse, empty_as_refused = [], [], []
        for r in recs:
            imp = classify(r.get("ori_output", ""))
            text = strip_special(r.get("ori_output", ""))
            q = r.get("prompt") or r.get("bad_q") or r.get("instruction") or ""
            item = {"q": q[:120], "a": text[:200]}
            if paper == "accepted" and imp == "refused":
                false_accept.append(item)
            elif paper == "refused" and imp == "accepted":
                false_refuse.append(item)
            elif paper == "refused" and imp == "empty":
                empty_as_refused.append(item)

        n = len(recs)
        fa, fr, em = len(false_accept), len(false_refuse), len(empty_as_refused)
        if fa or fr or em:
            report["per_file"][stem] = {
                "n": n, "paper_label": paper,
                "false_accept": fa, "false_refuse": fr, "empty_as_refused": em,
                "false_accept_pct": round(100 * fa / n, 1) if paper == "accepted" and n else 0.0,
                "contaminated_pct": round(100 * (fr + em) / n, 1) if paper == "refused" and n else 0.0,
                "examples": {
                    "false_accept": false_accept[:max_examples],
                    "false_refuse": false_refuse[:max_examples],
                    "empty_as_refused": empty_as_refused[:max_examples],
                },
            }
        report["totals"]["false_accept"] += fa
        report["totals"]["false_refuse"] += fr
        report["totals"]["empty_as_refused"] += em

    report["totals"] = dict(report["totals"])
    out = os.path.join(HERE, "output", f"{model}{model_size}", "audit_report.json")
    json.dump(report, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    # ---- summary table ----
    print(f"=== classification audit: {model}{model_size} ===")
    print(f"{'file':32s} {'n':>4s} {'paper':>9s} {'FALSE-ACC':>9s} {'FALSE-REF':>9s} {'EMPTY':>6s}")
    for stem, d in report["per_file"].items():
        tag = f"{d['false_accept_pct']}%" if d["paper_label"] == "accepted" else f"{d['contaminated_pct']}%"
        print(f"{stem:32s} {d['n']:>4d} {d['paper_label']:>9s} "
              f"{d['false_accept']:>9d} {d['false_refuse']:>9d} {d['empty_as_refused']:>6d}  {tag}")
    t = report["totals"]
    print(f"\nTOTAL false-accept={t['false_accept']}  false-refuse={t['false_refuse']}  "
          f"empty-as-refused={t['empty_as_refused']}")
    print(f"wrote {out}")
    return report


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen"
    model_size = sys.argv[2] if len(sys.argv) > 2 else "7b"
    audit(model, model_size)
