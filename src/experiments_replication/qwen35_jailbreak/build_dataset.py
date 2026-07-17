"""Turn accepted GCG suffixes into a pipeline-ready harmful-accepted dataset.

The Qwen3.5 thinking pipeline builds prompts with ``formatInp_thinking``, which inserts ``bad_q``
verbatim and has no ``adv_suffix`` field. So — exactly like the existing ``data/gcg-advsuffix.json`` —
we inline the suffix into ``bad_q`` (``bad_q = query + suffix``, no separator, matching where
``{optim_str}`` sat during optimization). Running the standard non-thinking pipeline over the output
file then reproduces the accepted completions and yields the harmful_accepted rows/activations.

Usage:
  python build_dataset.py [--in results/gennothink_suffixes.jsonl] \
                          [--out ../../../data/gcg_qwen35_advbench.json] [--all]
By default only rows with accepted=True are written; --all includes every attempt.
"""
import argparse
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path",
                    default=os.path.join(_HERE, "results", "gennothink_suffixes.jsonl"))
    ap.add_argument("--out", dest="out_path",
                    default=os.path.join(_ROOT, "data", "gcg_qwen35_advbench.json"))
    ap.add_argument("--all", action="store_true",
                    help="include every attempt, not just accepted ones")
    args = ap.parse_args()

    n_in = n_out = 0
    with open(args.in_path, encoding="utf-8") as f, open(args.out_path, "w", encoding="utf-8") as g:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            rec = json.loads(line)
            if not args.all and not rec.get("accepted"):
                continue
            row = {"bad_q": rec["query"] + rec["suffix"], "target": rec["target"]}
            g.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"read {n_in} attempts -> wrote {n_out} rows to {args.out_path}")


if __name__ == "__main__":
    main()
