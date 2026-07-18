"""Convert the GCG attack dump into a pipeline-ready thinking-generations file.

The GCG attack (gcg_attack.py) writes rows of {index, query, suffix, target, ..., generation}. The
qwen35 thinking pipeline instead consumes generations-format rows keyed on `bad_q` (the request) and
`ori_output` (the response), which `evaluate_thinking` / `easy_eval_thinking` and the activation
extraction (`formatInp_thinking`) read.

Crucially, these rows were attacked with `use_jb_prompt=True`, so the user turn the model actually saw
was `jailbreak_prompt.format(target_str=target, goal=query) + suffix` (gcg_attack.py:302-310) — the
jailbreak template with the suffix glued on where `{optim_str}` sat. We store exactly that in `bad_q`
so `formatInp_thinking(row, "gennothink", "qwen35")` reproduces the attack prompt verbatim. (Note:
build_dataset.py inlines `query + suffix` WITHOUT the jailbreak template — wrong for these rows.)

Usage:
  cd src/experiments_replication
  python qwen35_jailbreak/gcg_to_generations.py [--model_size 9b] \
         [--in ../../data/gennothink_gcg_advbench.json]
Writes output/qwen35<size>/datasets_outputs/generations/gcg_advbench_gennothink.json.
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXP = os.path.dirname(_HERE)                    # src/experiments_replication
_REPO = os.path.dirname(os.path.dirname(_EXP))   # repo root

sys.path.insert(0, _HERE)
sys.path.insert(0, _EXP)
from jailbreak_prompt import prompt as jb_template  # noqa: E402
from _00_run_inference import _read_rows, _write_pretty, _out_dirs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_size", default="9b",
                    help="qwen35 size the GCG attack was run at (activations must match). GCG "
                         "targeted Qwen3.5-9B, so default 9b.")
    ap.add_argument("--in", dest="in_path",
                    default=os.path.join(_REPO, "data", "gennothink_gcg_advbench.json"))
    args = ap.parse_args()

    rows = _read_rows(args.in_path)
    out_rows = []
    for r in rows:
        inner = jb_template.format(target_str=r["target"], goal=r["query"])
        out_rows.append({
            "bad_q": inner + r["suffix"],   # jb template + query + suffix == the attacked user turn
            "ori_output": r["generation"],
            "target": r["target"],
        })

    generations_dir, _, _ = _out_dirs("qwen35", args.model_size)
    os.makedirs(generations_dir, exist_ok=True)
    out_path = os.path.join(generations_dir, "gcg_advbench_gennothink.json")
    _write_pretty(out_path, out_rows)
    print(f"read {len(rows)} GCG rows -> wrote {len(out_rows)} generations to {out_path}")


if __name__ == "__main__":
    main()
