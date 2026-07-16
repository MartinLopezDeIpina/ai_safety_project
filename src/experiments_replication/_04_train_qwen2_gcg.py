"""Train per-prompt nanoGCG suffixes for Qwen2-7B-Instruct.

Outputs a JSON array compatible with _04_4_jailbreak_plot.py. Each row keeps the
original harmful request in source_bad_q, the optimized suffix in adv_suffix, and
the concatenated prompt in bad_q.
"""

import argparse
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import nanogcg
    from nanogcg import GCGConfig
except ImportError as exc:
    raise SystemExit(
        "nanogcg is not installed. Install it in the active env with "
        "`pip install nanogcg`, or submit the sbatch wrapper with INSTALL_NANOGCG=1."
    ) from exc


HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(HERE)
REPO_DIR = os.path.dirname(SRC_DIR)


def _read_rows(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if text.lstrip().startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _write_pretty(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def _target_for(row):
    if row.get("target"):
        return row["target"].strip() + "\n\n"
    prompt = row.get("bad_q") or row.get("instruction") or row.get("question") or ""
    return "Sure, here is a detailed response to your request:\n\n"


def train_suffixes(args):
    rows = _read_rows(args.input)
    rows = rows[max(args.left, 0): min(args.right, len(rows))]

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    config = GCGConfig(
        num_steps=args.num_steps,
        optim_str_init=args.optim_str_init,
        search_width=args.search_width,
        batch_size=args.batch_size,
        topk=args.topk,
        n_replace=args.n_replace,
        buffer_size=args.buffer_size,
        early_stop=args.early_stop,
        filter_ids=args.filter_ids,
        allow_non_ascii=args.allow_non_ascii,
        add_space_before_target=args.add_space_before_target,
        seed=args.seed,
        verbosity=args.verbosity,
    )

    out_rows = []
    for local_idx, row in enumerate(rows):
        source_bad_q = row.get("bad_q") or row.get("instruction") or row.get("question")
        if not source_bad_q:
            print(f"[skip] row {local_idx}: no prompt field", flush=True)
            continue

        target = _target_for(row)
        messages = [{"role": "user", "content": source_bad_q + " {optim_str}"}]

        print(f"[{local_idx + 1}/{len(rows)}] training GCG suffix", flush=True)
        result = nanogcg.run(model, tokenizer, messages, target, config)
        suffix = result.best_string
        print(f"  best_loss={getattr(result, 'best_loss', None)} suffix={suffix!r}", flush=True)

        out_rows.append({
            "bad_q": source_bad_q + " " + suffix,
            "source_bad_q": source_bad_q,
            "adv_suffix": suffix,
            "target": target,
            "gcg_model_id": args.model_id,
            "gcg_num_steps": args.num_steps,
            "gcg_search_width": args.search_width,
            "gcg_topk": args.topk,
            "gcg_seed": args.seed,
            "gcg_filter_ids": args.filter_ids,
        })
        _write_pretty(args.output, out_rows)

    print(f"saved {len(out_rows)} suffixes -> {args.output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=os.path.join(REPO_DIR, "data", "advbench.json"))
    parser.add_argument("--output", default=os.path.join(REPO_DIR, "data", "qwen2-gcg-advsuffix.json"))
    parser.add_argument("--model_id", default="Qwen/Qwen2-7B-Instruct")
    parser.add_argument("--left", type=int, default=0)
    parser.add_argument("--right", type=int, default=20)
    parser.add_argument("--num_steps", type=int, default=80)
    parser.add_argument("--search_width", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--n_replace", type=int, default=1)
    parser.add_argument("--buffer_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--optim_str_init", default="! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !")
    parser.add_argument("--filter_ids", type=int, default=0)
    parser.add_argument("--allow_non_ascii", type=int, default=0)
    parser.add_argument("--add_space_before_target", type=int, default=0)
    parser.add_argument("--early_stop", action="store_true")
    parser.add_argument("--verbosity", default="INFO")
    args = parser.parse_args()
    args.filter_ids = bool(args.filter_ids)
    args.allow_non_ascii = bool(args.allow_non_ascii)
    args.add_space_before_target = bool(args.add_space_before_target)
    train_suffixes(args)


if __name__ == "__main__":
    main()
