"""Run all inference passes, then classify responses into the 6 category pools."""

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(HERE)
REPO_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(REPO_DIR, "data")


def _read_rows(path):
    """Read a list of records from either an indented JSON array or JSONL."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if text.lstrip().startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _write_pretty(path, rows):
    """Write records as an indented JSON array (readable; still loadable by read_row)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def run_inference_on_dataset(
    input_file,
    output_file,
    do_not_use_last_inst_tok,
    *,
    model="qwen",
    model_size="0.5b",
    left=0,
    right=10,
):
    """Invoke src/inference.py on one dataset, writing JSONL to output_file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    cmd = [
        sys.executable,
        "inference.py",
        "--input", input_file,
        "--output_file_name", output_file,
        "--model", model,
        "--model_size", model_size,
        "--left", str(left),
        "--right", str(right),
        "--use_template", "1",
        "--use_jb", "0",
        "--use_inversion", "0",
        "--record_prob_max_pos", "0",
        "--max_len", "128",
        "--do_not_use_last_inst_tok", str(do_not_use_last_inst_tok),
    ]
    # cwd=SRC_DIR so inference.py's `from utils import ...` resolves.
    subprocess.run(cmd, check=True, cwd=SRC_DIR)

    # inference.py writes JSONL; rewrite as an indented array for readability.
    _write_pretty(output_file, _read_rows(output_file))


def _out_dirs(model, model_size):
    """Output dirs for a given model, e.g. output/qwen0.5b/{dataset_outputs,buckets}."""
    out_dir = os.path.join(HERE, "output", f"{model}{model_size}")
    return os.path.join(out_dir, "dataset_outputs"), os.path.join(out_dir, "buckets")


# (dataset file, output name, do_not_use_last_inst_tok)
# do_not_use_last_inst_tok: 1 = t_inst config, 0 = t_post config.
RUNS = [
    # harmful x t_inst
    ("advbench.json", "advbench_tinst.json", 1),
    ("jbb.json", "jbb_tinst.json", 1),
    ("sorry-badq.json", "sorrybench_tinst.json", 1),
    # harmful x t_post
    ("advbench.json", "advbench_tpost.json", 0),
    ("jbb.json", "jbb_tpost.json", 0),
    ("sorry-badq.json", "sorrybench_tpost.json", 0),
    # harmless x t_post (single config)
    ("xstest-harmless.json", "xstest.json", 0),
    ("alpaca_data_instruction.json", "alpaca.json", 0),
]

# pool name -> (source output names, label to keep)
# label: '5' = accepted, '0' = refused (easy_eval refusal mode).
# NOTE (see EXPERIMENT_LOG.md): corrected, paper-faithful bucket sources for Figure 2:
#  - tinst_accepted_harmful uses advbench+jbb only. Sorry-Bench's conversational prompts sit on the
#    harmless side of the advbench/jbb harmfulness axis at tinst and reverse the coral line.
#  - tpost_accepted_harmful uses Sorry-Bench only. advbench/jbb are ~98% refused at tpost, so their
#    "accepted" examples are substring false-negatives (actually refusals) that poison the coral line.
# For the definitive figures we instead re-bucket offline with `rebucket_store.py`, which also excludes
# empty responses and lets the harmless anchor be Alpaca-only (removes the xstest self-similarity
# confound that masks the tpost refusal flip).
POOLS = {
    "tinst_accepted_harmful": (["advbench_tinst.json", "jbb_tinst.json"], "5"),
    "tinst_refused_harmful": (["advbench_tinst.json", "jbb_tinst.json"], "0"),
    "tpost_accepted_harmful": (["sorrybench_tpost.json"], "5"),
    "tpost_refused_harmful": (["advbench_tpost.json", "jbb_tpost.json"], "0"),
    "accepted_harmless": (["xstest.json", "alpaca.json"], "5"),
    "refused_harmless": (["xstest.json"], "0"),
}


def run_all_inference(model, model_size, left, right):
    """Generate responses for all 8 dataset/config combinations."""
    dataset_out_dir, _ = _out_dirs(model, model_size)
    for dataset_file, out_name, no_last_tok in RUNS:
        run_inference_on_dataset(
            input_file=os.path.join(DATA_DIR, dataset_file),
            output_file=os.path.join(dataset_out_dir, out_name),
            do_not_use_last_inst_tok=no_last_tok,
            model=model,
            model_size=model_size,
            left=left,
            right=right,
        )


def evaluate(model, model_size):
    """Classify each response and route it into the 6 category pools."""
    sys.path.insert(0, SRC_DIR)
    from eval import easy_eval

    dataset_out_dir, buckets_dir = _out_dirs(model, model_size)
    os.makedirs(buckets_dir, exist_ok=True)

    # Classify every dataset output once; cache scored items per output name.
    scored = {}
    for _, out_name, _ in RUNS:
        data = _read_rows(os.path.join(dataset_out_dir, out_name))
        labels = easy_eval(data, tag="ori_output", mode="refusal")
        scored[out_name] = list(zip(data, labels))

    for pool_name, (source_names, keep_label) in POOLS.items():
        pool = [
            item
            for name in source_names
            for item, label in scored[name]
            if label == keep_label
        ]
        _write_pretty(os.path.join(buckets_dir, pool_name + ".json"), pool)
        print(f"{pool_name}: {len(pool)} items")
