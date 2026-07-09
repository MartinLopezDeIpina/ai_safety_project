"""Run the paper's inference.py on a single dataset with given parameters."""

import os
import subprocess
import sys

# Repo layout: this file is at <repo>/src/experiments_replication/
HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(HERE)


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
