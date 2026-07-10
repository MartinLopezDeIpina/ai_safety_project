"""Compute per-classified-generation activations by calling the paper's extract_hidden.py.

extract_hidden.py computes a mean-diff over two files; with --extract_only 1 it only
extracts (and saves) the first file's activations. We use that to grab one classified
generation's activations per call. The saved tensor (output_pth_harmful) holds activations
at both t_inst and t_post positions, regardless of the generation config the examples came
from (extract_hidden.py always re-tokenizes with the full template).
"""

import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(HERE)
EXTRACT_SCRIPT = os.path.join(SRC_DIR, "extract_hidden.py")


def _dirs(model, model_size):
    """classified_generations/ (input) and activations/ (output) for a given model."""
    base = os.path.join(HERE, "output", f"{model}{model_size}", "datasets_outputs")
    return (
        os.path.join(base, "classified_generations"),
        os.path.join(base, "activations"),
    )


def run_extract_hidden(
    harmful_pth,
    harmless_pth,
    output_pth_harmful,
    output_pth_harmless,
    output_pth,
    extract_only=1,
    *,
    model="qwen",
    model_size="0.5b",
    left=0,
    right=100000,
):
    """Wrapper around extract_hidden.py. With extract_only=1 only harmful_pth is extracted.

    Remaining knobs are left at the paper's get_diff_mean.sh defaults.
    """
    os.makedirs(os.path.dirname(output_pth_harmful), exist_ok=True)
    cmd = [
        sys.executable,
        EXTRACT_SCRIPT,
        "--model", model,
        "--model_size", model_size,
        "--harmful_pth", harmful_pth,
        "--harmless_pth", harmless_pth,
        "--output_pth_harmful", output_pth_harmful,
        "--output_pth_harmless", output_pth_harmless,
        "--output_pth", output_pth,
        "--extract_only", str(extract_only),
        "--extract_hidden_inst_token", "1",
        "--extract_harmful_token_only", "0",
        "--batch_size", "1",
        "--ret_whole_seq", "0",
        "--left", str(left),
        "--right", str(right),
        "--mode_dir", "hf",
    ]
    # cwd=HERE so extract_hidden.py's relative 'output/tmp_*.pt' land here, not in src/.
    subprocess.run(cmd, check=True, cwd=HERE)

    # extract_hidden.py leaves behind: output_pth_harmless (None under extract_only), an
    # output_pth prompts-log, and internal scratch tmp_*.pt in cwd/output. None are useful.
    debris = [
        output_pth_harmless,
        output_pth.replace(".pt", "_prompts_used.json"),
        os.path.join(HERE, "output", "tmp_mean_activations_harmful.pt"),
        os.path.join(HERE, "output", "tmp_full_activations_harmful.pt"),
    ]
    for path in debris:
        if os.path.exists(path):
            os.remove(path)


def compute_all_activations(model, model_size):
    """Extract activations for each classified generation into activations/."""
    classified_dir, acts_dir = _dirs(model, model_size)
    os.makedirs(acts_dir, exist_ok=True)

    for src_pth in sorted(glob.glob(os.path.join(classified_dir, "*.json"))):
        name = os.path.splitext(os.path.basename(src_pth))[0]
        with open(src_pth, encoding="utf-8") as f:
            if not json.load(f):
                print(f"{name}: empty, skipping")
                continue
        run_extract_hidden(
            harmful_pth=src_pth,
            harmless_pth=src_pth,  # read but unused under extract_only=1
            output_pth_harmful=os.path.join(acts_dir, name + ".pt"),
            output_pth_harmless=os.path.join(acts_dir, "_unused_harmless.pt"),
            output_pth=os.path.join(acts_dir, "_unused_dir_" + name + ".pt"),
            extract_only=1,
            model=model,
            model_size=model_size,
        )
        print(f"{name}: activations saved")
