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


def _is_split_file(name):
    """True for a classified-generation split (…_accepted / …_refused). Excludes non-split JSON that
    can share the dir — notably the judge's judge_log.json audit file, which is a {summary, samples}
    dict, not a list of rows, and would crash extract_hidden.py."""
    return name.endswith("_accepted") or name.endswith("_refused")


def _dirs(model, model_size, classified_subdir="classified_generations",
          acts_subdir="activations"):
    """Input classified-generations dir and output activations dir for a given model.

    Defaults are the standard classified_generations/ -> activations/. In judge mode the caller passes
    classified_subdir="judge_classifications", acts_subdir="judge_activations" so activations are
    extracted from the judge-corrected splits into a separate folder, leaving activations/ untouched.
    """
    base = os.path.join(HERE, "output", f"{model}{model_size}", "datasets_outputs")
    return (
        os.path.join(base, classified_subdir),
        os.path.join(base, acts_subdir),
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


def run_extract_hidden_thinking(
    harmful_pth,
    output_pth_harmful,
    thinking_mode,
    *,
    model="qwen35",
    model_size="0.8b",
    left=0,
    right=100000,
):
    """Wrapper around extract_hidden.py --thinking 1: forward on prompt+generation and store the
    fixed 22-slot (L, N, 22, H) tensor. Leaves no scratch/debris (unlike the mean-diff path)."""
    os.makedirs(os.path.dirname(output_pth_harmful), exist_ok=True)
    cmd = [
        sys.executable, EXTRACT_SCRIPT,
        "--model", model,
        "--model_size", model_size,
        "--harmful_pth", harmful_pth,
        "--output_pth_harmful", output_pth_harmful,
        "--thinking", "1",
        "--thinking_mode", thinking_mode,
        "--left", str(left),
        "--right", str(right),
    ]
    # expandable_segments reduces fragmentation OOMs on long full-trace forward passes.
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    subprocess.run(cmd, check=True, cwd=HERE, env=env)


def _thinking_mode_of(name):
    """thinking_mode from a classified filename. Order matters, longest first: each name is a
    substring of the next ('gennothink_stripped' is neither 'genthink' nor plain 'gennothink', and
    'gennothink_stripped_v2' contains 'gennothink_stripped')."""
    if "gennothink_stripped_v2" in name:
        return "gennothink_stripped_v2"
    if "gennothink_stripped" in name:
        return "gennothink_stripped"
    return "genthink" if "genthink" in name else "gennothink"


def compute_all_activations_thinking(model, model_size,
                                     classified_subdir="classified_generations",
                                     acts_subdir="activations", only_modes=None,
                                     max_acts_per_bucket=None):
    """Extract 22-slot activations for each thinking classified generation into acts_subdir.

    thinking_mode is read from the filename (genthink / gennothink / gennothink_stripped): gennothink
    leaves the CoT slots null (its </think> is in the prompt), while genthink and gennothink_stripped
    keep the (model-generated) CoT. Pass judge_classifications/judge_activations for the judge-corrected
    splits.

    only_modes: comma-separated thinking modes (e.g. "gennothink_stripped") to restrict extraction to;
    None extracts every classified split.
    max_acts_per_bucket: cap the number of rows extracted PER bucket (per classified split file) to the
    first N — bounds each .pt to (L, N, 22, H) so figure/bucket loading doesn't OOM. None = no cap. The
    cap is per-file, so buckets are never mixed.
    """
    classified_dir, acts_dir = _dirs(model, model_size, classified_subdir, acts_subdir)
    os.makedirs(acts_dir, exist_ok=True)

    wanted = {m.strip() for m in (only_modes.split(",") if isinstance(only_modes, str)
                                  else (only_modes or [])) if m and m.strip()}
    right = max_acts_per_bucket if max_acts_per_bucket else 100000  # extract_hidden slices rows[:right]

    for src_pth in sorted(glob.glob(os.path.join(classified_dir, "*.json"))):
        name = os.path.splitext(os.path.basename(src_pth))[0]
        if not _is_split_file(name):
            continue  # skip non-split JSON (e.g. judge_log.json)
        thinking_mode = _thinking_mode_of(name)
        if wanted and thinking_mode not in wanted:
            continue
        with open(src_pth, encoding="utf-8") as f:
            if not json.load(f):
                print(f"{name}: empty, skipping")
                continue
        run_extract_hidden_thinking(
            harmful_pth=src_pth,
            output_pth_harmful=os.path.join(acts_dir, name + ".pt"),
            thinking_mode=thinking_mode,
            model=model,
            model_size=model_size,
            right=right,
        )
        print(f"{name}: activations saved ({thinking_mode}, cap={right if max_acts_per_bucket else 'none'})")


def compute_all_activations(model, model_size,
                            classified_subdir="classified_generations",
                            acts_subdir="activations", max_acts_per_bucket=None):
    """Extract activations for each classified generation into acts_subdir.

    Pass judge_classifications/judge_activations for the judge-corrected splits.
    max_acts_per_bucket: cap the rows extracted PER bucket (per split file) to the first N; None = no
    cap. Per-file, so buckets are never mixed.
    """
    classified_dir, acts_dir = _dirs(model, model_size, classified_subdir, acts_subdir)
    os.makedirs(acts_dir, exist_ok=True)
    right = max_acts_per_bucket if max_acts_per_bucket else 100000  # extract_hidden slices rows[:right]

    for src_pth in sorted(glob.glob(os.path.join(classified_dir, "*.json"))):
        name = os.path.splitext(os.path.basename(src_pth))[0]
        if not _is_split_file(name):
            continue  # skip non-split JSON (e.g. judge_log.json)
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
            right=right,
        )
        print(f"{name}: activations saved (cap={right if max_acts_per_bucket else 'none'})")
