"""Section 4 jailbreak analysis: harmfulness vs refusal scatter.

This module reuses the Figure 3 scoring setup, but plots jailbreak prompt
families against a refused-harmful baseline. It intentionally does not classify
the jailbreak generations: every jailbreak dataset is treated as a fixed prompt
set and scored from its hidden states.

Default comparison:
  - refused harmful baseline from the normal train/test buckets
  - GPTFuzzer-50-adv as the adversarial prompting-template jailbreak
  - human-seed-50-adv as the evil-persona-style prompt

Typical flow:
  python _04_4_jailbreak_plot.py qwen 7b --stages infer acts plot --right 50
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

from dynamic_bucket_formation import gen_buckets

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(HERE)
REPO_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(REPO_DIR, "data")
EXTRACT_SCRIPT = os.path.join(SRC_DIR, "extract_hidden.py")

POSITION_INDEX = {"tinst": 1, "tpost": -1}


def _load_module(filename, name):
    path = os.path.join(HERE, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


figure3 = _load_module("_03_3.2_figure3.py", "figure3")
ANCHORS = figure3.ANCHORS
_anchor_center = figure3._anchor_center
_delta = figure3._delta

JAILBREAKS = {
    "template jailbreak": {
        "file": "GPTFuzzer-50-adv.json",
        "stem": "gptfuzzer_template",
        "color": "#6A51A3",
        "marker": "^",
    },
    "evil persona": {
        "file": "human-seed-50-adv.json",
        "stem": "evil_persona",
        "color": "#D95F02",
        "marker": "D",
    },
    "gcg suffix": {
        "file": "gcg-advsuffix.json",
        "stem": "gcg_suffix",
        "color": "#7570B3",
        "marker": "v",
    },
    "qwen2 gcg suffix": {
        "file": "qwen2-gcg-advsuffix.json",
        "stem": "qwen2_gcg_suffix",
        "color": "#4C78A8",
        "marker": "X",
    },
    "persuasion": {
        "file": "pap-persuasion.json",
        "stem": "persuasion",
        "color": "#E7298A",
        "marker": "P",
    },
}

DEFAULT_ALL_JAILBREAKS = [
    "template jailbreak",
    "evil persona",
    "gcg suffix",
    "persuasion",
]

BASELINE_STYLE = {
    "refused harmful": {
        "color": "#EF7C6E",
        "marker": "o",
        "s": 58,
        "alpha": 0.55,
        "edgecolors": "none",
    },
}


def _read_rows(path):
    """Read a list of records from either an indented JSON array or JSONL."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if text.lstrip().startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _write_pretty(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def _write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")


def _out_dirs(model, model_size):
    base = os.path.join(HERE, "output", f"{model}{model_size}", "section4_jailbreak")
    return {
        "prepared": os.path.join(base, "prepared"),
        "generations": os.path.join(base, "generations"),
        "activations": os.path.join(base, "activations"),
        "figures": base,
    }


def selected_jailbreaks(names):
    if names == ["all"]:
        return DEFAULT_ALL_JAILBREAKS
    unknown = [name for name in names if name not in JAILBREAKS]
    if unknown:
        raise ValueError(f"Unknown jailbreak set(s): {unknown}. Choices: {list(JAILBREAKS)} or all")
    return names


def prepare_jailbreak_inputs(model, model_size, names, left, right):
    """Slice the source jailbreak files into stable experiment inputs."""
    dirs = _out_dirs(model, model_size)
    prepared = {}
    for name in names:
        cfg = JAILBREAKS[name]
        rows = _read_rows(os.path.join(DATA_DIR, cfg["file"]))
        rows = rows[max(left, 0): min(right, len(rows))]
        out_path = os.path.join(dirs["prepared"], cfg["stem"] + ".json")
        _write_jsonl(out_path, rows)
        prepared[name] = out_path
        print(f"{name}: prepared {len(rows)} prompts -> {out_path}")
    return prepared


def run_inference(model, model_size, names, left, right, max_len=128, force=False):
    """Generate model outputs for jailbreak sets, without classifying them."""
    dirs = _out_dirs(model, model_size)
    prepared = prepare_jailbreak_inputs(model, model_size, names, left, right)
    for name, input_path in prepared.items():
        output_path = os.path.join(dirs["generations"], JAILBREAKS[name]["stem"] + ".json")
        expected = len(_read_rows(input_path))
        if os.path.exists(output_path) and not force:
            try:
                existing = _read_rows(output_path)
            except Exception:
                existing = []
            if len(existing) >= expected:
                print(f"{name}: generations already exist ({len(existing)} rows), skipping")
                continue
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cmd = [
            sys.executable,
            "inference.py",
            "--input", input_path,
            "--output_file_name", output_path,
            "--model", model,
            "--model_size", model_size,
            "--left", "0",
            "--right", str(len(_read_rows(input_path))),
            "--use_template", "1",
            "--use_jb", "0",
            "--use_inversion", "0",
            "--record_prob_max_pos", "0",
            "--max_len", str(max_len),
            "--do_not_use_last_inst_tok", "0",
        ]
        subprocess.run(cmd, check=True, cwd=SRC_DIR)
        _write_jsonl(output_path, _read_rows(output_path))
        print(f"{name}: generations saved -> {output_path}")


def compute_activations(model, model_size, names, left, right):
    """Extract t_inst/t_post activations for jailbreak prompt sets."""
    dirs = _out_dirs(model, model_size)
    prepared = {}
    for name in names:
        prepared_path = os.path.join(dirs["prepared"], JAILBREAKS[name]["stem"] + ".json")
        if not os.path.exists(prepared_path):
            prepared_path = prepare_jailbreak_inputs(model, model_size, [name], left, right)[name]
        prepared[name] = prepared_path

    for name, input_path in prepared.items():
        stem = JAILBREAKS[name]["stem"]
        output_pth_harmful = os.path.join(dirs["activations"], stem + ".pt")
        output_pth_harmless = os.path.join(dirs["activations"], "_unused_harmless.pt")
        output_pth = os.path.join(dirs["activations"], "_unused_dir_" + stem + ".pt")
        os.makedirs(os.path.dirname(output_pth_harmful), exist_ok=True)
        cmd = [
            sys.executable,
            EXTRACT_SCRIPT,
            "--model", model,
            "--model_size", model_size,
            "--harmful_pth", input_path,
            "--harmless_pth", input_path,
            "--output_pth_harmful", output_pth_harmful,
            "--output_pth_harmless", output_pth_harmless,
            "--output_pth", output_pth,
            "--extract_only", "1",
            "--extract_hidden_inst_token", "1",
            "--extract_harmful_token_only", "0",
            "--batch_size", "1",
            "--ret_whole_seq", "0",
            "--left", "0",
            "--right", str(len(_read_rows(input_path))),
            "--mode_dir", "hf",
        ]
        subprocess.run(cmd, check=True, cwd=HERE)
        for debris in (
            output_pth_harmless,
            output_pth.replace(".pt", "_prompts_used.json"),
            os.path.join(HERE, "output", "tmp_mean_activations_harmful.pt"),
            os.path.join(HERE, "output", "tmp_full_activations_harmful.pt"),
        ):
            if os.path.exists(debris):
                os.remove(debris)
        print(f"{name}: activations saved -> {output_pth_harmful}")


def _jailbreak_points(path, mu):
    """Compute (Delta_harmful, Delta_refuse) for one jailbreak activation tensor."""
    tensor = torch.load(path, map_location="cpu").float()  # (L, N, T, H)
    tinst = tensor[:, :, POSITION_INDEX["tinst"], :]
    tpost = tensor[:, :, POSITION_INDEX["tpost"], :]
    delta_harmful = _delta(tinst, mu["harmful"], mu["harmless"])
    delta_refuse = _delta(tpost, mu["refused"], mu["accepted"])
    n = min(len(delta_harmful), len(delta_refuse))
    return delta_harmful[:n], delta_refuse[:n]


def _baseline_points(buckets, mu):
    test = buckets["test"]
    tinst = test.get("refused_harmful_tinst")
    tpost = test.get("refused_harmful_tpost")
    if tinst is None or tpost is None:
        return None
    dx = _delta(tinst, mu["harmful"], mu["harmless"])
    dy = _delta(tpost, mu["refused"], mu["accepted"])
    n = min(len(dx), len(dy))
    return dx[:n], dy[:n]


def plot_section4(model, model_size, names, bucket_config, out_path=None):
    """Plot refused-harmful baseline and jailbreak prompt sets in Figure-3 space."""
    dirs = _out_dirs(model, model_size)
    buckets = gen_buckets(model, model_size, bucket_config)
    train = buckets["train"]
    mu = {name: _anchor_center(train, sources) for name, sources in ANCHORS.items()}

    points = {}
    baseline = _baseline_points(buckets, mu)
    if baseline is not None:
        points["refused harmful"] = baseline

    for name in names:
        path = os.path.join(dirs["activations"], JAILBREAKS[name]["stem"] + ".pt")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing activations for {name}: {path}. "
                "Run with --stages acts first."
            )
        points[name] = _jailbreak_points(path, mu)

    all_x = np.concatenate([dx for dx, _ in points.values()])
    all_y = np.concatenate([dy for _, dy in points.values()])
    xpad = max(0.02, float(np.ptp(all_x)) * 0.08)
    ypad = max(0.03, float(np.ptp(all_y)) * 0.10)

    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    ax.axhline(0, color="0.55", linestyle="--", linewidth=1.0, zorder=1)
    ax.axvline(0, color="0.55", linestyle="--", linewidth=1.0, zorder=1)

    if "refused harmful" in points:
        dx, dy = points["refused harmful"]
        ax.scatter(dx, dy, label="refused harmful", zorder=2, **BASELINE_STYLE["refused harmful"])

    for name in names:
        dx, dy = points[name]
        cfg = JAILBREAKS[name]
        ax.scatter(
            dx,
            dy,
            label=name,
            color=cfg["color"],
            marker=cfg["marker"],
            s=72,
            alpha=0.82,
            linewidths=1.0,
            edgecolors="white",
            zorder=3,
        )

    ax.set_xlim(float(all_x.min()) - xpad, float(all_x.max()) + xpad)
    ax.set_ylim(float(all_y.min()) - ypad, float(all_y.max()) + ypad)
    ax.set_xlabel(r"$\Delta$ harmful", fontsize=18)
    ax.set_ylabel(r"$\Delta$ refuse", fontsize=18)
    ax.set_title("Section 4 jailbreak analysis", fontsize=16, pad=10)
    ax.grid(True, linestyle="-", linewidth=0.7, alpha=0.32, zorder=0)
    ax.tick_params(labelsize=12)
    ax.legend(loc="best", fontsize=11, framealpha=0.92, edgecolor="0.85")

    save_path = out_path or os.path.join(dirs["figures"], "section4_jailbreak_scatter.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {save_path}")
    return save_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", default="qwen")
    parser.add_argument("model_size", nargs="?", default="7b")
    parser.add_argument("--left", type=int, default=0)
    parser.add_argument("--right", type=int, default=50)
    parser.add_argument(
        "--jailbreaks",
        nargs="+",
        default=["template jailbreak", "evil persona"],
        help=f"Jailbreak sets to include. Choices: {list(JAILBREAKS)} or all",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["plot"],
        choices=["infer", "acts", "plot"],
        help="Run any subset of inference, activation extraction, and plotting.",
    )
    parser.add_argument("--bucket_config", default="bucket_config_clean.json")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    names = selected_jailbreaks(args.jailbreaks)
    if "infer" in args.stages:
        run_inference(args.model, args.model_size, names, args.left, args.right, args.max_len, args.force)
    if "acts" in args.stages:
        compute_activations(args.model, args.model_size, names, args.left, args.right)
    if "plot" in args.stages:
        plot_section4(args.model, args.model_size, names, args.bucket_config, args.out)


if __name__ == "__main__":
    main()
