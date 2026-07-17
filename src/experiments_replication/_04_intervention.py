"""Section 3.4 / Figure 4: eliciting refusal by steering along the harmfulness direction.

Port of the `origin/interventions` branch onto the current pipeline, targeting the Qwen3.5
thinking track. Four stages:

    vectors    (CPU)  bucket families -> v_hf, v_refuse -> steering_vectors/<name>.pt
    intervene  (GPU)  subprocess src/intervention.py per experiment -> per-layer JSONs
    judge4     (GPU)  label each per-layer JSON with the judge LLM
    fig4       (CPU)  refusal rate vs intervention layer

The steering directions are differences of the SAME bucket family pair read at two different
token slots -- which is what the paper's four cluster centers reduce to (Appendix B populates
mu_harmful/mu_refuse from advbench+jbb-refused and mu_harmless/mu_accept from harmless-accepted):

    v_hf     = mu(refused_harmful @ t_inst) - mu(accepted_harmless @ t_inst)
    v_refuse = mu(refused_harmful @ t_post) - mu(accepted_harmless @ t_post)

The branch's get_intervene_vectors.py cannot be reused here: it maps t_post to token index -1,
which on the 25-slot thinking layout is slot 24 (</think>), not slot 1 (t_post), and its
_gentinst/_gentpost source names do not exist on this track. Sourcing from the buckets instead
also avoids its accepted/refused pooling (which put accepted-harmful on the refuse side).

Usage:
    python _04_intervention.py --stage vectors --config configs/intervene/intervene_config_qwen35_think.json
"""

import argparse
import glob
import importlib.util
import json
import os
import re
import subprocess
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from utils import read_row
from dynamic_bucket_formation import gen_buckets_thinking


def _load_module(filename, name):
    """Import a sibling module whose filename isn't a valid identifier (e.g. has a dot)."""
    path = os.path.join(HERE, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_figure2 = _load_module("_02_3.1_figure2.py", "figure2")
# (L, N, 25, H) -> (L, n, H) at one slot, dropping null (zero-norm) rows and upcasting to fp32.
_slice_pos = _figure2._slice_pos
SLOT_LABELS = _figure2.SLOT_LABELS


def load_config(path):
    """Read an intervene config json (relative paths resolve against this dir)."""
    p = path if os.path.isabs(path) else os.path.join(HERE, path)
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Layout: everything this stage produces lives under the model's own output dir, alongside the
# figure-2/3 artefacts:
#
#   output/<model><size>/
#     intervention/
#       intervention_vectors/<name>.pt          the steering directions (~540KB each)
#       generations/<key>/*-intervene<L>.json   steered generations, one file per layer
#                        /*-intervene<L>.labels.json   judge labels (gitignored: regenerable)
#     figure4_<model>_<mode>.png                with the other figures
#
# NOTE for the Modal image: modal_run.py ignores "**/output" and "*.pt" because that tree's .pt are
# the multi-GB activation tensors. The vectors now live under output/ too, so modal_intervene.py
# mounts output/ while ignoring datasets_outputs/ (where the heavy tensors actually are) instead of
# ignoring .pt wholesale.
# ---------------------------------------------------------------------------

def intervention_dir(model, model_size):
    return os.path.join(HERE, "output", f"{model}{model_size}", "intervention")


def vectors_dir(model, model_size, thinking_mode=""):
    """intervention_vectors/<thinking_mode>/ — the mode is part of the path, not optional.

    Vectors are named only hf.pt / refusal.pt, so without this a gennothink `vectors` run would
    silently OVERWRITE the genthink ones, and every later sweep would steer with whichever mode
    happened to be built last. The two are genuinely different directions: they come from
    different activations (*_genthink_* vs *_gennothink_*).
    """
    return os.path.join(intervention_dir(model, model_size), "intervention_vectors",
                        thinking_mode or "instruct")


def generations_dir(model, model_size):
    return os.path.join(intervention_dir(model, model_size), "generations")


def run_dir(model, model_size, key):
    """generations/<key>/ — one dir per (dataset, vector, direction, rows, coeff, layer band)."""
    return os.path.join(generations_dir(model, model_size), key)


def build_vectors(model, model_size, cfg, save=True):
    """Compose steering directions from bucket families -> {name: (L, H) fp32}.

    Each cfg["vectors"] entry is [name, pos_family, neg_family, slot]: the direction is the
    difference of the two families' per-layer means at that slot. Anchors come from cfg["split"]
    (default "train", matching Figure 2's anchors).
    """
    buckets = gen_buckets_thinking(model, model_size, cfg["bucket_config"],
                                   use_judged=cfg.get("use_judged", True))
    split_name = cfg.get("split", "train")
    if split_name not in buckets:
        raise KeyError(f"split {split_name!r} not in buckets (have {sorted(buckets)})")
    split = buckets[split_name]

    vectors = {}
    for name, pos_family, neg_family, slot in cfg["vectors"]:
        for family in (pos_family, neg_family):
            if family not in split:
                raise KeyError(
                    f"vector {name!r}: family {family!r} missing from the {split_name} split "
                    f"(have {sorted(split)}). Check bucket_config's families_{split_name}."
                )
        pos = _slice_pos(split[pos_family], slot)  # (L, n, H) fp32
        neg = _slice_pos(split[neg_family], slot)
        for family, tensor in ((pos_family, pos), (neg_family, neg)):
            if tensor.shape[1] == 0:
                raise ValueError(
                    f"vector {name!r}: family {family!r} has no non-null rows at slot {slot} "
                    f"({SLOT_LABELS.get(slot, '?')}). That slot is null for every row."
                )
        vectors[name] = pos.mean(dim=1) - neg.mean(dim=1)  # (L, H)
        print(f"  {name:16s} = mu({pos_family}) - mu({neg_family}) @ slot {slot} "
              f"({SLOT_LABELS.get(slot, '?')})  [{pos.shape[1]} vs {neg.shape[1]} rows] "
              f"-> {tuple(vectors[name].shape)}")

    if save:
        out_dir = vectors_dir(model, model_size, cfg.get("thinking_mode", ""))
        os.makedirs(out_dir, exist_ok=True)
        for name, tensor in vectors.items():
            path = os.path.join(out_dir, f"{name}.pt")
            torch.save(tensor, path)
            print(f"  saved -> {path}")

    return vectors


def report_vectors(vectors):
    """Verification step 1: shape/finiteness/norm per vector, and pairwise cosine per layer.

    The paper reports the harmfulness and refusal directions are near-orthogonal (~0.1 average
    cosine on Llama2). A cosine near 1.0 means the two families collapsed -- i.e. the config is
    pointing both vectors at the same contrast -- and everything downstream would be meaningless.
    """
    ok = True
    for name, v in vectors.items():
        norms = v.norm(dim=-1)
        finite = bool(torch.isfinite(v).all())
        zero_layers = (norms == 0).nonzero().flatten().tolist()
        print(f"\n{name}: shape={tuple(v.shape)} finite={finite} "
              f"norm min={norms.min():.4f} max={norms.max():.4f}")
        if not finite:
            print(f"  FAIL: {name} contains non-finite values")
            ok = False
        # Index 0 is the embedding output, which depends only on the token id. A vector taken at a
        # slot holding a FIXED template token (e.g. slot 1 = "assistant") therefore has identical
        # embeddings in both families and a difference of exactly zero there. That is expected, and
        # it doubles as a check that index 0 really is the embedding. Layers >= 1 see context, so a
        # zero there is a genuine failure.
        if 0 in zero_layers:
            print(f"  note: {name} is zero at index 0 (embedding) -- expected for a fixed "
                  f"template slot; index 0 is outside the steering band anyway")
        contextual_zeros = [l for l in zero_layers if l > 0]
        if contextual_zeros:
            print(f"  FAIL: {name} has zero-norm contextual layers: {contextual_zeros}")
            ok = False

    names = list(vectors)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            va, vb = vectors[a], vectors[b]
            cos = torch.nn.functional.cosine_similarity(va, vb, dim=-1)  # (L,)
            print(f"\ncos({a}, {b}) per layer: mean={cos.mean():.4f} "
                  f"min={cos.min():.4f} max={cos.max():.4f}")
            print("  " + " ".join(f"{c:+.2f}" for c in cos.tolist()))
            if cos.mean() > 0.9:
                print(f"  WARNING: {a} and {b} are nearly identical (mean cos {cos.mean():.3f}). "
                      f"Expected near-orthogonal. Check that the two vectors use different slots.")
                ok = False
    return ok


# ---------------------------------------------------------------------------
# intervene: steered generation, by subprocessing the paper's src/intervention.py
#
# intervention.py loops layer_s..layer_e itself and writes one JSON per layer, so one subprocess
# call covers a whole band. Local counterpart of modal_intervene.py (same relationship as
# _01_compute_activations vs modal_run.py); the two must agree on the run key.
# ---------------------------------------------------------------------------

INTERVENTION_SCRIPT = os.path.join(SRC, "intervention.py")


def run_intervention_local(model, model_size, cfg, dataset, vector, reverse, prompt_key,
                           ctx_only, all_tokens, use_inversion):
    """Subprocess src/intervention.py for one experiment -> intervention/generations/<key>/*.json."""
    key = run_key(model, model_size, dataset, vector, reverse, cfg["left"], cfg["right"],
                  use_inversion, cfg["coeff"], cfg["layer_s"], cfg["layer_e"],
                  cfg.get("thinking_mode", ""), cfg["max_decode_step_while_intervene"])
    out_dir = run_dir(model, model_size, key)
    os.makedirs(out_dir, exist_ok=True)
    output_pth = os.path.join(out_dir, f"{dataset}-{'less' if reverse else 'more'}.json")

    vector_pth = os.path.join(vectors_dir(model, model_size, cfg.get("thinking_mode", "")),
                              f"{vector}.pt")
    if not os.path.exists(vector_pth):
        raise FileNotFoundError(f"steering vector not found: {vector_pth}. Run --stage vectors first.")

    cmd = [
        sys.executable, "-u", INTERVENTION_SCRIPT,
        "--test_data_pth", os.path.join(SRC, os.pardir, "data", f"{dataset}.json"),
        "--output_pth", output_pth,
        "--intervention_vector", vector_pth,
        "--reverse_intervention", str(reverse),
        "--intervene_context_only", str(ctx_only),
        "--intervene_all", str(all_tokens),
        "--arg_key_prompt", prompt_key,
        "--model", model,
        "--model_size", model_size,
        "--left", str(cfg["left"]),
        "--right", str(cfg["right"]),
        "--layer_s", str(cfg["layer_s"]),
        "--layer_e", str(cfg["layer_e"]),
        "--coeff_select", str(cfg["coeff"]),
        "--max_token_generate", str(cfg["max_token_generate"]),
        "--max_decode_step_while_intervene", str(cfg["max_decode_step_while_intervene"]),
        "--use_inversion", str(use_inversion),
        "--batch_size", str(cfg.get("batch_size", 1)),
    ]
    if cfg.get("thinking_mode"):
        cmd += ["--thinking_mode", cfg["thinking_mode"]]

    print(f"  -> {key}")
    # cwd=SRC so intervention.py's `from utils import ...` and `from template_inversion import ...`
    # resolve, matching how _00/_01 invoke the other reference scripts.
    subprocess.run(cmd, check=True, cwd=SRC)
    return out_dir


def run_all_interventions(model, model_size, cfg, skip_existing=True):
    """Run every experiment in the config, skipping any whose layer files are already complete.

    skip_existing matters because configs share arms: e.g. two gennothink experiments that differ
    only in the refusal slot use the SAME hf vector, so they produce the same run key. Without this,
    running both configs regenerates that sweep twice — hours of GPU for a byte-identical result.
    """
    n_layers = cfg["layer_e"] - cfg["layer_s"]
    for dataset, vector, reverse, prompt_key, ctx, all_, use_inv in cfg["experiments"]:
        key = run_key(model, model_size, dataset, vector, reverse, cfg["left"], cfg["right"],
                      use_inv, cfg["coeff"], cfg["layer_s"], cfg["layer_e"],
                      cfg.get("thinking_mode", ""), cfg["max_decode_step_while_intervene"])
        d = run_dir(model, model_size, key)
        if skip_existing and os.path.isdir(d) and len(_layer_files(d)) >= n_layers:
            print(f"  already complete, skipped: {key}")
            continue
        run_intervention_local(model, model_size, cfg, dataset, vector, reverse, prompt_key,
                               ctx, all_, use_inv)


# ---------------------------------------------------------------------------
# judge4: label the per-layer intervention JSONs
#
# easy_eval is unusable on this track: a thinking response contains the reasoning trace, so its
# refusal substrings ("I cannot", "I'm sorry") match the model REASONING about a refusal even when
# the final answer complies. We split on </think> and judge only the answer, matching what
# judge_llm._response_text does for the pipeline's own classified generations.
#
# intervention.py's record shape is {'prompt': <row dict>, 'response': str, 'tokens', 'probs'} --
# note 'response', not the 'ori_output' that judge_llm._request_text/_response_text expect, so those
# two helpers are re-implemented here for this shape rather than reused.
# ---------------------------------------------------------------------------

_LAYER_RE = re.compile(r"-intervene(\d+)\.json$")


def _record_query(record):
    """The user's request. Field varies by dataset: harmful -> bad_q, alpaca -> instruction."""
    p = record.get("prompt")
    if isinstance(p, dict):
        return p.get("bad_q") or p.get("instruction") or p.get("prompt") or ""
    return p or ""


def _record_answer(record, strip_think=True):
    """The answer to judge, or None if the generation never produced one.

    For genthink, drop the reasoning trace and keep the post-</think> text. </think> survives
    intervention.py's skip_special_tokens=True decode (verified against the Qwen3.5 tokenizer: the
    think tags are ordinary vocab tokens, not special ones).

    Returns None when the trace never closed: at max_token_generate=2048 about 8.5% of 9b genthink
    rows are still reasoning when generation stops (median CoT ~1017 tokens, max ~3477), so there is
    no answer at all. Those rows must NOT be judged -- passing the truncated reasoning through as if
    it were the answer would score the model's DELIBERATION ("I cannot help with...") as its
    behaviour, which is exactly the failure the judge exists to avoid.
    """
    resp = record.get("response", "") or ""
    if not strip_think:
        return resp
    if "</think>" not in resp:
        return None
    return resp.split("</think>")[-1].strip()


def _layer_files(run_dir):
    """(layer, path) for every per-layer JSON in a run dir, ascending."""
    out = []
    for path in glob.glob(os.path.join(run_dir, "*-intervene*.json")):
        m = _LAYER_RE.search(os.path.basename(path))
        if m:
            out.append((int(m.group(1)), path))
    return sorted(out)


def labels_path(json_path):
    return json_path.replace(".json", ".labels.json")


def judge_run(run_dir, judge_config="configs/eval/judge_config_thinking.json", overwrite=False,
              judge_batch_size=None):
    """Label every per-layer JSON in run_dir with the judge LLM -> <stem>.labels.json.

    Cached: a layer whose .labels.json exists is skipped unless overwrite=True (the judge is a
    GPU pass over every generation, so re-running the plot must not re-judge).

    judge_batch_size overrides the config's batch_size for this run only. judge_config_thinking.json
    uses 64, which is sized for the A100/B200 the pipeline's own judge pass runs on; with
    max_new_tokens=4096 that OOMs a 16GB local card. Overriding here keeps the shared config
    untouched rather than tuning it down for the smallest machine that might run it.
    """
    import judge_llm

    files = _layer_files(run_dir)
    if not files:
        raise FileNotFoundError(f"no *-intervene*.json under {run_dir}")

    todo = [(l, p) for l, p in files if overwrite or not os.path.exists(labels_path(p))]
    if not todo:
        print(f"  all {len(files)} layers already judged in {os.path.basename(run_dir)}")
        return

    judge_llm.load_judge_config(judge_config)
    strip_think = judge_llm.STRIP_THINK
    batch_size = judge_batch_size or judge_llm.JUDGE_BATCH_SIZE
    print(f"  judging {len(todo)}/{len(files)} layers in {os.path.basename(run_dir)} "
          f"with {judge_llm.JUDGE_MODEL_NAME} (strip_think={strip_think}, batch={batch_size})")
    model, tokenizer = judge_llm.load_judge_model()

    for layer, path in todo:
        rows = read_row(path)
        if not rows:
            print(f"    layer {layer}: empty, skipped")
            continue
        # Rows whose CoT never closed have no answer: label TRUNCATED without spending judge GPU.
        answers = [_record_answer(r, strip_think) for r in rows]
        judged_idx = [i for i, a in enumerate(answers) if a is not None]
        labels = ["TRUNCATED"] * len(rows)
        if judged_idx:
            pairs = [(_record_query(rows[i]), answers[i]) for i in judged_idx]
            details = judge_llm.judge_accepted_batch(
                model, tokenizer, pairs,
                max_new_tokens=judge_llm.JUDGE_MAX_NEW_TOKENS,
                batch_size=batch_size,
                thinking=judge_llm.JUDGE_THINKING,
                return_details=True,
            )
            for i, d in zip(judged_idx, details):
                labels[i] = d["label"]
        with open(labels_path(path), "w", encoding="utf-8") as f:
            json.dump(labels, f)
        counts = _count(labels)
        print(f"    layer {layer}: {counts} -> refusal {_refusal_rate(counts):.1f}%")


def _count(labels):
    counts = {"ACCEPT": 0, "REJECT": 0, "NULL": 0, "TRUNCATED": 0}
    for l in labels:
        counts[l if l in counts else "NULL"] += 1
    return counts


def _refusal_rate(counts):
    """REJECT / (ACCEPT + REJECT), as a percentage. NULL and TRUNCATED are EXCLUDED.

    judge_accepted_batch's own `accepted` field is `label != "REJECT"`, which silently folds NULL
    (no parseable verdict -- e.g. degenerate output) into ACCEPT. That would make a broken run look
    like a compliant one. TRUNCATED (CoT never closed, so there is no answer) is likewise not an
    acceptance. Both are tracked separately and surfaced rather than folded into the rate.

    Caveat worth remembering when reading the figure: excluding TRUNCATED is only unbiased if
    truncation is independent of steering. If steering lengthens the CoT, more truncation at the
    steered layers means the surviving rows there are a non-random subset. Watch the per-layer
    TRUNCATED counts for a trend before trusting a shape.
    """
    decided = counts["ACCEPT"] + counts["REJECT"]
    if not decided:
        return float("nan")
    return counts["REJECT"] / decided * 100


def read_rates(run_dir):
    """(layers, rates, dropped) from a judged run dir; dropped = NULL + TRUNCATED per layer."""
    layers, rates, dropped = [], [], []
    for layer, path in _layer_files(run_dir):
        lp = labels_path(path)
        if not os.path.exists(lp):
            print(f"  [warn] not judged, skipped: {os.path.basename(path)}")
            continue
        with open(lp, encoding="utf-8") as f:
            counts = _count(json.load(f))
        layers.append(layer)
        rates.append(_refusal_rate(counts))
        dropped.append(counts["NULL"] + counts["TRUNCATED"])
    return layers, rates, dropped


# ---------------------------------------------------------------------------
# fig4: refusal rate vs intervention layer, one line per steering vector
# (ported from the branch's _04_figure4.py; the branch scored with easy_eval and hardcoded
#  NUM_LAYERS=28 -- here the layers come from whatever the run dirs actually contain.)
# ---------------------------------------------------------------------------

COLORS = ["#E8766C", "#4A6FA5", "#158A8A"]  # harmfulness, refusal, third vector if present


def fig4_path(model, model_size, config_path):
    """output/<model><size>/figure4<suffix>.png, suffix from the intervene config stem.

    Derived from the CONFIG, not from thinking_mode: several experiments share a mode but differ in
    which token slots the vectors come from, and they would all collide on one filename otherwise.
    Mirrors _fig_path's convention in main.py for figures 2/3.
    """
    stem = os.path.splitext(os.path.basename(config_path))[0]
    prefix = "intervene_config"
    suffix = stem[len(prefix):] if stem.startswith(prefix) else f"_{stem}"
    return os.path.join(HERE, "output", f"{model}{model_size}", f"figure4{suffix}.png")


def run_key(model, model_size, dataset, vector, reverse, left, right, use_inversion,
            coeff, layer_s, layer_e, thinking_mode="", max_decode_step=1):
    """Mirror of modal_intervene._key -- the local dir name a collected run lands in.

    thinking_mode is in the key: genthink and gennothink build different prompts from different
    vectors, but intervention.py names its output per LAYER, so without the mode a gennothink sweep
    would overwrite the genthink files layer-for-layer with no warning.
    """
    direction = "less" if reverse else "more"
    mode = thinking_mode or "instruct"
    # max_decode_step is in the key: d1 steers the PROMPT only (prefill), d-1 steers the prompt AND
    # every generated token. Same vector, same layers, materially different experiment -- without it
    # the two overwrite each other layer-for-layer.
    return (f"{model}{model_size}-{mode}-{dataset}-{vector}-{direction}-{left}-{right}"
            f"-inv{use_inversion}-c{coeff:g}-L{layer_s}_{layer_e}-d{max_decode_step}")


def plot_figure4(model, model_size, cfg, out_path):
    """One line per experiment: judged refusal rate against intervention layer."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    plotted = 0
    for i, (dataset, vector, reverse, _prompt, _ctx, _all, use_inv) in enumerate(cfg["experiments"]):
        key = run_key(model, model_size, dataset, vector, reverse, cfg["left"], cfg["right"],
                      use_inv, cfg["coeff"], cfg["layer_s"], cfg["layer_e"],
                      cfg.get("thinking_mode", ""), cfg["max_decode_step_while_intervene"])
        d = run_dir(model, model_size, key)
        if not os.path.isdir(d):
            print(f"  [warn] missing run dir, skipped: {key}")
            continue
        layers, rates, dropped = read_rates(d)
        if not layers:
            print(f"  [warn] no judged layers in {key}")
            continue
        label = {"hf": "harmfulness dir", "refusal": "refusal dir"}.get(vector, vector)
        if reverse:
            label = "reverse " + label
        ax.plot(layers, rates, color=COLORS[i % len(COLORS)], label=label,
                linewidth=2, marker="o", markersize=4)
        plotted += 1
        if sum(dropped):
            # Rows excluded from the rate (unclosed CoT, or no parseable verdict). Printed
            # per-layer, not just as a total: a FLAT count is benign, but a count that climbs with
            # the steered layers means truncation correlates with the intervention, so the
            # surviving rows there are a biased subset and the curve's shape is suspect.
            print(f"  [warn] {key}: {sum(dropped)} rows excluded from the rate "
                  f"(unclosed CoT / unlabelled). Per layer:")
            print("         " + "  ".join(f"L{l}:{d}" for l, d in zip(layers, dropped)))

    if not plotted:
        print("nothing to plot -- run the intervene and judge4 stages first")
        return

    ax.set_title(f"{model}{model_size} — steering harmless instructions ({cfg['thinking_mode']})",
                 fontsize=13)
    ax.set_xlabel("Intervention layer", fontsize=12)
    ax.set_ylabel("Refusal rate (%)", fontsize=12)
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Section 3.4 / Figure 4 interventions")
    parser.add_argument("--stage", required=True,
                        choices=["vectors", "intervene", "judge4", "fig4"],
                        help="vectors/fig4 are CPU; intervene and judge4 need a GPU. `intervene` "
                             "runs locally -- use modal_intervene.py for the remote equivalent.")
    parser.add_argument("--config", required=True, type=str,
                        help="path to an intervene config json (relative to this dir is fine)")
    parser.add_argument("--model", default="qwen35", type=str)
    parser.add_argument("--model-size", default="9b", type=str)
    parser.add_argument("--run-dir", default=None, type=str,
                        help="judge4: a single intervention_outputs/<key> dir. Default: every dir "
                             "implied by the config's experiments.")
    parser.add_argument("--judge-config", default="configs/eval/judge_config_thinking.json", type=str)
    parser.add_argument("--judge-batch-size", default=None, type=int,
                        help="judge4: override the judge config's batch_size. The shared config uses "
                             "64 (sized for A100/B200); a 16GB local card needs ~4 at 4096 tokens.")
    parser.add_argument("--overwrite", action="store_true",
                        help="judge4: re-judge layers that already have a .labels.json")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.stage == "vectors":
        print(f"building steering vectors for {args.model}{args.model_size} "
              f"from {cfg['bucket_config']} ({cfg.get('split', 'train')} split, "
              f"use_judged={cfg.get('use_judged', True)})")
        vectors = build_vectors(args.model, args.model_size, cfg)
        ok = report_vectors(vectors)
        print("\n" + ("vectors OK" if ok else "VECTOR CHECKS FAILED -- do not proceed to GPU"))
        sys.exit(0 if ok else 1)

    if args.stage == "intervene":
        print(f"steering {args.model}{args.model_size}, layers "
              f"{cfg['layer_s']}..{cfg['layer_e']}, coeff {cfg['coeff']}, "
              f"rows {cfg['left']}:{cfg['right']}")
        run_all_interventions(args.model, args.model_size, cfg)

    if args.stage == "judge4":
        if args.run_dir:
            run_dirs = [args.run_dir if os.path.isabs(args.run_dir)
                        else run_dir(args.model, args.model_size, args.run_dir)]
        else:
            run_dirs = [
                run_dir(args.model, args.model_size, run_key(
                    args.model, args.model_size, dataset, vector, reverse,
                    cfg["left"], cfg["right"], use_inv, cfg["coeff"],
                    cfg["layer_s"], cfg["layer_e"], cfg.get("thinking_mode", ""),
                    cfg["max_decode_step_while_intervene"]))
                for dataset, vector, reverse, _p, _c, _a, use_inv in cfg["experiments"]
            ]
        for d in run_dirs:
            if not os.path.isdir(d):
                print(f"  [warn] missing, skipped: {d}")
                continue
            judge_run(d, args.judge_config, overwrite=args.overwrite,
                      judge_batch_size=args.judge_batch_size)

    if args.stage == "fig4":
        plot_figure4(args.model, args.model_size, cfg,
                     fig4_path(args.model, args.model_size, args.config))


if __name__ == "__main__":
    main()
