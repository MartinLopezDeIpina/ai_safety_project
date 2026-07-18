#!/usr/bin/env python
"""launch_intervention.py -- the full "are the directions separate?" experiment suite for ONE
(harmfulness_slot, refusal_slot) token pair, on the Qwen3.5 thinking track.

WHAT SEPARATION REQUIRES (the reason each run below exists)
----------------------------------------------------------
Figure 4 alone does NOT prove separation: if both directions elicit refusal they look SIMILAR, and
"one works, one doesn't" is not separation either. Separation is proved by an OPPOSITION in the
reply-inversion task (Figure 5):

    direction     Fig 4: elicits refusal?     Fig 5: flips the harm judgement?
    harmfulness   yes                          YES  ("No" -> "Certainly")
    refusal       yes                          NO   (stays "No")

That mismatch -- refusal elicits refusal but does not change the harm judgement, while harmfulness
does -- IS the separation. So you need both directions alive and doing different things.

Every steered line needs TWO baselines, learned the hard way on this model:
  * coeff=0            -- the unsteered baseline (Fig4 ~0% refusal; Fig5 ~100% "No").
  * random, norm-matched -- a random direction with the SAME per-layer norm as the real one.
    On Qwen3.5-9b a random vector flipped 62% of inversion answers to a clean "Certainly" while the
    real harmfulness direction flipped ~1%. Without this control the effect is indistinguishable
    from generic perturbation damage. The paper runs no such control; do not skip it.

WHAT THIS SCRIPT DOES
---------------------
--mode launch  (default):
  1. writes one self-describing config per (figure, coeff) under configs/intervene/suite_<tag>/
  2. builds the 4 steering vectors once  (CPU: hf, refusal, random-matched-to-each)
  3. fires every generation run on Modal, detached.

--mode collect:
  pulls each finished run, scores it (Fig4 -> LLM judge on Modal; Fig5 -> the strict No/Certainly
  matcher, CPU), and plots each figure. Re-run until everything is judged.

Token slots (25-slot thinking layout): 0=t_inst 1=t_post 3=<think> 4=\\n\\n(gennothink)/\\n
  5..19=CoT 20=gen1 24=</think>. The paper's directions are harmfulness@0 (t_inst),
  refusal@1 (t_post).

Examples
--------
  # the paper's own tokens, coeff ladder, 200 examples:
  python launch_intervention.py --hf-slot 0 --refusal-slot 1 --coeffs 2,4 --right 200

  # print the plan without launching anything:
  python launch_intervention.py --hf-slot 0 --refusal-slot 1 --dry-run

  # later, pull + score + plot:
  python launch_intervention.py --hf-slot 0 --refusal-slot 1 --coeffs 2,4 --right 200 --mode collect
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MODAL_SCRIPT = os.path.join(HERE, "modal_intervene.py")
INTERVENE_SCRIPT = os.path.join(HERE, "_04_intervention.py")
PY = sys.executable

# thinking mode -> the bucket config that defines harmful_all / harmless_all for that mode.
BUCKET_CONFIG = {
    "genthink": "configs/bucketing/bucket_config_qwen35_think_intervene.json",
    "gennothink": "configs/bucketing/bucket_config_qwen35_nothink_intervene.json",
    "gennothink_stripped": "configs/bucketing/bucket_config_qwen35_nothink_stripped_intervene.json",
}
SLOT_LABEL = {0: "t_inst", 1: "t_post", 3: "<think>", 4: "newline-after-think",
              10: "mid1", 15: "last1", 20: "gen1", 24: "</think>"}

# Modes whose </think> is in the PROMPT: the answer is emitted immediately, so the inversion verdict
# (No/Certainly) is the first token. The others GENERATE a reasoning trace first (~1k tokens median,
# up to ~3.5k), so the verdict only appears after the trace closes -- inv-tokens must cover it or
# every inversion row scores TRUNCATED. CoT slots 5-19 are also null for the NO_COT modes.
NO_COT_MODES = {"gennothink"}  # (gennothink_stripped_v2 too, but it has no bucket config here)


def default_inv_tokens(mode):
    return 100 if mode in NO_COT_MODES else 4096


def default_gen_tokens(mode):
    # Fig4 (no inversion) generation. genthink needs ~4096 for 100% of rows to reach an answer
    # (2048 -> ~91% closed, rest TRUNCATED). gennothink answers are short, but generation stops at
    # EOS so a high cap is a harmless upper bound.
    return 2048 if mode in NO_COT_MODES else 4096


def _load_04():
    spec = importlib.util.spec_from_file_location("i04", INTERVENE_SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------------------
# Experiment plan for one (hf_slot, refusal_slot) pair
# ---------------------------------------------------------------------------
def make_plan(args):
    """Return (vectors_cfg, [figure_cfg, ...]).

    vectors_cfg builds the 4 .pt (hf, refusal, and a random twin norm-matched to each). Each
    figure_cfg is one plottable panel at one coefficient; coeff=0 controls are separate configs
    because the run key encodes the coefficient (a coeff=0 line cannot share a config with coeff=c).
    """
    hf, rf = f"hf_s{args.hf_slot}", f"refusal_s{args.refusal_slot}"
    rhf, rrf = f"random_{hf}", f"random_{rf}"
    bucket = args.bucket_config or BUCKET_CONFIG[args.thinking_mode]
    HARMLESS, HARMFUL = args.harmless_dataset, args.harmful_dataset

    common = dict(
        bucket_config=bucket, use_judged=True, split="train",
        thinking_mode=args.thinking_mode,
        vectors=[[hf, "harmful_all", "harmless_all", args.hf_slot],
                 [rf, "harmful_all", "harmless_all", args.refusal_slot]],
        random_vectors=[[rhf, hf], [rrf, rf]], random_seed=args.random_seed,
        layer_s=args.layer_s, layer_e=args.layer_e,
        max_decode_step_while_intervene=args.decode_step,
        batch_size=args.batch_size, left=args.left, right=args.right,
    )

    # experiment tuple = [dataset, vector, reverse, prompt_key, ctx_only, all_tokens, use_inversion]
    # ctx_only is always 0: trimming the trailing template for the inversion question is qwen35- and
    # mode-specific and unimplemented, so intervention.py refuses ctx_only=1 on the thinking track.
    def E(ds, vec, rev, pk, inv):
        return [ds, vec, rev, pk, 0, 1, inv]

    figs = []
    for c in args.coeffs:
        # ---- Figure 4: each direction's ability to elicit refusal on HARMLESS prompts ----
        figs.append(dict(common, _figure="fig4", coeff=float(c),
                         max_token_generate=args.gen_tokens,
                         _comment=(f"Figure 4 (elicit refusal, harmless), coeff={c}. Lines: "
                                   f"{hf}, {rf}, and their norm-matched random twins. A real "
                                   f"direction that rises while its random twin stays flat is a "
                                   f"directional effect."),
                         experiments=[E(HARMLESS, hf, 0, "instruction", 0),
                                      E(HARMLESS, rf, 0, "instruction", 0),
                                      E(HARMLESS, rhf, 0, "instruction", 0),
                                      E(HARMLESS, rrf, 0, "instruction", 0)]))
        # ---- Figure 5a: reply inversion on HARMLESS prompts ----
        figs.append(dict(common, _figure="fig5a", coeff=float(c), eval_mode="inversion",
                         max_token_generate=args.inv_tokens,
                         _comment=(f"Figure 5a (reply inversion, harmless), coeff={c}. Model should "
                                   f"say 'No'. Prediction: harmfulness dir DROPS 'No' (judgement "
                                   f"flips); refusal dir KEEPS 'No'. Random twins must stay flat."),
                         experiments=[E(HARMLESS, hf, 0, "instruction", 1),
                                      E(HARMLESS, rf, 0, "instruction", 1),
                                      E(HARMLESS, rf, 1, "instruction", 1),   # reverse refusal
                                      E(HARMLESS, rhf, 0, "instruction", 1),
                                      E(HARMLESS, rrf, 0, "instruction", 1)]))
        # ---- Figure 5b: reply inversion on HARMFUL prompts ----
        # Point --harmful-dataset at your GCG-accepted set; naive harmful prompts the model won't
        # engage with give a degenerate baseline.
        figs.append(dict(common, _figure="fig5b", coeff=float(c), eval_mode="inversion",
                         max_token_generate=args.inv_tokens,
                         _comment=(f"Figure 5b (reply inversion, harmful), coeff={c}. Model should "
                                   f"say 'Certainly'. Prediction: REVERSE harmfulness flips it to "
                                   f"'No' (now seen as harmless); reverse refusal fails to."),
                         experiments=[E(HARMFUL, hf, 1, "bad_q", 1),         # reverse harmfulness
                                      E(HARMFUL, rf, 0, "bad_q", 1),
                                      E(HARMFUL, rf, 1, "bad_q", 1),         # reverse refusal
                                      E(HARMFUL, rhf, 0, "bad_q", 1),
                                      E(HARMFUL, rrf, 0, "bad_q", 1)]))

    # ---- coeff=0 controls: one per (dataset, inversion). vector is irrelevant at coeff 0. ----
    figs.append(dict(common, _figure="ctl_fig4", coeff=0.0, max_token_generate=args.gen_tokens,
                     _comment="Unsteered baseline for Figure 4 (harmless). Expect ~0% refusal.",
                     experiments=[E(HARMLESS, hf, 0, "instruction", 0)]))
    figs.append(dict(common, _figure="ctl_fig5a", coeff=0.0, eval_mode="inversion",
                     max_token_generate=args.inv_tokens,
                     _comment="Unsteered baseline for Figure 5a (harmless). Expect ~100% 'No'.",
                     experiments=[E(HARMLESS, hf, 0, "instruction", 1)]))
    figs.append(dict(common, _figure="ctl_fig5b", coeff=0.0, eval_mode="inversion",
                     max_token_generate=args.inv_tokens,
                     _comment="Unsteered baseline for Figure 5b (harmful). Expect mostly 'Certainly'.",
                     experiments=[E(HARMFUL, hf, 0, "bad_q", 1)]))

    vectors_cfg = dict(common, _figure="vectors",
                       _comment=f"Vector-build config for the {hf} / {rf} pair.",
                       coeff=1.0, max_token_generate=args.gen_tokens, experiments=[])
    return vectors_cfg, figs


def suite_dir(args):
    tag = f"{args.thinking_mode}_hf{args.hf_slot}_ref{args.refusal_slot}"
    return os.path.join(HERE, "configs", "intervene", f"suite_{tag}"), tag


def write_configs(args, vectors_cfg, figs):
    d, tag = suite_dir(args)
    os.makedirs(d, exist_ok=True)
    paths = {}
    for cfg in [vectors_cfg] + figs:
        name = cfg["_figure"] if cfg["_figure"] == "vectors" else f"{cfg['_figure']}_c{cfg['coeff']:g}"
        p = os.path.join(d, f"{name}.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        paths[name] = p
    print(f"wrote {len(paths)} configs to {os.path.relpath(d, HERE)}/")
    return d, tag, paths


# ---------------------------------------------------------------------------
# launch
# ---------------------------------------------------------------------------
def modal_cmd(args, cfg, dataset, vector, reverse, prompt_key, use_inv):
    """One `modal run --detach` command for a single steered run."""
    return [PY, "-m", "modal", "run", "--detach", MODAL_SCRIPT,
            "--model", args.model, "--model-size", args.model_size,
            "--thinking-mode", args.thinking_mode,
            "--dataset", dataset, "--vector", vector,
            "--reverse-intervention", str(reverse),
            "--arg-key-prompt", prompt_key,
            "--intervene-context-only", "0", "--intervene-all", "1",
            "--use-inversion", str(use_inv), "--inversion-prompt-idx", str(args.inversion_prompt_idx),
            "--layer-s", str(args.layer_s), "--layer-e", str(args.layer_e),
            "--coeff-select", str(cfg["coeff"]),
            "--left", str(args.left), "--right", str(args.right),
            "--max-token-generate", str(cfg["max_token_generate"]),
            "--batch-size", str(args.batch_size),
            "--max-decode-step-while-intervene", str(args.decode_step),
            "--gpu", args.gpu, "--no-wait"]


def unique_runs(figs):
    """Dedup runs across figures by their identity (a vector can appear in several panels)."""
    seen, out = set(), []
    for cfg in figs:
        for ds, vec, rev, pk, _ctx, _all, inv in cfg["experiments"]:
            key = (ds, vec, rev, inv, cfg["coeff"])
            if key in seen:
                continue
            seen.add(key)
            out.append((cfg, ds, vec, rev, pk, inv))
    return out


def launch(args, vectors_cfg, figs, vectors_path):
    if not args.skip_vectors:
        print("\n=== building steering vectors (CPU) ===")
        cmd = [PY, INTERVENE_SCRIPT, "--stage", "vectors", "--config", vectors_path,
               "--model", args.model, "--model-size", args.model_size]
        if args.dry_run:
            print("  " + " ".join(cmd))
        else:
            subprocess.run(cmd, check=True, cwd=HERE)

    runs = unique_runs(figs)
    print(f"\n=== launching {len(runs)} generation runs (detached, {args.gpu}) ===")
    for i, (cfg, ds, vec, rev, pk, inv) in enumerate(runs, 1):
        cmd = modal_cmd(args, cfg, ds, vec, rev, pk, inv)
        direction = "reverse" if rev else "forward"
        print(f"  [{i:>2}/{len(runs)}] {cfg['_figure']:9s} {vec:18s} {direction:7s} "
              f"coeff={cfg['coeff']:g} inv={inv} {ds}")
        if args.dry_run:
            print("       " + " ".join(cmd))
            continue
        subprocess.run(cmd, check=True, cwd=os.path.dirname(HERE) + "/..")
        time.sleep(args.stagger)
    if not args.dry_run:
        print(f"\nAll launched. Collect + score + plot later with the SAME flags plus "
              f"--mode collect.")


# ---------------------------------------------------------------------------
# collect: pull, score, plot
# ---------------------------------------------------------------------------
def collect(args, figs, paths):
    m04 = _load_04()
    runs = unique_runs(figs)

    print(f"\n=== pulling {len(runs)} runs ===")
    for cfg, ds, vec, rev, pk, inv in runs:
        cmd = [PY, "-m", "modal", "run", MODAL_SCRIPT, "--collect-only",
               "--model", args.model, "--model-size", args.model_size,
               "--thinking-mode", args.thinking_mode, "--dataset", ds, "--vector", vec,
               "--reverse-intervention", str(rev), "--left", str(args.left),
               "--right", str(args.right), "--use-inversion", str(inv),
               "--coeff-select", str(cfg["coeff"]), "--layer-s", str(args.layer_s),
               "--layer-e", str(args.layer_e),
               "--max-decode-step-while-intervene", str(args.decode_step)]
        if args.dry_run:
            print("  " + " ".join(cmd)); continue
        subprocess.run(cmd, cwd=os.path.dirname(HERE) + "/..")

    if args.dry_run:
        return

    # Score every run dir. Fig5 (inversion) is CPU via the strict matcher; Fig4 (no inversion) needs
    # the LLM judge on Modal -- launched here detached, so a second `collect` pass picks up labels.
    print("\n=== scoring ===")
    for cfg, ds, vec, rev, pk, inv in runs:
        key = m04.run_key(args.model, args.model_size, ds, vec, rev, args.left, args.right, inv,
                          cfg["coeff"], args.layer_s, args.layer_e, args.thinking_mode,
                          args.decode_step)
        d = m04.run_dir(args.model, args.model_size, key)
        if not os.path.isdir(d):
            print(f"  [missing] {key}"); continue
        if cfg.get("eval_mode") == "inversion":
            m04.score_inversion_run(d, thinking_mode=args.thinking_mode, strict=True)
        else:
            # already judged? (labels present) -> skip. else launch the Modal judge, detached.
            if any(f.endswith(".labels.json") for f in os.listdir(d)):
                continue
            jcmd = [PY, "-m", "modal", "run", "--detach", MODAL_SCRIPT,
                    "--model", args.model, "--model-size", args.model_size,
                    "--thinking-mode", args.thinking_mode, "--dataset", ds, "--vector", vec,
                    "--reverse-intervention", str(rev), "--left", str(args.left),
                    "--right", str(args.right), "--use-inversion", str(inv),
                    "--coeff-select", str(cfg["coeff"]), "--layer-s", str(args.layer_s),
                    "--layer-e", str(args.layer_e),
                    "--max-decode-step-while-intervene", str(args.decode_step),
                    "--judge-only", "--judge-batch-size", str(args.judge_batch_size),
                    "--gpu", args.gpu, "--no-wait"]
            print(f"  [judge] {key}")
            subprocess.run(jcmd, cwd=os.path.dirname(HERE) + "/..")

    # Plot every non-control figure config that has labels on disk.
    print("\n=== plotting ===")
    for name, p in paths.items():
        if name.startswith("ctl_") or name == "vectors":
            continue
        cmd = [PY, INTERVENE_SCRIPT, "--stage", "fig4", "--config", p,
               "--model", args.model, "--model-size", args.model_size]
        subprocess.run(cmd, cwd=HERE)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf-slot", type=int, required=True, help="harmfulness token slot (paper: 0)")
    ap.add_argument("--refusal-slot", type=int, required=True, help="refusal token slot (paper: 1)")
    ap.add_argument("--thinking-mode", default="gennothink",
                    choices=list(BUCKET_CONFIG), help="which activations / prompt template")
    ap.add_argument("--bucket-config", default=None,
                    help="override the default bucket config (which datasets build the vectors), "
                         "e.g. configs/bucketing/bucket_config_qwen35_nothink_intervene_gcg.json")
    ap.add_argument("--model", default="qwen35")
    ap.add_argument("--model-size", default="9b")
    ap.add_argument("--coeffs", default="2", help="comma list of steering coefficients, e.g. 2,4")
    ap.add_argument("--layer-s", type=int, default=8)
    ap.add_argument("--layer-e", type=int, default=20, help="exclusive")
    ap.add_argument("--decode-step", type=int, default=1,
                    help="1 = steer prompt only (prefill); -1 = steer every generated token")
    ap.add_argument("--harmless-dataset", default="alpaca_data_instruction")
    ap.add_argument("--harmful-dataset", default="advbench",
                    help="Fig5b dataset; point at your GCG-accepted set for real acceptances")
    ap.add_argument("--left", type=int, default=0)
    ap.add_argument("--right", type=int, default=50)
    ap.add_argument("--gen-tokens", type=int, default=None,
                    help="max_new_tokens for Fig4 generation (default: mode-aware, 2048/4096)")
    ap.add_argument("--inv-tokens", type=int, default=None,
                    help="max_new_tokens for inversion (default: 100 for gennothink; 4096 for "
                         "modes that generate a reasoning trace before answering)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--judge-batch-size", type=int, default=64)
    ap.add_argument("--inversion-prompt-idx", type=int, default=1)
    ap.add_argument("--random-seed", type=int, default=0)
    ap.add_argument("--gpu", default="B300")
    ap.add_argument("--stagger", type=float, default=3.0, help="seconds between launches")
    ap.add_argument("--figures", default="fig4,fig5a,fig5b",
                    help="comma subset of {fig4,fig5a,fig5b} to run. fig5a alone is the cheapest "
                         "decisive test: inversion is CPU-scored (no LLM judge), and its coeff=0 + "
                         "random controls tell you if a direction moves behaviour at all.")
    ap.add_argument("--mode", choices=["launch", "collect"], default="launch")
    ap.add_argument("--skip-vectors", action="store_true", help="vectors already built")
    ap.add_argument("--dry-run", action="store_true", help="print commands, do nothing")
    args = ap.parse_args()
    args.coeffs = [c.strip() for c in args.coeffs.split(",") if c.strip()]

    if args.thinking_mode not in BUCKET_CONFIG:
        sys.exit(f"no bucket config for mode {args.thinking_mode!r}")

    # Mode-aware token defaults: the inversion verdict is immediate for gennothink but sits after a
    # ~1k-token reasoning trace for genthink, so a fixed 100 would score every genthink row TRUNCATED.
    if args.inv_tokens is None:
        args.inv_tokens = default_inv_tokens(args.thinking_mode)
    if args.gen_tokens is None:
        args.gen_tokens = default_gen_tokens(args.thinking_mode)

    has_cot = args.thinking_mode not in NO_COT_MODES
    # CoT slots (5..19) are only populated when the model generates a trace.
    for which, slot in (("hf", args.hf_slot), ("refusal", args.refusal_slot)):
        if 5 <= slot <= 19 and not has_cot:
            sys.exit(f"slot {slot} ({which}) is a CoT slot, which is NULL for mode "
                     f"{args.thinking_mode!r}. Use a genthink/gennothink_stripped mode, or a "
                     f"non-CoT slot (0-4, 20-24).")

    print(f"harmfulness @ slot {args.hf_slot} ({SLOT_LABEL.get(args.hf_slot, '?')}) | "
          f"refusal @ slot {args.refusal_slot} ({SLOT_LABEL.get(args.refusal_slot, '?')}) | "
          f"mode {args.thinking_mode} | coeffs {args.coeffs} | "
          f"inv_tokens {args.inv_tokens} | gen_tokens {args.gen_tokens}")
    if has_cot and args.decode_step == 1:
        print("  NOTE: --decode-step 1 steers the PROMPT only. On a model that reasons for ~1k "
              "tokens before answering, the steer may wash out before the verdict. If the curves "
              "come out flat with a clean coeff=0 control, retry with --decode-step -1.")

    vectors_cfg, figs = make_plan(args)

    # Keep only the requested figures (plus their coeff=0 controls). Lets you run e.g. just fig5a --
    # the cheapest decisive probe -- before paying for fig4's judge or fig5b's second panel.
    selected = {f.strip() for f in args.figures.split(",") if f.strip()}
    unknown = selected - {"fig4", "fig5a", "fig5b"}
    if unknown:
        sys.exit(f"unknown figure(s): {sorted(unknown)}; choose from fig4, fig5a, fig5b")

    def _base(fig):
        return fig[len("ctl_"):] if fig.startswith("ctl_") else fig
    figs = [f for f in figs if _base(f["_figure"]) in selected]

    _, tag, paths = write_configs(args, vectors_cfg, figs)
    vectors_path = paths["vectors"]

    if args.mode == "launch":
        launch(args, vectors_cfg, figs, vectors_path)
    else:
        collect(args, figs, paths)


if __name__ == "__main__":
    main()
