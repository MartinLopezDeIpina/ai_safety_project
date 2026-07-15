"""Entry point: run inference, split into classified generations, compute their activations,
then compose dynamic train/test buckets and plot Figures 2/3."""

import importlib.util
import os

from _00_run_inference import (run_all_inference, evaluate,
                               run_all_inference_thinking, evaluate_thinking)
from _01_compute_activations import compute_all_activations, compute_all_activations_thinking
from dynamic_bucket_formation import gen_buckets, gen_buckets_thinking


def _load_module(filename, name):
    """Import a sibling module whose filename isn't a valid identifier (e.g. has a dot)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_figure2 = _load_module("_02_3.1_figure2.py", "figure2")
plot_figure2 = _figure2.plot_figure2
plot_figure2_thinking = _figure2.plot_figure2_thinking
THINK_POSITIONS = _figure2.THINK_POSITIONS
NOTHINK_POSITIONS = _figure2.NOTHINK_POSITIONS
plot_figure3 = _load_module("_03_3.2_figure3.py", "figure3").plot_figure3


def _fig_path(model, model_size, fig, bucket_config):
    """output/<model><size>/<fig>[<suffix>].png (config suffix keeps parallel configs from clashing).

    The suffix is the config stem past the shared "bucket_config" prefix, so the default
    bucket_config.json adds nothing (figure2.png) and bucket_config_alt.json -> figure2_alt.png.
    """
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "output", f"{model}{model_size}")
    suffix = ""
    if bucket_config:
        stem = os.path.splitext(os.path.basename(bucket_config))[0]
        suffix = stem[len("bucket_config"):] if stem.startswith("bucket_config") else f"_{stem}"
    return os.path.join(out_dir, f"{fig}{suffix}.png")


def _main_thinking(model, model_size, left, right, stages,
                   bucket_config, bucket_config_nothink, max_len, batch_size, sampling_config,
                   use_judge=False, judge_config=None, only_datasets=None, only_modes=None,
                   use_judged_classifications=False, max_acts_per_bucket=None):
    """Qwen3.5 thinking track (see main's docstring)."""
    cfg_think = bucket_config or "bucket_config_qwen35_think.json"
    cfg_nothink = bucket_config_nothink or "bucket_config_qwen35_nothink.json"
    cfg_judge = judge_config or "configs/eval/judge_config_thinking.json"

    if "infer" in stages:
        run_all_inference_thinking(model, model_size, left, right,
                                   max_len=max_len, batch_size=batch_size,
                                   sampling_config=sampling_config, only_datasets=only_datasets,
                                   only_modes=only_modes)
    if "eval" in stages:
        evaluate_thinking(model, model_size, use_judge=use_judge, judge_config=cfg_judge,
                          only_modes=only_modes)
    if "acts" in stages:
        if use_judge:
            # activations from the judge-corrected splits -> judge_activations/ (activations/ untouched)
            compute_all_activations_thinking(model, model_size,
                                             classified_subdir="judge_classifications",
                                             acts_subdir="judge_activations", only_modes=only_modes,
                                             max_acts_per_bucket=max_acts_per_bucket)
        else:
            compute_all_activations_thinking(model, model_size, only_modes=only_modes,
                                             max_acts_per_bucket=max_acts_per_bucket)

    if any(s in stages for s in ("gen_buckets", "fig")):
        # figure2 (the only figure on this track) optionally from judge_activations/ (judge-corrected).
        # Build one grid at a time and free it before the next: each is a full set of fp16
        # (L,N,22,H) buckets, and holding both think+nothink at once OOMs at N=512.
        buckets_think = gen_buckets_thinking(model, model_size, cfg_think,
                                             use_judged=use_judged_classifications)
        if "fig" in stages:
            plot_figure2_thinking(model, model_size, buckets_think, THINK_POSITIONS,
                                  _fig_path(model, model_size, "figure2", cfg_think))
            del buckets_think
        buckets_nothink = gen_buckets_thinking(model, model_size, cfg_nothink,
                                               use_judged=use_judged_classifications)
        if "fig" in stages:
            plot_figure2_thinking(model, model_size, buckets_nothink, NOTHINK_POSITIONS,
                                  _fig_path(model, model_size, "figure2", cfg_nothink))
            del buckets_nothink


def main(model="qwen", model_size="0.5b", left=0, right=10,
         stages=("infer", "eval", "acts", "gen_buckets", "fig", "fig3"),
         bucket_config=None, thinking=False, bucket_config_nothink=None, max_len=512, batch_size=8,
         sampling_config="sampling_config.json", use_judge=False,
         judge_config=None, only_datasets=None, only_modes=None,
         use_judged_classifications=False, max_acts_per_bucket=None):
    """Run the pipeline for one model/config. `stages` selects which stages run.

    bucket_config: path to a bucket config json (relative paths resolve against this dir), or None
    for the module defaults. Consumed by the gen_buckets step and used to tag figure filenames.

    use_judge: after the eval split, re-verify the two opposing buckets (harmful->accepted,
    harmless->refused) with the judge LLM and rebucket disagreements into a parallel
    classified_generations_judge/ dir (classified_generations/ is left untouched). judge_config
    defaults per track — configs/eval/judge_config.json for instruct, judge_config_thinking.json for
    the thinking track; pass a path to override. Re-run `acts` against the judge dir for judged figures.

    thinking=True runs the Qwen3.5 thinking track instead: genthink+gennothink generation, the
    22-slot extraction, and two Figure-2 grids (think = all 22 slots, nothink = the 7 meaningful
    ones). It uses bucket_config (genthink sources) and bucket_config_nothink; both default to the
    checked-in bucket_config_qwen35_{think,nothink}.json. Inference sampling params (temperature,
    top_p, top_k, min_p, presence_penalty, do_sample) come from sampling_config (a json path,
    resolved against this dir; default sampling_config.json).

    only_datasets: comma-separated dataset names (e.g. "alpaca") to restrict the `infer` stage to; None
    generates all. A named dataset runs all of its generation configs (gentinst/gentpost or
    genthink/gennothink/gennothink_stripped). Only affects `infer`.

    only_modes: (thinking track only) comma-separated thinking modes (e.g. "gennothink_stripped") to
    restrict `infer` and `eval` to; None runs all. Scope both stages to a single generation type.

    use_judged_classifications: build FIGURE 2 from the judge-corrected splits — sources bucket
    activations from judge_activations/ instead of activations/ (produced by an eval+acts run with
    use_judge=True). Only affects figure2; figure3 still uses the standard activations/.

    max_acts_per_bucket: cap the `acts` stage to the first N rows PER bucket (per classified/judged
    split file), bounding each .pt so figure/bucket loading doesn't OOM. None = no cap; per-file, so
    buckets are never mixed.
    """
    if thinking:
        _main_thinking(model, model_size, left, right, stages,
                       bucket_config, bucket_config_nothink, max_len, batch_size, sampling_config,
                       use_judge=use_judge, judge_config=judge_config, only_datasets=only_datasets,
                       only_modes=only_modes, use_judged_classifications=use_judged_classifications,
                       max_acts_per_bucket=max_acts_per_bucket)
        return

    if "infer" in stages:
        run_all_inference(model, model_size, left, right, only_datasets=only_datasets)
    if "eval" in stages:
        evaluate(model, model_size, use_judge=use_judge,
                 judge_config=judge_config or "configs/eval/judge_config.json")
    if "acts" in stages:
        if use_judge:
            # activations from the judge-corrected splits -> judge_activations/ (activations/ untouched)
            compute_all_activations(model, model_size,
                                    classified_subdir="judge_classifications",
                                    acts_subdir="judge_activations",
                                    max_acts_per_bucket=max_acts_per_bucket)
        else:
            compute_all_activations(model, model_size, max_acts_per_bucket=max_acts_per_bucket)

    buckets = None
    if any(s in stages for s in ("gen_buckets", "fig", "fig3")):
        buckets = gen_buckets(model, model_size, bucket_config)
    if "fig" in stages:
        # figure2 optionally from the judge-corrected activations (judge_activations/); figure3 stays
        # on the standard activations/.
        fig2_buckets = (gen_buckets(model, model_size, bucket_config, use_judged=True)
                        if use_judged_classifications else buckets)
        plot_figure2(model, model_size, fig2_buckets,
                     out_path=_fig_path(model, model_size, "figure2", bucket_config))
    if "fig3" in stages:
        plot_figure3(model, model_size, buckets,
                     save_path=_fig_path(model, model_size, "figure3", bucket_config))


if __name__ == "__main__":
    # Experiments-only smoke for qwen 0.5b (activations already computed). Uncomment the GPU
    # stages to regenerate generations/classified_generations/activations from scratch.
    # main("qwen", "0.5b", stages=("infer", "eval", "acts"))
    main("qwen35", "9b",
         stages=("fig",),  # note the trailing comma
         bucket_config="configs/bucketing/bucket_config_qwen35_think.json",
         bucket_config_nothink="configs/bucketing/bucket_config_qwen35_nothink.json",
         thinking=True,
         use_judged_classifications=False)
    """
    main("qwen", "7b",
         stages=("fig", "fig3"),  # note the trailing comma
         bucket_config="configs/bucketing/bucket_config_alt.json",
         thinking=False,
         use_judged_classifications=True)
    """

    # Qwen3.5 thinking track (new model -> GPU stages are expected to run). Local RTX 5080 smoke:
    # main("qwen35", "0.8b", left=0, right=8, thinking=True,
    #      stages=("infer", "eval", "acts", "gen_buckets", "fig"))

    # gennothink_stripped smoke (generation + eval only for that one type), local GPU:
    # main("qwen35", "0.8b", left=0, right=50, thinking=True,
    #      stages=("infer", "eval"), only_modes="gennothink_stripped")

