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
                   bucket_config, bucket_config_nothink, max_len, batch_size, sampling_config):
    """Qwen3.5 thinking track (see main's docstring)."""
    cfg_think = bucket_config or "bucket_config_qwen35_think.json"
    cfg_nothink = bucket_config_nothink or "bucket_config_qwen35_nothink.json"

    if "infer" in stages:
        run_all_inference_thinking(model, model_size, left, right,
                                   max_len=max_len, batch_size=batch_size,
                                   sampling_config=sampling_config)
    if "eval" in stages:
        evaluate_thinking(model, model_size)
    if "acts" in stages:
        compute_all_activations_thinking(model, model_size)

    if any(s in stages for s in ("gen_buckets", "fig")):
        buckets_think = gen_buckets_thinking(model, model_size, cfg_think)
        buckets_nothink = gen_buckets_thinking(model, model_size, cfg_nothink)
    if "fig" in stages:
        plot_figure2_thinking(model, model_size, buckets_think, THINK_POSITIONS,
                              _fig_path(model, model_size, "figure2", cfg_think))
        plot_figure2_thinking(model, model_size, buckets_nothink, NOTHINK_POSITIONS,
                              _fig_path(model, model_size, "figure2", cfg_nothink))


def main(model="qwen", model_size="0.5b", left=0, right=10,
         stages=("infer", "eval", "acts", "gen_buckets", "fig", "fig3"),
         bucket_config=None, thinking=False, bucket_config_nothink=None, max_len=512, batch_size=8,
         sampling_config="sampling_config.json"):
    """Run the pipeline for one model/config. `stages` selects which stages run.

    bucket_config: path to a bucket config json (relative paths resolve against this dir), or None
    for the module defaults. Consumed by the gen_buckets step and used to tag figure filenames.

    thinking=True runs the Qwen3.5 thinking track instead: genthink+gennothink generation, the
    22-slot extraction, and two Figure-2 grids (think = all 22 slots, nothink = the 7 meaningful
    ones). It uses bucket_config (genthink sources) and bucket_config_nothink; both default to the
    checked-in bucket_config_qwen35_{think,nothink}.json. Inference sampling params (temperature,
    top_p, top_k, min_p, presence_penalty, do_sample) come from sampling_config (a json path,
    resolved against this dir; default sampling_config.json).
    """
    if thinking:
        _main_thinking(model, model_size, left, right, stages,
                       bucket_config, bucket_config_nothink, max_len, batch_size, sampling_config)
        return

    if "infer" in stages:
        run_all_inference(model, model_size, left, right)
    if "eval" in stages:
        evaluate(model, model_size)
    if "acts" in stages:
        compute_all_activations(model, model_size)

    buckets = None
    if any(s in stages for s in ("gen_buckets", "fig", "fig3")):
        buckets = gen_buckets(model, model_size, bucket_config)
    if "fig" in stages:
        plot_figure2(model, model_size, buckets,
                     out_path=_fig_path(model, model_size, "figure2", bucket_config))
    if "fig3" in stages:
        plot_figure3(model, model_size, buckets,
                     save_path=_fig_path(model, model_size, "figure3", bucket_config))


if __name__ == "__main__":
    # Experiments-only smoke for qwen 0.5b (activations already computed). Uncomment the GPU
    # stages to regenerate generations/classified_generations/activations from scratch.
    # main("qwen", "0.5b", stages=("infer", "eval", "acts"))
    main("qwen", "7b", stages=("fig", "fig3"),
         bucket_config="bucket_config_clean.json")

    # Qwen3.5 thinking track (new model -> GPU stages are expected to run). Local RTX 5080 smoke:
    # main("qwen35", "0.8b", left=0, right=8, thinking=True,
    #      stages=("infer", "eval", "acts", "gen_buckets", "fig"))

