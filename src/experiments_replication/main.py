"""Entry point: run inference, split into classified generations, compute their activations,
then compose dynamic train/test buckets and plot Figures 2/3."""

import importlib.util
import os

from _00_run_inference import run_all_inference, evaluate
from _01_compute_activations import compute_all_activations
from dynamic_bucket_formation import gen_buckets


def _load_module(filename, name):
    """Import a sibling module whose filename isn't a valid identifier (e.g. has a dot)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


plot_figure2 = _load_module("_02_3.1_figure2.py", "figure2").plot_figure2
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


def main(model="qwen", model_size="0.5b", left=0, right=10,
         stages=("infer", "eval", "acts", "gen_buckets", "fig", "fig3"),
         bucket_config=None):
    """Run the pipeline for one model/config. `stages` selects which stages run.

    bucket_config: path to a bucket config json (relative paths resolve against this dir), or None
    for the module defaults. Consumed by the gen_buckets step and used to tag figure filenames.
    """
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
    main("qwen", "7b", stages=("gen_buckets", "fig", "fig3"),
         bucket_config="bucket_config_alt.json")

