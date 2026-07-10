"""Entry point: run inference, build the 6 category pools, compute their activations."""

import importlib.util
import os

from _00_run_inference import run_all_inference, evaluate
from _01_compute_activations import compute_all_activations
import rebucket_store


def _load_module(filename, name):
    """Import a sibling module whose filename isn't a valid identifier (e.g. has a dot)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


plot_figure2 = _load_module("_02_3.1_figure2.py", "figure2").plot_figure2
plot_figure3 = _load_module("_03_3.2_figure3.py", "figure3").plot_figure3

def main(model="qwen", model_size="0.5b", left=0, right=10,
         stages=("infer", "eval", "acts", "fig")):
    """Run the pipeline for one model/config. `stages` selects which stages run."""
    if "infer" in stages:
        run_all_inference(model, model_size, left, right)
    if "eval" in stages:
        evaluate(model, model_size)
    if "acts" in stages:
        compute_all_activations(model, model_size)
    if "rebucket" in stages:
        # corrected, CPU-only re-bucketing from existing activations (see EXPERIMENT_LOG.md);
        # writes buckets_activations_v2/. Reuses saved activations, no GPU needed.
        rebucket_store.main(model, model_size,
                            tinst_accepted_sources=("advbench", "jbb"),
                            tpost_accepted_sources=("sorrybench",),
                            harmless_sources=("alpaca",))
    if "fig" in stages:
        plot_figure2(model, model_size)
    if "fig_v2" in stages:
        # definitive Figure 2 from the corrected v2 buckets
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", f"{model}{model_size}")
        plot_figure2(model, model_size,
                     acts_dir=os.path.join(base, "buckets_activations_v2"),
                     out_path=os.path.join(base, "figure2.png"))
    if "fig3" in stages:
        plot_figure3(model, model_size)


if __name__ == "__main__":
    # Regenerate the definitive Figure 2 for qwen7b from existing activations (no GPU):
    main("qwen_before", "7b", stages=("fig"))

