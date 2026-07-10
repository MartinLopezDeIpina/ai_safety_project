"""Entry point: run inference, split into classified generations, compute their activations."""

import importlib.util
import os

from _00_run_inference import run_all_inference, evaluate
from _01_compute_activations import compute_all_activations


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
    # NOTE: the fig/fig3 stages still read the old buckets_activations/ layout and will be
    # migrated to dynamic_bucket_formation.build_splits in a later change.
    if "fig" in stages:
        plot_figure2(model, model_size)
    if "fig3" in stages:
        plot_figure3(model, model_size)


if __name__ == "__main__":
    # qwen 0.5b smoke test of the GPU stages (writes datasets_outputs/{generations,
    # classified_generations,activations}/); then run dynamic_bucket_formation.py to split.
    main("qwen", "0.5b", stages=("infer", "eval", "acts"))

