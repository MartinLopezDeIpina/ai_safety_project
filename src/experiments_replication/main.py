"""Entry point: run inference, build the 6 category pools, compute their activations."""

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

MODEL = "qwen"
MODEL_SIZE = "0.5b"
LEFT = 0
RIGHT = 10

if __name__ == "__main__":
    # run_all_inference(MODEL, MODEL_SIZE, LEFT, RIGHT)  # already run
    # evaluate(MODEL, MODEL_SIZE)  # already run
    #compute_all_activations(MODEL, MODEL_SIZE)  # already run
    plot_figure2(MODEL, MODEL_SIZE)
