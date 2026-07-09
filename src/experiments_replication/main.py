"""Entry point: run inference, build the 6 category pools, compute their activations."""

from _00_run_inference import run_all_inference, evaluate
from _01_compute_activations import compute_all_activations

MODEL = "qwen"
MODEL_SIZE = "0.5b"
LEFT = 0
RIGHT = 10

if __name__ == "__main__":
    # run_all_inference(MODEL, MODEL_SIZE, LEFT, RIGHT)  # already run
    evaluate(MODEL, MODEL_SIZE)
    compute_all_activations(MODEL, MODEL_SIZE)
