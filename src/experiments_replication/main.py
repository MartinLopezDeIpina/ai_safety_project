"""Entry point: run inference on all datasets, then build the 6 category pools."""

from run_inference import run_all_inference, evaluate

MODEL = "qwen"
MODEL_SIZE = "0.5b"
LEFT = 0
RIGHT = 10

if __name__ == "__main__":
    run_all_inference(MODEL, MODEL_SIZE, LEFT, RIGHT)
    evaluate(MODEL, MODEL_SIZE)
