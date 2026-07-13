"""Run or collect all intervention experiments on Modal.
"""

import argparse
import subprocess
import time

SCRIPT = "src/experiments_replication/modal_intervene.py"
DELAY = 5  # seconds between modal run calls

# Datasets grouped by type. Harmful datasets use "bad_q"; harmless use the noted prompt key.
HARMFUL_DATASETS = ["advbench", "jbb"]
HARMLESS_DATASETS = ["alpaca_data_instruction", "xstest-harmless"]
HARMLESS_PROMPT_KEYS = {
    "alpaca_data_instruction": "instruction",
    "xstest-harmless": "bad_q"
}

# First flag is for context-only, second is for all tokens.
INTERVENTION_SCOPE = {"hf": (1, 0), "refusal": (0, 1)}

# Harmful datasets: reverse hf (less-harm), reverse refusal (less-refusal), refusal (more-refusal).
# Harmless datasets: hf (more-harm), refusal (more-refusal), reverse refusal (less-refusal).
HARMFUL_VARIANTS = [("hf", 1), ("refusal", 1), ("refusal", 0)]
HARMLESS_VARIANTS = [("hf", 0), ("refusal", 0), ("refusal", 1)]


def build_experiments() -> list[tuple]:
    """Generate the experiment table from the dataset/vector definitions above."""
    experiments = []
    for ds in HARMFUL_DATASETS:
        for vector, reverse in HARMFUL_VARIANTS:
            ctx, all_ = INTERVENTION_SCOPE[vector]
            # prompt key for harmful datasets is always "bad_q"
            experiments.append((ds, vector, reverse, "bad_q", ctx, all_, 1))
    for ds in HARMLESS_DATASETS:
        for vector, reverse in HARMLESS_VARIANTS:
            ctx, all_ = INTERVENTION_SCOPE[vector]
            experiments.append((ds, vector, reverse, HARMLESS_PROMPT_KEYS[ds], ctx, all_, 1))
    # No-inversion runs on alpaca: steer along hf and refusal without the inversion prompt.
    for vector, reverse in [("hf", 0), ("refusal", 0)]:
        ctx, all_ = INTERVENTION_SCOPE[vector]
        experiments.append(("alpaca_data_instruction", vector, reverse, "instruction", ctx, all_, 0))
    return experiments

EXPERIMENTS = build_experiments()


def main():
    parser = argparse.ArgumentParser(description="Run or collect intervention experiments on Modal")
    parser.add_argument("--mode", choices=["run", "collect"], required=True,
                        help="'run' to launch detached, 'collect' to pull results")
    parser.add_argument("--runs", default="qwen:7b:0:500", type=str,
                        help="Run spec: model:size[:left[:right]] (default: qwen:7b:0:500)")
    parser.add_argument("--delay", default=DELAY, type=int,
                        help=f"Seconds between modal calls (default: {DELAY})")
    args = parser.parse_args()

    is_run = args.mode == "run"

    for i, (dataset, vector, reverse, prompt, ctx, all_, use_inv) in enumerate(EXPERIMENTS, 1):
        direction = "less" if reverse else "more"
        action = "Launching" if is_run else "Collecting"
        print(f"=== {action} {i}. {dataset} / {vector} / {direction} ===")

        cmd = ["modal", "run"]
        if is_run:
            cmd.append("--detach")
        cmd.append(SCRIPT)
        cmd += [
            "--runs", args.runs,
            "--datasets", dataset,
            "--vectors", vector,
            "--reverse-intervention", str(reverse),
            "--arg-key-prompt", prompt,
            "--intervene-context-only", str(ctx),
            "--intervene-all", str(all_),
            "--use-inversion", str(use_inv),
        ]
        if is_run:
            cmd.append("--no-wait")
        else:
            cmd.append("--collect-only")

        subprocess.run(cmd, check=True)

        if i < len(EXPERIMENTS):
            print(f"Sleeping {args.delay}s to avoid rate limit...")
            time.sleep(args.delay)

    if is_run:
        print(f"\nAll {len(EXPERIMENTS)} runs launched detached.")
        print("Collect results later with:")
        print(f"  uv run python {__file__} --mode collect --runs \"{args.runs}\"")
    else:
        print(f"\nDone. Results are in src/experiments_replication/intervention_outputs/")


if __name__ == "__main__":
    main()
