"""Run or collect all intervention experiments on Modal.
"""

import argparse
import subprocess
import time

SCRIPT = "src/experiments_replication/modal_intervene.py"
DELAY = 10  # seconds between modal run calls

# Each experiment tuple:
#   (dataset, vector, reverse, prompt_key, ctx_only, all_tokens, use_inversion)
EXPERIMENTS = [
    # Harmful datasets (prompt key: "bad_q")
    ("advbench",              "hf",      1, "bad_q",       1, 0, 1),  # less-harm
    ("advbench",              "refusal", 1, "bad_q",       0, 1, 1),  # less-refusal
    ("advbench",              "refusal", 0, "bad_q",       0, 1, 1),  # more-refusal
    ("jbb",                   "hf",      1, "bad_q",       1, 0, 1),  # less-harm
    ("jbb",                   "refusal", 1, "bad_q",       0, 1, 1),  # less-refusal
    ("jbb",                   "refusal", 0, "bad_q",       0, 1, 1),  # more-refusal

    # Harmless datasets
    ("alpaca_data_instruction", "hf",      0, "instruction", 1, 0, 1),  # more-harm
    ("alpaca_data_instruction", "refusal", 0, "instruction", 0, 1, 1),  # more-refusal
    ("alpaca_data_instruction", "refusal", 1, "instruction", 0, 1, 1),  # less-refusal
    ("xstest-harmless",         "hf",      0, "bad_q",       1, 0, 1),  # more-harm
    ("xstest-harmless",         "refusal", 0, "bad_q",       0, 1, 1),  # more-refusal
    ("xstest-harmless",         "refusal", 1, "bad_q",       0, 1, 1),  # less-refusal

    # # No-inversion runs on alpaca (respond directly, no inversion question)
    # ("alpaca_data_instruction", "hf",      0, "instruction", 0, 1, 0),  # more-harm (no inversion)
    # ("alpaca_data_instruction", "refusal", 0, "instruction", 0, 1, 0),  # more-refusal (no inversion)
]


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
