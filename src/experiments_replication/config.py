"""
Shared configuration for all replication experiments.

All paths are absolute and derived from the location of this file,
so scripts can be run from any working directory.
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "../.."))

# Directory containing original paper source code (extract_hidden.py, utils.py, etc.)
SRC_DIR = os.path.join(REPO_ROOT, "src")

# Raw data files
DATA_DIR = os.path.join(REPO_ROOT, "data")

# All outputs go here (.pt tensors, .json results, .png plots)
RESULTS_DIR = os.path.join(_HERE, "results")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2-7B-Instruct"
MODEL_TYPE = "qwen"
MODEL_SIZE = "7b"

# ---------------------------------------------------------------------------
# Token positions
# ---------------------------------------------------------------------------
# The post-instruction marker for Qwen2 is:
#   <|im_end|>  \n  <|im_start|>  assistant  \n   (5 tokens)
#
# tinst     = position of <|im_end|>  (the END of the user's instruction)
# tpost-inst = position of last \n    (the last token before model generation)
#
# With the full input `<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n`:
#   positions [-5] = <|im_end|>   → tinst
#   positions [-1] = \n (last)    → tpost-inst

INST_TOKEN_LEN = 5       # verified via tokenizer('<|im_end|>\n<|im_start|>assistant\n')
POS_TINST      = [-INST_TOKEN_LEN]   # [-5] → <|im_end|>
POS_TPOSTINST  = [-1]                # last input token before generation

# ---------------------------------------------------------------------------
# Data files
# ---------------------------------------------------------------------------

HARMFUL_DATA    = os.path.join(DATA_DIR, "advbench.json")                 # key: bad_q
HARMLESS_DATA   = os.path.join(DATA_DIR, "alpaca_data_instruction.json")  # key: instruction
XSTEST_DATA     = os.path.join(DATA_DIR, "xstest-harmless.json")          # key: bad_q
SORRY_DATA      = os.path.join(DATA_DIR, "sorry-badq.json")               # key: bad_q
GPTZFUZZER_DATA = os.path.join(DATA_DIR, "GPTFuzzer-50-adv.json")         # key: bad_q
HUMAN_SEED_DATA = os.path.join(DATA_DIR, "human-seed-50-adv.json")        # key: bad_q

# Sample sizes
N_BEHAVIOR   = 100   # samples per dataset for behavior collection
XSTEST_N     = 250   # use all xstest prompts to find over-refusals
N_JAILBREAK  = 50    # jailbreak samples per type for Figure 6
N_TRAIN      = 50    # samples per category for direction extraction
N_TEST       = 30    # samples per category for evaluation

# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------

# Must be 1: extract_hidden.get_mean_activations uses torch.stack across batches,
# which fails when the last batch is smaller (different tensor shapes).
BATCH_SIZE = 1

# ---------------------------------------------------------------------------
# Steering experiments (§3.4 Figure 4, §3.5 Figure 5)
# ---------------------------------------------------------------------------

# Single coefficient used for ALL per-layer steering experiments.
# Large enough to clearly elicit behavior changes.
STEERING_COEFF = 20

