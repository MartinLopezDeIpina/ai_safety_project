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

#MODEL_NAME = "Qwen/Qwen2-7B-Instruct"
#MODEL_TYPE = "qwen"
#MODEL_SIZE = "7b"
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_TYPE = "qwen"
MODEL_SIZE = "1.5b"

# ---------------------------------------------------------------------------
# LLM-judge verification of accepted-harmful classifications
# ---------------------------------------------------------------------------
# The substring classifier (is_refusal / REFUSAL_PHRASE) mislabels ~10-15% of
# harmful cases: responses that restate the task then refuse, or give a safe
# workaround, carry no leading refusal phrase and are wrongly bucketed as
# "accepted". When JUDGE is True, every substring-"accepted" response on a
# HARMFUL source is re-checked by an LLM judge; if the judge says it actually
# refused, the sample is moved to "refused". Substring-"refused" samples are
# trusted as-is (the judge only rescues false-accepts). Harmless sources are
# never judged.
#
# The judge model is SEPARATE from the inference model (MODEL_NAME) above — it is
# loaded on its own (load_judge_model()) only when judging: by eval_judge.py and
# by 00_collect_behaviors.py when JUDGE is on. Set it to whatever you want to
# judge with, e.g. a larger or a "thinking" model.
JUDGE = True
JUDGE_MODEL_NAME = "Qwen/Qwen3.5-4B"   # judge model (may differ from MODEL_NAME)
JUDGE_THINKING = True                   # judge emits <think>…</think>; parse the answer after it
# Thinking judges need room to reason before the verdict; non-thinking judges only
# need to emit ACCEPT / REFUSE.
JUDGE_MAX_NEW_TOKENS = 512 if JUDGE_THINKING else 5
# Judge generation batch size. Keep small for big judges on a 16GB GPU — the 4B
# Qwen3.5 (hybrid linear-attention) OOMs at 32. Raise for small judges / big GPUs.
JUDGE_BATCH_SIZE = 8

# ---------------------------------------------------------------------------
# Figure 2 faithful replication — harmful bucket sourcing
# ---------------------------------------------------------------------------
# When True, the harmful buckets (refused_harmful / accepted_harmful) are built from
# ADVBENCH ONLY, matching the paper. This keeps the harmful pole and the harmful test
# line (accepted_harmful) in the same homogeneously-harmful corpus, so:
#   - the difference-in-means axis measures harmfulness, not advbench-vs-other style;
#   - the acceptance filter cannot self-select a benign tail (advbench has none).
# sorry-badq is still collected but routed to jailbreak_persuasion ONLY (for Fig 6),
# and catqa is skipped (not a paper source; can't be filtered to genuinely-harmful
# without the prompt-harmfulness judge, which is a separate later step).
# Set False to restore the multi-source harmful buckets (sorry-badq + catqa).
FAITHFUL_HARMFUL_BUCKETS = True

# ---------------------------------------------------------------------------
# Token positions
# ---------------------------------------------------------------------------
# The prompt built by utils.formatInp_llama_persuasion(model='qwen') is:
#   <|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant
# There is NO trailing "\n" after "assistant", so the post-instruction suffix is
# 4 tokens: <|im_end|>, \n, <|im_start|>, assistant.
#
# Tokenizing the full prompt, the last tokens are:
#   ...  {last instruction token}  <|im_end|>  \n  <|im_start|>  assistant
#         -5                        -4          -3   -2           -1
#
# tinst     = position -5 = the last token of the instruction content
#             (the paper's "last instruction token"; it is NOT <|im_end|>).
# tpost-inst = position -1 = the "assistant" token, the last input token before the
#             model starts generating (there is no trailing \n to land on).
#
# These match the original extract_hidden.py exactly: with inst_token
# '<|im_end|>\n<|im_start|>assistant' (4 tokens) and NUM_TOKEN_HIDDEN=2, its 'hf' mode
# reads merged[NUM_TOKEN_HIDDEN-1] = position -5, and its 'refuse' mode reads
# merged[-1] = position -1.  Do not change these values without re-deriving the offsets.

INST_TOKEN_LEN = 5       # offset of the last instruction token from the sequence end
POS_TINST      = [-INST_TOKEN_LEN]   # [-5] → last instruction content token
POS_TPOSTINST  = [-1]                # [-1] → "assistant" (last input token before generation)

# ---------------------------------------------------------------------------
# Data files
# ---------------------------------------------------------------------------

HARMFUL_DATA    = os.path.join(DATA_DIR, "advbench.json")                 # key: bad_q
HARMLESS_DATA   = os.path.join(DATA_DIR, "alpaca_data_instruction.json")  # key: instruction
XSTEST_DATA     = os.path.join(DATA_DIR, "xstest-harmless.json")          # key: bad_q
SORRY_DATA      = os.path.join(DATA_DIR, "sorry-badq.json")               # key: bad_q
GPTZFUZZER_DATA = os.path.join(DATA_DIR, "GPTFuzzer-50-adv.json")         # key: bad_q
HUMAN_SEED_DATA = os.path.join(DATA_DIR, "human-seed-50-adv.json")        # key: bad_q

# CatQA harmful-question sets (12 category files, key: bad_q). These are blatantly
# harmful direct questions; any the model accepts land in accepted_harmful with a
# genuinely-harmful signature at t_inst (unlike the milder sorry-badq acceptances),
# which is exactly what the Figure 2 red line needs. The *_test split is excluded.
import glob as _glob
CATQA_DATA = sorted(
    f for f in _glob.glob(os.path.join(DATA_DIR, "catqa_*.json"))
    if not f.endswith("_test.json")
)

# Sample sizes
# Harmful behavior collection: run the FULL harmful sets so more prompts get a chance
# to be accepted (advbench ~3% / sorry-badq ~15% acceptance on Qwen2.5-1.5B), growing
# accepted_harmful well beyond the ~18 we got at N=100. None = use all rows in the file.
N_HARMFUL    = None  # advbench (520) + sorry-badq (440) + catqa (12×50) in full
N_HARMLESS   = 150   # alpaca rows for accepted_harmless (already plentiful)
XSTEST_N     = 250   # all xstest prompts — the sole over-refusal (refused_harmless) source
N_JAILBREAK  = 50    # jailbreak samples per type for Figure 6
N_TRAIN      = 100   # samples per category for cluster centers / direction extraction
N_BEHAVIOR   = 100   # (legacy) default per-dataset cap, still used by older callers
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

