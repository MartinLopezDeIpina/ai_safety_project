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

# Judge generation parameters — abstracted here so they can be tuned without editing
# model_utils. These are Qwen3's recommended THINKING-mode params for precise tasks
# (temp 0.6 / top_p 0.95 / top_k 20 / min_p 0), plus penalties to break the thinking
# judge's self-repetition ("Wait… Wait…") loop that otherwise never closes </think>.
#
# NOTE: transformers (5.x) has NO native `presence_penalty`; judge_accepted_batch
# applies it via a custom logits processor (OpenAI semantics: subtract the penalty
# from the logit of any token already present in the GENERATED continuation). All
# other keys go straight to model.generate().
JUDGE_GEN = {
    "do_sample": True,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "repetition_penalty": 1.0,   # native (multiplicative)
    "presence_penalty": 1.5,     # custom processor (additive, generated tokens only)
}

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

# t_post position experiment (§3.5 investigation).
# The default t_post = the pre-generation "assistant" token (POS_TPOSTINST=[-1]),
# which encodes the prompt's refusal-PROPENSITY rather than the realised accept/refuse
# decision. When True, t_post instead = the FIRST GENERATED token: we greedily generate
# one response token, append it, and read the hidden state AT that token — where the
# accept/refuse decision actually manifests. Only affects the t_post extraction in 01
# (tinst is unchanged). See extract_first_gen_token_states in model_utils.
TPOST_FIRST_GEN_TOKEN = False

# ---------------------------------------------------------------------------
# Data files
# ---------------------------------------------------------------------------

HARMFUL_DATA    = os.path.join(DATA_DIR, "advbench.json")                 # key: bad_q
JBB_DATA        = os.path.join(DATA_DIR, "jbb.json")                      # key: bad_q — JailbreakBench harmful (100)
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
# Dataset -> bucket routing (stage 00)   [config-driven; edit here, not the code]
# ---------------------------------------------------------------------------
# Each row routes a dataset's greedy-decoded responses into behaviour buckets by the
# substring refusal label:
#   (name, path, key, n, refused_buckets, accepted_buckets)
# refused_buckets / accepted_buckets are LISTS of bucket names ([] = drop those
# responses). A dataset may feed several buckets (e.g. sorry-badq accepts feed both a
# within-corpus refusal pole AND the Figure-6 persuasion category). This replaces the
# old hardcoded per-dataset blocks in 00_collect_behaviors.py, so re-sourcing any bucket
# is a config edit. To restore multi-source harmful buckets, add advbench/sorry/catqa to
# refused_harmful / accepted_harmful here.
# Paper §2 "Datasets": harmful instructions = Advbench + JBB + Sorry-Bench; harmless =
# ALPACA; over-refusal (harmless) = Xstest. So ALL harmful sources feed the harmful
# buckets (refused_harmful / accepted_harmful) by behaviour — NOT advbench-only (that
# was a prior modification, not the paper). NOTE: JBB (JailbreakBench) is NOT in data/;
# add its file + a row here to fully match the paper.
DATASET_ROUTING = [
    # name          path             key            n           refused_buckets        accepted_buckets
    ("advbench",   HARMFUL_DATA,    "bad_q",       N_HARMFUL,   ["refused_harmful"],   ["accepted_harmful"]),
    ("jbb",        JBB_DATA,        "bad_q",       N_HARMFUL,   ["refused_harmful"],   ["accepted_harmful"]),
    ("sorry-badq", SORRY_DATA,      "bad_q",       N_HARMFUL,   ["refused_harmful"],   ["accepted_harmful", "jailbreak_persuasion"]),
    ("alpaca",     HARMLESS_DATA,   "instruction", N_HARMLESS,  [],                    ["accepted_harmless"]),  # alpaca "refusals" are artifacts → drop
    ("xstest",     XSTEST_DATA,     "bad_q",       XSTEST_N,    ["refused_harmless"],  ["accepted_harmless"]),
    ("human-seed", HUMAN_SEED_DATA, "bad_q",       N_JAILBREAK, [],                    ["jailbreak_adversarial"]),
    ("gptzfuzzer", GPTZFUZZER_DATA, "bad_q",       N_JAILBREAK, [],                    ["jailbreak_template"]),
]

# The judge (when JUDGE=True) re-checks this bucket's accepts and moves false-accepts
# to JUDGE_MOVE_TO. Only this harmful-pool bucket is judged; poles/jailbreaks are not.
JUDGE_BUCKET   = "accepted_harmful"
JUDGE_MOVE_TO  = "refused_harmful"

# ---------------------------------------------------------------------------
# Figure 2 poles & test lines (stages 01 extract these; 03 plots them)
# ---------------------------------------------------------------------------
# Buckets to extract hidden states for in stage 01 (at BOTH t_inst and t_post),
# capped at N_TRAIN each. Must include every pole and test-line bucket named below.
EXTRACT_CATEGORIES = [
    "refused_harmful", "accepted_harmless", "accepted_harmful", "refused_harmless",
]

# Figure 2 axis poles (bucket names). Paper §3.2 uses the SAME diagonal poles at both
# positions: Cl_refused_harmful vs Cl_accepted_harmless. t_inst axis reads as
# harmfulness; the same axis at t_post reads as refusal.
FIG2_TINST_POLE_POS = "refused_harmful"      # harmful  pole @ t_inst
FIG2_TINST_POLE_NEG = "accepted_harmless"    # harmless pole @ t_inst
FIG2_TPOST_POLE_POS = "refused_harmful"      # refused  pole @ t_post
FIG2_TPOST_POLE_NEG = "accepted_harmless"    # accepted pole @ t_post
# (These are configurable so you can still try a within-corpus refusal axis at t_post
#  — e.g. add "refused_sorry"/"accepted_sorry" buckets via DATASET_ROUTING +
#  EXTRACT_CATEGORIES and point the two t_post poles at them.)

# The misbehaving categories plotted as the two lines.
FIG2_TEST_CATEGORIES = ["accepted_harmful", "refused_harmless"]

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

