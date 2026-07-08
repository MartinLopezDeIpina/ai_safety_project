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

# All outputs go here (.pt tensors, .json results, .png plots).
# Defaults to ./results next to this file; override with REPL_RESULTS_DIR (e.g. to route a
# per-model run into its own dir). Model-agnostic — local runs with the env unset are unaffected.
RESULTS_DIR = os.environ.get("REPL_RESULTS_DIR") or os.path.join(_HERE, "results")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2-7B-Instruct"
MODEL_TYPE = "qwen"
MODEL_SIZE = "7b"
# --- Option: Llama-3-8B-Instruct (gated → needs HF_TOKEN; modal_run.py injects it).
#     Uncomment this block to switch. Token positions auto-derive from MODEL_TYPE via
#     model_utils.get_token_positions (POST_INST_SUFFIX below), and model_utils.load_model
#     loads MODEL_NAME directly for the "8b" size. Everything else is model-agnostic. ---
#MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
#MODEL_TYPE = "llama3"
#MODEL_SIZE = "8b"
# --- Option: Qwen2.5-1.5B-Instruct (fits a 16GB GPU) ---
#MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
#MODEL_TYPE = "qwen"
#MODEL_SIZE = "1.5b"

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
# NOTE: Qwen3.5 (arch `qwen3_5`) is NOT recognised by the transformers version on the Modal
# image → it crashes load_judge_model. Use a well-supported judge (Qwen2.5-7B, `qwen2` arch).
# Non-thinking: the judge just emits ACCEPT/REFUSE (the thinking-CoT machinery was tuned for
# Qwen3.5; a 7B non-thinking judge is still far better than the 1.5B substring baseline).
JUDGE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"   # judge model (may differ from MODEL_NAME)
JUDGE_THINKING = False                  # emit ACCEPT/REFUSE directly (no <think> CoT)
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
# Token positions  (derived per-model — see model_utils.get_token_positions)
# ---------------------------------------------------------------------------
# t_inst and t_post-inst are model-specific because they depend on how many tokens the
# prompting template appends AFTER the instruction. Rather than hardcoding offsets (valid
# for only one model family), we keep the post-instruction SUFFIX string per family here as
# the single source of truth, and model_utils.get_token_positions(tokenizer) tokenizes it:
#   n = len(tokenizer(suffix, add_special_tokens=False))
#   POS_TINST     = [-(n+1)]   # last instruction content token, just before the suffix
#   POS_TPOSTINST = [-1]       # last input token before generation (last suffix token)
# So switching MODEL_TYPE (e.g. qwen -> llama3) re-derives the positions automatically — no
# manual editing. Mirrors src/extract_hidden.py's per-model inst_token logic exactly.
#
# Worked examples (n verified by tokenizing):
#   qwen  : "<|im_end|>\n<|im_start|>assistant"                          -> 4 tok -> POS_TINST=[-5]
#   llama3: "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"  -> 5 tok -> POS_TINST=[-6]
POST_INST_SUFFIX = {
    "llama3": "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
    "qwen":   "<|im_end|>\n<|im_start|>assistant",
    "vicuna": "ASSISTANT:\n",
}
DEFAULT_POST_INST_SUFFIX = "[/INST]"   # llama2 / any unlisted family

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
GPTZFUZZER_DATA = os.path.join(DATA_DIR, "GPTFuzzer-50-adv.json")         # key: bad_q — template jailbreaks (Yu et al. 2023)
HUMAN_SEED_DATA = os.path.join(DATA_DIR, "human-seed-50-adv.json")        # key: bad_q — (unused; human-written template seeds)
# Paper §2 jailbreak taxonomy = adv-suffix (GCG) + persuasion (PAP) + template (GPTFuzzer).
# GCG and PAP have no ready file in the paper's repo — they are generated ON THE TARGET
# model (Qwen2-7B) by gen_jailbreaks.py and stored full-prompt-in-`bad_q`, same structure
# as GPTFuzzer-50-adv.json.
GCG_DATA        = os.path.join(DATA_DIR, "gcg-advsuffix.json")            # key: bad_q — Advbench + GCG suffix (adv-suffix; Zou et al. 2023)
PAP_DATA        = os.path.join(DATA_DIR, "pap-persuasion.json")           # key: bad_q — persuasion-paraphrased Advbench (PAP; Zeng et al. 2024)

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
# Trimmed for a faster, preemption-tolerant run: cap each harmful source at 250 (was None =
# full ~960) so stage 00 finishes in ~20-25 min (fits inside a Modal preemption window); still
# yields plenty of accepted_harmful (~15% of Sorry-Bench) and refused_sorry for the augmented
# Latent-Guard centroid. Set back to None for the full-fidelity sample.
N_HARMFUL    = 250   # per harmful source (advbench / jbb / sorry-badq)
N_HARMLESS   = 150   # alpaca rows for accepted_harmless (already plentiful)
XSTEST_N     = 250   # all xstest prompts — the sole over-refusal (refused_harmless) source
N_JAILBREAK  = 50    # jailbreak samples per type for Figure 6
N_TRAIN      = 100   # samples per category for cluster centers / direction extraction
N_BEHAVIOR   = 100   # (legacy) default per-dataset cap, still used by older callers
N_TEST       = 20    # samples per category for steering evaluation (was 30; speeds up 05/06)
#N_HARMFUL    = 1
#N_HARMLESS   = 1
#XSTEST_N     = 1
#N_JAILBREAK  = 1
#N_TRAIN      = 1
#N_BEHAVIOR   = 1
#N_TEST       = 1

# ---------------------------------------------------------------------------
# Llama Guard 3 baseline for Table 3 (§5) — used by 08b_llama_guard_baseline.py
# ---------------------------------------------------------------------------
# The paper's Table 3 compares Latent Guard against Llama Guard 3 8B. We load it via
# transformers (GATED: needs an HF token with the Llama license accepted, read from
# HF_TOKEN / HUGGING_FACE_HUB_TOKEN). 08b classifies the SAME per-category samples that
# Latent Guard scored (counts read from table3.json) and merges accuracy back in. If the
# model is inaccessible, 08b records it as unavailable rather than failing the pipeline.
LLAMA_GUARD_BASELINE = True
LLAMA_GUARD_MODEL    = "meta-llama/Llama-Guard-3-8B"

# Latent Guard centroid (§5 / Appendix B). The base μ_harmful is the Fig-2/3 harmfulness
# pole (Advbench/JBB refused). The paper reports that for the Latent Guard specifically it
# ALSO adds "harmful instructions accepted at t_post from Advbench/JBB, OR refused harmful
# examples from Sorry-Bench" to the μ_harmful sampling pool, which improves detection of the
# mild Sorry-Bench accepts (that otherwise read harmless at t_inst on Qwen2-7B). When True,
# 08 builds an AUGMENTED μ_harmful = mean over t_inst of {refused_harmful ∪ refused_sorry ∪
# accepted_harmful-from-AUGMENT_SOURCES} and uses it as the primary Latent-Guard pole
# (falling back to the base pole if the extra activations are absent). Fig-2/3 poles are
# unchanged — this only affects Table 3.
LATENT_GUARD_AUGMENT         = True
LATENT_GUARD_AUGMENT_SOURCES = {"advbench", "jbb"}   # which accepted_harmful sources to add

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
# NOTE on the jailbreak taxonomy (Figure 6 / Table 3): the paper's three methods map to
#   jailbreak_adversarial ← GCG adv-suffix  (gcg-advsuffix.json)
#   jailbreak_persuasion  ← PAP persuasion  (pap-persuasion.json)
#   jailbreak_template    ← GPTFuzzer templates (GPTFuzzer-50-adv.json)
# Only ACCEPTED responses feed the jailbreak_* buckets, so these datasets must contain
# prompts that actually jailbreak Qwen2-7B → they are generated on the target model.
# sorry-badq now feeds ONLY accepted_harmful (naive Sorry-Bench harmful; it is NOT
# persuasion — that was a prior mislabel). human-seed is dropped (it is a second template
# set, redundant with GPTFuzzer).
DATASET_ROUTING = [
    # name          path             key            n           refused_buckets        accepted_buckets
    ("advbench",   HARMFUL_DATA,    "bad_q",       N_HARMFUL,   ["refused_harmful"],   ["accepted_harmful"]),
    ("jbb",        JBB_DATA,        "bad_q",       N_HARMFUL,   ["refused_harmful"],   ["accepted_harmful"]),
    ("sorry-badq", SORRY_DATA,      "bad_q",       N_HARMFUL,   ["refused_harmful", "refused_sorry"], ["accepted_harmful"]),
    ("alpaca",     HARMLESS_DATA,   "instruction", N_HARMLESS,  [],                    ["accepted_harmless"]),  # alpaca "refusals" are artifacts → drop
    ("xstest",     XSTEST_DATA,     "bad_q",       XSTEST_N,    ["refused_harmless"],  ["accepted_harmless"]),
    ("gcg",        GCG_DATA,        "bad_q",       N_JAILBREAK, [],                    ["jailbreak_adversarial"]),
    ("pap",        PAP_DATA,        "bad_q",       N_JAILBREAK, [],                    ["jailbreak_persuasion"]),
    ("gptzfuzzer", GPTZFUZZER_DATA, "bad_q",       N_JAILBREAK, [],                    ["jailbreak_template"]),
]

# The judge (when JUDGE=True) re-checks this bucket's accepts and moves false-accepts
# to JUDGE_MOVE_TO. Only this harmful-pool bucket is judged; poles/jailbreaks are not.
JUDGE_BUCKET   = "accepted_harmful"
JUDGE_MOVE_TO  = "refused_harmful"

# ---------------------------------------------------------------------------
# Position-specific accepted-harmful set for the Figure 2 t_inst panel
# ---------------------------------------------------------------------------
# Paper Appendix B builds the accepted-harmful set DIFFERENTLY per token position:
#   • t_post panel: acceptance under the FULL template → Advbench/JBB almost all refuse,
#     so the set is ~all (mild) Sorry-Bench. That is the `accepted_harmful` bucket above.
#   • t_inst panel: "we can more easily find accepted harmful examples at the t_inst
#     position because the refusal rate drops greatly when prompting WITHOUT post-
#     instruction tokens" (§3.1 / Table 1). Stripping the post-inst tokens collapses
#     refusal, so strongly-harmful Advbench/JBB prompts now count as accepted and enter
#     the red line — and their t_inst hidden states read positive (harmful).
#
# When True, stage 00 runs an EXTRA decode pass over the harmful accepted-sources under
# the no-post-inst template (format_prompt_no_postinst) and routes the substring-accepted
# responses into a separate `accepted_harmful_tinst` bucket used ONLY by the top panel.
# The t_inst hidden state of a prompt is identical with or without post-inst tokens
# (causal attention), so this pass only changes WHICH prompts are labelled accepted; the
# vectors are still extracted from the normal full template at POS_TINST=[-5] in stage 01.
# This set is substring-labelled and NOT judged (paper convention; a false-accept still
# has a genuinely-harmful t_inst vector, so judging is unnecessary here).
TINST_ACCEPTED_NO_POSTINST = True

# Datasets that feed the t_inst accepted-harmful red line (the no-post-inst pass). Appendix
# B aggregates Advbench+JBB+Sorry-Bench for accepted-harmful, but states Sorry-Bench is
# needed specifically for the T_POST panel (Advbench/JBB almost all refuse there); at
# t_inst Advbench/JBB are the abundant accepted source once post-inst tokens are removed.
# On Qwen2-7B the mild Sorry-Bench acceptances read harmless at t_inst (they sit on the
# alpaca pole), so restricting the t_inst red line to advbench+jbb keeps it genuinely
# harmful. Sorry-Bench still drives the T_POST red line (`accepted_harmful`). Set to None
# to use every accepted_harmful source.
TINST_ACCEPTED_SOURCES = {"advbench", "jbb"}

# Datasets whose refusals form the harmfulness/refusal cluster CENTERS (μ_harmful,
# μ_refuse). Appendix B: "We sample 100 harmful instructions refused ... from Advbench and
# JBB to compute the center of the harmfulness cluster μ_harmful and the refusal cluster
# μ_refuse" — Sorry-Bench is HELD OUT from centroid computation. refused_harmful pools all
# harmful refusals (incl. sorry-badq and judge-moved samples); stage 00 reorders it so
# these pole sources lead, and the first-N_TRAIN slice 01 averages into the poles is
# advbench+jbb only. Set to None to disable the reorder (pole = whatever leads naturally).
POLE_HARMFUL_SOURCES = {"advbench", "jbb"}

# ---------------------------------------------------------------------------
# Figure 2 poles & test lines (stages 01 extract these; 03 plots them)
# ---------------------------------------------------------------------------
# Buckets to extract hidden states for in stage 01 (at BOTH t_inst and t_post),
# capped at N_TRAIN each. Must include every pole and test-line bucket named below.
EXTRACT_CATEGORIES = [
    "refused_harmful", "accepted_harmless", "accepted_harmful", "refused_harmless",
    "accepted_harmful_tinst",
    "refused_sorry",   # Sorry-Bench refusals — t_inst only, for the §5 augmented Latent-Guard centroid
]

# Buckets extracted at t_inst ONLY (stage 01 skips the t_post pass for these). The
# no-post-inst accepted set has no post-instruction tokens, so t_post is meaningless for it;
# refused_sorry is only used to augment the t_inst harmfulness centroid for Latent Guard.
EXTRACT_TINST_ONLY = {"accepted_harmful_tinst", "refused_sorry"}

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

# The misbehaving categories plotted as the two lines — position-specific (Appendix B).
# Top panel (t_inst) uses the no-post-inst accepted set so the red line carries genuinely
# harmful Advbench/JBB content; bottom panel (t_post) uses the full-template accepted set.
FIG2_TINST_TEST_CATEGORIES = ["accepted_harmful_tinst", "refused_harmless"]
FIG2_TPOST_TEST_CATEGORIES = ["accepted_harmful", "refused_harmless"]

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

