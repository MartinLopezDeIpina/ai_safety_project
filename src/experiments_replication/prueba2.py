"""
Qwen3.5-0.8B: dump token sequences for a prompt + real generation,
with and without the reasoning trace. Verbose logging at every step so
a silent hang can be located immediately.

Run unbuffered:   python -u qwen35_think_tokens.py
Force offline:     HF_HUB_OFFLINE=1 python -u qwen35_think_tokens.py
"""

import os
import sys
import time
import logging

# ---- logging setup: timestamps + elapsed, flushed immediately ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d | +%(relativeCreated)7.0fms | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("qwen")


def step(msg):
    log.info(msg)
    sys.stdout.flush()


step("script start")
step(f"python={sys.version.split()[0]}  HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE','0')}")

step("importing torch ...")
import torch
step(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    step(f"gpu: {torch.cuda.get_device_name(0)}")

step("importing transformers ...")
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
step(f"transformers {transformers.__version__}")

MODEL_ID = "Qwen/Qwen3.5-0.8B"
QUESTION = "What is 2 + 2?"
MAX_NEW = 256

# ---------------------------------------------------------------
# TOKENIZER  (cheap — needs no weights)
# ---------------------------------------------------------------
step(f"loading tokenizer: {MODEL_ID} ...")
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
step(f"tokenizer loaded in {time.time()-t0:.2f}s")

step("=== special token ids ===")
for s in ["<think>", "</think>", "<|im_start|>", "<|im_end|>"]:
    ids = tokenizer(s, add_special_tokens=False).input_ids
    step(f"  {s!r:<16} -> {ids}")


def build_prompt(enable_thinking):
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": QUESTION}],
            tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        step("  (enable_thinking kwarg rejected; retrying via chat_template_kwargs)")
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": QUESTION}],
            tokenize=False, add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": enable_thinking},
        )


step("building both prompt strings (tokenizer-only, no model needed) ...")
prompt_no = build_prompt(False)
prompt_th = build_prompt(True)
step("NON-THINKING prompt repr:")
step(f"  {prompt_no!r}")
step("THINKING prompt repr:")
step(f"  {prompt_th!r}")


def dump(label, ids, tail=None):
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    n = len(ids)
    toks = tokenizer.convert_ids_to_tokens(ids)
    lo = 0 if tail is None else max(0, n - tail)
    print(f"\n===== {label}  (len={n}) =====", flush=True)
    print(f"{'idx':>4} {'neg':>5}  {'id':>8}  token", flush=True)
    for i in range(lo, n):
        print(f"{i:>4} {i - n:>5}  {ids[i]:>8}  {toks[i]!r}", flush=True)
    sys.stdout.flush()


# dump input tokens for both modes right away — still no model
for tag, prompt in [("NON-THINKING", prompt_no), ("THINKING", prompt_th)]:
    ids = tokenizer(prompt, add_special_tokens=False).input_ids
    dump(f"{tag} :: INPUT PROMPT", ids)
    ptoks = tokenizer.convert_ids_to_tokens(ids)
    step(f"{tag}: t_post-inst = prompt[-1] = {ptoks[-1]!r} ; tail = {ptoks[-10:]}")

# ---------------------------------------------------------------
# MODEL  (expensive — this is where a hang usually is)
# ---------------------------------------------------------------
step("about to load MODEL weights — if this is the last line you see, "
     "the hang is in from_pretrained (network check or disk load)")
step(f"loading model: {MODEL_ID} ...")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float16,          # note: `dtype`, not deprecated `torch_dtype`
    device_map="auto",
    trust_remote_code=True,
)
step(f"model loaded in {time.time()-t0:.2f}s  device={next(model.parameters()).device}")
model.eval()
step("model set to eval()")


def generate(tag, prompt):
    step(f"[{tag}] tokenizing prompt for generation ...")
    enc = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
    step(f"[{tag}] prompt has {enc.input_ids.shape[1]} tokens; starting generate "
         f"(max_new_tokens={MAX_NEW}, greedy) ...")
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=MAX_NEW, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    dt = time.time() - t0
    gen_ids = out[0][enc.input_ids.shape[1]:]
    step(f"[{tag}] generated {len(gen_ids)} tokens in {dt:.2f}s "
         f"({len(gen_ids)/dt:.1f} tok/s)")
    dump(f"{tag} :: GENERATED", gen_ids)
    step(f"[{tag}] decoded generation:")
    step(f"  {tokenizer.decode(gen_ids, skip_special_tokens=False)!r}")
    return gen_ids


generate("NON-THINKING", prompt_no)
generate("THINKING", prompt_th)

step("done.")