"""
Shared model and data utilities for all replication experiments.

Loads Qwen2-7B-Instruct via inference.load_model_and_tokenizer() and provides
helpers that wrap the original paper utilities from src/ with the correct
Qwen-specific arguments.
"""
import os
import sys

import numpy as np
import torch

# Make the original src/ directory importable from any working directory.
from config import SRC_DIR, MODEL_TYPE, MODEL_SIZE, RESULTS_DIR
sys.path.insert(0, SRC_DIR)

from utils import formatInp_llama_persuasion, REFUSAL_PHRASE, read_row  # noqa: E402


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model():
    """Load model via inference.load_model_and_tokenizer() when the size is supported,
    otherwise load directly (e.g. Qwen2.5-1.5B which inference.py doesn't know about)."""
    from config import MODEL_NAME
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _INFERENCE_SUPPORTED = {"7b", "13b", "14b"}
    if MODEL_SIZE in _INFERENCE_SUPPORTED:
        import inference as _inf
        model, tokenizer = _inf.load_model_and_tokenizer(MODEL_TYPE, MODEL_SIZE)
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, device_map="auto", trust_remote_code=True,
            dtype=torch.float16,
        )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def format_prompt_no_postinst(d):
    """
    Format a prompt WITHOUT the post-instruction tokens.

    Used in §3.1 (Table 1) to test how the model behaves when it must generate
    directly from tinst (the end of the user instruction) rather than from
    tpost-inst (the end of the full post-instruction block).

    For Qwen2, the full template is:
        <|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n
    The truncated version (do_not_use_last_inst_tok=True) is:
        <|im_start|>user\n{instruction}
    """
    if isinstance(d, str):
        d = {"instruction": d}
    return formatInp_llama_persuasion(d, model=MODEL_TYPE, use_template=True,
                                       do_not_use_last_inst_tok=True)


def format_prompt(d):
    """
    Format a data dict into a Qwen2 chat prompt string.

    Wraps formatInp_llama_persuasion with model='qwen' to get:
        <|im_start|>user
        {instruction}
        <|im_end|>
        <|im_start|>assistant

    Bug fix: the original generate_directions() in extract_hidden.py calls
    formatInp_llama_persuasion without the model= argument, defaulting to
    'llama2' format ([INST]…[/INST]) even for Qwen models. We always pass
    model='qwen' here.

    d can be a dict (with keys like 'instruction', 'bad_q', 'question') or
    a plain string (wrapped in {'instruction': d} automatically).
    """
    if isinstance(d, str):
        d = {"instruction": d}
    return formatInp_llama_persuasion(d, model=MODEL_TYPE, use_template=True)


def tokenize_fn(tokenizer, instructions):
    """
    Tokenize a list of instruction dicts/strings into a BatchEncoding.

    Returns a BatchEncoding with input_ids and attention_mask on CPU.
    Move to device yourself before the forward pass.
    """
    prompts = [format_prompt(i) for i in instructions]
    return tokenizer(prompts, padding=True, return_tensors="pt")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(path, key, n=None, offset=0):
    """
    Read a JSONL/JSON file and return the text strings at `key`.

    path   : path to the data file
    key    : which field to extract ('bad_q', 'instruction', 'prompt', …)
    n      : how many items to return (None = all)
    offset : skip the first `offset` items (useful for train/test splits)
    """
    rows = read_row(path)
    rows = rows[offset:]
    if n is not None:
        rows = rows[:n]
    return rows  # return full dicts so format_prompt can choose the right key


# ---------------------------------------------------------------------------
# Refusal detection
# ---------------------------------------------------------------------------

def is_refusal(text):
    """Return True if the model response contains a standard refusal phrase."""
    text_lower = text.lower()
    return any(phrase.lower() in text_lower for phrase in REFUSAL_PHRASE)


# ---------------------------------------------------------------------------
# Cosine similarity (used in 02, 05, plot_figure3)
# ---------------------------------------------------------------------------

def cosc(A, B, eps=1e-8):
    """
    Numerically-stable cosine similarity between two vectors or batches.

    Ported from classifier.ipynb (cell-1) by the paper authors.
    Works with numpy arrays or plain lists.
    """
    A = np.array(A, dtype=np.float64)
    B = np.array(B, dtype=np.float64)
    dot = np.clip(np.dot(A, B), -1e150, 1e150)
    norm_A = np.clip(np.linalg.norm(A), eps, 1e150)
    norm_B = np.clip(np.linalg.norm(B), eps, 1e150)
    return dot / (norm_A * norm_B)


# ---------------------------------------------------------------------------
# Activation extraction helper
# ---------------------------------------------------------------------------

def _make_compat_fwd_hook(layer, cache_full, positions, whole_seq=False, step=None):
    """
    Replacement for extract_hidden.get_mean_activations_fwd_hook that handles
    both old (tuple) and new (tensor) decoder layer output formats.

    In transformers ≤ 4.46, Qwen2DecoderLayer.forward() returned a tuple
    (hidden_states, optional_past_kv, …) so output[0] was the 3-D hidden states.
    In transformers 4.57 / 5.x it returns hidden_states directly as a Tensor,
    so output[0] slices the batch dimension and gives a 2-D tensor — causing
    "IndexError: too many indices for tensor of dimension 2".

    This hook handles both cases by checking isinstance(output, torch.Tensor).
    """
    import extract_hidden as _eh
    if step is None:
        step = _eh.NUM_TOKEN_HIDDEN

    def hook_fn(module, inp, output):
        activation = (output if isinstance(output, torch.Tensor) else output[0]).half()
        seq_len = activation.shape[1]
        if whole_seq:
            cache_full[layer].append(activation.clone().detach().cpu())
        else:
            if seq_len >= len(positions):
                context   = activation[:, -len(positions) - step:-len(positions), :]
                pos_acts  = activation[:, positions, :]
                merged    = torch.cat([context, pos_acts], dim=1)
                cache_full[layer].append(merged.clone().detach().cpu())
            else:
                print("seq_len<positions", seq_len, len(positions))
                exit()
    return hook_fn


def extract_hidden_states(model, tokenizer, data_dicts, positions=(-1,)):
    """
    Run a forward pass over data_dicts and return hidden states at `positions`.

    Calls get_mean_activations from the original extract_hidden.py with
    batch_size=1 (required – see config.py BATCH_SIZE comment).

    The forward hook in extract_hidden.py is monkey-patched with a version that
    tolerates both tuple and tensor decoder-layer outputs (see _make_compat_fwd_hook).

    Returns:
        mean_acts : [n_layers+1, len(positions)+NUM_TOKEN_HIDDEN, hidden_dim]  (float16 CPU)
        all_acts  : [n_layers+1, N, len(positions)+NUM_TOKEN_HIDDEN, hidden_dim]
    """
    import extract_hidden

    # Patch the forward hook in-place so get_mean_activations picks it up when
    # it looks up the name in the extract_hidden module namespace.
    _orig_fwd_hook = extract_hidden.get_mean_activations_fwd_hook
    extract_hidden.get_mean_activations_fwd_hook = _make_compat_fwd_hook
    try:
        def tok_fn(instructions, **_kwargs):
            return tokenize_fn(tokenizer, instructions)

        return extract_hidden.get_mean_activations(
            model,
            tokenizer,
            data_dicts,
            tok_fn,
            model.model.layers,
            batch_size=1,
            positions=list(positions),
        )
    finally:
        extract_hidden.get_mean_activations_fwd_hook = _orig_fwd_hook
