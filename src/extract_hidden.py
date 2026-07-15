import json
import torch
import os
import copy
import argparse
from typing import List, Tuple, Callable, Optional
from torch import Tensor
from tqdm import tqdm
from utils import read_row, formatInp_llama_persuasion, formatInp_thinking
from transformers import AutoModelForCausalLM, AutoTokenizer
import contextlib
import functools
import random


MODEL = ''
NUM_TOKEN_HIDDEN = 2  # by default, we extract NUM_TOKEN_HIDDEN tokens + all special post-instruction tokens

@contextlib.contextmanager
def add_hooks(
    module_forward_pre_hooks: List[Tuple[torch.nn.Module, Callable]],
    module_forward_hooks: List[Tuple[torch.nn.Module, Callable]],
    **kwargs
) -> None:
    """
    Context manager for temporarily adding forward hooks to a model.

    Args:
        module_forward_pre_hooks: A list of pairs: (module, fnc) The function will be registered as a
            forward pre hook on the module
        module_forward_hooks: A list of pairs: (module, fnc) The function will be registered as a
            forward hook on the module
        **kwargs: Additional keyword arguments to pass to the hooks
    """
    try:
        handles = []
        for module, hook in module_forward_pre_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_pre_hook(partial_hook))

        for module, hook in module_forward_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_hook(partial_hook))
        yield
    finally:
        for h in handles:
            h.remove()

def get_mean_activations_pre_hook(
    layer: int,
    cache_full: List[List[Tensor]],
    positions: List[int],
    whole_seq: bool = False,
    step: int = NUM_TOKEN_HIDDEN
) -> Callable:
    """
    Creates a hook function to collect mean activations.

    Args:
        layer: Layer number
        cache_full: Cache to store activations
        positions: Positions to extract activations from
        whole_seq: Whether to store whole sequence
        step: Number of tokens to consider

    Returns:
        Hook function that collects activations
    """
    def hook_fn(module: torch.nn.Module, input: Tuple[Tensor, ...]) -> None:
        activation = input[0].half()
        seq_len = activation.shape[1]
        
        if whole_seq:
            cache_full[layer].append(activation.clone().detach().cpu())
        else:
            if seq_len >= len(positions):
                print('extracting positions', positions)
                assert isinstance(positions[0], int)
                context = activation[:, -len(positions)-step:-len(positions), :]
                pos_activations = activation[:, positions, :]
                merged_activation = torch.cat([context, pos_activations], dim=1)
                cache_full[layer].append(merged_activation.clone().detach().cpu())
            else:
                print('seq_len<positions', seq_len, len(positions))
                exit()
    return hook_fn

def get_mean_activations_fwd_hook(
    layer: int,
    cache_full: List[List[Tensor]],
    positions: List[int],
    whole_seq: bool = False,
    step: int = NUM_TOKEN_HIDDEN
) -> Callable:
    """
    Creates a forward hook function to collect mean activations.

    Args:
        layer: Layer number
        cache_full: Cache to store activations
        positions: Positions to extract activations from
        whole_seq: Whether to store whole sequence
        step: Number of tokens to consider

    Returns:
        Hook function that collects activations
    """
    def hook_fn(module: torch.nn.Module, input: Tuple[Tensor, ...], output: Tuple[Tensor, ...]) -> None:
        # transformers >=5 returns a bare tensor here instead of a tuple.
        activation = (output[0] if isinstance(output, (tuple, list)) else output).half()
        seq_len = activation.shape[1]
        
        if whole_seq:
            cache_full[layer].append(activation.clone().detach().cpu())
        else:
            if seq_len >= len(positions):
                context = activation[:, -len(positions)-step:-len(positions), :]
                pos_activations = activation[:, positions, :]
                merged_activation = torch.cat([context, pos_activations], dim=1)
                cache_full[layer].append(merged_activation.clone().detach().cpu())
            else:
                print('seq_len<positions', seq_len, len(positions))
                exit()
    return hook_fn

def get_mean_activations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    instructions: List[str],
    tokenize_instructions_fn: Callable,
    block_modules: List[torch.nn.Module],
    batch_size: int = 32,
    positions: List[int] = [-1],
    ret_whole_seq: bool = False
) -> Tuple[Tensor, Tensor]:
    """
    Extracts mean activations from model for given instructions.

    Args:
        model: Model to extract activations from
        tokenizer: Tokenizer instance
        instructions: List of input instructions
        tokenize_instructions_fn: Function to tokenize instructions
        block_modules: List of model blocks to hook
        batch_size: Batch size for processing
        positions: Positions to extract activations from
        ret_whole_seq: Whether to return whole sequence

    Returns:
        Tuple of (mean activations, full activations)
    """
    torch.cuda.empty_cache()

    n_layers = model.config.num_hidden_layers
    full_activations = [[] for _ in range(n_layers + 1)]

    fwd_pre_hooks = [
        (block_modules[layer], get_mean_activations_pre_hook(
            layer=layer,
            cache_full=full_activations,
            positions=positions,
            whole_seq=ret_whole_seq
        )) for layer in range(n_layers)
    ]
    
    fwd_hooks = [
        (block_modules[n_layers-1], get_mean_activations_fwd_hook(
            layer=-1,
            cache_full=full_activations,
            positions=positions,
            whole_seq=ret_whole_seq
        ))
    ]

    for i in tqdm(range(0, len(instructions), batch_size)):
        inputs = tokenize_instructions_fn(instructions=instructions[i:i+batch_size])
        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
            model(
                input_ids=inputs.input_ids.to(model.device),
                attention_mask=inputs.attention_mask.to(model.device),
            )

    flat_list = [torch.stack(inner_list) for inner_list in full_activations]
    result = torch.stack(flat_list).squeeze()

    if len(result.shape) < 3:
        result = result.unsqueeze(1)

    mean_activations = result.mean(dim=1)
    print('mean shape', mean_activations.shape)

    return mean_activations, result

def get_mean_diff(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    harmful_instructions: List[str],
    harmless_instructions: List[str],
    tokenize_instructions_fn: Callable,
    block_modules: List[torch.nn.Module],
    batch_size: int = 32,
    positions: List[int] = [-1],
    extract_only: bool = False,
    use_persuade_harmful: bool = False,
    use_persuade_harmless: bool = False,
    use_sys_harmful: bool = False,
    ret_whole_seq: bool = False
) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    """
    Computes mean activation differences between harmful and harmless instructions.

    Args:
        model: Model to extract activations from
        tokenizer: Tokenizer instance
        harmful_instructions: List of harmful instructions
        harmless_instructions: List of harmless instructions
        tokenize_instructions_fn: Function to tokenize instructions
        block_modules: List of model blocks to hook
        batch_size: Batch size for processing
        positions: Positions to extract activations from
        extract_only: Whether to only extract harmful activations
        use_persuade_harmful: Whether to use persuasion for harmful
        use_persuade_harmless: Whether to use persuasion for harmless
        use_sys_harmful: Whether to use system prompt for harmful
        ret_whole_seq: Whether to return whole sequence

    Returns:
        Tuple of (harmful mean activations, harmless mean activations,
                harmful full activations, harmless full activations)
    """
    mean_activations_harmful, full_activations_harmful = get_mean_activations(
        model, tokenizer, harmful_instructions,
        functools.partial(tokenize_instructions_fn, use_persuade=use_persuade_harmful, use_sys=use_sys_harmful),
        block_modules, batch_size=batch_size, positions=positions, ret_whole_seq=ret_whole_seq
    )
    
    torch.save(mean_activations_harmful, 'output/tmp_mean_activations_harmful.pt')
    torch.save(full_activations_harmful, 'output/tmp_full_activations_harmful.pt')
    del mean_activations_harmful, full_activations_harmful
    torch.cuda.empty_cache()

    if not extract_only:
        mean_activations_harmless, full_activations_harmless = get_mean_activations(
            model, tokenizer, harmless_instructions,
            functools.partial(tokenize_instructions_fn, use_persuade=use_persuade_harmless),
            block_modules, batch_size=batch_size, positions=positions, ret_whole_seq=ret_whole_seq
        )
        mean_activations_harmful = torch.load('output/tmp_mean_activations_harmful.pt')
        full_activations_harmful = torch.load('output/tmp_full_activations_harmful.pt')

        print('mean_activations_harmful shape', mean_activations_harmful.shape)
        print('mean_activations_harmless shape', mean_activations_harmless.shape)
    else:
        # extract_only: harmful acts were del'd above; reload them (mirrors the branch above).
        mean_activations_harmful = torch.load('output/tmp_mean_activations_harmful.pt')
        full_activations_harmful = torch.load('output/tmp_full_activations_harmful.pt')
        mean_activations_harmless = None
        full_activations_harmless = None

    return mean_activations_harmful, mean_activations_harmless, full_activations_harmful, full_activations_harmless

def generate_directions(
    model_base: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    harmful_instructions: List[str],
    harmless_instructions: List[str],
    args: dict
) -> Optional[Tensor]:
    """
    Generates direction vectors from model activations.

    Args:
        model_base: Base model
        tokenizer: Tokenizer instance
        harmful_instructions: List of harmful instructions
        harmless_instructions: List of harmless instructions
        args: Arguments dictionary

    Returns:
        Mean difference tensor or None if computation fails
    """
    def tokenize_instructions_fn(instructions: List[str], use_persuade: bool = False, use_sys: bool = False) -> dict:
        inps = [formatInp_llama_persuasion(i, use_persuade, use_ss=use_sys, use_template=True, model=MODEL) for i in instructions]
        return tokenizer(inps, padding=True, return_tensors="pt")

    model_block_modules = model_base.model.layers
    mean_activations_harmful, mean_activations_harmless, all_harmful, all_harmless = get_mean_diff(
        model_base, tokenizer, harmful_instructions, harmless_instructions,
        tokenize_instructions_fn, model_block_modules, args['batch_size'],
        args['positions'], args['extract_only'], args['use_persuade_harmful'],
        args['use_persuade_harmless'], args['use_sys_harmful'], args['ret_whole_seq']
    )

    torch.save(all_harmful, args['output_pth_harmful'])
    torch.save(all_harmless, args['output_pth_harmless'])

    try:
        print('mean_activations_harmful shape', mean_activations_harmful.shape)
        print('mean_activations_harmless shape', mean_activations_harmless.shape)
        mean_diffs = mean_activations_harmful - mean_activations_harmless
        assert not mean_diffs.isnan().any()
        if args['mode_dir'] == 'hf':
            mean_diffs = mean_diffs[:,NUM_TOKEN_HIDDEN-1] 
        elif args['mode_dir'] == 'refuse':
            mean_diffs = mean_diffs[:,-1]
        torch.save(mean_diffs.to('cpu'), args['output_pth'])
    except Exception as e:
        print(e)
        mean_diffs = None

    return mean_diffs

# ---------------------------------------------------------------------------
# Qwen3.5 thinking-model extraction (clone path). Instead of the prompt-only hook machinery
# above, we run one forward pass on prompt+generation and gather a FIXED 25-slot token layout so
# both generation modes share a single index scheme (see the plan / CLAUDE-style docs):
#   0        t_inst (token before the first <|im_end|>)
#   1-4      assistant \n <think> \n     (slot 1 = t_post)
#   5-9      first 5 CoT tokens
#   10-14    middle 5 CoT tokens (region CoT[5:-5] split into 5 groups, middle of each)
#   15-19    last 5 CoT tokens
#   20-23    the 4 tokens right after </think>, taken CONTIGUOUSLY (no whitespace skip) so the
#            offset from </think> is identical across modes: genthink = 4 generated tokens;
#            gennothink = the template's \n\n + the first 2 generated answer tokens.
#   24       </think> itself (generated in genthink/gennothink_stripped; in the prompt for gennothink)
# gennothink has no CoT (its </think> is inside the prompt), so slots 5-19 stay null (zeros);
# missing/short-CoT slots are null too. dynamic_bucket_formation drops zero-norm rows per position.
# ---------------------------------------------------------------------------
THINK_SLOTS = 25


def _single_tid(tokenizer, s: str):
    ids = tokenizer(s, add_special_tokens=False).input_ids
    return ids[0] if len(ids) == 1 else None


def _place(slots, offset, values):
    for i, v in enumerate(values):
        if offset + i < len(slots):
            slots[offset + i] = v


def _sample_middle(region, k):
    """Split `region` (a list of indices) into k contiguous groups, return each group's middle
    index (None for an empty group). Always returns exactly k entries."""
    if not region:
        return [None] * k
    n = len(region)
    out = []
    for i in range(k):
        grp = region[i * n // k:(i + 1) * n // k]
        out.append(grp[len(grp) // 2] if grp else None)
    return out


def compute_positions_thinking(ids, tokenizer, mode):
    """Map a full prompt+generation token-id list to the 25 fixed slot indices (int or None)."""
    slots = [None] * THINK_SLOTS
    im_end = _single_tid(tokenizer, '<|im_end|>')
    th_open = _single_tid(tokenizer, '<think>')
    th_close = _single_tid(tokenizer, '</think>')

    # slot 0: t_inst = token before the first <|im_end|>
    if im_end is not None and im_end in ids:
        fe = ids.index(im_end)
        if fe - 1 >= 0:
            slots[0] = fe - 1

    # slots 1-4: assistant \n <think> \n  (block = <think> index minus 2 .. plus 1)
    tho = ids.index(th_open) if (th_open is not None and th_open in ids) else None
    if tho is not None:
        _place(slots, 1, [i if 0 <= i < len(ids) else None for i in range(tho - 2, tho + 2)])

        cot_start = tho + 2  # skip <think>\n
        thc = next((k for k in range(cot_start, len(ids)) if ids[k] == th_close), None)

        if mode != 'gennothink':
            end = thc if thc is not None else len(ids)
            cot = list(range(cot_start, end))
            if cot:
                _place(slots, 5, cot[:5])
                _place(slots, 10, _sample_middle(cot[5:len(cot) - 5], 5))
                _place(slots, 15, cot[-5:])

        if thc is not None:
            # slots 20-23: the 4 tokens right after </think>, taken CONTIGUOUSLY (no whitespace
            # skip) so the offset from </think> matches across modes (gennothink keeps its \n\n +
            # 2 generated tokens; genthink is 4 generated tokens).
            _place(slots, 20, [i if i < len(ids) else None for i in range(thc + 1, thc + 5)])
            slots[24] = thc  # slot 24: </think> itself
    return slots


def extract_thinking_activations(model, tokenizer, rows, thinking_mode):
    """Forward each (instruction, stored generation) once and gather the 25-slot layout.

    Returns (L, N, 25, H): L = n_layers+1 residual-stream points (output_hidden_states, equivalent
    to the pre/fwd-hook capture used elsewhere), N rows, 25 token slots, hidden size H.
    """
    all_acts = []
    for row in tqdm(rows):
        full = formatInp_thinking(row, thinking_mode=thinking_mode, model=MODEL) + row.get('ori_output', '')
        enc = tokenizer(full, return_tensors='pt', add_special_tokens=False)
        with torch.no_grad():
            out = model(input_ids=enc.input_ids.to(model.device), output_hidden_states=True)
        hs = torch.stack(out.hidden_states).squeeze(1).detach().cpu().half()  # (L, seq, H)
        L, seq, H = hs.shape
        idxs = compute_positions_thinking(enc.input_ids[0].tolist(), tokenizer, thinking_mode)
        slots = [hs[:, j, :] if (j is not None and 0 <= j < seq) else torch.zeros(L, H, dtype=hs.dtype)
                 for j in idxs]
        all_acts.append(torch.stack(slots, dim=1))  # (L, 25, H)
        # full reasoning traces can be long (thousands of tokens); free GPU memory each step so
        # peak stays at one forward pass and fragmentation doesn't accumulate across examples.
        del out, hs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return torch.stack(all_acts, dim=1)  # (L, N, 25, H)


def generate_directions_thinking(model, tokenizer, rows, args):
    result = extract_thinking_activations(model, tokenizer, rows, args['thinking_mode'])
    print('thinking activations shape', tuple(result.shape))
    torch.save(result, args['output_pth_harmful'])
    return result


def main() -> None:
    """Run the full pipeline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='llama', type=str, help="Model type")
    parser.add_argument("--model_size", default='7b', type=str, help="Model size (e.g. 7b, 0.5b)")
    parser.add_argument("--harmful_pth", default='data/medcq.json', type=str, help="Path to harmful examples")
    parser.add_argument("--harmless_pth", default='data/medcq.json', type=str, help="Path to harmless examples")
    parser.add_argument("--output_pth_harmful", default='output/mean_diff.pt', type=str, help="Output path for harmful activations")
    parser.add_argument("--output_pth_harmless", default='output/mean_diff_harmless.pt', type=str, help="Output path for harmless activations")
    parser.add_argument('--use_persuade_harmful', default=0, type=int, help='Use persuasion for harmful examples')
    parser.add_argument('--use_persuade_harmless', default=0, type=int, help='Use persuasion for harmless examples')
    parser.add_argument('--use_sys_harmful', default=0, type=int, help='Use system prompt for harmful examples')
    parser.add_argument('--left', default=0, type=int, help='Left index for data slicing')
    parser.add_argument('--right', default=10, type=int, help='Right index for data slicing')
    parser.add_argument('--random_sample_harmful', default=0, type=int, help='Randomly sample harmful examples')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size')
    parser.add_argument("--output_pth", default='output/dir.pt', type=str, help="Output path of generated directions")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument('--mode', default='diff-mean', type=str, help='Mode')
    parser.add_argument('--positions', default='-1', type=str, help='Positions to extract')
    parser.add_argument('--extract_only', default=0, type=int, help='Only extract harmful activations')
    parser.add_argument('--ret_whole_seq', default=0, type=int, help='Return whole sequence')
    parser.add_argument('--extract_hidden_inst_token', default=0, type=int, help="Extract hidden state of instruction tokens")
    parser.add_argument('--extract_harmful_token_only', default=0, type=int, help="Extract harmful token only")
    parser.add_argument('--mode_dir', default='hf', type=str, help="Mode for direction extraction: 'hf' or 'refuse'")
    parser.add_argument('--thinking', default=0, type=int, help="Use the Qwen3.5 thinking 22-slot extraction")
    parser.add_argument('--thinking_mode', default='genthink', type=str, help="genthink or gennothink")

    args = parser.parse_args()
    params = vars(args)
    
    global MODEL
    global NUM_TOKEN_HIDDEN
    
    params['positions'] = list(map(int, params['positions'].split()))
    MODEL = params['model']
    llama_2_model_path = "NousResearch/Llama-2-7b-chat-hf"
    if MODEL == 'llama':
        model = AutoModelForCausalLM.from_pretrained(
            llama_2_model_path,
            cache_dir='models/llama',
            torch_dtype=torch.float16,
            device_map="cuda",
        )
        tokenizer = AutoTokenizer.from_pretrained(
            llama_2_model_path,
            cache_dir='models/llama',
        )
    elif MODEL == 'llama3':
        llama3_model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
        model = AutoModelForCausalLM.from_pretrained(
            llama3_model_path,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(
            llama3_model_path,
        )
        tokenizer.pad_token = tokenizer.eos_token
    elif MODEL == 'qwen':
        if params['model_size'] == '0.5b':
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct", trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct", device_map="auto", trust_remote_code=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct", trust_remote_code=True, cache_dir='models/qwen')
            model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-7B-Instruct", device_map="auto", trust_remote_code=True, cache_dir='models/qwen')
    elif MODEL == 'qwen35':
        qwen35_path = f"Qwen/Qwen3.5-{params['model_size'].upper()}"
        # absolute cache dir (src/models/qwen35) so inference and extraction share one download.
        qwen35_cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'qwen35')
        tokenizer = AutoTokenizer.from_pretrained(qwen35_path, trust_remote_code=True, cache_dir=qwen35_cache)
        model = AutoModelForCausalLM.from_pretrained(qwen35_path, dtype=torch.float16, device_map="auto", trust_remote_code=True, cache_dir=qwen35_cache)

    if params['thinking']:
        # Thinking clone path: forward on prompt+generation, store the fixed 22-slot layout.
        rows = read_row(params['harmful_pth'])
        rows = rows[params['left']:params['right']] if params['left'] < len(rows) else rows
        generate_directions_thinking(model, tokenizer, rows, params)
        return

    if params['extract_hidden_inst_token']:
        inst_token = "[/INST]"
        if params['model'] == 'llama3':
            inst_token = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        elif params['model'] == 'vicuna':
            inst_token = 'ASSISTANT:\n'
        elif params['model'] == 'qwen':
            inst_token = '<|im_end|>\n<|im_start|>assistant'
            
        tokenized_inst = tokenizer(inst_token, return_tensors='pt', add_special_tokens=False)
        print('inst_token', tokenizer.decode(tokenized_inst.input_ids[0]))
        params['positions'] = [i for i in range(-len(tokenized_inst.input_ids[0]), 0, 1)]
        if params['extract_harmful_token_only']:
            params['positions'] = [-len(tokenized_inst.input_ids[0])-1]
            NUM_TOKEN_HIDDEN = 0

    harmful_train = read_row(params['harmful_pth'])

    if params['random_sample_harmful']:
        random.seed(params['left'] % len(harmful_train))
        harmful_train = random.sample(harmful_train, 1)
    else:
        if params['left'] < len(harmful_train):
            harmful_train = harmful_train[params['left']:params['right']]
        else:
            harmful_train = harmful_train[params['left'] % len(harmful_train):params['left'] % len(harmful_train)+1]
            
    harmless_train = read_row(params['harmless_pth'])[params['left']:params['right']]

    with open(params['output_pth'].replace('.pt', '_prompts_used.json'), 'w') as f:
        json.dump({'harmful': harmful_train, 'harmless': harmless_train}, f, indent=4)

    candidate_directions = generate_directions(model, tokenizer, harmful_train, harmless_train, params)
    if candidate_directions is not None:
        print(candidate_directions.shape)

if __name__ == "__main__":
    main()