
import os
import torch
import argparse
import math
import json
import logging
from typing import Dict, List, Tuple, Any

from transformers import (AutoModelForCausalLM, AutoTokenizer, GenerationConfig,
                          LogitsProcessor, LogitsProcessorList)
from tqdm import tqdm

from utils import read_row, formatInp_llama_persuasion, formatInp_thinking

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.cuda.set_device(0)
else:
    raise RuntimeError("CUDA is required for this script")

def evaluate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    generation_config: GenerationConfig,
    args: Dict[str, Any]
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Evaluate a model on a given prompt.
    
    Args:
        model: The language model to evaluate
        tokenizer: The tokenizer for the model
        prompt: Input prompt text
        generation_config: Configuration for text generation
        args: Additional arguments dictionary
        
    Returns:
        Tuple of (generated_text, probability_scores)
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    
    max_new_tokens = args['max_len']
    
    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=max_new_tokens,
        num_return_sequences=1
    )
    
    generated_scores = generation_output.scores
    logger.debug(f'Max tokens: {max_new_tokens}, Input length: {len(input_ids[0])}')
    logger.debug(f'Generated scores: {len(generated_scores)}, Generated IDs: {len(generation_output.sequences[0])}')
    
    # Extract probability scores
    probabilities = []
    for i, score_tensor in enumerate(generated_scores):
        if i >= args['record_prob_max_pos']:
            break
            
        probability = torch.softmax(score_tensor[0], dim=-1)
        all_probs = probability.detach().cpu().numpy().tolist()
        token_id = torch.argmin(probability).item()
        token = tokenizer.decode(token_id)
        probabilities.append({'prob': all_probs, 'token': token})
    
    # Decode the generated text
    output = tokenizer.decode(generation_output.sequences[0])
    return output, probabilities


def extract_model_output(
    full_output: str,
    model_type: str,
    input_prompt: str = ""
) -> str:
    """
    Extract the model's response from the full generated text.
    
    Args:
        full_output: Complete generated text
        model_type: Type of model used
        input_prompt: Original input prompt (for llama3)
        
    Returns:
        Extracted model response
    """
    if model_type == 'llama' or model_type == 'mistral':
        return full_output.split('[/INST]')[-1].replace('</s>', '').strip()
    elif model_type == 'vicuna':
        return full_output.split('ASSISTANT:')[-1].replace('</s>', '').strip()
    elif model_type == 'qwen':
        # tpost templates end with the assistant turn ("...<|im_start|>assistant"); the tinst
        # template ("<|im_start|>user\n{q}") has no assistant token, so split('assistant') would
        # leave the whole echoed prompt in the response. Strip the exact input prompt when present
        # (mirrors the llama3 branch); fall back to the assistant split otherwise.
        if input_prompt and input_prompt in full_output:
            response = full_output.split(input_prompt)[-1]
        else:
            response = full_output.split('assistant')[-1]
        return response.replace('</s>', '').strip()
    elif model_type == 'llama3':
        return full_output.split(input_prompt)[-1].strip()
    else:
        return full_output.strip()


def infer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    eval_data: List[Dict[str, Any]],
    args: Dict[str, Any]
) -> None:
    """
    Run inference on evaluation data.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        eval_data: List of evaluation data points
        args: Configuration arguments
    """
    model.eval()
    model.config.use_cache = True
    
    # Remove existing output file
    if os.path.exists(args['output_file_name']):
        os.remove(args['output_file_name'])
    
    # Configure generation strategy
    if args['do_sample_decode']:
        generation_config = GenerationConfig(
            do_sample=True,
            temperature=args['temperature'],
            top_p=args['top_p'],
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id
        )
    else:
        generation_config = GenerationConfig(
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    logger.info('Starting inference...')

    for i in tqdm(range(len(eval_data)), desc="Processing samples"):
        data_point = eval_data[i]
        input_prompt = formatInp_llama_persuasion(
            data_point,
            use_persuade=args['use_jb'],
            use_adv=args['use_adv_suffix'],
            use_ss=args['use_sys_prompt'],
            model=args['model'],
            use_template=args['use_template'],
            do_not_use_last_inst_tok=args['do_not_use_last_inst_tok'],
            use_inversion=args['use_inversion'],
            inversion_prompt_idx=args['inversion_prompt_idx']
        )
        
        output, probabilities = evaluate(model, tokenizer, input_prompt, generation_config, args)
        # Process and save results
        with open(args['output_file_name'], 'a') as f:
            if args['use_jb']:
                response = extract_model_output(output, args['model'])
                eval_data[i] = {'prompt': eval_data[i], 'response': response}
            else:
                eval_data[i]['ori_output'] = extract_model_output(output, args['model'], input_prompt)
            
            eval_data[i]['probs'] = probabilities
            json.dump(eval_data[i], f)
            f.write('\n')
        
        logger.debug(f'Output: {output[:100]}...')


def extract_model_output_thinking(full_output: str, input_prompt: str) -> str:
    """Clone of extract_model_output for the Qwen3.5 thinking family. Keeps the FULL generated
    continuation (reasoning trace + ``</think>`` + answer) so downstream can split on ``</think>``.
    Strips the echoed input prompt when present; otherwise falls back to the last ``<think>``.
    """
    if input_prompt and input_prompt in full_output:
        response = full_output.split(input_prompt)[-1]
    else:
        response = full_output.split('<think>')[-1]
    return response.replace('</s>', '').strip()


# Default thinking-inference sampling. A --sampling_config json may override any subset of these
# keys; presence_penalty is applied via PresencePenaltyLogitsProcessor (generate ignores it), the
# rest are passed straight to GenerationConfig.
_DEFAULT_SAMPLING = {"do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": 20,
                     "min_p": 0.0, "presence_penalty": 2.0}


def load_sampling_config(path: str) -> dict:
    """Merge a sampling-config json (path may be empty/missing) over the defaults."""
    cfg = dict(_DEFAULT_SAMPLING)
    if path and os.path.exists(path):
        with open(path) as f:
            cfg.update(json.load(f))
    return cfg


class PresencePenaltyLogitsProcessor(LogitsProcessor):
    """OpenAI/vLLM-style presence penalty (transformers' generate has no built-in one): subtract a
    flat `penalty` from the logit of every token that has already appeared in the *generated* part
    of the sequence (binary presence, count-independent). `prompt_len` is the padded input length,
    so only generated tokens are penalized, not the prompt.
    """
    def __init__(self, penalty: float, prompt_len: int):
        self.penalty = penalty
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores):
        generated = input_ids[:, self.prompt_len:]
        if generated.shape[1] == 0:
            return scores
        seen = torch.zeros_like(scores, dtype=torch.bool)
        seen.scatter_(1, generated, True)
        return scores - seen.to(scores.dtype) * self.penalty


def infer_thinking(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    eval_data: List[Dict[str, Any]],
    args: Dict[str, Any]
) -> None:
    """Batched clone of infer for the Qwen3.5 thinking family. Builds prompts with formatInp_thinking
    (genthink/gennothink), generates in batches of args['batch_size'], and stores the full
    reasoning+answer continuation in ori_output.

    Sampling params come from args['sampling_config'] (a json path) merged over _DEFAULT_SAMPLING;
    presence_penalty is applied via PresencePenaltyLogitsProcessor (generate ignores it), the rest
    go to GenerationConfig.
    """
    model.eval()
    model.config.use_cache = True

    if os.path.exists(args['output_file_name']):
        os.remove(args['output_file_name'])

    # left padding is required for correct batched decoder-only generation.
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    sampling = load_sampling_config(args.get('sampling_config', ''))
    presence_penalty = float(sampling.pop('presence_penalty', 0.0))
    generation_config = GenerationConfig(num_beams=1, pad_token_id=tokenizer.pad_token_id, **sampling)
    batch_size = int(args.get('batch_size', 1)) or 1
    max_new_tokens = args['max_len']

    prompts = [formatInp_thinking(d, thinking_mode=args['thinking_mode'], model=args['model'])
               for d in eval_data]

    logger.info('Starting thinking inference (%s), %d samples, batch_size=%d, sampling=%s, '
                'presence_penalty=%s...', args['thinking_mode'], len(eval_data), batch_size,
                sampling, presence_penalty)

    with open(args['output_file_name'], 'a') as f:
        for i in tqdm(range(0, len(eval_data), batch_size), desc="Batches"):
            batch_prompts = prompts[i:i + batch_size]
            enc = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)
            prompt_len = enc.input_ids.shape[1]
            logits_processor = (LogitsProcessorList(
                [PresencePenaltyLogitsProcessor(presence_penalty, prompt_len)])
                if presence_penalty else None)
            with torch.no_grad():
                out = model.generate(**enc, generation_config=generation_config,
                                     max_new_tokens=max_new_tokens,
                                     logits_processor=logits_processor)
            gen = out[:, prompt_len:]  # generated continuation only
            for j in range(gen.shape[0]):
                ids = gen[j].tolist()
                if tokenizer.eos_token_id in ids:  # trim eos + any trailing pad
                    ids = ids[:ids.index(tokenizer.eos_token_id)]
                text = tokenizer.decode(ids, skip_special_tokens=False)  # keep <think>/</think>
                eval_data[i + j]['ori_output'] = text.replace('</s>', '').strip()
                json.dump(eval_data[i + j], f)
                f.write('\n')
            f.flush()


def load_model_and_tokenizer(model_type: str, model_size: str, load_checkpoint: bool = False, checkpoint_path: str = "") -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load the specified model and tokenizer.
    
    Args:
        model_type: Type of model ('llama', 'llama3', 'qwen')
        model_size: Size of model ('7b', '13b', '14b')
        load_checkpoint: Whether to load a checkpoint
        checkpoint_path: Path to checkpoint if loading
        
    Returns:
        Tuple of (model, tokenizer)
    """
    try:
        if model_type == 'llama':
            #specify your model path here
            model_path = "NousResearch/Llama-2-7b-chat-hf" if model_size == '7b' else "Llama-2-13b-chat-hf/"
            
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                #cache_dir='models/llama',
                device_map="auto",
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                #cache_dir='models/llama',
            )
            
        elif model_type == 'llama3':
            model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                #cache_dir='models/llama3',
                device_map="auto",
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                #cache_dir='models/llama3',
            )
            
        elif model_type == 'qwen':
            if model_size == '7b':
                model_path = "Qwen/Qwen2-7B-Instruct"
                #cache_dir = 'models/qwen'
            elif model_size == '14b':
                model_path = "Qwen/Qwen2.5-14B-Instruct"
                #cache_dir = 'models/qwen-14b'
            elif model_size == '0.5b':
                model_path = "Qwen/Qwen2-0.5B-Instruct"
                # cache_dir = 'models/qwen-14b'
            else:
                raise ValueError(f"Unsupported model size: {model_size}")
                
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                #cache_dir=cache_dir
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                trust_remote_code=True,
                #cache_dir=cache_dir
            )

        elif model_type == 'qwen35':
            # Qwen3.5 thinking family (any size, e.g. '0.8b'). Cache under src/models/qwen35 via an
            # absolute path so inference and extract_hidden (run from different cwds) share one copy.
            model_path = f"Qwen/Qwen3.5-{model_size.upper()}"
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'qwen35')
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True, cache_dir=cache_dir,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path, dtype=torch.float16, device_map="auto",
                trust_remote_code=True, cache_dir=cache_dir,
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Load checkpoint if specified
        if load_checkpoint and checkpoint_path:
            model.load_adapter(checkpoint_path)
            
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Failed to load model {model_type}: {e}")
        raise


def main():
    """Main function to run the inference script."""
    parser = argparse.ArgumentParser(description="Language model inference script")
    
    # Model configuration
    parser.add_argument("--model", default='llama3', type=str, help="Model type (llama, llama3, qwen)")
    parser.add_argument("--model_size", default='7b', type=str, help="Model size (7b, 13b, 14b)")
    parser.add_argument("--peft_pth_ckpt", default='', type=str, help="Path to PEFT checkpoint")
    parser.add_argument("--load_ckpt", default=0, type=int, help="Whether to load checkpoint")
    
    # Data configuration
    parser.add_argument("--input", default='data/medcq.json', type=str, help="Input file path")
    parser.add_argument("--output_file_name", default='data/medcq.json', type=str, help="Output file path")
    parser.add_argument('--left', default=0, type=int, help='Left index for data slicing')
    parser.add_argument('--right', default=10, type=int, help='Right index for data slicing')
    
    # Generation configuration
    parser.add_argument("--max_len", default=4096, type=int, help="Maximum generation length")
    parser.add_argument("--temperature", default=0, type=float, help="Generation temperature")
    parser.add_argument("--top_p", default=0, type=float, help="Top-p sampling parameter")
    parser.add_argument("--do_sample_decode", default=0, type=int, help="Use sampling for decoding")
    parser.add_argument("--record_prob_max_pos", default=3, type=int, help="Max positions to record probabilities")
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size (thinking inference)")
    
    # Prompt configuration
    parser.add_argument("--use_jb", default=0, type=int, help="Use jailbroken prompt")
    parser.add_argument("--use_adv_suffix", default=0, type=int, help="Use adversarial suffix")
    parser.add_argument("--use_sys_prompt", default=0, type=int, help="Use system prompt")
    parser.add_argument("--use_template", default=0, type=int, help="Use default prompting template")
    parser.add_argument("--do_not_use_last_inst_tok", default=0, type=int, help="Don't use last instruction token")
    parser.add_argument("--use_inversion", default=0, type=int, help="Use inversion")
    parser.add_argument("--inversion_prompt_idx", default=0, type=int, help="Inversion prompt index")
    parser.add_argument("--thinking_mode", default='', type=str,
                        help="Qwen3.5 thinking template: '' (off), 'genthink', or 'gennothink'")
    parser.add_argument("--sampling_config", default='', type=str,
                        help="path to a json of thinking-inference sampling params "
                             "(temperature, top_p, top_k, min_p, presence_penalty, do_sample)")
    
    
    args = parser.parse_args()
    params = vars(args)
    
    logger.info(f'Model: {params["model"]}')
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        params['model'],
        params['model_size'],
        params['load_ckpt'],
        params['peft_pth_ckpt']
    )
    
    # Load and preprocess data
    test_data = read_row(params['input'])
    
    # Validate indices
    if params['left'] < 0:
        params['left'] = 0
    if params['right'] > len(test_data):
        params['right'] = len(test_data)
    
    if params['left'] >= params['right']:
        raise ValueError("Left index must be less than right index")
    
    # Filter data
    test_data = [
        d for d in test_data[params['left']:params['right']]
        if 'sample_rounds' not in d or d['sample_rounds'] != 'Failed'
    ]
    
    # Run inference (thinking family routes to the cloned path)
    if params['thinking_mode']:
        infer_thinking(model, tokenizer, test_data, params)
    else:
        infer(model, tokenizer, test_data, params)


if __name__ == "__main__":
    main()
