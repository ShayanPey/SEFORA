import sys
import torch, gc
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
from huggingface_hub import login
import matplotlib.pyplot as plt
import glob
import json
import numpy as np
import random
import os
from pathlib import Path
import re
from typing import List

login(token=os.environ["HUGGINGFACE_KEY"])


############################################################################################ MODEL ##############################################################################################
gc.collect()
torch.cuda.empty_cache()
model_name = "meta-llama/Llama-2-7b-hf"
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    use_fast=False,
    token=True
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    device_map="auto",
    use_auth_token=True,
    trust_remote_code=True
)
#model = model.to("cuda")
model.eval()
##################################################################################################################################################################################################
########################################################################################### PROMPT ################################################################################################
PROMPT_TEMPLATE = (
"""
You are a professional writing instructor providing very concise, useful feedback. You should spot obvious errors and mention surface-level improvements, but your main focus is to offer incisive, deep, insightful, and actionable advice that helps the student improve. When a passage is especially strong and offers little room for improvement, acknowledge that briefly. Your response should be concise, to the point, and limited to a few sentences, each addressing a specific improvement or commendation. While keeping the full essay in mind, focus only on the target paragraph. If something is wrong with a specific part, sentence, or word, explicitly identify that part.

### Essay:\n{context}
### Target Paragraph:\n{paragraph}
### Provide feedback to the student on the target paragraph:\n
"""
)
##################################################################################################################################################################################################
########################################################################################## NO EDIT ################################################################################################
print("CUDA device count:", torch.cuda.device_count())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
try:
    from pprint import pprint
    pprint(model.hf_device_map)
except Exception as e:
    print("No hf_device_map available:", e)
print(model.config._name_or_path)
print(PROMPT_TEMPLATE)
print(model.config.max_position_embeddings)
MAX_CONTEXT_LENGTH = min(model.config.max_position_embeddings, 8192)
print(MAX_CONTEXT_LENGTH)
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.device_count() > 0:
    print("CUDA device name:", torch.cuda.get_device_name(0))
else:
    print("No usable CUDA devices detected.")
print(os.system("nvidia-smi"))

import os
from typing import Optional

def fetch_rubric_from_filename(
    filename: str,
    dataset_root: str = "./SEFORA"
) -> Optional[str]:
    """
    Given a filename like:
    Batch_1_Course_1_Class_1_Essay_1_Initial_ID0110.json

    Walks from the deepest inferred directory upward and
    returns the contents of the first .txt file found.
    """

    name = os.path.basename(filename)
    if not name.endswith(".json"):
        raise ValueError(f"Not a JSON file: {filename}")
    parts = name[:-5].split("_")  # remove .json, split
    if len(parts) < 10:
        print("RUBRIC NOT FOUND RETURNING EMPTY!")
        return ""

    batch  = "_".join(parts[0:2])   # Batch_#
    course = "_".join(parts[2:4])   # Course_#
    clazz  = "_".join(parts[4:6])   # Class_#
    essay  = "_".join(parts[6:8])   # Essay_#
    stage  = parts[8]               # STAGE (single token)

    current_dir = os.path.abspath(
        os.path.join(dataset_root, batch, course, clazz, essay, stage)
    )
    dataset_root = os.path.abspath(dataset_root)

    # Walk upward until dataset_root
    while True:
        if os.path.isdir(current_dir):
            for fname in os.listdir(current_dir):
                if fname.lower().endswith(".txt"):
                    rubric_path = os.path.join(current_dir, fname)
                    with open(rubric_path, "r", encoding="utf-8") as f:
                        return f.read()

        if current_dir == dataset_root:
            break

        parent = os.path.dirname(current_dir)
        if parent == current_dir:
            break

        current_dir = parent

    return None
def remove_half_sentence(text: str) -> str:
    """
    Remove the last incomplete sentence from the text if it doesn't end
    with a sentence terminator (., !, ?, ;) optionally followed by quotes or brackets.
    If no terminator is found anywhere, returns the original text.
    """
    if not text:
        return text

    stripped = text.rstrip()

    if re.search(r'[\.!?;][\'"\)\]\}]*\Z', stripped):
        return text  # Already ends with a complete sentence

    last_pos = max(
        stripped.rfind('.'),
        stripped.rfind('!'),
        stripped.rfind('?'),
        stripped.rfind(';')
    )
    if last_pos == -1:
        return text  # No terminator at all
    truncated = stripped[:last_pos + 1]
    return truncated
    

def make_prompt(paragraphs: list, idx: int, tokenizer: PreTrainedTokenizer, max_tokens: int, prompt_overhead: int, rubric: str) -> str:
    """
    Build a prompt for paragraph at index idx, including preceding and following context paragraphs
    truncated to fit within the model's context window by removing the largest-side paragraphs first.
    """
    target = paragraphs[idx]

    # Build initial context list: all paragraphs including target
    all_ctx = []
    # include preceding paragraphs (in order)
    for j in range(idx + 1):
        all_ctx.append(paragraphs[j])
    # include following paragraphs
    for j in range(idx + 1, len(paragraphs)):
        all_ctx.append(paragraphs[j])

    # Helper to count words in a list of paragraphs
    def word_count(paras):
        return sum(len(p.split()) for p in paras)

    # Trim paragraphs until the token limit is satisfied
    while True:
        # Generate the context string
        context = "\n".join(all_ctx).strip()
        # Compute token lengths
        ctx_len = tokenizer(context, return_tensors="pt")["input_ids"].size(1)
        tgt_len = tokenizer(target, return_tensors="pt")["input_ids"].size(1)
        
        CONTEXT_BUDGET = int(0.37 * max_tokens) 

        if ctx_len + prompt_overhead <= CONTEXT_BUDGET:
            break 
        if len(all_ctx) == 0:
            break
        # Determine words before and after the target in all_ctx
        tgt_pos = all_ctx.index(target)
        before = word_count(all_ctx[:tgt_pos])
        after = word_count(all_ctx[tgt_pos+1:])
        # Remove from the side with more words;
        if before > after and tgt_pos > 0:
            all_ctx.pop(0)
        elif len(all_ctx) - 1 > tgt_pos:
            all_ctx.pop()
        else:
            all_ctx.pop(0)

    final_context = "\n".join(all_ctx).strip()
    if "{rubric}" in PROMPT_TEMPLATE:
        return PROMPT_TEMPLATE.format(context=final_context, paragraph=target, rubric=rubric)
    return PROMPT_TEMPLATE.format(context=final_context, paragraph=target)

def process_json_files(json_dir: str,
                       num_files: int = None,
                       continue_ind: int = None,
                       batch_size: int = 8,
                       seed: int = 42,
                       setting: str = "default",
                       save_dir: str = None,
                       tokenizer: PreTrainedTokenizer = None,
                       model: PreTrainedModel = None,
                       use_rubric: bool = False) -> list:
    """
    Entry-point: sample files and generate feedback.
    """
    # collect filepaths
    json_dir = os.path.expanduser(json_dir)    
    paths = glob.glob(os.path.join(json_dir, '**', '*.json'), recursive=True)
    if num_files is not None:
        random.seed(seed)
        paths = random.sample(paths, min(num_files, len(paths)))
    # delegate
    #print(paths)
    #return
    if continue_ind is None:
        continue_ind = 0
    return run_exp(paths, continue_ind, tokenizer, model, setting, batch_size, seed, save_dir, use_rubric)

#################################################################################################################################################################################################
######################################################################################### PRE & POST PROCESS ####################################################################################
def post_process(text:str) -> str:
    return text
OUTPUT_PREFIX = ''
OUTPUT_PREFIX = OUTPUT_PREFIX
#################################################################################################################################################################################################
########################################################################################### RUN_EXP CoT? ########################################################################################

GEN_PARAMS = {
    "max_new_tokens": 250, # For faster testing
    "temperature": 0.1,
    "top_p": 1,
    "repetition_penalty": 1.2,
} 


def run_exp(filepaths: list,
            continue_ind: int,
            tokenizer: PreTrainedTokenizer,
            model: PreTrainedModel,
            setting: str = "default",
            batch_size: int = 8,
            seed: int = 42,
            save_dir: str = None,
            use_rubric: bool = False) -> list:
    """
    Process a list of JSON files, generating feedback for each paragraph in true parallel batches.
    """

    # Verify pad_token and padding_side
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(tokenizer, 'padding_side', None) != 'left':
        tokenizer.padding_side = 'left'

    # Determine context window and prompt overhead
    max_tokens = MAX_CONTEXT_LENGTH
    empty_prompt = ""
    if use_rubric:
        empty_prompt = PROMPT_TEMPLATE.format(context="", paragraph="", rubric="")
    else:
        empty_prompt = PROMPT_TEMPLATE.format(context="", paragraph="")
    prompt_overhead = tokenizer(empty_prompt, return_tensors="pt")["input_ids"].size(1)

    random.seed(seed)
    results = []

    cnt = 0
    for filepath in filepaths:
        if cnt < continue_ind:
            cnt += 1
            continue

        cnt += 1
        print("Processing ", cnt, '/', len(filepaths), filepath)
        os.system("nvidia-smi > ~/logs/GPU_Stat_" + setting)

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        original_paragraphs = data.get('paragraphs', [])

        rubric = ""
        if use_rubric:
            rubric = fetch_rubric_from_filename(filepath)
            
        # FULL bodies list — used ONLY for context
        bodies_all = [
            p.get('body', '').strip() or p.get('text', '').strip()
            for p in original_paragraphs
        ]

        # indices of paragraphs that have annotations (generation targets)
        target_indices = [
            i for i, p in enumerate(original_paragraphs)
            if p.get("annotations")
        ]

        # deep copy ENTIRE JSON so everything stays identical
        out_entry = json.loads(json.dumps(data))

        # nothing to generate
        if not target_indices:
            results.append(out_entry)
            if save_dir:
                save_dir = os.path.expanduser(save_dir)
                os.makedirs(save_dir, exist_ok=True)
                base = os.path.basename(filepath)
                out_path = os.path.join(save_dir, f"{base}")
                with open(out_path, 'w', encoding='utf-8') as wf:
                    json.dump(out_entry, wf, ensure_ascii=False, indent=2)
            continue

        # Process in true batches over TARGET indices
        for start in range(0, len(target_indices), batch_size):
            batch_orig_indices = target_indices[start:start + batch_size]

            # Build prompts USING FULL CONTEXT
            prompts = [
                make_prompt(
                    bodies_all,
                    orig_idx,
                    tokenizer,
                    max_tokens,
                    prompt_overhead,
                    rubric
                )
                for orig_idx in batch_orig_indices
            ]

            #print(prompts[0])

            # Tokenize
            batch_inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            batch_inputs = {k: v.to(model.device) for k, v in batch_inputs.items()}

            input_size = batch_inputs['input_ids'].shape[1]

            if seed is not None:
                torch.manual_seed(seed)

            with torch.no_grad():
                output_ids = model.generate(
                    **batch_inputs,
                    **GEN_PARAMS,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            # Decode and WRITE BACK IN PLACE
            for i, orig_idx in enumerate(batch_orig_indices):
                seq = output_ids[i]
                feedback_tokens = seq[input_size:]
                feedback = tokenizer.decode(
                    feedback_tokens,
                    skip_special_tokens=True
                ).strip()

                feedback = remove_half_sentence(feedback)

                entry = out_entry['paragraphs'][orig_idx]

                if 'model' not in entry:
                    entry['model'] = []

                for model_entry in entry['model']:
                    if model_entry.get('setting') == setting:
                        model_entry.setdefault('feedback', []).append(feedback)
                        break
                else:
                    entry['model'].append({
                        'setting': setting,
                        'feedback': [feedback]
                    })

        results.append(out_entry)

        # save if needed
        if save_dir:
            save_dir = os.path.expanduser(save_dir)
            os.makedirs(save_dir, exist_ok=True)
            base = os.path.basename(filepath)
            out_path = os.path.join(save_dir, f"{base}")
            with open(out_path, 'w', encoding='utf-8') as wf:
                json.dump(out_entry, wf, ensure_ascii=False, indent=2)

    return results

#################################################################################################################################################################################################
################################################################################################# RUN ############################################################################################
results = process_json_files(
    #json_dir="~/AllJsons/ALL",
    json_dir="~/ALL",
    num_files=None,
    continue_ind=None,
    batch_size=6,
    seed=42,
    save_dir="~/s1",
    tokenizer=tokenizer,
    model=model,
    setting = "rq2_" + "unguided_" + "zero_" + "prompt1_"+ "llama3.1_8B",
    use_rubric = False
    #setting = "rq2_unguided_zero-shot_prompt1_llama2"
)