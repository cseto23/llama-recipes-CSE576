# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# from accelerate import init_empty_weights, load_checkpoint_and_dispatch

import fire
import os
import sys
import time
import json
import torch
from transformers import LlamaTokenizer

from llama_recipes.inference.safety_utils import get_safety_checker
from llama_recipes.inference.model_utils import load_model, load_peft_model
from datasets import load_dataset
import pandas as pd
from datasets import load_dataset
from transformers import default_data_collator
import tqdm

import copy
import json

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

def main(
    model_name,
    peft_model: str=None,
    quantization: bool=False,
    max_new_tokens =100, #The maximum numbers of tokens to generate
    output_file: str="predictions.json",
    seed: int=42, #seed value for reproducibility
    do_sample: bool=False, #Whether or not to use sampling ; use greedy decoding otherwise.
    min_length: int=None, #The minimum length of the sequence to be generated, input prompt + min_new_tokens
    use_cache: bool=True,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float=1.0, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float=0, # [optional] The value used to modulate the next token probabilities.
    top_k: int=50, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=1.0, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation. 
    max_padding_length: int=None, # the max padding length to be used with tokenizer padding the prompts.
    use_fast_kernels: bool = False, # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    split:str = "test", #train or test
    train_idx_start:int = 0, #train index start
    train_idx_end:int = 52002, #train index end
    **kwargs
):
    
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    if split=="test":
        all_lines = open("/home/dfulop/CSE_576_2023F_project_1/combined_test_set.jsonl").readlines()
    elif split=="train":
        fp = open("/home/dfulop/CSE_576_2023F_project_1/diverse_examples.json", "r")
        examples = json.load(fp)
        fp.close()
        examples = examples['examples']
        alpaca_ds = list(load_dataset("tatsu-lab/alpaca")['train'])[train_idx_start:train_idx_end]
        all_lines = []
        for idx,d in enumerate(alpaca_ds):
            if idx+train_idx_start not in examples:
                if d.get("input", "") == "":
                    prompt = PROMPT_DICT["prompt_no_input"].format_map(d)
                else:
                    prompt = PROMPT_DICT["prompt_input"].format_map(d)
                all_lines.append({"prompt":prompt, "ground_truth":d.get("output",""), "alpaca_idx":(idx+train_idx_start)})

    # Set the seeds for reproducibility
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    
    model = load_model(model_name, quantization)
    if peft_model:
        model = load_peft_model(model, peft_model)

    model.eval()
    
    if use_fast_kernels:
        """
        Setting 'use_fast_kernels' will enable
        using of Flash Attention or Xformer memory-efficient kernels 
        based on the hardware being used. This would speed up inference when used for batched inputs.
        """
        try:
            from optimum.bettertransformer import BetterTransformer
            model = BetterTransformer.transform(model)    
        except ImportError:
            print("Module 'optimum' not found. Please install 'optimum' it before proceeding.")

    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    
    all_predictions = []
    
    for idx, line in enumerate(tqdm.tqdm(all_lines)):
        if split=="test":
            data = json.loads(line)
        elif split=="train":
            data = line
        user_prompt = data["prompt"]
                
        batch = tokenizer(user_prompt, padding='max_length', truncation=True, max_length=max_padding_length, return_tensors="pt")
        number_of_input_tokens = len(batch["input_ids"][0])
        batch = {k: v.to("cuda") for k, v in batch.items()}
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                min_length=min_length,
                use_cache=use_cache,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                output_hidden_states= True, return_dict_in_generate=True,
                **kwargs 
            )
        e2e_inference_time = (time.perf_counter()-start)*1000
        number_of_output_tokens = len(outputs.sequences[0])
        number_of_new_tokens = number_of_output_tokens - number_of_input_tokens
        print(f"Total inference time is {e2e_inference_time} ms for {number_of_new_tokens} tokens")
        print(f"Per Token inference time is {e2e_inference_time/number_of_new_tokens} ms")
        output_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

        data_with_prediction = data.copy()
        data_with_prediction["response"] = output_text[len(user_prompt):]
        data_with_prediction["total_response_time"] = e2e_inference_time
        data_with_prediction["new_tokens_generated"] = number_of_new_tokens
        data_with_prediction["avg_response_time_per_token"] = e2e_inference_time/number_of_new_tokens
        
        
        
        all_predictions.append(data_with_prediction)

        json_object = json.dumps({
            "parameters": {
                "model_name": model_name,
                "peft_model": peft_model,
                "quantization": quantization,
                "max_new_tokens": max_new_tokens,
                "output_file": output_file,
                "seed": seed,
                "do_sample": do_sample,
                "min_length": min_length,
                "use_cache": use_cache,
                "top_p": top_p,
                "temperature": temperature,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "length_penalty": length_penalty,
                "max_padding_length": max_padding_length,
                "use_fast_kernels": use_fast_kernels,

            }, 
        "predictions": all_predictions
        }, indent=4)
        
        with open(output_file, "w") as outfile:
            outfile.write(json_object)

        
if __name__ == "__main__":
    fire.Fire(main)