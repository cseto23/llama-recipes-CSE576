import argparse
import re

from transformers import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer
)
import torch
from fastchat.serve.inference import *


def prepare_logits_processor(
    temperature: float, repetition_penalty: float, top_p: float, top_k: int
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    # TemperatureLogitsWarper doesn't accept 0.0, 1.0 makes it a no-op so we skip two cases.
    if temperature >= 1e-5 and temperature != 1.0:
        processor_list.append(TemperatureLogitsWarper(temperature))
    if repetition_penalty > 1.0:
        processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    if 1e-8 <= top_p < 1.0:
        processor_list.append(TopPLogitsWarper(top_p))
    if top_k > 0:
        processor_list.append(TopKLogitsWarper(top_k))
    return processor_list


def compare_outputs(model, tokenizer, device: str, question: str, model_1_output: str, model_2_output: str) -> list:
    prompt = (
        f"I will give you a question and two responses. Rate the two responses on a scale of 1 to 10 for RESPONSE 1 "
        f"and RESPONSE 2 separated by a '|' respectively. For example 'SCORE1|SCORE2'.\n"
        f"QUESTION: '{question}'\n"
        f"RESPONSE 1: '{model_1_output}'\n"
        f"RESPONSE 2: '{model_2_output}'"
    )
    model_path = model.config._name_or_path
    conv = get_conversation_template(model_path)
    gen_params = {
        "model": model_path,
        "prompt": prompt,
        "temperature": 0.1,
        "repetition_penalty": 1.0,
        "max_new_tokens": 512,
        "stop": conv.stop_str,
        "stop_token_ids": conv.stop_token_ids,
        "echo": False,
    }
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    return list(generate_stream(
        model,
        tokenizer,
        gen_params,
        device,
        context_len=get_context_length(model.config),
        judge_sent_end=True,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    args = parser.parse_args()

    vicuna_model, vicuna_tokenizer = load_model(args.model_path)
    for i in range(10):
        result = compare_outputs(
            model=vicuna_model,
            tokenizer=vicuna_tokenizer,
            device="cuda",
            question="Johnny only wears shirts of his favorite color and his shirt is blue. What is his favorite color?",
            model_1_output="Johnny's favorite color is red.",
            model_2_output="Johnny's favorite color is blue.",
        )
        with open("results.json", "w") as file:
            file.write(json.dumps(result))
        print(re.findall("\\d+\\|\\d+", result[-1]["text"]))

    # TODO: load next 1000 "diverse" samples, along with ground truths,
    #  and save comparisons.


