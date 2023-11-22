import argparse

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


def load_model(model_path: str):
    """
    Loads the Vicuna model.

    :param model_path:
    :return:
    """
    from_pretrained_kwargs = {"torch_dtype": torch.float16, "revision": "main"}
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
            revision="main",
            trust_remote_code=True,
        )
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=False, revision="main", trust_remote_code=True
        )
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            **from_pretrained_kwargs,
        )
    except NameError:
        model = AutoModel.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            **from_pretrained_kwargs,
        )

    return model, tokenizer


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


def compare_outputs(model, tokenizer, question: str, model_1_output: str, model_2_output: str) -> tuple[int, int]:
    # set parameters
    temperature = 1.0
    repetition_penalty = 1.0
    top_p = 1.0
    top_k = -1
    max_new_tokens = 256
    logprobs = None
    echo = True
    stop_str = None
    context_len = 2048
    stop_token_ids = []
    if tokenizer.eos_token_id not in stop_token_ids:
        stop_token_ids.append(tokenizer.eos_token_id)

    prompt = (
        f"Please rate two responses on a scale from 1 to 10 with 1 being a poor response and 10 being "
        f"a good response using a pipe '|' symbol to separate the scores i.e. '4,7' or '9,1'"
        f"The context to the question is: '{question}'."
        f"Model 1 response: '{model_1_output}'."
        f"Model 2 response: '{model_2_output}'."
    )

    logits_processor = prepare_logits_processor(
        temperature, repetition_penalty, top_p, top_k
    )
    input_ids = tokenizer(prompt).input_ids
    max_src_len = context_len - max_new_tokens - 1
    input_ids = input_ids[-max_src_len:]
    output_ids = list(input_ids)
    input_echo_len = len(input_ids)
    start_ids = torch.as_tensor([input_ids], device='cuda')
    out = model(input_ids=start_ids, use_cache=True)
    logits = out.logits
    past_key_values = out.past_key_values
    token_logprobs = [None]  # The first token has no logprobs.

    if logits_processor:
        if repetition_penalty > 1.0:
            tmp_output_ids = torch.as_tensor([output_ids], device=logits.device)
        else:
            tmp_output_ids = None
        last_token_logits = logits_processor(tmp_output_ids, logits[:, -1, :])[0]
    else:
        last_token_logits = logits[0, -1, :]

    if temperature < 1e-5 or top_p < 1e-8:  # greedy
        _, indices = torch.topk(last_token_logits, 2)
        tokens = [int(index) for index in indices.tolist()]
    else:
        probs = torch.softmax(last_token_logits, dim=-1)
        indices = torch.multinomial(probs, num_samples=2)
        tokens = [int(token) for token in indices.tolist()]
    token = tokens[0]
    output_ids.append(token)
    if logprobs is not None:
        # Cannot use last_token_logits because logprobs is based on raw logits.
        token_logprobs.append(
            torch.log_softmax(logits[0, -1, :], dim=-1)[token].tolist()
        )

    # Yield the output tokens
    if echo:
        tmp_output_ids = output_ids
    else:
        tmp_output_ids = output_ids[input_echo_len:]

    output = tokenizer.decode(
        tmp_output_ids,
        skip_special_tokens=True,
        spaces_between_special_tokens=False,
        clean_up_tokenization_spaces=True,
    )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    args = parser.parse_args()

    vicuna_model, vicuna_tokenizer = load_model(args.model_path)
    result = compare_outputs(
        model=vicuna_model,
        tokenizer=vicuna_tokenizer,
        question="Johnny only wears shirts of his favorite color and his shirt is blue. What is his favorite color?",
        model_1_output="Johnny's favorite color is red.",
        model_2_output="Johnny's favorite color is blue.",
    )

