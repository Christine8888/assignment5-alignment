from vllm import LLM, SamplingParams
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
import argparse
from typing import Callable, List
import json
from cs336_alignment.info import *

def evaluate_vllm(
        vllm_model: LLM,
        reward_fn: Callable[[str, str], dict[str, float]],
        prompts: List[str],
        answers: List[str],
        eval_sampling_params: SamplingParams,
        save_path: str = None,
        ) -> float:
    """
    Evaluate the VLLM model using the reward function.
    """
    # serialization
    results = []

    # generate text
    outputs = vllm_model.generate(prompts, eval_sampling_params)

    # grade outputs
    for prompt, output, answer in zip(prompts, outputs, answers):
        # get output response
        response = output.outputs[0].text
        scores = reward_fn(response, answer)
        results.append({
            "prompt": prompt,
            "model_output": response,
            "expected_answer": answer,
            "format_reward": scores["format_reward"],
            "answer_reward": scores["answer_reward"],
            "reward": scores["reward"]
        })
    
    if save_path is not None:
        with open(save_path, 'a') as f:
            # save json list of dicts
            json.dump(results, f, indent = 2)
    
    return results

def load_MATH(path: str):
    with open(path, 'r') as f:
        data = [json.loads(line) for line in f]
    
    return [d["problem"] for d in data], [d["answer"] for d in data]

def make_prompts(questions: List[str], prompt_path = PROMPT_PATH):
    with open(prompt_path, 'r') as f:
        prompt_txt = f.read()

    prompts = []
    for question in questions:
        prompts.append(prompt_txt.format(question = question))
    
    return prompts

def compute_metrics(results: List[dict], printout = True):
    # count number completely correct 
    n_correct = sum(1 for result in results if result["answer_reward"] == 1.0 and result["format_reward"] == 1.0)
    n_total = len(results)
    n_format_only = sum(1 for result in results if result["answer_reward"] == 0.0 and result["format_reward"] == 1.0)
    n_format = sum(1 for result in results if result["format_reward"] == 1.0)
    n_wrong = sum(1 for result in results if result["answer_reward"] == 0.0 and result["format_reward"] == 0.0)

    # print 10 wrong examples
    format_only_count = 0
    wrong_count = 0
    for result in results:
        if result["answer_reward"] == 0.0 and result["format_reward"] == 1.0:
            if format_only_count < 10 and printout:
                print("-" * 100)
                print("FORMAT ONLY")
                print(f"Prompt: {result['prompt']}")
                print(f"Model output: {result['model_output']}")
                print(f"Expected answer: {result['expected_answer']}")
                print("-" * 100)
            format_only_count += 1
        elif result["answer_reward"] == 0.0 and result["format_reward"] == 0.0:
            if wrong_count < 10 and printout:
                print("-" * 100)
                print("FULLY WRONG")
                print(f"Prompt: {result['prompt']}")
                print(f"Model output: {result['model_output']}")
                print(f"Expected answer: {result['expected_answer']}")
                print("-" * 100)
            wrong_count += 1

    print('MATH results:\n')
    print(f'Completely correct: {(n_correct / n_total):.3f}')
    print(f'Format only: {(n_format_only / n_total):.3f}')
    print(f'Format: {(n_format / n_total):.3f}')
    print(f'Wrong: {(n_wrong / n_total):.3f}')
    
    return n_correct / n_total, n_format / n_total, wrong_count / n_total

if __name__ == "__main__":
    # sampling setup
    sampling_params = SamplingParams(
        temperature = 1.0, 
        top_p = 1.0, 
        max_tokens = 1024, 
        stop = ["</answer>"], 
        include_stop_str_in_output = True,
    )
    llm = LLM(model = QWEN_25)

    math_qs, math_as = load_MATH()
    math_qs = make_prompts(math_qs)

    math_results = evaluate_vllm(vllm_model = llm, 
                                 reward_fn = r1_zero_reward_fn, 
                                 prompts = math_qs, 
                                 answers = math_as, 
                                 eval_sampling_params = sampling_params,
                                 save_path = 'math_results.jsonl')
    compute_metrics(math_results)

    # exit vllm process group
    llm.shutdown()

