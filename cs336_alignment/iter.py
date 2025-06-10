from cs336_alignment.sft import train_sft, train_setup, evaluate
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
import wandb
import json
import numpy as np
import cs336_alignment.baseline as baseline
from cs336_alignment.info import *
from typing import List
from vllm import LLM, SamplingParams
from transformers import PreTrainedModel, PreTrainedTokenizer
import torch
import random

def filter_responses(prompts: List[str], 
                     responses: List[str], 
                     answers: List[str], 
                     save_path: str = None) -> tuple[List[str], List[str], List[str]]:
    """
    Filter responses based on reward function.
    """
    save_list = []
    filtered_prompts = []
    filtered_responses = []
    filtered_answers = []
    for prompt, response, answer in zip(prompts, responses, answers):
        score = r1_zero_reward_fn(response, answer)
        if score["reward"] == 1.0:
            save_list.append({
                "prompt": prompt,
                "response": response,
                "ground_truth": answer
            })
            filtered_prompts.append(prompt)
            filtered_responses.append(response)
            filtered_answers.append(answer)
    
    if save_path is not None:
        with open(save_path, "w") as f:
            for item in save_list:
                f.write(json.dumps(item) + "\n")
    
    return filtered_prompts, filtered_responses, filtered_answers

def do_rollouts(prompts: List[str], 
                answers: List[str], 
                vllm_model: LLM, 
                train_sampling_params: SamplingParams, 
                n_rollouts: int = 1) -> tuple[List[str], List[str], List[str]]:
    """
    Do rollouts for each prompt/answer pair.
    """
    prompts = prompts * n_rollouts # repeat prompts n_rollouts times
    answers = answers * n_rollouts # repeat answers n_rollouts times
    responses = vllm_model.generate(prompts, train_sampling_params)
    return [r.outputs[0].text for r in responses], prompts, answers # return responses, prompts, answers

def iterate(config: dict, 
            train_prompts: List[str], 
            train_answers: List[str],
            val_prompts: List[str],
            val_answers: List[str],
            vllm_model: LLM, 
            model: PreTrainedModel,
            model_device: str,
            optimizer: torch.optim.Optimizer,
            tokenizer: PreTrainedTokenizer,
            sampling_params: SamplingParams, 
            start_train_step: int = 0, 
            end_eval: bool = False,
            iter: int = 0):
    
    g_responses, g_prompts, g_answers = do_rollouts(train_prompts, train_answers, vllm_model, sampling_params, config['n_rollouts'])
    save_path = f"./iter_results/iter_{iter}_{config['n_rollouts']}_{config['iter_batch_size']}_{config['n_epochs']}_train.jsonl"
    filtered_prompts, filtered_responses, filtered_answers = filter_responses(g_prompts, g_responses, g_answers, save_path=save_path)

    train_step = train_sft(train_prompts = filtered_prompts,
                           train_responses = filtered_responses,
                           train_ground_truths = filtered_answers,
                           val_prompts = val_prompts,
                           val_answers = val_answers,
                           vllm_model = vllm_model,
                           model = model,
                           model_device = model_device,
                           optimizer = optimizer,
                           tokenizer = tokenizer,
                           eval_sampling_params = sampling_params,
                           config = config,
                           start_train_step = start_train_step,
                           end_eval = end_eval)
    
    # sample and evaluate
    val_batch_indices = random.sample(range(len(val_prompts)), 4 * config['train_batch_size'])
    val_batch_prompts = [val_prompts[i] for i in val_batch_indices] 
    val_batch_answers = [val_answers[i] for i in val_batch_indices]

    correct_fraction, format_fraction = evaluate(prompts = val_batch_prompts, 
                                                        answers = val_batch_answers, 
                                                        vllm_model = vllm_model, 
                                                        eval_sampling_params = sampling_params, 
                                                        reward_fn = baseline.r1_zero_reward_fn)

    print(f'Logging metrics to wandb for iteration {iter}...')
    wandb.log({
        "iter/eval_correct": correct_fraction,
        "iter/eval_format": format_fraction,
        "iter/n_sft": len(filtered_prompts),
        "iter": iter,
    })

    return train_step


def run_iter(config, sampling_params, train_path = MATH_TRAIN_PATH, vllm_device = 'cuda', model_device = 'cuda'):
    # load model and tokenizer
    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr = config['learning_rate'], 
                                  weight_decay = config['weight_decay'], 
                                  betas = config['betas'])
    vllm_model, model, tokenizer = train_setup(config['model'], config['seed'], config['learning_rate'], vllm_device, model_device)
    
    # load validation and training data
    val_questions, val_answers = baseline.load_MATH(config['val_path'])
    val_prompts = baseline.make_prompts(val_questions)
    train_questions, train_answers = baseline.load_MATH(train_path)
    train_prompts = baseline.make_prompts(train_questions)
    
    train_step = 0
    for i in range(config['n_ei_steps']):
        # sample config['iter_batch_size'] items by index
        sampled_indices = np.random.choice(len(train_prompts), config['iter_batch_size'], replace=False)
        print(f'Sampled {len(sampled_indices)} examples')
        step_prompts = [train_prompts[i] for i in sampled_indices]
        step_answers = [train_answers[i] for i in sampled_indices]

        train_step = iterate(config = config, 
                             train_prompts = step_prompts, 
                             train_answers = step_answers, 
                             val_prompts = val_prompts, 
                             val_answers = val_answers, 
                             vllm_model = vllm_model, 
                             model = model, 
                             model_device = model_device, 
                             optimizer = optimizer, 
                             tokenizer = tokenizer, 
                             sampling_params = sampling_params, 
                             start_train_step = train_step, 
                             end_eval = (i == config['n_ei_steps'] - 1), # only do full evaluation on last step
                             iter = i)

if __name__ == "__main__":
    config = {
        'model': QWEN_25,
        'n_rollouts': 10,
        'n_ei_steps': 5,
        'n_epochs': 2,
        'n_unique': None, # always train on full SFT dataset
        'minibatch_size': 4,
        'iter_batch_size': 512, # number of examples to rollout per iteration
        'train_batch_size': 128,
        'learning_rate': 1e-4,
        'log_every_n': 10,
        'eval_every_n': 20,
        'val_path': MATH_VAL_PATH,
        'train_path': MATH_TRAIN_PATH,
        'seed': 42,
        'start_train_step': 0,
    }

    wandb.init(project = "cs336-alignment-iter", 
               name = f"iter_{config['n_epochs']}_{config['n_rollouts']}_{config['iter_batch_size']}", 
               config = config)
    
    run_iter(config = config, 
             sampling_params = EVAL_SAMPLING_PARAMS, 
             train_path = MATH_TRAIN_PATH, 
             vllm_device = 'cuda', 
             model_device = 'cuda')

#if __name__ == "__main__":
#    train_prompts, train_responses, train_ground_truths = load_SFT(MATH_TRAIN_PATH)
#    filter_responses(train_prompts, train_responses, train_ground_truths, save_path="filtered_train.jsonl")