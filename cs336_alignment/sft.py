from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
import torch
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from vllm import LLM, SamplingParams
import wandb
import json
import cs336_alignment.baseline as baseline
import cs336_alignment.utils as utils
import random
from unittest.mock import patch
from typing import List, Callable
from cs336_alignment.info import *

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    """
    Start the inference process, here we use vLLM to hold a model on
    a GPU separate from the policy
    """
    vllm_set_random_seed(seed)
    
    # Monkeypatch from TRL:
    # https://github.com/huggingface/trl/blob/
    # 22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py
    # Patch vLLM to make sure we can
    # (1) place the vLLM model on the desired device (world_size_patch) and
    # (2) avoid a test that is not designed for our setting (profiling_patch).
    
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )
    
def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    Copied from https://github.com/huggingface/trl/blob/
    22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

def train_setup(model_string: str, 
                seed: int = 42, 
                learning_rate: float = 1e-4, 
                vllm_device: str = 'cuda', 
                model_device: str = 'cuda') -> tuple[LLM, PreTrainedModel, PreTrainedTokenizer, torch.optim.Optimizer]:
    
    # initialize vllm onto 1st GPU
    print("Initializing vllm...")
    vllm_model = init_vllm(model_string, device = vllm_device, seed = seed, gpu_memory_utilization=0.2)

    # load model and tokenizer onto 1st GPU
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        model_string,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map = model_device
    )

    # get tokenizer, optimizer
    tokenizer = AutoTokenizer.from_pretrained(model_string)
    optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

    # load policy into vllm
    print("Loading policy into vllm...")
    load_policy_into_vllm_instance(model, vllm_model)

    return vllm_model, model, tokenizer, optimizer

def full_train_step(prompts: List[str], 
                    answers: List[str], 
                    tokenizer: PreTrainedTokenizer, 
                    model: PreTrainedModel, 
                    gradient_accumulation_steps: int, 
                    do_backward: bool = True, 
                    device: str = 'cuda:0') -> tuple[float, float]:
    """
    Do a full SFT training step from prompts/answers
    """
    # tokenize prompts and responses --> input_ids, labels, response_mask
    tokenized_results = utils.tokenize_prompt_and_output(prompts, answers, tokenizer, device = device)

    # get log probs and entropy
    lp_dict = utils.get_response_log_probs(model, tokenized_results["input_ids"], tokenized_results["labels"], return_token_entropy = True)
    log_probs = lp_dict["log_probs"]
    token_entropy = lp_dict["token_entropy"]

    # compute loss and do backward pass
    loss, metadata = utils.sft_microbatch_train_step(policy_log_probs = log_probs,
                                                        response_mask = tokenized_results["response_mask"],
                                                        gradient_accumulation_steps = gradient_accumulation_steps,
                                                        normalize_constant = 1.0,
                                                        do_backward = do_backward)
    
    return loss.item(), token_entropy.mean().item()

def evaluate_loss(prompts: List[str], answers: List[str], tokenizer: PreTrainedTokenizer, model: PreTrainedModel, minibatch_size: int):
    """
    Evaluate the model on a set of prompts/answers
    """
    
    # set model to eval mode
    model.eval()
    avg_loss = 0
    avg_entropy = 0
    n_minibatches = len(prompts) // minibatch_size

    with torch.no_grad():
        for i in range(0, len(prompts), minibatch_size):
            minibatch_prompts = prompts[i:i+minibatch_size]
            minibatch_answers = answers[i:i+minibatch_size]
            
            loss, entropy = full_train_step(minibatch_prompts, 
                            minibatch_answers, 
                            tokenizer, model, 
                            gradient_accumulation_steps = 1, 
                            do_backward = False)
            avg_loss += loss.item()
            avg_entropy += entropy
    
    model.train()
    
    # average loss and entropy over minibatches
    avg_loss /= n_minibatches
    avg_entropy /= n_minibatches

    return avg_loss, avg_entropy

def evaluate(prompts: List[str], answers: List[str], vllm_model: LLM, eval_sampling_params: SamplingParams, reward_fn: Callable[[str, str], dict[str, float]]):
    """
    Evaluate the model on a set of prompts/answers
    """
    results = baseline.evaluate_vllm(vllm_model, 
                           reward_fn, 
                           prompts, 
                           answers, 
                           eval_sampling_params)

    # compute metrics
    correct_fraction, format_only_fraction, wrong_fraction = baseline.compute_metrics(results, printout = False)

    return correct_fraction, format_only_fraction


def load_SFT(path: str):
    with open(path, 'r') as f:
        data = [json.loads(line) for line in f]
    
    return [d["prompt"] for d in data], [d["response"] for d in data], [d["ground_truth"] for d in data]

def train_run(config: dict,
              eval_sampling_params: SamplingParams = None,
              end_eval: bool = True):
    
    vllm_device = 'cuda:0'
    model_device = 'cuda:0'

    # do training setup
    vllm_model, model, tokenizer, optimizer = train_setup(config['model'], config['seed'], config['learning_rate'], vllm_device, model_device)

    # load training and validation sets
    print("Loading training and validation sets...")
    train_prompts, train_responses, train_ground_truths = load_SFT(config['train_path'])
    print(f"Loaded {len(train_prompts)} training examples")

    # sample subset of unique prompts to use
    if config['n_unique']: # use random sample of unique prompts
        train_idxs = random.sample(range(len(train_prompts)), config['n_unique'])
        train_prompts = [train_prompts[i] for i in train_idxs]
        train_responses = [train_responses[i] for i in train_idxs]
        train_ground_truths = [train_ground_truths[i] for i in train_idxs]
    
    val_questions, val_answers = baseline.load_MATH(config['val_path'])
    val_prompts = baseline.make_prompts(val_questions)

    # run SFT training
    train_sft(train_prompts = train_prompts, 
            train_responses = train_responses, 
            train_ground_truths = train_ground_truths, 
            val_prompts = val_prompts, 
            val_answers = val_answers, 
            vllm_model = vllm_model, 
            model = model, 
            model_device = model_device, 
            optimizer = optimizer, 
            tokenizer = tokenizer, 
            eval_sampling_params = eval_sampling_params, 
            config = config,
            start_train_step = config['start_train_step'],
            end_eval = end_eval)

def train_sft(train_prompts: List[str], 
            train_responses: List[str], 
            train_ground_truths: List[str], 
            val_prompts: List[str], 
            val_answers: List[str], 
            vllm_model: LLM, 
            model: PreTrainedModel, 
            model_device: str,
            optimizer: torch.optim.Optimizer, 
            tokenizer: PreTrainedTokenizer, 
            eval_sampling_params: SamplingParams,
            config: dict,
            start_train_step: int = 0,
            end_eval: bool = True):
    
    minibatch_size = config['minibatch_size']
    batch_size = config['train_batch_size']
    n_epochs = config['n_epochs']
    log_every_n = config['log_every_n']
    eval_every_n = config['eval_every_n']
    n_unique = config['n_unique']
    learning_rate = config['learning_rate']
    
    print("Running SFT training...")
    gradient_accumulation_steps = batch_size // minibatch_size
    n_minibatches = len(train_prompts) // config['minibatch_size']
    train_step = start_train_step
    mini_train_step = 0
    log_train = True 
    log_eval = True
    
    print(f"Training for {n_epochs} epochs...")
    for epoch in range(n_epochs):
        # shuffle train indices before each epoch
        train_indices = list(range(len(train_prompts)))
        random.shuffle(train_indices)
        train_prompts = [train_prompts[i] for i in train_indices]
        train_responses = [train_responses[i] for i in train_indices]
        train_ground_truths = [train_ground_truths[i] for i in train_indices]
        
        # run training steps
        n_minibatches = len(train_prompts) // minibatch_size
        print(f'Training on {len(train_indices)} examples in {n_minibatches} minibatches')
        
        for minibatch_idx in range(n_minibatches):
            minibatch_prompts = train_prompts[minibatch_idx * minibatch_size:(minibatch_idx + 1) * minibatch_size]
            minibatch_responses = train_responses[minibatch_idx * minibatch_size:(minibatch_idx + 1) * minibatch_size]

            loss_val, avg_entropy = full_train_step(minibatch_prompts, 
                            minibatch_responses, 
                            tokenizer, model, 
                            gradient_accumulation_steps, 
                            do_backward = True,
                            device = model_device)
            
            # backwards pass
            if (mini_train_step + 1) % gradient_accumulation_steps == 0:
                # do gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # perform gradient descent once accumulated
                optimizer.step()
                optimizer.zero_grad()
                train_step += 1

                print("Train step: ", train_step)
                if train_step % log_every_n == 0:
                    log_train = True
                if train_step % eval_every_n == 0:
                    log_eval = True
            
            # logging
            if log_train:
                log_train = False
                wandb.log({
                    "train/loss": loss_val,
                    "train/avg_entropy": avg_entropy,
                }, step = train_step)

            if log_eval:
                multiplier = 2
                log_eval = False
                print("Training step: ", train_step)
                print(f"Evaluating on {batch_size * multiplier} prompts...")

                # load policy into vllm
                print("Loading policy into vllm...")
                load_policy_into_vllm_instance(model, vllm_model)

                # select random double-batch of eval prompts/answers
                val_batch_indices = random.sample(range(len(val_prompts)), batch_size * multiplier)
                val_batch_prompts = [val_prompts[i] for i in val_batch_indices] 
                val_batch_answers = [val_answers[i] for i in val_batch_indices]

                correct_fraction, format_fraction = evaluate(prompts = val_batch_prompts, 
                                                                   answers = val_batch_answers, 
                                                                   vllm_model = vllm_model, 
                                                                   eval_sampling_params = eval_sampling_params, 
                                                                   reward_fn = baseline.r1_zero_reward_fn)
                print('Logging metrics to wandb...')
                wandb.log({
                    "eval/correct": correct_fraction,
                    "eval/format": format_fraction,
                }, step = train_step)

                log_indices = random.sample(range(len(val_prompts)), minibatch_size)
                log_prompts = [val_prompts[i] for i in log_indices]
                log_answers = [val_answers[i] for i in log_indices]
                print(f"Logging generations for {len(log_prompts)} prompts...")
                utils.log_generations(vllm_model = vllm_model, 
                                        reward_fn = baseline.r1_zero_reward_fn, 
                                        prompts = log_prompts, 
                                        answers = log_answers, 
                                        sampling_params = eval_sampling_params,
                                        log_file = f'sft_results/sft_{n_unique}_{batch_size}_{minibatch_size}_{learning_rate}.txt')
                
            mini_train_step += 1
    
    remaining_steps = gradient_accumulation_steps - (mini_train_step % gradient_accumulation_steps)
    print(f"Remaining steps: {remaining_steps}")
    if remaining_steps > (gradient_accumulation_steps // 2):
        print("Performing partial gradient update...")
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        train_step += 1
    else:
        print("Not performing gradient update...")
        optimizer.zero_grad()

    print("Training complete!")
    load_policy_into_vllm_instance(model, vllm_model) # load policy into vllm

    # do full val set evaluation
    if end_eval:
        print("Evaluating on full validation set...")
        correct_fraction, format_fraction = evaluate(prompts = val_prompts, 
                                                                answers = val_answers, 
                                                                vllm_model = vllm_model, 
                                                                eval_sampling_params = eval_sampling_params, 
                                                                reward_fn = baseline.r1_zero_reward_fn)
        print('Logging metrics to wandb...')
        wandb.log({
            "eval/correct": correct_fraction,
            "eval/format": format_fraction,
        }, step = train_step)

    return train_step
    
if __name__ == "__main__":
    train_path = "./filtered_train.jsonl"
    n_examples_full = sum(1 for _ in open(train_path, 'r'))

    eval_sampling_params = SamplingParams(
        temperature = 1.0, 
        top_p = 1.0, 
        max_tokens = 1024, 
        stop = ["</answer>"], 
        include_stop_str_in_output = True,
    )
    n_steps = 64
    config = {
        "model": QWEN_25,
        "n_unique": None,
        "minibatch_size": 8,
        "train_batch_size": 128,
        "learning_rate": 1e-4,
        "seed": 42,
        "log_every_n": 10,
        "eval_every_n": 20,
        "train_path": train_path,
        "val_path": MATH_VAL_PATH,
    }
    if config['n_unique'] is None:
        config['n_unique'] = n_examples_full
    
    config['n_epochs'] = n_steps // (config['n_unique'] // config['train_batch_size'])

    wandb.init(project = "cs336-alignment-sft", 
               name = f"filtered_sft_{config['n_unique']}_{config['train_batch_size']}_{config['learning_rate']}", 
               config = config)
    
    train_run(config = config,
              eval_sampling_params = eval_sampling_params,
              end_eval = True)
