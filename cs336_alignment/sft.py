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

QWEN_25 = "/data/a5-alignment/models/Qwen2.5-Math-1.5B"
PROMPT_PATH = './prompts/r1_zero.prompt'
MATH_TRAIN_PATH = '/data/a5-alignment/MATH/sft.jsonl'
MATH_VAL_PATH = '/data/a5-alignment/MATH/validation.jsonl'

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

def wandb_setup():
    """
    Set up wandb folders and x-axis metrics
    """
    wandb.define_metric("train_step") # the x‑axis for training
    wandb.define_metric("eval_step") # the x‑axis for evaluation
    # everything that starts with train/ is tied to train_step
    wandb.define_metric("train/*", step_metric="train_step")
    # everything that starts with eval/ is tied to eval_step
    wandb.define_metric("eval/*", step_metric="eval_step")

def do_rollouts(prompts, vllm_model, train_sampling_params, n_rollouts = 1):
    prompts = [[prompt] * n_rollouts for prompt in prompts]
    responses = vllm_model.generate(prompts, train_sampling_params)
    return [r.outputs[0].text for r in responses], prompts

def train_setup(model_string: str, seed: int = 42, learning_rate: float = 1e-4, vllm_device: str = 'cuda:0', model_device: str = 'cuda:1'):
    # initialize vllm onto 1st GPU
    print("Initializing vllm...")
    vllm_model = init_vllm(model_string, device = vllm_device, seed = seed)

    # load model and tokenizer onto 1st GPU
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_25,
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

def full_train_step(prompts: List[str], answers: List[str], tokenizer: PreTrainedTokenizer, model: PreTrainedModel, gradient_accumulation_steps: int, do_backward: bool = True, device: str = 'cuda:1'):
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
            
            loss, avg_entropy = full_train_step(minibatch_prompts, 
                            minibatch_answers, 
                            tokenizer, model, 
                            gradient_accumulation_steps = 1, 
                            do_backward = False)
            avg_loss += loss.item()
            avg_entropy += avg_entropy
    
    model.train()
    
    # average loss and entropy over minibatches
    avg_loss /= n_minibatches
    avg_entropy /= n_minibatches

    return avg_loss, avg_entropy

def evaluate(prompts: List[str], answers: List[str], tokenizer: PreTrainedTokenizer, vllm_model: LLM, minibatch_size: int, eval_sampling_params: SamplingParams, reward_fn: Callable[[str, str], dict[str, float]]):
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

def train_run(model_string: str, 
              n_unique: int = None, 
              n_epochs: int = 10, 
              minibatch_size = 32, 
              batch_size: int = 128, 
              learning_rate: float = 1e-4,
              log_every_n: int = 100,
              eval_every_n: int = 100,
              eval_sampling_params: SamplingParams = None,
              seed: int = 42):
    vllm_device = 'cuda:0'
    model_device = 'cuda:1'

    # do training setup
    vllm_model, model, tokenizer, optimizer = train_setup(model_string, seed, learning_rate, vllm_device, model_device)

    # load training and validation sets
    print("Loading training and validation sets...")
    train_prompts, train_responses, train_ground_truths = load_SFT(MATH_TRAIN_PATH)

    # sample subset of unique prompts to use
    if n_unique: # use random sample of unique prompts
        train_idxs = random.sample(range(len(train_prompts)), n_unique)
        train_prompts = [train_prompts[i] for i in train_idxs]
        train_responses = [train_responses[i] for i in train_idxs]
        train_ground_truths = [train_ground_truths[i] for i in train_idxs]
    
    val_questions, val_answers = baseline.load_MATH(MATH_VAL_PATH)
    val_prompts = baseline.make_prompts(val_questions)

    # run SFT training
    print("Running SFT training...")
    gradient_accumulation_steps = batch_size // minibatch_size
    n_minibatches = len(train_prompts) // minibatch_size
    train_step = 0
    eval_step = 0
    
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
            if (train_step + 1) % gradient_accumulation_steps == 0:
                # do gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # perform gradient descent once accumulated
                optimizer.step()
                optimizer.zero_grad()
            
            # logging
            if (train_step + 1) % log_every_n == 0:
                wandb.log({
                    "train/loss": loss_val,
                    "train/avg_entropy": avg_entropy,
                    "train_step": train_step,
                })

            if (train_step + 1) % eval_every_n == 0:
                # select random batch eval prompts/answers
                val_batch_indices = random.sample(range(len(val_prompts)), batch_size * 8)
                val_batch_prompts = [val_prompts[i] for i in val_batch_indices]
                val_batch_answers = [val_answers[i] for i in val_batch_indices]

                correct_fraction, format_only_fraction = evaluate(prompts = val_batch_prompts, 
                                                                   answers = val_batch_answers, 
                                                                   tokenizer = tokenizer, 
                                                                   vllm_model = vllm_model, 
                                                                   minibatch_size = minibatch_size, 
                                                                   eval_sampling_params = eval_sampling_params, 
                                                                   reward_fn = baseline.r1_zero_reward_fn)
                wandb.log({
                    "eval/correct": correct_fraction,
                    "eval/format_only": format_only_fraction,
                    "eval_step": train_step,
                })

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
                
                eval_step += 1

            train_step += 1
        
        # load policy into vllm
        load_policy_into_vllm_instance(model, vllm_model)

    print("Training complete!")
    # do full val set evaluation
    print("Evaluating on full validation set...")
    baseline.evaluate_vllm(vllm_model, 
                           baseline.r1_zero_reward_fn, 
                           val_prompts, 
                           val_answers, 
                           eval_sampling_params, 
                           save_path = f'sft_results/sft_{n_unique}_{batch_size}_{minibatch_size}_{learning_rate}.jsonl')
    
if __name__ == "__main__":
    eval_sampling_params = SamplingParams(
        temperature = 1.0, 
        top_p = 1.0, 
        max_tokens = 1024, 
        stop = ["</answer>"], 
        include_stop_str_in_output = True,
    )
    config = {
        "model": QWEN_25,
        "n_unique": None,
        "n_epochs": 10,
        "minibatch_size": 16, # 32 causes OOM
        "train_batch_size": 128,
        "val_batch_size": 128,
        "learning_rate": 1e-4,
    }
    wandb.init(project = "cs336-alignment", name = f"sft_{config['n_unique']}_{config['minibatch_size']}_{config['train_batch_size']}_{config['learning_rate']}", config = config)
    wandb_setup()
    train_run(model_string = config['model'],
              n_unique = config['n_unique'],
              n_epochs = config['n_epochs'],
              minibatch_size = config['minibatch_size'],
              batch_size = config['train_batch_size'],
              learning_rate = config['learning_rate'],
              log_every_n = 10,
              eval_every_n = 20,
              eval_sampling_params = eval_sampling_params)