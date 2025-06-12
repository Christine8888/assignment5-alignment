import cs336_alignment.rl_utils as rl_utils
from vllm import LLM
from transformers import PreTrainedModel, PreTrainedTokenizer
from cs336_alignment.info import *
import torch
from typing import Literal, Callable, List
import cs336_alignment.sft as sft
import cs336_alignment.baseline as baseline
from vllm import SamplingParams
import cs336_alignment.utils as utils
import wandb
import cs336_alignment.iter as iter
import numpy as np

class GRPOConfig():
    n_grpo_steps: int = 200
    rollout_batch_size: int = 256
    train_batch_size: int = 256
    eval_batch_size: int = 1024
    micro_batch_size: int = 2
    gradient_accumulation_steps: int = 128 # microbatch size is 2, will fit on H100
    
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.95)
    
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"] = "reinforce_with_baseline"
    advantage_eps: float = 1e-6
    group_size: int = 8
    epochs_per_rollout_batch: int = 1 # 1 = on-policy
    use_std_normalization: bool = True # normalize by std of rewards
    length_normalize: bool = True # divide by sequence length
    cliprange: float = 0.2
    
    gpu_memory_utilization: float = 0.3 # limit vllm memory usage
    vllm_device: str = "cuda"
    model_device: str = "cuda"
    seed: int = 42
    model_string: str = QWEN_25
    eval_interval: int = 5
    run_name: str = "default"

    def __init__(self, config: dict | None = None):
        if config is not None:
            self.init_from_dict(config)
        else:
            print("No config provided, using default values")

    def init_from_dict(self, config: dict):
        for key, value in config.items():
            setattr(self, key, value) # override default values
            
        return self
    
    def __str__(self):
        return f"GRPOConfig(n_grpo_steps={self.n_grpo_steps}, rollout_batch_size={self.rollout_batch_size}, train_batch_size={self.train_batch_size}, eval_batch_size={self.eval_batch_size}, micro_batch_size={self.micro_batch_size}, gradient_accumulation_steps={self.gradient_accumulation_steps}, learning_rate={self.learning_rate}, weight_decay={self.weight_decay}, betas={self.betas}, loss_type={self.loss_type}, advantage_eps={self.advantage_eps}, group_size={self.group_size}, epochs_per_rollout_batch={self.epochs_per_rollout_batch}, use_std_normalization={self.use_std_normalization}, length_normalize={self.length_normalize}, cliprange={self.cliprange}, gpu_memory_utilization={self.gpu_memory_utilization}, vllm_device={self.vllm_device}, model_device={self.model_device}, seed={self.seed}, model_string={self.model_string}, eval_interval={self.eval_interval})"

class GRPODataset():
    def __init__(self,
                 train_path: str,
                 val_path: str,
                 prompt_path: str,
                 load_function: Callable = baseline.load_MATH,
                 make_prompts_function: Callable = baseline.make_prompts):
        self.train_path = train_path
        self.val_path = val_path
        self.prompt_path = prompt_path
        self.load_function = load_function
        self.make_prompts_function = make_prompts_function
        self.train_prompts, self.train_as = self.load_prompts(file_path = train_path, prompt_path = prompt_path)
        self.val_prompts, self.val_as = self.load_prompts(file_path = val_path, prompt_path = prompt_path)
        self.train_idx = 0
        self.val_idx = 0
    
    def load_prompts(self, prompt_path: str = PROMPT_PATH, file_path: str = MATH_TRAIN_PATH):
        """
        Load and format prompts and answers from train/val sets
        """
        questions, answers = self.load_function(file_path)

        prompts = self.make_prompts_function(questions, prompt_path = prompt_path)

        # log number of prompts
        print(f"Loaded {len(prompts)} prompts")

        # shuffle prompts and answers
        shuffle_idx = torch.randperm(len(prompts))
        prompts = [prompts[i] for i in shuffle_idx]
        answers = [answers[i] for i in shuffle_idx]

        return prompts, answers
    
    def sample(self, split: Literal["train", "val"], n_samples: int):
        if split == "train":
            if self.train_idx + n_samples > len(self.train_prompts):
                # just reshuffle and reset idx
                self.train_idx = 0
                self.train_prompts, self.train_as = self.load_prompts(file_path = self.train_path, prompt_path = self.prompt_path)
            
            sample_prompts = self.train_prompts[self.train_idx:self.train_idx + n_samples]
            sample_as = self.train_as[self.train_idx:self.train_idx + n_samples]
            self.train_idx += n_samples
        else:
            if self.val_idx + n_samples > len(self.val_prompts):
                # just reshuffle and reset idx
                self.val_idx = 0
                self.val_prompts, self.val_as = self.load_prompts(file_path = self.val_path, prompt_path = self.prompt_path)
            
            sample_prompts = self.val_prompts[self.val_idx:self.val_idx + n_samples]
            sample_as = self.val_as[self.val_idx:self.val_idx + n_samples]
            self.val_idx += n_samples

        return sample_prompts, sample_as

class GRPOTrainer():
    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 vllm_model: LLM,
                 optimizer: torch.optim.Optimizer,
                 config: GRPOConfig,
                 sampling_params: SamplingParams,
                 reward_fn: Callable = baseline.r1_zero_reward_fn,
                 verbose: bool = True):
        
        self.model = model
        self.tokenizer = tokenizer
        self.vllm_model = vllm_model
        self.optimizer = optimizer
        self.config = config
        self.buffer = []
        self.sampling_params = sampling_params
        self.reward_fn = reward_fn
        
        # override gradient accumulation steps to batch_size / microbatch_size
        self.config.gradient_accumulation_steps = self.config.train_batch_size // self.config.micro_batch_size
        print(f"Updating gradient accumulation steps to {self.config.gradient_accumulation_steps}")

        # check if we can evenly divide train batch size into steps
        assert self.config.train_batch_size % self.config.gradient_accumulation_steps == 0, (
            f"train_batch_size must be divisible by gradient_accumulation_steps, but got {self.config.train_batch_size} and {self.config.gradient_accumulation_steps}"
        )

        # check if we can evenly divide rollout batch into groups
        assert self.config.rollout_batch_size % self.config.group_size == 0, (
            f"rollout_batch_size must be divisible by group_size, but got {self.config.rollout_batch_size} and {self.config.group_size}"
        )
        # check if train batch size is greater than or equal to group size
        assert self.config.train_batch_size >= self.config.group_size, (
            f"train_batch_size must be greater than or equal to group_size, but got {self.config.train_batch_size} and {self.config.group_size}"
        )
        
        # check if we are using clipping if going off-policy
        assert self.config.loss_type == "grpo_clip" or self.config.loss_type == "grpo_no_clip" if self.config.epochs_per_rollout_batch > 1 else True, (
            "grpo_clip loss type must be used if epochs_per_rollout_batch > 1"
        )

        if self.config.rollout_batch_size % self.config.train_batch_size != 0:
            print("WARNING: rollout_batch_size is not divisible by train_batch_size")

        self.verbose = verbose
        if self.verbose:
            print(f"GRPOTrainer initialized with config: {self.config}")
    
    def rollout(self, prompts: List[str], answers: List[str], group_size: int) -> tuple[List[str], List[str], List[str]]:
        """
        Run group_size VLLM generations for each prompt, answer pair

        Returns:
            responses: List[str]
            repeated_prompts: List[str]
            repeated_answers: List[str]
        """
        repeated_prompts = [[prompt] * group_size for prompt in prompts] # shape: (n_prompts, group_size)
        repeated_answers = [[answer] * group_size for answer in answers] # shape: (n_prompts, group_size)
        # flatten repeated prompts and answers
        repeated_prompts = [item for sublist in repeated_prompts for item in sublist]
        repeated_answers = [item for sublist in repeated_answers for item in sublist]

        outputs = self.vllm_model.generate(repeated_prompts, sampling_params = self.sampling_params)
        responses = [output.outputs[0].text for output in outputs]
        return responses, repeated_prompts, repeated_answers

    def full_step(self, grpo_step: int, train_step: int, dataset: GRPODataset) -> tuple[int, dict]:
        """
        Run a full GRPO step, including generating rollouts, computing rewards, and training

        Returns:
            train_step: int
            grpo_metadata: dict
        """

        n_prompts_per_rollout_batch = self.config.rollout_batch_size // self.config.group_size
        n_microbatches_per_rollout_batch = self.config.rollout_batch_size // self.config.micro_batch_size

        # sample batch of prompts and answers
        sample_prompts, sample_as = dataset.sample("train", n_prompts_per_rollout_batch)
        if self.verbose: print(f"Sampled {n_prompts_per_rollout_batch} prompts and answers")

        # collect all rollouts with vllm
        responses, repeated_prompts, repeated_answers = self.rollout(prompts = sample_prompts, 
                                                                     answers = sample_as, 
                                                                     group_size = self.config.group_size)

        # tokenized responses padded; response mask shows which tokens are actually part of the response
        tokenized_responses = utils.tokenize_prompt_and_output(repeated_prompts, 
                                                               responses, 
                                                               tokenizer = self.tokenizer, 
                                                               device = self.config.model_device)
        response_mask = tokenized_responses["response_mask"]
        response_lengths = torch.sum(response_mask, dim = -1, dtype = torch.float32) # sum over sequence length
        input_ids = tokenized_responses["input_ids"].long()
        labels = tokenized_responses["labels"].long()
        
        # compute raw rewards and get grpo advantage, collect metadata
        advantages, raw_rewards, grpo_metadata = rl_utils.compute_group_normalized_rewards(
            reward_fn = self.reward_fn,
            rollout_responses = responses,
            repeated_ground_truths = repeated_answers,
            group_size = self.config.group_size,
            advantage_eps = self.config.advantage_eps,
            normalize_by_std = self.config.use_std_normalization
        )

        grpo_metadata["average_response_length"] = torch.mean(response_lengths).detach().item()
        raw_rewards = raw_rewards.to(self.config.model_device).unsqueeze(-1) # shape: (rollout_batch_size, 1)
        advantages = advantages.to(self.config.model_device).unsqueeze(-1) # shape: (rollout_batch_size, 1)

        if self.verbose: print(f"Reward statistics at GRPO step {grpo_step}: {grpo_metadata}")

        # get old_log_probs FIRST
        old_log_probs = torch.zeros_like(input_ids, dtype = torch.float32).detach().to(self.config.model_device)

        if self.config.loss_type == "grpo_clip":
            with torch.inference_mode():
                for i in range(n_microbatches_per_rollout_batch):
                    start_idx = i * self.config.micro_batch_size
                    end_idx = start_idx + self.config.micro_batch_size

                    return_dict = utils.get_response_log_probs(model = self.model, 
                                                                input_ids = input_ids[start_idx:end_idx], 
                                                                labels = labels[start_idx:end_idx], 
                                                                return_token_entropy = True)
                    old_log_probs[start_idx:end_idx] = return_dict["log_probs"]

        if self.config.epochs_per_rollout_batch > 1:
            if self.verbose: print(f"GOING OFF-POLICY: running {self.config.epochs_per_rollout_batch} epochs per rollout batch")
        
        if self.verbose:
            n_train_steps_per_rollout_batch = self.config.epochs_per_rollout_batch * (self.config.rollout_batch_size / self.config.train_batch_size)
            print(f"Performing {n_train_steps_per_rollout_batch} total training steps")

        # do multiple epochs per rollout batch if going off-policy
        for epoch in range(self.config.epochs_per_rollout_batch):
            if self.verbose: print(f"Epoch {epoch} of {self.config.epochs_per_rollout_batch}")

            # record metadata for each train step
            train_metadata = {"entropy": 0.0, "pg_loss": 0.0, "grad_norm": 0.0, "clip_fraction": 0.0}

            microbatch_step = 0 # keep count across epochs

            for i in range(n_microbatches_per_rollout_batch): 
                start_idx = i * self.config.micro_batch_size
                end_idx = start_idx + self.config.micro_batch_size

                return_dict = utils.get_response_log_probs(model = self.model, 
                                                                input_ids = input_ids[start_idx:end_idx], 
                                                                labels = labels[start_idx:end_idx], 
                                                                return_token_entropy = True)
                log_probs = return_dict["log_probs"]
                entropy = return_dict["token_entropy"] # on model device by default
                entropy = entropy * response_mask[start_idx:end_idx] # mask out padding tokens
                
                # divide by sequence length to get per-token entropy by sequence
                entropy = torch.sum(entropy, dim = -1) / response_lengths[start_idx:end_idx]
                # get per-token entropy, normalized by number of gradient accumulation steps
                train_metadata["entropy"] += torch.mean(entropy).detach().item() / self.config.gradient_accumulation_steps

                # run grpo microbatch train step
                microbatch_loss, microbatch_metadata = rl_utils.grpo_microbatch_train_step(
                    policy_log_probs = log_probs,
                    response_mask = response_mask[start_idx:end_idx],
                    gradient_accumulation_steps = self.config.gradient_accumulation_steps,
                    loss_type = self.config.loss_type,
                    raw_rewards = raw_rewards[start_idx:end_idx],
                    advantages = advantages[start_idx:end_idx],
                    old_log_probs = old_log_probs[start_idx:end_idx], # no differentiation w.r.t. old log probs
                    cliprange = self.config.cliprange,
                    length_normalize = self.config.length_normalize,
                )

                train_metadata["pg_loss"] += microbatch_loss.item() # already detached, normalized by gradient accumulation steps
                
                # record clip fraction
                if "clip_fraction" in microbatch_metadata:
                    train_metadata["clip_fraction"] += microbatch_metadata["clip_fraction"] / n_microbatches_per_rollout_batch
                else: train_metadata["clip_fraction"] = 0.0 # duh, no clipping

                # DO TRAINING UPDATE; we have checked that we can evenly divide train batch size into microbatches
                if (microbatch_step + 1) % self.config.gradient_accumulation_steps == 0:
                    # grad clipping
                    train_metadata["grad_norm"] = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = 1.0)

                    # update optimizer
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    # log train step metadata to wandb
                    wandb.log({
                        "pg_loss": train_metadata["pg_loss"],
                        "grad_norm": train_metadata["grad_norm"],
                        "entropy": train_metadata["entropy"],
                        "clip_fraction": train_metadata["clip_fraction"],
                        "grpo_step": grpo_step,
                        "train_step": train_step
                    })

                    # increment train step
                    print(f"Incrementing train step to {train_step}")
                    train_step += 1
                
                microbatch_step += 1
        
        return train_step, grpo_metadata
    
    def eval(self, dataset: GRPODataset, save_path: str, eval_batch_size: int):
        
        eval_prompts, eval_as = dataset.sample("val", eval_batch_size)
        results = baseline.evaluate_vllm(vllm_model = self.vllm_model, 
                                         reward_fn = self.reward_fn, 
                                         prompts = eval_prompts, 
                                         answers = eval_as, 
                                         eval_sampling_params = self.sampling_params,
                                         save_path = save_path)
        
        correct_fraction, format_fraction, wrong_fraction = baseline.compute_metrics(results, printout = False)

        if self.verbose:
            print(f"Eval results: correct_fraction = {correct_fraction}, format_fraction = {format_fraction}")
        
        return correct_fraction, format_fraction
        
    
    def train(self, dataset: GRPODataset):
        eval_save_path = f"./grpo_results/grpo_{self.config.rollout_batch_size}_{self.config.train_batch_size}_{self.config.learning_rate}.json"
        train_step = 0
        
        for i in range(self.config.n_grpo_steps):
            if self.verbose: print(f"GRPO step {i} of {self.config.n_grpo_steps}")
            # load weights into vllm model -- OOPS
            sft.load_policy_into_vllm_instance(self.model, self.vllm_model)

            train_step, grpo_metadata = self.full_step(dataset = dataset, grpo_step = i, train_step = train_step)

            # log grpo step metadata to wandb
            wandb.log({
                "grpo_step": i,
                "average_response_length": grpo_metadata["average_response_length"],
                "train_step": train_step,
                "per_group_mean_reward": grpo_metadata["per_group_mean_reward"],
                "per_group_std_reward": grpo_metadata["per_group_std_reward"],
                "per_group_min_reward": grpo_metadata["per_group_min_reward"],
                "per_group_max_reward": grpo_metadata["per_group_max_reward"],
                "mean_advantage": grpo_metadata["mean_advantage"],
                "std_advantage": grpo_metadata["std_advantage"],
                "min_advantage": grpo_metadata["min_advantage"],
                "max_advantage": grpo_metadata["max_advantage"]
            })

            if i % self.config.eval_interval == 0:
                if self.verbose: print(f"Evaluating at GRPO step {i}")
                correct_fraction, format_only_fraction = self.eval(dataset = dataset, 
                                                                   save_path = eval_save_path,
                                                                   eval_batch_size = self.config.eval_batch_size)
                # log to wandb
                wandb.log({
                    "eval_correct_fraction": correct_fraction,
                    "eval_format": format_only_fraction,
                    "grpo_step": i
                })
        
        # do final evaluation
        if self.verbose: print("Doing final evaluation")
        correct_fraction, format_only_fraction = self.eval(dataset = dataset, 
                                                           save_path = eval_save_path,
                                                           eval_batch_size = len(dataset.val_prompts))
        wandb.log({
            "eval_correct_fraction": correct_fraction,
            "eval_format": format_only_fraction,
            "grpo_step": self.config.n_grpo_steps
        })

def run_math_grpo(config: GRPOConfig, 
                  sampling_params: SamplingParams, 
                  verbose: bool = True,
                  prompt_path: str = PROMPT_PATH,
                  train_path: str = MATH_TRAIN_PATH,
                  val_path: str = MATH_VAL_PATH,
                  reward_fn: Callable = baseline.r1_zero_reward_fn,
                  iter_warmup: bool = False):
    wandb.init(project = "cs336-alignment-grpo", 
               name = f"grpo_{config.run_name}_{config.model_string}_{config.n_grpo_steps}_{config.rollout_batch_size}_{config.train_batch_size}_{config.learning_rate}", 
               config = config)
    
    print("Setting up model, tokenizer, and vllm model")
    vllm_model, model, tokenizer = sft.train_setup(model_string = config.model_string, 
                                                             seed = config.seed, 
                                                             vllm_device = config.vllm_device, 
                                                             model_device = config.model_device,
                                                             gpu_memory_utilization = config.gpu_memory_utilization)
    
    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr = config.learning_rate, 
                                  weight_decay = config.weight_decay, 
                                  betas = config.betas)
    
    if iter_warmup:
        iter_config = {
            'model': QWEN_25,
            'n_rollouts': 12,
            'n_ei_steps': 5,
            'n_epochs': 1,
            'n_unique': None, # always train on full SFT dataset
            'minibatch_size': 4,
            'iter_batch_size': 512,
            'train_batch_size': 128,
            'learning_rate': 1e-4,
            'log_every_n': 10,
            'eval_every_n': 100,
            'val_path': MATH_VAL_PATH,
            'train_path': MATH_TRAIN_PATH,
            'seed': 42,
            'start_train_step': 0,
        }

        # set learning rate for expert iteration to 1e-4
        for param_group in optimizer.param_groups:
            param_group['lr'] = iter_config['learning_rate']
        
        print("Running expert iteration for warmup")
        val_questions, val_answers = baseline.load_MATH(val_path)
        val_prompts = baseline.make_prompts(val_questions, prompt_path = prompt_path)
        train_questions, train_answers = baseline.load_MATH(train_path)
        train_prompts = baseline.make_prompts(train_questions, prompt_path = prompt_path)
        
        train_step = 0
        for i in range(1): # 1 iteration of expert iteration
            # sample config['iter_batch_size'] items by index
            sampled_indices = np.random.choice(len(train_prompts), iter_config['iter_batch_size'], replace=False)
            print(f'Sampled {len(sampled_indices)} examples')
            step_prompts = [train_prompts[i] for i in sampled_indices]
            step_answers = [train_answers[i] for i in sampled_indices]

            train_step = iter.iterate(config = iter_config, 
                                train_prompts = step_prompts, 
                                train_answers = step_answers, 
                                val_prompts = val_prompts, 
                                val_answers = val_answers, 
                                vllm_model = vllm_model, 
                                model = model, 
                                model_device = config.model_device, 
                                optimizer = optimizer, 
                                tokenizer = tokenizer, 
                                sampling_params = sampling_params, 
                                start_train_step = train_step, 
                                end_eval = False, # only do full evaluation on last step
                                iter = i)
            
            # checkpoint model

            # load policy into vllm
            sft.load_policy_into_vllm_instance(model, vllm_model)
    
    if iter_warmup:
        model.save_pretrained(f"./grpo_results/grpo_iter_warmup_model_{i}")
    # reset optimizer
    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr = config.learning_rate, 
                                  weight_decay = config.weight_decay, 
                                  betas = config.betas)
    
    dataset = GRPODataset(train_path = train_path, 
                          val_path = val_path, 
                          prompt_path = prompt_path, 
                          load_function = baseline.load_MATH, 
                          make_prompts_function = baseline.make_prompts)
    
    trainer = GRPOTrainer(model = model, 
                          tokenizer = tokenizer, 
                          vllm_model = vllm_model, 
                          optimizer = optimizer, 
                          config = config, 
                          sampling_params = sampling_params,
                          verbose = verbose,
                          reward_fn = reward_fn)
    
    print("Training with GRPO")
    trainer.train(dataset = dataset)

    print("GRPO training complete")

if __name__ == "__main__":
    sampling_params = EVAL_SAMPLING_PARAMS
    config = GRPOConfig()
    run_math_grpo(config = config, sampling_params = sampling_params, verbose = True)