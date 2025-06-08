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
    
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"] = "grpo_clip"
    advantage_eps: float = 1e-6
    group_size: int = 8
    epochs_per_rollout_batch: int = 1 # 1 = on-policy
    use_std_normalization: bool = True # normalize by std of rewards
    constant_normalize_factor: float = None # if not None, will not divide by sequence length
    cliprange: float = 0.2
    
    gpu_memory_utilization: float = 0.2 # limit vllm memory usage
    vllm_device: str = "cuda"
    model_device: str = "cuda"
    seed: int = 42
    model_string: str = QWEN_25
    eval_interval: int = 5

    def __init__(self, config: dict | None = None):
        if config is not None:
            self.init_from_dict(config)
        else:
            print("No config provided, using default values")

    def init_from_dict(self, config: dict):
        for key, value in config.items():
            if key in self.__dict__:
                setattr(self, key, value) # override default values
            else:
                raise ValueError(f"Invalid config key: {key}")
        return self

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
        self.train_prompts, self.train_as, self.val_prompts, self.val_as = self.load_prompts()
        self.train_idx = 0
        self.val_idx = 0
    
    def load_prompts(self, prompt_path: str = PROMPT_PATH, train_path: str = MATH_TRAIN_PATH, val_path: str = MATH_VAL_PATH):
        """
        Load and format prompts and answers from train/val sets
        """
        train_qs, train_as = self.load_function(train_path)
        val_qs, val_as = self.load_function(val_path)

        train_prompts = self.make_prompts_function(train_qs, prompt_path = prompt_path)
        val_prompts = self.make_prompts_function(val_qs, prompt_path = prompt_path)

        # log number of prompts
        print(f"Loaded {len(train_prompts)} train questions and {len(val_prompts)} val questions")

        # shuffle prompts and answers
        train_shuffle_idx = torch.randperm(len(train_prompts))
        val_shuffle_idx = torch.randperm(len(val_prompts))
        train_prompts = train_prompts[train_shuffle_idx]
        train_as = train_as[train_shuffle_idx]
        val_prompts = val_prompts[val_shuffle_idx]
        val_as = val_as[val_shuffle_idx]

        return train_prompts, train_as, val_prompts, val_as
    
    def sample(self, split: Literal["train", "val"], n_samples: int):
        if split == "train":
            if self.train_idx + n_samples > len(self.train_prompts):
                # just reshuffle and reset idx
                self.train_idx = 0
                self.train_prompts, self.train_as = self.load_prompts()
            
            sample_prompts = self.train_prompts[self.train_idx:self.train_idx + n_samples]
            sample_as = self.train_as[self.train_idx:self.train_idx + n_samples]
            self.train_idx += n_samples
        else:
            if self.val_idx + n_samples > len(self.val_prompts):
                # just reshuffle and reset idx
                self.val_idx = 0
                self.val_prompts, self.val_as = self.load_prompts()
            
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
                 sampling_params: SamplingParams):
        
        self.model = model
        self.tokenizer = tokenizer
        self.vllm_model = vllm_model
        self.optimizer = optimizer
        self.config = config
        self.buffer = []
        self.sampling_params = sampling_params
        
        # check if we can evenly divide train batch size into steps
        assert self.config.train_batch_size % self.config.gradient_accumulation_steps == 0, (
            f"train_batch_size must be divisible by gradient_accumulation_steps, but got {self.config.train_batch_size} and {self.config.gradient_accumulation_steps}"
        )
        self.config.micro_batch_size = self.config.train_batch_size // self.config.gradient_accumulation_steps
        # check if we can evenly divide rollout batch into groups
        assert self.config.rollout_batch_size % self.config.group_size == 0, (
            f"rollout_batch_size must be divisible by group_size, but got {self.config.rollout_batch_size} and {self.config.group_size}"
        )
        # check if train batch size is greater than or equal to group size
        assert self.config.train_batch_size >= self.config.group_size, (
            f"train_batch_size must be greater than or equal to group_size, but got {self.config.train_batch_size} and {self.config.group_size}"
        )
    
    def rollout(self, prompts: List[str], answers: List[str], group_size: int):
        repeated_prompts = [[prompt] * group_size for prompt in prompts] # shape: (n_prompts, group_size)
        repeated_answers = [[answer] * group_size for answer in answers] # shape: (n_prompts, group_size)
        # flatten repeated prompts and answers
        repeated_prompts = [item for sublist in repeated_prompts for item in sublist]
        repeated_answers = [item for sublist in repeated_answers for item in sublist]

        outputs = self.vllm_model.generate(repeated_prompts, sampling_params = self.sampling_params)
        responses = [output.outputs[0].text for output in outputs]
        return responses, repeated_prompts, repeated_answers

    def full_step(self, dataset: GRPODataset):
        full_metadata = {}

        n_prompts_per_rollout_batch = self.config.rollout_batch_size // self.config.group_size
        n_microbatches_per_rollout_batch = self.config.rollout_batch_size // self.config.micro_batch_size
        sample_prompts, sample_as = dataset.sample("train", n_prompts_per_rollout_batch)

        # collect all rollouts with vllm
        responses, repeated_prompts, repeated_answers = self.rollout(prompts = sample_prompts, 
                                                                     answers = sample_as, 
                                                                     group_size = self.config.group_size)
        
        # tokenized responses padded; response mask shows which tokens are actually part of the response
        empty_prompts = [""] * len(responses)
        tokenized_responses = utils.tokenize_prompt_and_output(empty_prompts, responses, tokenizer = self.tokenizer, device = "cuda")
        response_mask = tokenized_responses["response_mask"]
        
        # compute raw rewards and get grpo advantage, collect metadata
        raw_rewards, advantages, metadata = rl_utils.compute_group_normalized_rewards(
            reward_fn = baseline.r1_zero_reward_fn,
            rollout_responses = responses,
            repeated_ground_truths = repeated_answers,
            group_size = self.config.group_size,
            advantage_eps = self.config.advantage_eps,
            normalize_by_std = self.config.use_std_normalization
        )

        full_metadata.update(metadata) # add reward metadata
        
        log_probs, entropy = utils.get_response_log_probs(model = self.model, 
                                                          input_ids = tokenized_responses["input_ids"], 
                                                          labels = tokenized_responses["labels"], 
                                                          return_token_entropy = True)
        
        # record entropy
        full_metadata["entropy"] = torch.mean(entropy).detach().item()

        pg_loss = 0

        # do minibatch train steps
        for i in range(n_microbatches_per_rollout_batch):
            start_idx = i * self.config.micro_batch_size
            end_idx = start_idx + self.config.micro_batch_size

            microbatch_loss, microbatch_metadata = rl_utils.grpo_microbatch_train_step(
                policy_log_probs = log_probs[start_idx:end_idx],
                response_mask = response_mask[start_idx:end_idx],
                gradient_accumulation_steps = self.config.gradient_accumulation_steps,
                loss_type = self.config.loss_type,
                raw_rewards = raw_rewards[start_idx:end_idx],
                advantages = advantages[start_idx:end_idx],
                old_log_probs = log_probs[start_idx:end_idx], # no off-policy support yet
                cliprange = self.config.cliprange,
                constant_normalize_factor = self.config.constant_normalize_factor,
            )

            pg_loss += torch.mean(microbatch_loss).detach()
             
            # figure out what to do with microbatch metadata
        
        full_metadata["pg_loss"] = pg_loss.item()
        
        # grad clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = 1.0)

        # update optimizer
        self.optimizer.step()
        self.optimizer.zero_grad()

        return full_metadata
    
    def eval(self, dataset: GRPODataset, save_path: str):
        eval_prompts, eval_as = dataset.sample("val", self.config.eval_batch_size)
        results = baseline.evaluate_vllm(vllm_model = self.vllm_model, 
                                         reward_fn = baseline.r1_zero_reward_fn, 
                                         prompts = eval_prompts, 
                                         answers = eval_as, 
                                         eval_sampling_params = self.sampling_params,
                                         save_path = save_path)
        
        correct_fraction, format_only_fraction, wrong_fraction = baseline.compute_metrics(results, printout = False)
        return correct_fraction, format_only_fraction
        
    
    def train(self, dataset: GRPODataset):
        eval_save_path = f"./grpo_results/grpo_{self.config.rollout_batch_size}_{self.config.train_batch_size}_{self.config.learning_rate}.json"
        for i in range(self.config.n_grpo_steps):
            full_metadata = self.full_step(dataset = dataset)

            # log to wandb
            wandb.log({
                "pg_loss": full_metadata["pg_loss"],
                "entropy": full_metadata["entropy"],
                
                "train_step": i
            })

            if i % self.config.eval_interval == 0:
                correct_fraction, format_only_fraction = self.eval(dataset = dataset, save_path = eval_save_path)
                # log to wandb
                wandb.log({
                    "correct_fraction": correct_fraction,
                    "format_only_fraction": format_only_fraction,
                    "train_step": i
                })

def run_math_grpo(config: GRPOConfig, sampling_params: SamplingParams):
    wandb.init(project = "cs336-alignment-grpo", 
               name = f"grpo_{config.model_string}_{config.n_grpo_steps}_{config.rollout_batch_size}_{config.train_batch_size}_{config.learning_rate}", 
               config = config)
    
    model, tokenizer, vllm_model = sft.train_setup(model_string = config.model_string, 
                                                             seed = config.seed, 
                                                             vllm_device = config.vllm_device, 
                                                             model_device = config.model_device,
                                                             gpu_memory_utilization = config.gpu_memory_utilization)
    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr = config.learning_rate, 
                                  weight_decay = config.weight_decay, 
                                  betas = config.betas)
    
    dataset = GRPODataset(train_path = MATH_TRAIN_PATH, 
                          val_path = MATH_VAL_PATH, 
                          prompt_path = PROMPT_PATH, 
                          load_function = baseline.load_MATH, 
                          make_prompts_function = baseline.make_prompts)
    
    trainer = GRPOTrainer(model = model, 
                          tokenizer = tokenizer, 
                          vllm_model = vllm_model, 
                          optimizer = optimizer, 
                          config = config, 
                          sampling_params = sampling_params)
    
    trainer.train(dataset = dataset)

if __name__ == "__main__":
    sampling_temperature: float = 1.0
    sampling_min_tokens: int = 4 # disallow empty string responses
    sampling_max_tokens: int = 1024
    gpu_memory_utilization: float = 0.2 # limit vllm memory usage