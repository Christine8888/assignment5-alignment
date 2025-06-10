import submitit
from cs336_alignment.grpo import run_math_grpo, GRPOConfig
from cs336_alignment.info import EVAL_SAMPLING_PARAMS
import argparse
from cs336_alignment.info import PROMPT_PATH
from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn
from typing import Callable

def launch_from_config(config: dict, prompt_path: str = PROMPT_PATH, reward_fn: Callable = r1_zero_reward_fn, iter_warmup: bool = False):
    """Launch GRPO training."""
    grpo_config = GRPOConfig(config)
    run_math_grpo(grpo_config, EVAL_SAMPLING_PARAMS, prompt_path = prompt_path, reward_fn = reward_fn, iter_warmup = iter_warmup)

def ablate_std_normalization(default_config: dict = None):
    config = default_config.copy()
    config['use_std_normalization'] = False
    launch_from_config(config)

def sweep_lr(task_id: int, learning_rates: list[float], default_config: dict = None):
    if default_config is None:
        default_config = {}
    
    config = default_config.copy()
    config['learning_rate'] = learning_rates[task_id]
    launch_from_config(config)

def sweep_off_policy(task_id, configs: list[dict]):
    launch_from_config(configs[task_id])

def submit_experiment(experiment_name: str, executor: submitit.AutoExecutor):
    if experiment_name == "grpo_sweep_lr":
        default_config = {}
        learning_rates = [8e-6, 1e-5, 2e-5, 3e-5] 
        total_configs = len(learning_rates)
        
        jobs = executor.map_array(
            lambda task_id: sweep_lr(task_id, learning_rates, default_config),
            range(total_configs)
        )
        
        print(f"Submitted learning rate sweep with {total_configs} configurations")
        print(f"Learning rates: {learning_rates}")
        
    elif experiment_name == "grpo_baseline":
        default_config = {"learning_rate": 1.5e-5, 
                          "run_name": "baselining",
                          "loss_type": "default"}
        jobs = executor.submit(launch_from_config, default_config)
        
        print(f"Submitted baseline experiment")
    
    elif experiment_name == "grpo_length_normalization":
        default_config = {"learning_rate": 1.5e-5, 
                          "run_name": "no_length_normalization",
                          "length_normalize": False,
                          "n_grpo_steps": 50}
        jobs = executor.submit(launch_from_config, default_config)
        
        print(f"Submitted length normalization experiment")
    
    elif experiment_name == "grpo_std_normalization":
        default_config = {"learning_rate": 1.5e-5, 
                          "run_name": "no_std_normalization",
                          "use_std_normalization": False,
                          "length_normalize": False, # this was more stable
                          "n_grpo_steps": 50}
        jobs = executor.submit(ablate_std_normalization, default_config)
        
        print(f"Submitted standard deviation normalization experiment")
        
    elif experiment_name == "grpo_off_policy_small":
        lrs = [2e-5]
        train_batch_sizes = [64, 128, 512]  
        epochs_per_rollout_batch = [2, 4, 8] 

        default_config = {"run_name": "off_policy",
                          "n_grpo_steps": 50,
                          "loss_type": "grpo_clip",
                          "micro_batch_size": 2,
                          "length_normalize": False,
                          "use_std_normalization": False}

        configs = []

        for lr in lrs:
            for train_batch_size in train_batch_sizes:
                for eprb in epochs_per_rollout_batch:
                    config = default_config.copy()
                    config['learning_rate'] = lr
                    config['train_batch_size'] = train_batch_size
                    config['epochs_per_rollout_batch'] = eprb
                    configs.append(config)
        
        n_configs = len(configs)
        jobs = executor.map_array(lambda task_id: sweep_off_policy(task_id, configs), range(n_configs))
    
    elif experiment_name == "grpo_no_clip":
        default_config = {"learning_rate": 2e-5, 
                          "run_name": "no_clip",
                          "loss_type": "grpo_no_clip",
                          "n_grpo_steps": 50,
                          "micro_batch_size": 2,
                          "length_normalize": False,
                          "use_std_normalization": False,
                          "train_batch_size": 128,
                          "epochs_per_rollout_batch": 2}
        jobs = executor.submit(launch_from_config, default_config)
    
    elif experiment_name == "prompt_ablation":
        default_config = {"learning_rate": 3e-5, 
                          "run_name": "prompt_ablation",
                          "loss_type": "reinforce_with_baseline",
                          "n_grpo_steps": 50,
                          "micro_batch_size": 2,
                          "length_normalize": False,
                          "use_std_normalization": False}
        jobs = executor.submit(launch_from_config, default_config, 
                               prompt_path = "/home/c-cye/assignment5-alignment/cs336_alignment/prompts/question_only.prompt",
                               reward_fn = question_only_reward_fn)
    
    elif experiment_name == "leaderboard":
        onpolicy_config = {"learning_rate": 2.5e-5, 
                          "run_name": "on_policy_leaderboard",
                          "n_grpo_steps": 200,
                          "loss_type": "grpo_clip",
                          "epochs_per_rollout_batch": 1,
                          "train_batch_size": 128,
                          "micro_batch_size": 2,
                          "length_normalize": False,
                          "use_std_normalization": False}
        jobs = executor.submit(launch_from_config, onpolicy_config)

        offpolicy_config = {"learning_rate": 2e-5, 
                          "run_name": "off_policy_leaderboard",
                          "loss_type": "grpo_clip",
                          "epochs_per_rollout_batch": 2,
                          "train_batch_size": 128,
                          "n_grpo_steps": 200,
                          "micro_batch_size": 2,
                          "length_normalize": False,
                          "use_std_normalization": False}
        jobs = executor.submit(launch_from_config, offpolicy_config)
        
        print(f"Submitted leaderboard experiments")
    elif experiment_name == "final_leaderboard":
        onpolicy_config = {"learning_rate": 2.5e-5, 
                          "run_name": "on_policy_leaderboard",
                          "n_grpo_steps": 100,
                          "loss_type": "grpo_clip",
                          "epochs_per_rollout_batch": 1,
                          "train_batch_size": 128,
                          "micro_batch_size": 2,
                          "length_normalize": False,
                          "use_std_normalization": False}
        
        jobs = executor.submit(launch_from_config, onpolicy_config, iter_warmup = True) # use EI for warmup
    else:
        raise ValueError(f"Unknown experiment: {experiment_name}")
    
    return jobs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True)
    args = parser.parse_args()

    experiment = args.experiment
    
    executor = submitit.AutoExecutor(folder="./logs")
    executor.update_parameters(
        timeout_min=120,
        slurm_gpus_per_node=1,
        slurm_partition="a5-batch",
        slurm_qos="a5-batch-qos",
        slurm_nodes=1,
        slurm_job_name=f"grpo-{experiment}",
        slurm_exclude="ad12a3ca-02",
        mem_gb=50,
    )
    
    jobs = submit_experiment(experiment, executor)

if __name__ == "__main__":
    main()