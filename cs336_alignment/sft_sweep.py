import submitit
from cs336_alignment.sft import train_run
from cs336_alignment.sft import wandb_setup
from vllm import SamplingParams
import wandb
import math

N_SFT_STEPS = 64
QWEN_25 = "/data/a5-alignment/models/Qwen2.5-Math-1.5B"
MATH_SFT_PATH = "/data/a5-alignment/MATH/sft.jsonl"
MATH_VAL_PATH = "/data/a5-alignment/MATH/validation.jsonl"
BATCH_SIZES = [256]
LRS = [1e-4]
N_UNIQUE = [512]
MINIBATCH_SIZE = 8 # fixed

EVAL_SAMPLING_PARAMS = SamplingParams(
        temperature = 1.0, 
        top_p = 1.0, 
        max_tokens = 1024, 
        stop = ["</answer>"], 
        include_stop_str_in_output = True,
        min_tokens = 4,
    )

def run_sft(config):
    wandb.init(project = "cs336-alignment-sft", name = f"sft_{config['n_unique']}_{config['minibatch_size']}_{config['train_batch_size']}_{config['learning_rate']}", config = config)
    train_run(config = config,
              eval_sampling_params = EVAL_SAMPLING_PARAMS,
              end_eval = True)

def run_sft_task(task_id, train_path: str):
    """Run a single SFT task based on task_id (SLURM array index)"""
    # get size of train path
    n_examples_full = sum(1 for _ in open(train_path, 'r'))
    
    configs = []
    for batch_size in BATCH_SIZES:
        for lr in LRS:
            for n_unique in N_UNIQUE:
                if n_unique is None: n_examples = n_examples_full
                else: n_examples = n_unique
                
                # steps per epoch: examples / batch_size
                steps_per_epoch = math.ceil(n_examples / batch_size)
                
                # epochs needed: total_steps / steps_per_epoch (rounded up)
                n_epochs = math.ceil(N_SFT_STEPS / steps_per_epoch)
                
                config = {
                    'model': QWEN_25,
                    'n_unique': n_unique,
                    'n_epochs': n_epochs,
                    'minibatch_size': MINIBATCH_SIZE,
                    'train_batch_size': batch_size,
                    'learning_rate': lr,
                    'train_path': train_path,
                    'val_path': MATH_VAL_PATH,
                    'seed': 42,
                    'log_every_n': 10,
                    'eval_every_n': 20,
                }
                configs.append(config)
    
    # get the config for this task
    config = configs[task_id]
    run_sft(config)

if __name__ == "__main__":
    # calculate total number of configurations
    total_configs = len(BATCH_SIZES) * len(LRS) * len(N_UNIQUE)
    
    # initialize submitit executor
    executor = submitit.AutoExecutor(folder="./logs")
    executor.update_parameters(
        timeout_min = 60,
        slurm_gpus_per_node = 1,
        slurm_partition = "a5-batch",
        slurm_qos = "a5-batch-qos",
        slurm_nodes = 1, # only 1 node
        slurm_job_name = "sft-sweep",
        mem_gb=50,
    )
    
    # submit array job
    jobs = executor.map_array(run_sft_task, range(total_configs), MATH_SFT_PATH)
    
    print(f"Submitted SLURM array job with {total_configs} tasks")
    print(f"Job IDs: {[job.job_id for job in jobs]}")
