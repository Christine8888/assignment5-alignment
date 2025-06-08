import submitit
from cs336_alignment.iter import run_iter
from cs336_alignment.sft import wandb_setup
import wandb
from cs336_alignment.info import *

N_EI_STEPS = 5
N_ROLLOUTS = [3, 6, 12]
N_EPOCHS = [1, 2, 4] 
ITER_BATCH_SIZE = [512, 1024] 

def iter(config, train_path: str):
    print(train_path)
    wandb.init(project = "cs336-alignment-iter", 
               name = f"iter_{config['n_epochs']}_{config['n_rollouts']}_{config['iter_batch_size']}", 
               config = config)
    wandb_setup()
    
    run_iter(config = config, 
             sampling_params = EVAL_SAMPLING_PARAMS, 
             train_path = train_path, 
             vllm_device = 'cuda', 
             model_device = 'cuda')

def run_iter_task(task_id):
    configs = []
    for n_rollouts in N_ROLLOUTS:
        for n_epochs in N_EPOCHS:
            for iter_batch_size in ITER_BATCH_SIZE:
                config = {
                    'model': QWEN_25,
                    'n_rollouts': n_rollouts,
                    'n_ei_steps': N_EI_STEPS,
                    'n_epochs': n_epochs,
                    'iter_batch_size': iter_batch_size,
                    'n_unique': None, # always train on full SFT dataset
                    'minibatch_size': 4,
                    'train_batch_size': 128,
                    'learning_rate': 1e-4,
                    'log_every_n': 10,
                    'eval_every_n': 20,
                    'val_path': MATH_VAL_PATH,
                    'train_path': MATH_TRAIN_PATH,
                    'seed': 42,
                    'start_train_step': 0,
                }
                configs.append(config)
    
    # get the config for this task
    config = configs[task_id]
    iter(config, MATH_TRAIN_PATH)

if __name__ == "__main__":
    # calculate total number of configurations
    total_configs = len(N_ROLLOUTS) * len(N_EPOCHS) * len(ITER_BATCH_SIZE)
    
    # initialize submitit executor
    executor = submitit.AutoExecutor(folder="./logs")
    executor.update_parameters(
        timeout_min = 240,
        slurm_gpus_per_node = 1,
        slurm_partition = "a5-batch",
        slurm_qos = "a5-batch-qos",
        slurm_nodes = 1, # only 1 node
        slurm_job_name = "iter-sweep",
        mem_gb=50,
    )
    
    # submit array job
    jobs = executor.map_array(run_iter_task, range(total_configs)) # pass train_path as argument
    
    print(f"Submitted SLURM array job with {total_configs} tasks")
    print(f"Job IDs: {[job.job_id for job in jobs]}")
