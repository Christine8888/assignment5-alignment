#!/bin/bash
#SBATCH --job-name=debug
#SBATCH --partition=a5-batch
#SBATCH --qos=a5-batch-qos
#SBATCH -c 4
#SBATCH --gpus=1
#SBATCH --mem=50G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

PYTHON_SCRIPT="$1"

uv run python ${PYTHON_SCRIPT}