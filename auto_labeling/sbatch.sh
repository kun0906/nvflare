#!/bin/bash
#SBATCH --job-name=param_combinations_job   # Job name
#SBATCH --account=kunyang_nvflare_py31012_0001
#SBATCH --output=log/output_%A_%a.out            # Standard output (job ID and array task ID)
#SBATCH --error=log/error_%A_%a.err              # Standard error
#SBATCH --ntasks=1                           # Number of tasks
#SBATCH --mem=16G                            # Memory per node
#SBATCH --gres=gpu:1                         # Request 1 GPU (adjust if needed)
#SBATCH --time=05:00:00                      # Time limit in hrs:min:sec
#SBATCH --array=0-1                         # Array range for combinations

# Define the parameter combinations (distill_weight, epochs)
params1=(0.1)
param2=(1000)

# Calculate the total number of combinations
num_combinations=${#params1[@]} * ${#param2[@]}
$num_combinations

# Get the parameter combination for the current task (SLURM_ARRAY_TASK_ID)
param1_index=$((SLURM_ARRAY_TASK_ID / ${#param2[@]}))  # Index for params1
param2_index=$((SLURM_ARRAY_TASK_ID % ${#param2[@]}))  # Index for param2

param1_value=${params1[$param1_index]}
param2_value=${param2[$param2_index]}

# Combine the selected parameters into the format for flags
PARAM="-r $param1_value -n $param2_value"

# Load necessary modules (e.g., Python and CUDA)
module load conda
conda activate nvflare-3.10

# Run your script with the selected parameters
cd ~/nvflare
pwd
PYTHONPATH=. python3 auto_labeling/gnn_fl_cvae_attention.py $PARAM

