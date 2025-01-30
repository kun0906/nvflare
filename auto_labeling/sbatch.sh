#!/bin/bash
#SBATCH --job-name=param_combinations_job    # Job name
#SBATCH --account=kunyang_nvflare_py31012_0001
#SBATCH --output=log/output_%A_%a.out        # Standard output (job ID and array task ID)
#SBATCH --error=log/error_%A_%a.err          # Standard error
#SBATCH --ntasks=1                           # Number of tasks per array job
#SBATCH --mem=16G                            # Memory allocation per node
#SBATCH --gres=gpu:1                         # Request 1 GPU
#SBATCH --time=05:00:00                      # Time limit (hrs:min:sec)
#SBATCH --array=0-100

# Define parameter combinations
param1=(0. 0.5 1.0 5.0)                                 # Example distillation weight values
epochs_values=(10000 50000)             # Number of epochs
hidden_values=(0 1 2 3 5)                          # Hidden layer sizes
patience_values=(0.001 0.0001)        # Patience values for early stopping

# Calculate the total number of parameter combinations
total_combinations=$(( ${#param1[@]} * ${#epochs_values[@]} * ${#hidden_values[@]} * ${#patience_values[@]} ))

## SBATCH --array=0-$((total_combinations - 1)) # Array range for parameter combinations

# Check if the task ID exceeds the total combinations
if [ $SLURM_ARRAY_TASK_ID -ge $total_combinations ]; then
  echo "SLURM_ARRAY_TASK_ID exceeds the total parameter combinations. Exiting."
  exit 1
fi

# Compute indices for the current task
dw_index=$((SLURM_ARRAY_TASK_ID / (${#epochs_values[@]} * ${#hidden_values[@]} * ${#patience_values[@]})))
remaining=$((SLURM_ARRAY_TASK_ID % (${#epochs_values[@]} * ${#hidden_values[@]} * ${#patience_values[@]})))
epochs_index=$((remaining / (${#hidden_values[@]} * ${#patience_values[@]})))
remaining=$((remaining % (${#hidden_values[@]} * ${#patience_values[@]})))
hidden_index=$((remaining / ${#patience_values[@]}))
patience_index=$((remaining % ${#patience_values[@]}))

# Get the values for the current combination
distill_weight=${param1[$dw_index]}
epochs=${epochs_values[$epochs_index]}
hidden=${hidden_values[$hidden_index]}
patience=${patience_values[$patience_index]}

# Combine selected parameters for the Python script
PARAMS="-r $distill_weight -n $epochs -l $hidden -p $patience"

# Load necessary modules
module load conda
conda activate nvflare-3.10

# Navigate to the working directory
cd ~/nvflare/auto_labeling || exit
pwd

# Run the script with the selected parameters
#PYTHONPATH=. python3 gnn_fl_vaes_attention_link_jaccard.py $PARAMS
PYTHONPATH=. python3 cgan_fl_generate.py $PARAMS
