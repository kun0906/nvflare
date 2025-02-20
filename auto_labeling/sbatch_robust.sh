#!/bin/bash
#SBATCH --job-name=robust    # Job name
#SBATCH --account=kunyang_nvflare_py31012_0001
#SBATCH --output=log/output_%A_%a.out        # Standard output (job ID and array task ID)
#SBATCH --error=log/error_%A_%a.err          # Standard error
#SBATCH --ntasks=1                           # Number of tasks per array job
#SBATCH --mem=16G                            # Memory allocation per node
#SBATCH --gres=gpu:1                         # Request 1 GPU
#SBATCH --time=1:00:00                      # Time limit (hrs:min:sec)
#SBATCH --array=0-40

# Define parameter combinations
#labeling_rates=(-5 -2.5 -1 -0.5 -0.1 -0.05 -0.01 -0.001 -0.0001 0 0.0001 0.001 0.01 0.05 0.1 0.5 1 2.5 5)                            # Labeling rate
#labeling_rates=(0.5 0.7 1.0 2 5 10 20)                            # random_noise rate
labeling_rates=(5)                            # Labeling rate
epochs_values=(5)                           # Number of server epochs
benign_values=(4)                          # Number of benign clients
aggregation_values=('refined_krum' 'krum' 'median' 'mean')        # Aggregation method

# Calculate the total number of parameter combinations
total_combinations=$(( ${#labeling_rates[@]} * ${#epochs_values[@]} * ${#benign_values[@]} * ${#aggregation_values[@]} ))

## SBATCH --array=0-$((total_combinations - 1)) # Array range for parameter combinations

# Check if the task ID exceeds the total combinations
if [ $SLURM_ARRAY_TASK_ID -ge $total_combinations ]; then
  echo "SLURM_ARRAY_TASK_ID exceeds the total parameter combinations. Exiting."
  exit 1
fi

# Compute indices for the current task
lrate_index=$((SLURM_ARRAY_TASK_ID / (${#epochs_values[@]} * ${#benign_values[@]} * ${#aggregation_values[@]})))
remaining=$((SLURM_ARRAY_TASK_ID % (${#epochs_values[@]} * ${#benign_values[@]} * ${#aggregation_values[@]})))
epochs_index=$((remaining / (${#benign_values[@]} * ${#aggregation_values[@]})))
remaining=$((remaining % (${#benign_values[@]} * ${#aggregation_values[@]})))
benign_index=$((remaining / ${#aggregation_values[@]}))
aggregation_index=$((remaining % ${#aggregation_values[@]}))

# Get the values for the current combination
labeling_rate=${labeling_rates[$lrate_index]}
epochs=${epochs_values[$epochs_index]}
benign=${benign_values[$benign_index]}
aggregation=${aggregation_values[$aggregation_index]}

# Combine selected parameters for the Python script
PARAMS="-r $labeling_rate -n $epochs -b $benign -a $aggregation"
$PARAMS

# Load necessary modules
module load conda
conda activate nvflare-3.10

# Navigate to the working directory
cd ~/nvflare/auto_labeling || exit
pwd

# Run the script with the selected parameters
#PYTHONPATH=. python3 fl_cnn_robust_aggregation.py $PARAMS
PYTHONPATH=. python3 fl_cnn_robust_aggregation_label_flipping.py $PARAMS
#PYTHONPATH=. python3 fl_cnn_robust_aggregation_large_values.py $PARAMS
#PYTHONPATH=. python3 fl_cnn_robust_aggregation_sign_flipping.py $PARAMS
#PYTHONPATH=. python3 fl_cnn_robust_aggregation_random_noise.py $PARAMS

