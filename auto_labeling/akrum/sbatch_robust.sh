#!/bin/bash
#SBATCH --job-name=akrum    # Job name
#SBATCH --account=kunyang_nvflare_py31012_0001
#SBATCH --output=/projects/kunyang/nvflare_py31012/nvflare/log/out_%A_%a.out        # Standard output (job ID and array task ID)
#SBATCH --error=/projects/kunyang/nvflare_py31012/nvflare/log/error_%A_%a.err          # Standard error
#SBATCH --ntasks=1                           # Number of tasks per array job
#SBATCH --mem=16G                            # Memory allocation per node
#SBATCH --gres=gpu:1                         # Request 1 GPU
#SBATCH --time=6:00:00                      # Time limit (hrs:min:sec)
#SBATCH --array=0-140

# Define parameter combinations
#labeling_rates=(-5 -2.5 -1 -0.5 -0.1 -0.05 -0.01 -0.001 -0.0001 0 0.0001 0.001 0.01 0.05 0.1 0.5 1 2.5 5)                            # Labeling rate
#labeling_rates=(0.01 0.05 0.1)                            # random_noise rate
#labeling_rates=(10 30 45 90 135 270 5)          # degree
#labeling_rates=(0.001 0.01 0.1 0.3 0.5 0.8 1.0)          # percent of parameters will be changed
#labeling_rates=(0.01 0.1 1 3 5 10 50)          # larger values
#tunable_type='different_f'
#labeling_rates=(0 0.01 0.05 0.1 0.2 0.3 0.4 0.45 0.5)          # larger values
#labeling_rates=(0.1 0.5 1 2 3 4 5)      # different byzantine location
#tunable_type='different_n'
#labeling_rates=(5 10 20 30 40 50 60 70 80 90 100)      # different number of clients
#tunable_type='different_d'
#labeling_rates=(2 5 10 25 50 75 100)      # different number of byzantine clients
# labeling_rates=(5 10 50 100 200 400 768)      # different number of byzantine clients
#tunable_type='different_mu'
#labeling_rates=(0.001 0.005 0.01 0.05 0.1 0.5 1 2 5 10)      # different number of byzantine clients
#tunable_type='different_var'    # different variance
#labeling_rates=(0 0.001 0.005 0.01 0.05 0.1 0.5 1 5 10 50)      # different number of byzantine clients
tunable_type='different_c_ratio'    # different c_ratio for random projection,
# project model updates (not training data dimension) onto a lower dimension
# reduced_d = C * np.log(N), where C = c_ratio * (D/np.log(N))
# here maximum C = D / np.log(N), which mean, using all features.
labeling_rates=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1)      # different number of byzantine clients
server_epochs_values=(100)                           # Number of server epochs
num_clients_values=(50)                          # Number of total clients
aggregation_values=('adaptive_krum' 'krum' 'adaptive_krum+rp' 'krum+rp' 'medoid' 'median' 'mean'
'adaptive_krum_avg' 'krum_avg' 'adaptive_krum+rp_avg' 'krum+rp_avg' 'medoid_avg' 'trimmed_mean' 'geometric_median')
# Aggregation method

# Calculate the total number of parameter combinations
total_combinations=$(( ${#labeling_rates[@]} * ${#server_epochs_values[@]} * ${#num_clients_values[@]} * ${#aggregation_values[@]} ))

## SBATCH --array=0-$((total_combinations - 1)) # Array range for parameter combinations

# Check if the task ID exceeds the total combinations
if [ $SLURM_ARRAY_TASK_ID -ge $total_combinations ]; then
  echo "SLURM_ARRAY_TASK_ID exceeds the total parameter combinations. Exiting."
  exit 1
fi

# Compute indices for the current task
lrate_index=$((SLURM_ARRAY_TASK_ID / (${#server_epochs_values[@]} * ${#num_clients_values[@]} * ${#aggregation_values[@]})))
remaining=$((SLURM_ARRAY_TASK_ID % (${#server_epochs_values[@]} * ${#num_clients_values[@]} * ${#aggregation_values[@]})))
epochs_index=$((remaining / (${#num_clients_values[@]} * ${#aggregation_values[@]})))
remaining=$((remaining % (${#num_clients_values[@]} * ${#aggregation_values[@]})))
num_clients_index=$((remaining / ${#aggregation_values[@]}))
aggregation_index=$((remaining % ${#aggregation_values[@]}))

# Get the values for the current combination
labeling_rate=${labeling_rates[$lrate_index]}
server_epochs=${server_epochs_values[$epochs_index]}
num_clients=${num_clients_values[$num_clients_index]}
aggregation=${aggregation_values[$aggregation_index]}

# Combine selected parameters for the Python script
PARAMS="-r $labeling_rate -s $server_epochs -n $num_clients -a $aggregation -t $tunable_type -R 10"
$PARAMS

# Load necessary modules
module load conda
conda activate nvflare-3.10

# Navigate to the working directory
cd ~/nvflare/auto_labeling || exit
pwd

# Run the script with the selected parameters

# replicate neurips results
#PYTHONPATH=. python3 nsf/ragg_random_spambase_nips_paper.py $PARAMS
#PYTHONPATH=. python3 nsf/ragg_random_mnist_nips_paper.py $PARAMS
#PYTHONPATH=. python3 nsf/ragg_random_noise_model_nips_paper1.py $PARAMS
#PYTHONPATH=. python3 nsf/fl_ragg.py $PARAMS

# Data Poisoning
#PYTHONPATH=. python3 akrum/ragg_random_noise_data.py $PARAMS
#PYTHONPATH=. python3 akrum/ragg_label_flipping.py $PARAMS
#PYTHONPATH=. python3 akrum/ragg_rotation_data.py $PARAMS

# Model Poisoning
#PYTHONPATH=. python3 akrum/ragg_random_noise_model.py $PARAMS
#PYTHONPATH=. python3 akrum/ragg_sign_flipping.py $PARAMS
PYTHONPATH=. python3 akrum/ragg_model_large_value.py $PARAMS



