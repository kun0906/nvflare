#!/bin/bash
#SBATCH --job-name=robust
#SBATCH --account=kunyang_nvflare_py31012_0001
#SBATCH --output=log/output_%A_%a.out
#SBATCH --error=log/error_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --array=0-12

labeling_rates=(0.5 10)
# server_epochs_values=(2)
# num_clients_values=(10)
server_epochs_values=(200)
num_clients_values=(100)
aggregation_values=('adaptive_krum' 'krum' 'adaptive_krum_avg' 'krum_avg' 'median' 'mean')

total_combinations=$(( ${#labeling_rates[@]} * ${#server_epochs_values[@]} * ${#num_clients_values[@]} * ${#aggregation_values[@]} ))

if [ $SLURM_ARRAY_TASK_ID -ge $total_combinations ]; then
  echo "SLURM_ARRAY_TASK_ID exceeds the total parameter combinations. Exiting."
  exit 1
fi

lrate_index=$((SLURM_ARRAY_TASK_ID / (${#server_epochs_values[@]} * ${#num_clients_values[@]} * ${#aggregation_values[@]})))
remaining=$((SLURM_ARRAY_TASK_ID % (${#server_epochs_values[@]} * ${#num_clients_values[@]} * ${#aggregation_values[@]})))
epochs_index=$((remaining / (${#num_clients_values[@]} * ${#aggregation_values[@]})))
remaining=$((remaining % (${#num_clients_values[@]} * ${#aggregation_values[@]})))
num_clients_index=$((remaining / ${#aggregation_values[@]}))
aggregation_index=$((remaining % ${#aggregation_values[@]}))

labeling_rate=${labeling_rates[$lrate_index]}
server_epochs=${server_epochs_values[$epochs_index]}
num_clients=${num_clients_values[$num_clients_index]}
aggregation=${aggregation_values[$aggregation_index]}

PARAMS="-r $labeling_rate -s $server_epochs -n $num_clients -a $aggregation"

module load conda
conda activate nvflare-3.10

cd ~/nvflare/auto_labeling || exit
pwd


PYTHONPATH=. python3 ICONIP/ragg_model_random_noise_spambase.py $PARAMS

