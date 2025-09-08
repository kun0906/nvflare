"""
    python3 sbatch_all.py

    For spambase,
        for each case, 100 clients takes 1GB. 2 cases,
    For MNIST,
        for each case, 100 clients takes ~5GB. 3 cases,
    For Sentiment140,
        for each case, 100 clients takes ~5GB. 3 cases,
        (For 30 parameters, it takes ~150GB.)

    For one parameter (labeling_rate, num_clients_value, num_clients_value)
        We have 6 methods, so total is 6*(2*1 + 5*3 + 5*3) = ~180GB

"""
import os
import subprocess

# SLURM script header
input_str = """#!/bin/bash
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

"""

# Define dataset, attack type, and corresponding script
cases = {
    # 'spambase_paper': {
    #     'gaussian': 'PYTHONPATH=. python3 ICONIP/replication_neurips/ragg_random_spambase_gaussian.py $PARAMS',
    #     'omniscient': 'PYTHONPATH=. python3 ICONIP/replication_neurips/ragg_random_spambase_omniscient.py $PARAMS',
    # },
    'spambase': {
        'random_noise': 'PYTHONPATH=. python3 ICONIP/ragg_model_random_noise_spambase.py $PARAMS',
        'large_value': 'PYTHONPATH=. python3 ICONIP/ragg_model_large_value_spambase.py $PARAMS',
        'label_flipping': 'PYTHONPATH=. python3 ICONIP/ragg_label_flipping_spambase.py $PARAMS',
    },
    # 'mnist': {
    #     'random_noise': 'PYTHONPATH=. python3 ICONIP/ragg_model_random_noise_mnist.py $PARAMS',
    #     'large_value': 'PYTHONPATH=. python3 ICONIP/ragg_model_large_value_mnist.py $PARAMS',
    #     'label_flipping': 'PYTHONPATH=. python3 ICONIP/ragg_label_flipping_mnist.py $PARAMS',
    # },
    # 'sentiment140': {
    #     'random_noise': 'PYTHONPATH=. python3 ICONIP/ragg_model_random_noise_sentiment140.py $PARAMS',
    #     'large_value': 'PYTHONPATH=. python3 ICONIP/ragg_model_large_value_sentiment140.py $PARAMS',
    #     'label_flipping': 'PYTHONPATH=. python3 ICONIP/ragg_label_flipping_sentiment140.py $PARAMS',
    # }
}
job_txt = 'job.txt'
res = {}
# Loop over combinations and write batch files
for dataset, attack_types in cases.items():
    for attack_name, run_cmd in attack_types.items():
        sbatch_content = f"{input_str}\n{run_cmd}\n\n"
        sbatch_filename = f"_{dataset}_{attack_name}.sh"

        with open(sbatch_filename, 'w') as f:
            f.write(sbatch_content)

        print(f"Submitting: {sbatch_filename}", flush=True)
        # os.system(f"sbatch {sbatch_filename}")

        try:
            # Submit and capture output
            result = subprocess.run(
                ["sbatch", sbatch_filename],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            if result.returncode == 0:
                output = result.stdout.strip()
                print(f"Submission output: {output}")
                # Extract job ID (e.g., "Submitted batch job 123456")
                if output.startswith("Submitted batch job"):
                    job_id = output.split()[-1]
                    print(f"Job ID: {job_id}")
            else:
                print(f"Failed to submit {sbatch_filename}")
                print(f"Error: {result.stderr}")
                output = ''

            res[sbatch_filename] = output
        except Exception as e:
            print(f"Failed to submit {sbatch_filename}")

with open(job_txt, 'w') as f:
    for k, v in res.items():
        f.write(f'{k}: {v}\n')
