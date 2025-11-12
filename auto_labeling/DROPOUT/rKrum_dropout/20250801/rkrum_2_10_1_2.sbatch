#!/bin/bash
#SBATCH --job-name=rkrum_2_10_1_2
#SBATCH --account=kunyang_nvflare_py31012_0001
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
# #SBATCH --nodelist=bcm-dgxa100-0002,bcm-dgxa100-0005
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=out/%N_%j.log
## SBATCH --output=/dev/null
#SBATCH --error=out/%N_%j.err
#SBATCH --mem=128G           # Request 64 GB of RAM
echo "Running with job name: 2_2_1_2"

#scontrol show job 
#free -h

OUT_DIR=out
NUM_GPUS=1    # each client has one GPU
NUM_CLIENTS=2
NUM_ROUNDS=10
echo "GPUs: 1, Clients: 2, NUM_CLIENTS_PER_NODE: 1, Rounds: 10, OUT_DIR: out"


# === Print SLURM environment variables ===
echo "=== SLURM Environment Variables ==="
env | grep SLURM
echo "==================================="

# === Print Node information ===
echo "=== Node Information ==="
#hostname
#nvidia-smi
#echo "========================"
#
## === Print Task Information ===
#echo "=== Task Information ==="
#echo "Job ID        : $SLURM_JOB_ID"
#echo "Job Name      : $SLURM_JOB_NAME"
#echo "Number of Nodes: $SLURM_NNODES"
#echo "Node List     : $SLURM_JOB_NODELIST"
#echo "Number of GPUs: $SLURM_GPUS_ON_NODE"
#echo "Tasks/Node    : $SLURM_TASKS_PER_NODE"
#echo "CPUs per Task : $SLURM_CPUS_PER_TASK"
#echo "Submit Dir    : $SLURM_SUBMIT_DIR"
#echo "Work Dir      : $(pwd)"
#echo "================================="

echo "===2_nodes0.sh==="
echo "hostname: $(hostname)"
IP=$(getent hosts "$(hostname)" | awk '{print $1}')
echo "IP: ${IP}"
uname -a
pwd

# Get the node list allocated by SLURM
echo "Running on nodes: $SLURM_NODELIST"
echo "Job ID: $SLURM_JOB_ID"

echo "Running with job name: ${SLURM_JOB_NAME}"
#OUT_DIR=$OUT_DIR
#NUM_GPUS=$NUM_GPUS
#NUM_CLIENTS=$NUM_CLIENTS
#NUM_ROUNDS=$NUM_ROUNDS
NODELIST=$SLURM_NODELIST #SLURM nodelist
echo "GPUs: ${NUM_GPUS}, Clients: ${NUM_CLIENTS}, Rounds: ${NUM_ROUNDS}, OUT_DIR: $OUT_DIR, NODE_NAME: $NODELIST"

#  kunyang@slogin-01:~$ srun -A kunyang_nvflare_py31012_0001 -G 1 -t 2000 --nodelist=bcm-dgxa100-0016 --pty $SHELL
ENV_NAME="flnlp_3.9.21"
PROJECT_DIR="$HOME/FLNLP"
CUSTOM_CODE="Code at client and server folder/custom"
DATE=$(date +%Y%m%d_%H%M%S)
WORKSPACE_DIR="${OUT_DIR}/ws_${NUM_CLIENTS}_${NUM_ROUNDS}_${NUM_GPUS}_${SLURM_NNODES}"
#
module load conda
conda activate "$ENV_NAME"

# Extract hostnames from the allocated node list
SERVER_NODE=$(scontrol show hostnames $SLURM_NODELIST | sed -n 1p)
#SERVER_NODE_IP=$(hostname -I | awk '{print $1}')
SERVER_NODE_IP=$(getent hosts "$SERVER_NODE" | awk '{print $1}')
echo "SERVER_NODE: $SERVER_NODE, IP: ${SERVER_NODE_IP}"
CLIENT_NODE=$(scontrol show hostnames $SLURM_NODELIST | sed -n 2p)

#############################################################################################################
# Run server and first half of clients on SERVER_NODE
echo "Launching server and first half of clients on $SERVER_NODE..."
#chmod +x ./2nodes_1.sh
srun --nodes=1 --ntasks=1 --exclusive -w "$SERVER_NODE" python3 2nodes_1.sh "$SERVER_NODE_IP" "$WORKSPACE_DIR" "$NUM_CLIENTS" "$NUM_ROUNDS"  "$NUM_GPUS" &

#############################################################################################################
# Now launch the second half of the clients
echo "Launching second half of clients on $CLIENT_NODE..."
srun --nodes=1 --ntasks=1 --exclusive -w "$CLIENT_NODE" python3 2nodes_2.sh "$SERVER_NODE_IP" "$WORKSPACE_DIR" "$NUM_CLIENTS" "$NUM_ROUNDS"  "$NUM_GPUS" &


while true; do
  echo "hostname: $(hostname)"
  IP=$(getent hosts "$(hostname)" | awk '{print $1}')
  echo "IP: ${IP}"
  uname -a
  echo "ps -aux | grep server"
  ps -aux | grep server

  uname -a
  echo "⏳ Sleeping for 100 seconds before next check..."
  sleep 1
done



pkill -f "./2nodes_1.sh"
pkill -f "./2nodes_2.sh"

sleep 10
echo "✅ All nvflare processes terminated."


