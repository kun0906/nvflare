#!/bin/bash

SERVER_NODE=$1
WORKSPACE_DIR=$2
NUM_CLIENTS=$3
NUM_ROUNDS=$4
NUM_GPUS=$5

echo "===2nodes_2.sh==="
echo "hostname: $(hostname)"
IP=$(getent hosts "$(hostname)" | awk '{print $1}')
echo "IP: ${IP}"
uname -a
pwd

# === Kill existing python processes safely ===
echo "Checking and killing existing python processes..."
while pgrep -f "python3 client.py" > /dev/null; do
  echo "python3 client.py  detected. Attempting to kill..."
  pgrep -f "python3 client.py" | xargs -r kill
  sleep 10
done
echo "✅All python processes terminated."


# === Launch clients ===
for ((i=$((NUM_CLIENTS / 2 + 1)); i<=NUM_CLIENTS; i++)); do
#  cuda_idx=$((i - NUM_CLIENTS / 2))    # each client run on 1 GPU, we only have 8 clients
  if [ "$NUM_GPUS" -eq 1 ]; then
    cuda_idx=0
  else
    cuda_idx=$(( (i - 1) % NUM_GPUS ))
  fi
  client_idx=$i
  echo "🚀 Launching client $i... cuda_idx: ${cuda_idx}, client_idx: ${client_idx}"
  CUDA_VISIBLE_DEVICES=$cuda_idx python3 client.py --server_ip "$SERVER_NODE_IP" --workspace_dir "$WORKSPACE_DIR" --client_idx "$client_idx" &
  if [[ $? -ne 0 ]]; then
    echo "❌ Failed to start client site-${i}"
  else
    echo "✅ Started ./start.sh for client site-${i} with PID $!"
  fi
done

while true; do
#  uname -a
#  echo "⏳ Sleeping for 100 seconds"
  sleep 100
done
