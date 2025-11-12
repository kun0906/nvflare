#!/bin/bash

SERVER_NODE=$1
WORKSPACE_DIR=$2
NUM_CLIENTS=$3
NUM_ROUNDS=$4
NUM_GPUS=$5

echo "===2nodes_1.sh==="
echo "hostname: $(hostname)"
IP=$(getent hosts "$(hostname)" | awk '{print $1}')
echo "IP: ${IP}"
uname -a
pwd


# === Kill existing python processes safely ===
echo "Checking and killing existing python processes..."
while pgrep -f "python3 server.py" > /dev/null; do
  echo "python3 server.py  detected. Attempting to kill..."
  pgrep -f "python3 server.py" | xargs -r kill
  sleep 10
done
echo "✅All python processes terminated."

echo "🚀 Starting server...${SERVER_DIR}"
pwd

#parser.add_argument("--server_ip", type=str, default="127.0.0.1", help="IP address of the server")
#parser.add_argument("--workspace_dir", type=str, default=".", help="Workspace directory")
#parser.add_argument("--num_clients", type=int, default=2, help="Number of clients")
#parser.add_argument("--num_rounds", type=int, default=1, help="Number of rounds")

python3 server.py --server_ip $SERVER_NODE --workspace_dir $WORKSPACE_DIR --num_clients $NUM_CLIENTS  --num_rounds $NUM_ROUNDS &
#
# Wait until NVFLARE server is running
echo "Waiting for server to start..."
while true; do
  if pgrep -f "python3 server.py" > /dev/null; then
    echo "python3 server.py is now running."
    echo "server started" >> server_ready.log
    break
  else
    echo "Server not up yet... retrying in 5 seconds."
    sleep 5
  fi
done

sleep 10
# === Only Launch the first half clients ===
for ((i=1; i<=$((NUM_CLIENTS / 2)); i++)); do
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

