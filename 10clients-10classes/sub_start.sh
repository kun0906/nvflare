#!/bin/bash
# Start nvflare: server and clients

# Load the conda environment with NVFlare
#module load conda
if [ $? -ne 0 ]; then
  echo "Error loading conda module"
  exit 1
fi
conda -V
which conda

#conda activate nvflare-3.10
if [ $? -ne 0 ]; then
  echo "Error activating conda environment nvflare-3.10"
  exit 1
fi
nvflare -V
which nvflare

python3 -V
which python3

pwd

N=10
yes | nvflare poc prepare -n $N

# Set the root directory
#echo $WORK
ROOT_DIR="/tmp/nvflare/poc/example_project/prod_00"

# Start server
echo "Starting server..."
$ROOT_DIR/server/startup/start.sh &

# Start clients
for CLIENT_ID in $(seq 1 $N); do  # Replace N with the actual number of clients
  echo "Starting client $CLIENT_ID..."
  $ROOT_DIR/site-${CLIENT_ID}/startup/start.sh &
done

echo "NVFlare started."

$SHELL
