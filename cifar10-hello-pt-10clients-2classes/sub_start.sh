#!/bin/bash

# Load the conda environment with NVFlare
module load conda
if [ $? -ne 0 ]; then
  echo "Error loading conda module"
  exit 1
fi
which conda
conda --version

conda activate nvflare-3.10
if [ $? -ne 0 ]; then
  echo "Error activating conda environment nvflare-3.10"
  exit 1
fi

nvflare -V
N=10
#nvflare poc prepare -n $N

pwd
python3 -V
nvflare -V

# Define the root directory for NVFlare
ROOT_DIR='/tmp/nvflare/poc/example_project/prod_00'

# Start the server
echo "Starting the server..."
$ROOT_DIR/server/startup/start.sh &

# Start the clients
for CLIENT_ID in $(seq 1 $N); do  # Replace N with the actual number of clients
  echo "Starting client $CLIENT_ID..."
  $ROOT_DIR/site-${CLIENT_ID}/startup/start.sh &
done



echo "NVFlare started."


$SHELL
