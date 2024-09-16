#!/bin/bash

## Load the conda environment with NVFlare
#module load conda
#if [ $? -ne 0 ]; then
#  echo "Error loading conda module"
#  exit 1
#fi
#which conda
#conda --version
#
#conda activate nvflare-3.10
#if [ $? -ne 0 ]; then
#  echo "Error activating conda environment nvflare-3.10"
#  exit 1
#fi
#
#pwd
#python3 -V
#nvflare -V

# Define the root directory for NVFlare
ROOT_DIR='/tmp/nvflare/poc/example_project/prod_00'

# Stop the server
echo "Stoping the server..."
#$ROOT_DIR/server/startup/stop_fl.sh &
echo "Y" | $ROOT_DIR/server/startup/stop_fl.sh

# Stop the clients
N=10
for CLIENT_ID in $(seq 1 $N); do  # Replace N with the actual number of clients
  echo "Stoping client $CLIENT_ID..."
  #$ROOT_DIR/site-${CLIENT_ID}/startup/stop_fl.sh &
  echo "Y" | $ROOT_DIR/site-${CLIENT_ID}/startup/stop_fl.sh
done



echo "NVFlare stopped."


#$SHELL
