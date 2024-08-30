#!/bin/bash

# Log in to a specific computing node
# srun -G 1 -t 600 --nodelist=bcm-dgxa100-0008 --pty $SHELL

# Define the root directory for NVFlare
ROOT_DIR='/tmp/nvflare/poc/example_project/prod_00'

# Stop the server
echo "Stopping the server..."
each "Y" | $ROOT_DIR/server/startup/stop.sh

# Stop the clients
N=2
for CLIENT_ID in $(seq 1 $N); do  # Replace N with the actual number of clients
  echo "Stopping client $CLIENT_ID..."
  echo "Y" | $ROOT_DIR/site-${CLIENT_ID}/startup/stop.sh
done

echo "NVFlare stopped."
