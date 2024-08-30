#!/bin/bash

pwd
python3 -V
nvflare -V

# Define the root directory for NVFlare
ROOT_DIR='/tmp/nvflare/poc/example_project/prod_00'

# Start the server
echo "Starting the server..."
$ROOT_DIR/server/startup/start.sh &

# Start the clients
N=3
for CLIENT_ID in $(seq 0 $N); do  # Replace N with the actual number of clients
  echo "Starting client $CLIENT_ID..."
  $ROOT_DIR/site-${CLIENT_ID}/startup/start.sh &
done

echo "NVFlare started."
