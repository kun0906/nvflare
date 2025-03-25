#!/bin/bash
# Stop server and clients

#$WORK
# Set the root directory for NVFlare
# Single quotes ('): Prevent variable expansion, so $WORK will be treated as a literal string.
ROOT_DIR="/tmp/nvflare/poc/example_project/prod_00"

# Stop the server
echo "Stopping the server..."
#$ROOT_DIR/server/startup/stop_fl.sh &
echo "Y" | $ROOT_DIR/server/startup/stop_fl.sh

# Stop the clients
N=10
for CLIENT_ID in $(seq 1 $N); do  # Replace N with the actual number of clients
  echo "Stopping client $CLIENT_ID..."
  #$ROOT_DIR/site-${CLIENT_ID}/startup/stop_fl.sh &
  echo "Y" | $ROOT_DIR/site-${CLIENT_ID}/startup/stop_fl.sh
done

echo "NVFlare stopped."

$SHELL
