#!/bin/bash

# Log in to a specific computing node
srun -G 1 -t 600 --nodelist=bcm-dgxa100-0008 --pty $SHELL

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

chmod 755 sub_start.sh
echo "Running sub_start.sh..."
./sub_start.sh
if [ $? -ne 0 ]; then
  echo "Error running sub_start.sh"
  exit 1
fi

wait

wait



