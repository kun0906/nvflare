#!/bin/bash

echo "Load conda, activate nvflare, and request for a computing node."

module load conda
conda -V
which conda

conda activate nvflare-3.10
nvflare -V
which nvflare

python3 -V
which python3

# Request for a specific computing node from HPC
#srun -G 1 -t 600 --nodelist=bcm-dgxa100-0018 --pty sub_start.sh
# Log in to a specific computing node with your account, t: how much time you need for one CPU, not two GPUs
srun -A kunyang_nvflare_py31012_0001 -G 2 -t 300 --nodelist=bcm-dgxa100-0016 --pty $SHELL
#srun -A kunyang_nvflare_py31012_0001 -G 2 -t 600 --nodelist=bcm-dgxa100-0016 --pty $SHELL



