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
srun -G 2 -t 600 --nodelist=bcm-dgxa100-0016 --pty $SHELL



