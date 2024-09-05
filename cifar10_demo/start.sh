#!/bin/bash

# Log in to a specific computing node
#srun -G 1 -t 600 --nodelist=bcm-dgxa100-0018 --pty sub_start.sh

module load conda
conda -V
conda activate nvflare-3.10
nvflare -V

srun -G 1 -t 600 --nodelist=bcm-dgxa100-0018 --pty $SHELL



