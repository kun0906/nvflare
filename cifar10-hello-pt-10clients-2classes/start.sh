#!/bin/bash

# Log in to a specific computing node
srun -G 1 -t 600 --nodelist=bcm-dgxa100-0020 --pty sub_start.sh
