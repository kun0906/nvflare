#!/bin/bash

# Useful commands

# Log in to a specific login node.
ssh kunyang@slogin-02.superpod.smu.edu

nvidia-smi

ps aux | grep nvflare


cd ~/cifar10-hello-pt-10clients-2classes/transfer
# /Users/49751124/cifar10-hello-pt-10clients-2classes/transfer
job_id=77c7db60-2617-4855-a38b-c75d3a8e18ce
tensorboard --logdir=$job_id/workspace/tb_events
 tensorboard --logdir=de4460ff-edf0-49a9-991c-0604cf8b6092/workspace/tb_events



tmux ls
tmux new -s nvflare
tmux attach -t nvflare
tmux kill-session -t nvflare


echo $WORK
$WORK/nvflare/poc/example_project/prod_00/admin@nvidia.com/startup/fl_admin.sh




