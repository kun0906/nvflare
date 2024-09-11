#!/bin/bash

# Useful commands

nvidia-smi

ps aux | grep nvflare


job_id=77c7db60-2617-4855-a38b-c75d3a8e18ce
tensorboard --logdir=$job_id/workspace/tb_events

