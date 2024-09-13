#!/bin/bash

echo "Using \"nvflare simulator\" to test job!"

JOB_NAME=10clients-2classes
nvflare simulator -w simulator/$JOB_NAME -n 3 -t 2 jobs/$JOB_NAME




