#!/bin/bash

echo "Using \"nvflare simulator\" to test job!"

JOB_NAME=10clients-10classes
nvflare simulator -w simulator/$JOB_NAME -n 10 -t 10 jobs/$JOB_NAME




