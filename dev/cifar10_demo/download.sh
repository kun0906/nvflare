#!/bin/bash

# download files from a computing node to the login node
node_id=bcm-dgxa100-0018
scp -r kunyang@${node_id}:/tmp/nvflare/poc/example_project/prod_00/admin@nvidia.com/transfer/ ./
$ls

# download files from the login node to local
job_id=def38e5d-b037-41e2-9566-ec7c7f3074b7
$scp -r kunyang@superpod.smu.edu:~/cifar10-hello-pt-10clients-2classes/${job_id} ./
