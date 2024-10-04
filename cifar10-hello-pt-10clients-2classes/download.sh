#!/bin/bash

pwd

# Step 1: Log in to HPC and execute commands
PROJECT_DIR="cifar10-hello-pt-10clients-2classes"
NODE_ID="bcm-dgxa100-0016"
WORK="/work/users/kunyang"
echo
echo -e "Step 1: Log in to HPC, change directory, and perform actions"
ssh -t kunyang@superpod.smu.edu "
cd ~/${PROJECT_DIR} &&
pwd &&
#ls

echo
echo -e \"Step 2: Define NODE_ID and perform rsync\"
# rsync -avP kunyang@${NODE_ID}:/tmp/nvflare/poc/example_project/prod_00/admin@nvidia.com/transfer ~/${PROJECT_DIR}
rsync -avP kunyang@${NODE_ID}:$WORK/nvflare/poc/example_project/prod_00/admin@nvidia.com/transfer ~/${PROJECT_DIR}
"

echo
echo -e "Step 3: Download files from the login node to the local machine"
LOCAL_PROJECT_DIR=~
echo "Download files from /users/kunyang/${PROJECT_DIR} to ${LOCAL_PROJECT_DIR}"
rsync -avP kunyang@superpod.smu.edu:/users/kunyang/${PROJECT_DIR} $LOCAL_PROJECT_DIR

echo
echo "Finished!"

python3 format_json.py

