#!/bin/bash

<<<<<<< HEAD
<<<<<<< HEAD
ssh kunyang@superpod.smu.edu

cd ~/cifar10-hello-pt-10clients-2classes

=======
>>>>>>> 814d36a (v0.0.7-3:Update download.sh)
pwd

# Step 1: Log in to HPC and execute commands
PROJECT_DIR="cifar10-hello-pt-10clients-2classes"
NODE_ID="bcm-dgxa100-0016"
echo
echo -e "Step 1: Log in to HPC, change directory, and perform actions"
ssh -t kunyang@superpod.smu.edu "
cd ~/${PROJECT_DIR} &&
pwd &&
#ls

echo
echo -e \"Step 2: Define NODE_ID and perform rsync\"
rsync -avP kunyang@${NODE_ID}:/tmp/nvflare/poc/example_project/prod_00/admin@nvidia.com/transfer ~/${PROJECT_DIR}
"

echo
echo -e "Step 3: Download files from the login node to the local machine"
LOCAL_PROJECT_DIR=~
echo "Download files from /users/kunyang/${PROJECT_DIR} to ${LOCAL_PROJECT_DIR}"
rsync -avP kunyang@superpod.smu.edu:/users/kunyang/${PROJECT_DIR} $LOCAL_PROJECT_DIR

echo
echo "Finished!"

python3 format_json.py

<<<<<<< HEAD
rsync -avP kunyang@superpod.smu.edu:/users/kunyang/cifar10-hello-pt-10clients-2classes ~
=======
# download files from a computing node to the login node
node_id=bcm-dgxa100-0018
rsync -avP kunyang@${node_id}:/tmp/nvflare/poc/example_project/prod_00/admin@nvidia.com/transfer .
ls

# download files from the login node to local
job_id=def38e5d-b037-41e2-9566-ec7c7f3074b7
rsync -avP kunyang@superpod.smu.edu:~/cifar10-hello-pt-10clients-2classes/${job_id} .
# download all the folder to local
rsync -avP kunyang@superpod.smu.edu:~/cifar10-hello-pt-10clients-2classes .
>>>>>>> 45127c9 (v0.0.7-1:sync with different devices)
=======
>>>>>>> 814d36a (v0.0.7-3:Update download.sh)
