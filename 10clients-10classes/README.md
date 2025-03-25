
Step 1 
```shell 
%Upload the local project to SMU remote server (SuperPOD)
$cd PycharmProjects
$rsync -avP nvflare kunyang@slogin-01.superpod.smu.edu:/users/kunyang
```

Step 2: Start Federated Learning Environment
```shell
$ssh kunyang@slogin-01.superpod.smu.edu
$module load conda && conda activate nvflare-3.10
$sinfo
% copy folder from login node to computing node
$rsync -avP nvflare/ kunyang@bcm-dgxa100-0008:/users/kunyang/
% The job code needs at least 2 GPUS, nvidia-smi
$srun -A kunyang_nvflare_py31012_0001 -t 60 -G 2 -w bcm-dgxa100-0008 --pty $SHELL
% srun -A kunyang_nvflare_py31012_0001 -t 60 -G 1 --pty $SHELL
$module load conda && conda activate nvflare-3.10
$cd nvflare/10clients-10classes 
$python3 gen_sites.py
$python3 cifar10_data.py
$./sub_start.sh
```

Step 3: Check the status 
```shell
$ps aux | grep nvflare*
$killall -u kunyang python3
$ps aux | grep start*
$killall start* 
$./stop.sh 

```


Step 4: Submit jobs by Admin 
```shell
$srun -A kunyang_nvflare_py31012_0001 -t 60 -G 1 -w bcm-dgxa100-0008 --pty $SHELL
$conda activate nvflare-3.10
$/tmp/nvflare/poc/example_project/prod_00/admin@nvidia.com/startup/fl_admin.sh

$ps aux | grep nvflare*
$ps aux | grep start*
$killall start* 
$./stop.sh 

% Admin@nvidia.com on admin side 
$check_status server 
$check_status client 
% submit_job job_path (absolute path)
%submit_job /Users/49751124/PycharmProjects/nvflare/10clients-10classes/jobs/10clients-10classes
$submit_job /users/kunyang/nvflare/10clients-10classes/jobs/10clients-10classes
$list_jobs 
% download job from server to admin 
$download_job 6cdfc15c-1cc2-418a-bf44-94847e5f4a8b
$bye 
% download from remote to local (after exiting from the admin node)
$rsync -avP kunyang@slogin-01.superpod.smu.edu:/users/kunyang/nvflare/10clients-10classes/6cdfc15c-1cc2-418a-bf44-94847e5f4a8b ~/PycharmProjects/nvflare/10clients-10classes

```

Step 5: tensorboard (at local machine)
```shell
% Download results from the remote server to your local machine
% download job from (remote) admin to local machine 
$tensorboard --logdir=58a8b77d-6052-4a7d-bebb-ed30d1b41528/workspace/tb_events

$sshfs kunyang@superpod.smu.edu:/users/kunyang/ superpod -o volname=superpod
$sshfs kunyang@slogin-01.superpod.smu.edu:/tmp/nvflare/ data
```

#Main steps
1. Use start.sh to log into a computing node 
2. Use sub_start.sh to start clients and server 
3. Use $WORK/nvflare/poc/example_project/prod_00/admin@nvidia.com/startup/fl_admin.sh to start admin console 
4. Submit your job: submit_job /users/kunyang/cifar10-hello-pt-10clients-2classes/jobs/10clients-2classes_byzantine 
5. check_status client
6. list_jobs 

