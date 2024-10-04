
#Main steps 
1. Use start.sh to log into a computing node 
2. Use sub_start.sh to start clients and server 
3. Use $WORK/nvflare/poc/example_project/prod_00/admin@nvidia.com/startup/fl_admin.sh to start admin console 
4. Submit your job: submit_job /users/kunyang/cifar10-hello-pt-10clients-2classes/jobs/10clients-2classes_byzantine 
5. check_status client
6. list_jobs 


```shell
$module load conda && conda activate nvflare-3.10
$sinfo
$srun -t 600 -G 1 -w bcm-dgxa100-0020 --pty $SHELL
$nvflare poc prepare -n 10 
$python3 gen_sites.py
$python3 cifar10_data.py
$./sub_start.sh


$ps aux | grep nvflare*
$ps aux | grep start*
$killall start* 
$./stop.sh 


# Admin@nvidia.com on admin side 
$check_status sever 
$check_status client 
$submit_job job_path
$list_jobs 
$download_job job_id


$sshfs kunyang@superpod.smu.edu:/users/kunyang/ superpod -o volname=superpod
$sshfs kunyang@slogin-01.superpod.smu.edu:/tmp/nvflare/ data


```
