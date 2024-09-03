
#Main steps 

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
 
```
