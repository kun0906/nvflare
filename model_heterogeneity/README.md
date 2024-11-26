
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
$module load conda && conda activate nvflare-3.10
$cd nvflare/model_heterogeneity
$python3 kl_model_centralized.py
```
