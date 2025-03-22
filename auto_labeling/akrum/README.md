aKrum: 
- dataset
  - 


$ssh kunyang@slogin-01.superpod.smu.edu
$sinfo 
$srun -A kunyang_nvflare_py31012_0001 -t 60 -G 1 -w bcm-dgxa100-0008 --pty $SHELL
$srun -A kunyang_nvflare_py31012_0001 -t 60 -G 1 --pty $SHELL
$module load conda
$conda activate nvflare-3.10
$cd nvflare/auto_labeling


