Semisupervised ML: 
- dataset
  - Tiny labeled data + large amount of unlabeled data



$sinfo 
$srun -A kunyang_nvflare_py31012_0001 -t 60 -G 1 -w bcm-dgxa100-0008 --pty $SHELL
$conda activate nvflare-3.10
$cd nvflare/auto_labeling


