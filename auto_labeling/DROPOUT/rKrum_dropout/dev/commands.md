

ssh kunyang@superpod.smu.edu

ssh kunyang@slogin-01.superpod.smu.edu

cd /users/kunyang/FLNLP_Dropout/rKrum_dropout

sbatch rkrum_8_30_1_1.sbatch


srun -A kunyang_nvflare_py31012_0001 -G 1 -t 1000 --nodelist=bcm-dgxa100-0016 --pty bash

srun -A kunyang_nvflare_py31012_0001 -G 1 -t 10 --pty bash

