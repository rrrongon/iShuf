#!/bin/bash
#$ -cwd
#$ -l rt_F=64
#$ -l h_rt=2:00:00
#$ -N par_256
#$ -o ./logs/data_partition/$JOB_ID.$JOB_NAME.log
#$ -j y
#$ -l USE_SSH=1
#$ -v SSH_PORT=2299

mpirun -n 4  python data_partition.py -f ../natural_image/data_2/train/ -np 16 -o ../natural_image/Partition_Folder -cf ../natural_image/data_2/train_class_idx.txt
