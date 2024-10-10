#!/usr/bin/env bash
#
#SBATCH --job-name=task2
#SBATCH --partition=common
#SBATCH --qos=4gpu1h
#SBATCH --time=5
#SBATCH --output=output2b.txt
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=3

srun /bin/hostname
