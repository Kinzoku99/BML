#!/usr/bin/env bash
#
#SBATCH --job-name=task2a
#SBATCH --partition=common
#SBATCH --qos=4gpu1h
#SBATCH --time=5
#SBATCH --output=output2a.txt
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1

srun /bin/hostname
