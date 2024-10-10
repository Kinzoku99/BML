#!/usr/bin/env bash
#
#SBATCH --job-name=task2c_bruce
#SBATCH --partition=a6000
#SBATCH --qos=4gpu1h
#SBATCH --time=00:05:00
#SBATCH --output=output_bruce.txt
#SBATCH --nodelist=bruce
#SBATCH --ntasks-per-node=3

# Use srun to run the command on each task
srun /bin/hostname