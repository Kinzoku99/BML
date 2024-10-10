#!/usr/bin/env bash
#
#SBATCH --job-name=task2c_arnold
#SBATCH --partition=common
#SBATCH --qos=4gpu1h
#SBATCH --time=5
#SBATCH --output=output_arnold.txt
#SBATCH --nodelist=arnold
#SBATCH --ntasks-per-node=3

# Use srun to run the command on each task
srun /bin/hostname