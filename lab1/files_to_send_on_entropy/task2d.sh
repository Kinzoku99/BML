#!/usr/bin/env bash
#
#SBATCH --job-name=task2d
#SBATCH --partition=common
#SBATCH --qos=4gpu1h
#SBATCH --time=00:05:00
#SBATCH --output=output2d.txt
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=2

echo "Srun with params:"
srun --nodes=2 --ntasks-per-node=2 /bin/hostname
echo "srun without params:"
srun  /bin/hostname
