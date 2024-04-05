#!/usr/bin/env bash
# slurm template for serial jobs
# SLURM options set below
#SBATCH --job-name=ASL_jliem # Job name
#SBATCH --output=log-%j.out
# Standard output and error log
#SBATCH --mail-user=jliem@middlebury.edu
# Where to send email
#SBATCH --mail-type=FAIL
#SBATCH --mem=10GB
#SBATCH --partition=gpu-standard
# Partition (queue)
#SBATCH --gres=gpu:1
# Number of GPUs
#SBATCH --time=02:00:00
# Adjust time as needed < 24:00:00

# Use a conda environment to manage dependencies. You may find the dependencies in the README.md file.
conda info --envs
conda activate ML