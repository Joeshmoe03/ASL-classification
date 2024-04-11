#!/usr/bin/env bash
# slurm template for serial jobs
# SLURM options set below
#SBATCH --job-name=ASLjliem # Job name
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

# Use a conda environment to manage dependencies. You may find the dependencies in the requirements.txt. Like such:
conda info --envs
conda activate ML

# Run the job: 
# NOTE: You may need to change the python command to python3 or python2 depending on your setup with SLURM
# NOTE: You may change the arguments to the python script as needed
python train.py -nepoch 1 -batchSize 64 -lr 0.001 -metric accuracy precision recall f1_score
