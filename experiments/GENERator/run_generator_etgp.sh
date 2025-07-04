#!/bin/bash
#SBATCH -p gpu-large
#SBATCH --gres=gpu:A100:1
#SBATCH -w oven-0-13
#SBATCH --ntasks-per-node=1
#SBATCH --mem=48GB
#SBATCH --job-name generator
#SBATCH -o logs/generator_etgp-%J.out
#SBATCH -e logs/generator_etgp-%J.err
#SBATCH --cpus-per-task=8
#SBATCH -t 2-0:00:00

# Activate conda environment

eval "$(conda shell.bash hook)"
conda activate dnalongbench

python generator_etgp.py 