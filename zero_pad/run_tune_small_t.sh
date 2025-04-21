#!/bin/bash
#SBATCH --job-name=GT
#SBATCH --partition=a100-4 
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out

module load conda
source activate /scratch.global/$USER/myenv
python t_small_GT.py
