#!/bin/bash
#SBATCH --mem=50gb
#SBATCH --requeue
#SBATCH --job-name=GATv2_ctrp
#SBATCH --output=logs/GATv2_ctrp.out
#SBATCH --error=logs/GATv2_ctrp.err
#SBATCH --partition gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=inoue019@umn.edu
#SBATCH --time=10-00:00:00

source /data/$USER/conda/etc/profile.d/conda.sh && source /data/$USER/conda/etc/profile.d/mamba.sh
conda activate genex

python ../run_GAT.py --method GATv2 --data ctrp
