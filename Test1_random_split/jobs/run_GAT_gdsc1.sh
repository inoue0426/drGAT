#!/bin/bash
#SBATCH --mem=50gb
#SBATCH --requeue
#SBATCH --job-name=GAT_gdsc1
#SBATCH --output=logs/GAT_gdsc1.out
#SBATCH --error=logs/GAT_gdsc1.err
#SBATCH --partition gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=inoue019@umn.edu
#SBATCH --time=10-00:00:00

source /data/$USER/conda/etc/profile.d/conda.sh && source /data/$USER/conda/etc/profile.d/mamba.sh
conda activate genex

python ../run_GAT.py --method GAT --data gdsc1
