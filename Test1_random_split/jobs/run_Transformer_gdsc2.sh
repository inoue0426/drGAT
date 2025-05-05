#!/bin/bash
#SBATCH --mem=50gb
#SBATCH --requeue
#SBATCH --job-name=Transformer_gdsc2
#SBATCH --output=logs/Transformer_gdsc2.out
#SBATCH --error=logs/Transformer_gdsc2.err
#SBATCH --partition gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=inoue019@umn.edu
#SBATCH --time=10-00:00:00

source /data/$USER/conda/etc/profile.d/conda.sh && source /data/$USER/conda/etc/profile.d/mamba.sh
conda activate genex

python ../run_GAT.py --method Transformer --data gdsc2
