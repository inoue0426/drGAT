methods = ["GAT", "GATv2", "Transformer"]
datasets = ["nci", "ctrp", "gdsc1", "gdsc2"]

template = """#!/bin/bash
#SBATCH --mem=50gb
#SBATCH --requeue
#SBATCH --job-name={method}_{data}
#SBATCH --output=logs/{method}_{data}.out
#SBATCH --error=logs/{method}_{data}.err
#SBATCH --partition gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=inoue019@umn.edu
#SBATCH --time=10-00:00:00

source /data/$USER/conda/etc/profile.d/conda.sh && source /data/$USER/conda/etc/profile.d/mamba.sh
conda activate genex

python ../run_GAT.py --method {method} --data {data}
"""

for m in methods:
    for d in datasets:
        with open(f"jobs/run_{m}_{d}.sh", "w") as f:
            f.write(template.format(method=m, data=d))
