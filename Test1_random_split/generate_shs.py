import os

# 空のLogsとjobsディレクトリを作成
os.makedirs("Logs", exist_ok=True)
os.makedirs("jobs", exist_ok=True)

methods = ["GAT", "GATv2", "Transformer"]
datasets = ["nci", "ctrp", "gdsc1", "gdsc2"]

# 略称マッピング
method_short = {
    "GAT": "G",
    "GATv2": "v",
    "Transformer": "t"
}

template = """#!/bin/bash
#SBATCH --mem=50gb
#SBATCH --requeue
#SBATCH --job-name={short}_{data}
#SBATCH --partition gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=inoue019@umn.edu
#SBATCH --time=10-00:00:00
#SBATCH --output=Logs/%x_%j.out
#SBATCH --error=Logs/%x_%j.err

source /data/$USER/conda/etc/profile.d/conda.sh && source /data/$USER/conda/etc/profile.d/mamba.sh
conda activate genex

python ../run_GAT.py --method {method} --data {data}
"""

for m in methods:
    for d in datasets:
        short = method_short[m]
        with open(f"jobs/run_{short}_{d}.sh", "w") as f:
            f.write(template.format(method=m, data=d, short=short))

