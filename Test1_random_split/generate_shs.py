import argparse
import os

# 引数
parser = argparse.ArgumentParser(description="Generate Slurm job scripts.")
parser.add_argument(
    "-g",
    "--gpu",
    type=str,
    default="v100x",
    choices=["a100", "v100x"],
    help="GPU type to request (a100 or v100x)",
)
parser.add_argument(
    "-p",
    "--partition",
    type=str,
    default="gpu",
    help="Partition to submit to (default: gpu)",
)
args = parser.parse_args()

# ディレクトリ
os.makedirs("Logs", exist_ok=True)
os.makedirs("jobs", exist_ok=True)

methods = ["GAT", "GATv2", "Transformer"]
datasets = ["nci", "ctrp", "gdsc1", "gdsc2"]

# 略称マップ
method_short = {"GAT": "G", "GATv2": "v", "Transformer": "t"}
data_short = {"nci": "n", "ctrp": "c", "gdsc1": "g1", "gdsc2": "g2"}
gpu_short = {"a100": "a", "v100x": "v1"}

# テンプレート
template = """#!/bin/bash
#SBATCH --mem=50gb
#SBATCH --requeue
#SBATCH --job-name={m}_{d}.{g}.{p}
#SBATCH --partition={partition}
#SBATCH --gres=gpu:{gpu_type}:1
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=inoue019@umn.edu
#SBATCH --time=10-00:00:00
#SBATCH --output=Logs/slurm-%j.out

cd $SLURM_SUBMIT_DIR  

source /data/$USER/conda/etc/profile.d/conda.sh && source /data/$USER/conda/etc/profile.d/mamba.sh
conda activate genex

python run_drGAT.py --method {method} --data {data}
"""

# スクリプト生成
for m in methods:
    for d in datasets:
        m_short = method_short[m]
        d_short = data_short[d]
        g_short = gpu_short[args.gpu]
        p_short = args.partition

        with open(f"jobs/run_{m_short}_{d_short}.sh", "w") as f:
            f.write(
                template.format(
                    m=m_short,
                    d=d_short,
                    g=g_short,
                    p=p_short,
                    method=m,
                    data=d,
                    gpu_type=args.gpu,
                    partition=args.partition,
                )
            )
