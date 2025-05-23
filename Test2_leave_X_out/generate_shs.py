import argparse
import os

# 引数
parser = argparse.ArgumentParser(description="Generate Slurm job scripts.")
parser.add_argument(
    "-g",
    "--gpu",
    type=str,
    default="a100",
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

parser.add_argument(
    "-j",
    "--n_jobs",
    type=int,
    default=3,
    help="Number of jobs to pass to run_drGAT.py (default: 3)",
)

args = parser.parse_args()

# ディレクトリ
os.makedirs("Logs", exist_ok=True)
os.makedirs("jobs", exist_ok=True)

methods = ["GAT", "GATv2", "Transformer"]
datasets = ["nci", "ctrp", "gdsc1", "gdsc2"]
target_dims = [0, 1]

# 略称マップ
method_short = {"GAT": "G", "GATv2": "v", "Transformer": "t"}
data_short = {"nci": "n", "ctrp": "c", "gdsc1": "g1", "gdsc2": "g2"}
gpu_short = {"a100": "a", "v100x": "v1"}
tasks = ['cell', 'drug']

# テンプレート
template = """#!/bin/bash
#SBATCH --mem=40gb
#SBATCH --requeue
#SBATCH --job-name={m}_{d}_{t}.{g}.{p}
#SBATCH --partition={partition}
#SBATCH --gres=gpu:{gpu_type}:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mail-type=ALL
#SBATCH --mail-user=inouey2slurm@gmail.com
#SBATCH --time=10-00:00:00
#SBATCH --output=Logs/slurm-%j.out

umask 002

cd $SLURM_SUBMIT_DIR  
module load singularity

singularity exec --nv --bind /data/LunaLab/drGAT:/workspace ../../pyg_container.sif python /workspace/Test2_leave_X_out/run_drGAT.py --method {method} --data {data} --target_dim {target_dim} --n_jobs {n_jobs}
"""

# スクリプト生成
for m in methods:
    for d in datasets:
        for i in target_dims:
            m_short = method_short[m]
            d_short = data_short[d]
            g_short = gpu_short[args.gpu]
            p_short = args.partition
            task = tasks[i]
    
            with open(f"jobs/run_{m_short}_{d_short}_{task}.sh", "w") as f:
                f.write(
                    template.format(
                        m=m_short,
                        d=d_short,
                        g=g_short,
                        t=task,
                        p=p_short,
                        method=m,
                        data=d,
                        target_dim=i,
                        gpu_type=args.gpu,
                        partition=args.partition,
                        n_jobs=args.n_jobs,
                    )
                )
