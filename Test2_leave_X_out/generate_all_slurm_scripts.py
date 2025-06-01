from pathlib import Path

def generate_slurm_script(data_name, data_type, partition, output_dir="slurm_scripts"):
    job_name = f"{data_name}_{data_type}_{partition}"
    work_dir = "/home/inoue019/drGAT/Test2_leave_X_out"

    # GPUとパーティションに応じた設定
    if partition == "preempt-gpu":
        gres = "gpu:a40:1"
    else:
        gres = "gpu:a100:1"

    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --gres={gres}
#SBATCH --time=01:00:00
#SBATCH --mem=10G
#SBATCH --cpus-per-task=8
#SBATCH --requeue
#SBATCH --output=logs/%x.out

module load conda
source activate /panfs/jay/groups/33/kuangr/inoue019/conda-envs/myenv
cd {work_dir}
python run_no_atten_drgat.py --data_name {data_name} --data_type {data_type}
"""

    output_path = Path(output_dir) / f"{job_name}.sh"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(script)
    print(f"✅ {output_path} を生成しました。")

def main():
    data_names = ["gdsc1", "gdsc2", "ctrp", "nci"]
    data_types = ["cell", "drug"]
    partitions = ["preempt-gpu", "a100-4", "a100-8"]

    for partition in partitions:
        for data_name in data_names:
            for data_type in data_types:
                generate_slurm_script(data_name, data_type, partition)

if __name__ == "__main__":
    main()
