#!/bin/bash

SCRIPT_DIR="slurm_scripts"

# 引数チェック
if [ $# -ne 1 ]; then
  echo "Usage: $0 [4 | 8 | p]"
  echo "  4 = a100-4"
  echo "  8 = a100-8"
  echo "  p = preempt-gpu"
  exit 1
fi

# 引数をSlurmパーティション名にマッピング
case "$1" in
  4)
    PARTITION="a100-4"
    ;;
  8)
    PARTITION="a100-8"
    ;;
  p)
    PARTITION="preempt-gpu"
    ;;
  *)
    echo "❌ Invalid option: $1"
    echo "Valid options: 4, 8, p"
    exit 1
    ;;
esac

# ディレクトリ存在確認
if [ ! -d "$SCRIPT_DIR" ]; then
  echo "❌ Directory '$SCRIPT_DIR' does not exist."
  exit 1
fi

echo "🚀 Submitting SLURM scripts for partition: $PARTITION"

# 対象ファイルだけを sbatch で送信
for script in "$SCRIPT_DIR"/*_"$PARTITION".sh; do
  if [ -f "$script" ]; then
    echo "📤 Submitting: $script"
    sbatch "$script"
  fi
done

echo "✅ All matching scripts submitted."
