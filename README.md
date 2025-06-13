# 🧬 drGT

Official implementation of **drGT: Attention-Guided Gene Assessment for Drug Response in Drug-Cell-Gene Heterogeneous Network**\




drGT uses attention-based GNNs (e.g., GAT, GATv2, Transformer) to predict drug sensitivity and highlight gene-level importance in a heterogeneous network.

---

## 🚀 Quick Start

> Requires Python 3.10 or 3.11

```bash
git clone https://github.com/inoue0426/drGT.git
cd drGT
chmod +x run_drGT.py
./run_drGT.py --task test2 --data nci --method GATv2 --cell_or_drug cell
```

---

## 🧠 Using Pretrained Models

```python
from drGT import drGT
probs, labels, attn = drGT.predict('best_model.pt', sampler, params)
```

Make sure `params` matches the pretrained architecture.

---

## ⚙️ Environment Setup (with [uv](https://github.com/astral-sh/uv))

```bash
uv venv && source .venv/bin/activate
uv sync
jupyter notebook
```

> Kernel: **Python (drGT)**

---

## ⚡️ GPU Support

drGT supports PyTorch CUDA acceleration (tested on NVIDIA A100). \
It also runs on Apple Silicon Macs (e.g., M3), using the Metal backend for PyTorch. \
Use [PyTorch guide](https://pytorch.org/get-started/locally/) to install the right CUDA version.

---

## 📁 Directory Structure

```
drGT/                  # Model code
configs/               # YAML configs
Test1_random_split/    # Experiment: random masking
Test2_leave_X_out/     # Experiment: leave-out
preprocess/            # Data notebooks
data/                  # Input data
```

---

## 📦 Dependencies

**Option 1: **``** (lightweight)**\
Include `#!/usr/bin/env -S uv run --script` headers in scripts.

**Option 2: **``** (standard)**\
Install with pip:

```bash
python -m venv .venv && source .venv/bin/activate
pip install .
```

---

## 📖 Citation

```bibtex
@article{inoue2024drgat,
  title={drGT: Attention-Guided Gene Assessment of Drug Response Utilizing a Drug-Cell-Gene Heterogeneous Network},
  author={Inoue, Yoshitaka and Lee, Hunmin and Fu, Tianfan and Luna, Augustin},
  journal={arXiv preprint arXiv:2405.08979},
  year={2024}
}
```
