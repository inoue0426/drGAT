#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9, <3.12"
# dependencies = [
#     "numpy<2",
#     "pandas",
#     "torch",
#     "torch-geometric",
#     "scikit-learn",
#     "tqdm",
#     "pubchempy",
#     "seaborn",
#     "pyyaml",
#     "packaging",
#     "rdkit-pypi"
# ]
# ///

import argparse
import os
import sys

import pandas as pd
import torch
from sklearn.model_selection import KFold
from tqdm import tqdm

current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from drGAT import drGAT
from drGAT.load_data import load_data
from drGAT.metrics import compute_metrics_stats
from drGAT.myutils import get_all_edges_and_labels, get_model_params
from drGAT.sampler import BalancedSampler

parser = argparse.ArgumentParser()
parser.add_argument(
    "--method", type=str, choices=["GAT", "GATv2", "Transformer"], default="GATv2"
)
parser.add_argument(
    "--data", type=str, choices=["gdsc1", "gdsc2", "ctrp", "nci"], default="nci"
)
parser.add_argument("--task", type=str, choices=["test1", "test2"], default="test1")
parser.add_argument(
    "--cell_or_drug",
    type=str,
    choices=["cell", "drug"],
    required=False,
    help="Only required for test2",
)
args = parser.parse_args()

method = args.method
data = args.data
task = args.task
cell_or_drug = args.cell_or_drug

if task == "test2":
    if cell_or_drug is None:
        raise ValueError("`--cell_or_drug` is required for test2 task.")
    params = get_model_params(task, data, method, cell_or_drug)
else:
    params = get_model_params(task, data, method)


# Load data
(
    drugAct,
    null_mask,
    S_d,
    S_c,
    S_g,
    _,
    _,
    _,
    A_cg,
    A_dg,
) = load_data(data, is_zero_pad=params["is_zero_pad"])

# Update parameters
params.update(
    {
        "n_drug": S_d.shape[0],
        "n_cell": S_c.shape[0],
        "n_gene": S_g.shape[0],
        "gnn_layer": method,
    }
)

##### Sample #####

params.update(
    {
        "epochs": 3,
    }
)

# Training and evaluation
all_edges, all_labels = get_all_edges_and_labels(drugAct, null_mask)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

true_datas = pd.DataFrame()
predict_datas = pd.DataFrame()

for train_idx, test_idx in tqdm(kf.split(all_edges)):
    sampler = BalancedSampler(
        drugAct,
        all_edges,
        all_labels,
        train_idx,
        test_idx,
        null_mask,
        S_d,
        S_c,
        S_g,
        A_cg,
        A_dg,
    )

    (
        model,
        best_train_attention,
        best_val_attention,
        true_data,
        predict_data,
        _,
    ) = drGAT.train(sampler, params=params, device=device, verbose=True)

    true_datas = pd.concat([true_datas, pd.DataFrame(true_data).T], ignore_index=True)
    predict_datas = pd.concat(
        [predict_datas, pd.DataFrame(predict_data).T], ignore_index=True
    )

metrics_result = compute_metrics_stats(
    true=true_datas,
    pred=predict_datas,
    target_metrics=["AUROC", "AUPR", "F1", "ACC"],
)

for metric in ["ACC", "Precision", "Recall", "F1", "AUROC", "AUPR"]:
    if metric in metrics_result["formatted"]:
        print(f"{metric:14s}: {metrics_result['formatted'][metric]}")
