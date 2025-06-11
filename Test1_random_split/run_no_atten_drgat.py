#!/usr/bin/env python
# coding: utf-8

import argparse
import gc
import os
import sys

import numpy as np
import optuna
import pandas as pd
import rdkit
import torch
from rdkit import RDLogger
from sklearn.model_selection import KFold
from tqdm import tqdm

# RDKitのC++ログを抑制
RDLogger.DisableLog("rdApp.*")

current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from drGAT import No_atten_drGAT as drGAT
from drGAT.load_data import load_data
from drGAT.metrics import compute_metrics_stats
from drGAT.myutils import get_all_edges_and_labels
from drGAT.sampler import BalancedSampler


def suggest_hyperparams(trial, S_d, S_c, S_g):
    params = {
        "n_drug": S_d.shape[0],
        "n_cell": S_c.shape[0],
        "n_gene": S_g.shape[0],
        "dropout1": trial.suggest_float("dropout1", 0.1, 0.5),
        "dropout2": trial.suggest_float("dropout2", 0.1, 0.5),
        "hidden1": trial.suggest_int("hidden1", 128, 512),
        "hidden2": trial.suggest_int("hidden2", 64, 256),
        "hidden3": trial.suggest_int("hidden3", 32, 128),
        "activation": trial.suggest_categorical("activation", ["relu", "gelu"]),
        "optimizer": trial.suggest_categorical("optimizer", ["Adam", "AdamW"]),
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "scheduler": trial.suggest_categorical("scheduler", [None, "Cosine"]),
        "epochs": trial.suggest_int("epochs", 10, 100),
        "gnn_layer": trial.suggest_categorical("gnn_layer", ["GCN", "MPNN"]),
    }

    # CosineAnnealingLRのT_maxをスケジューラに応じて追加
    if params["scheduler"] == "Cosine":
        min_epoch_div = max(1, params["epochs"] // 5)
        max_epoch_div = max(min_epoch_div + 1, params["epochs"] // 2)
        params["T_max"] = trial.suggest_int("T_max", min_epoch_div, max_epoch_div)

    return params


def objective(trial, data_name):
    try:
        is_zero_pad = trial.suggest_categorical("is_zero_pad", [True, False])
        drugAct, null_mask, S_d, S_c, S_g, _, _, _, A_cg, A_dg = load_data(
            data_name, is_zero_pad=is_zero_pad
        )
        params = suggest_hyperparams(trial, S_d, S_c, S_g)

        all_edges, all_labels = get_all_edges_and_labels(drugAct, null_mask)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        true_datas = pd.DataFrame()
        predict_datas = pd.DataFrame()

        for train_idx, test_idx in kf.split(all_edges):
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

            _, true_labels, pred_probs, *_ = drGAT.train(
                sampler, params=params, device=device, verbose=False
            )

            true_datas = pd.concat(
                [true_datas, pd.DataFrame(true_labels).T], ignore_index=True
            )
            predict_datas = pd.concat(
                [predict_datas, pd.DataFrame(pred_probs).T], ignore_index=True
            )

        metrics = compute_metrics_stats(
            trial=trial,
            true=true_datas,
            pred=predict_datas,
            target_metrics=["AUROC", "AUPR", "F1", "ACC"],
        )
        return tuple(metrics["target_values"])

    except (ValueError, RuntimeError) as e:
        msg = str(e)
        if "NaN" in msg or "Input contains NaN" in msg:
            print(f"Pruned trial {trial.number}: NaN detected ({msg})")
            raise optuna.TrialPruned("NaN detected")
        if "CUDA out of memory" in msg:
            torch.cuda.empty_cache()
            gc.collect()
            print(f"Pruned trial {trial.number}: CUDA OOM")
            raise optuna.TrialPruned("CUDA OOM")
        raise e  # それ以外のエラーは通常通り上げる


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_name",
        type=str,
        required=True,
        choices=["gdsc1", "gdsc2", "ctrp", "nci"],
        help="Dataset name to use",
    )
    args = parser.parse_args()

    study = optuna.create_study(
        directions=["maximize"] * 4,
        sampler=optuna.samplers.NSGAIISampler(),
        pruner=optuna.pruners.HyperbandPruner(),
        storage=f"sqlite:///no_atten_{args.data_name}.sqlite3",
        study_name="MPNN_GCN",
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: objective(trial, args.data_name), n_trials=100, timeout=3600
    )
