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

from joblib import Parallel, delayed

from drGAT import No_atten_drGAT as drGAT
from drGAT.load_data import load_data
from drGAT.metrics import compute_metrics_stats
from drGAT.sampler import NewSampler
from drGAT.utility import filter_target

n_jobs = 1


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


def handle_optuna_errors(e, trial):
    msg = str(e)
    if "CUDA out of memory" in msg:
        print(f"Pruned trial {trial.number}: CUDA OOM")
        with torch.cuda.device("cuda"):
            torch.cuda.empty_cache()
        gc.collect()
        raise optuna.TrialPruned(f"OOM at trial {trial.number}")
    elif "Input contains NaN" in msg:
        print(f"Pruned trial {trial.number}: Input contains NaN")
        raise optuna.TrialPruned(f"NaN input at trial {trial.number}")
    elif isinstance(e, ZeroDivisionError):
        print(f"Pruned trial {trial.number}: ZeroDivisionError in CosineAnnealingLR")
        raise optuna.TrialPruned("ZeroDivisionError in CosineAnnealingLR")
    else:
        print(f"Unexpected error in trial {trial.number}: {msg}")
        raise e


def drGAT_new(
    res,
    null_mask,
    target_dim,
    target_index,
    S_d,
    S_c,
    S_g,
    A_cg,
    A_dg,
    params,
    device,
):
    sampler = NewSampler(
        res,
        null_mask,
        target_dim,
        target_index,
        S_d,
        S_c,
        S_g,
        A_cg,
        A_dg,
    )

    _, true_labels, pred_probs, *_ = drGAT.train(
        sampler, params=params, device=device, verbose=False
    )
    return true_labels, pred_probs


def objective(trial, data_name, data_type):
    target_dim = 1 if data_type == "cell" else 0
    try:
        is_zero_pad = trial.suggest_categorical("is_zero_pad", [True, False])
        (
            res,
            null_mask,
            S_d,
            S_c,
            S_g,
            drug_feature,
            gene_norm_gene,
            gene_norm_cell,
            A_cg,
            A_dg,
        ) = load_data(data_name, is_zero_pad=is_zero_pad)

        # Suggest Hyperparameters
        params = suggest_hyperparams(trial, S_d, S_c, S_g)

        def run_single_target(target_index):
            true_data, predict_data = drGAT_new(
                res=res,
                null_mask=null_mask.values,
                target_dim=target_dim,
                target_index=target_index,
                S_d=S_d,
                S_c=S_c,
                S_g=S_g,
                A_cg=A_cg,
                A_dg=A_dg,
                params=params,
                device=device,
            )
            return true_data, predict_data

        samples = res.shape[target_dim]

        passed_targets = []
        skipped_targets = []

        for target_index in range(samples):
            label_vec = (
                res.iloc[target_index] if target_dim == 0 else res.iloc[:, target_index]
            )
            passed, reason, pos, neg, total = filter_target(label_vec)

            if passed:
                passed_targets.append(target_index)
            else:
                skipped_targets.append((target_index, reason, pos, neg, total))

        # Display skipped targets
        print(f"\n🚫 Skipped Targets: {len(skipped_targets)}")
        for idx, reason, pos, neg, total in skipped_targets:
            print(
                f"Target {idx}: skipped because {reason} (total={total}, pos={pos}, neg={neg})"
            )

        # Joblib並列実行（注意: GPUメモリに余裕がないなら n_jobs=1）
        try:
            results = Parallel(n_jobs=n_jobs)(
                delayed(run_single_target)(target_index)
                for target_index in tqdm(passed_targets)
            )
        except Exception as e:
            handle_optuna_errors(e, trial)

        # 結果を統合
        true_datas = pd.DataFrame()
        predict_datas = pd.DataFrame()

        for true_data, predict_data in results:
            true_datas = pd.concat(
                [true_datas, pd.DataFrame(true_data).T], ignore_index=True
            )
            predict_datas = pd.concat(
                [predict_datas, pd.DataFrame(predict_data).T], ignore_index=True
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
    parser.add_argument(
        "--data_type",
        type=str,
        required=True,
        choices=["cell", "drug"],
        help="Specify whether the task is cell or drug prediction",
    )

    args = parser.parse_args()

    storage_name = f"sqlite:///no_atten_{args.data_name}_{args.data_type}.sqlite3"

    study = optuna.create_study(
        directions=["maximize"] * 4,
        sampler=optuna.samplers.NSGAIISampler(),
        pruner=optuna.pruners.HyperbandPruner(),
        storage=storage_name,
        study_name="MPNN_GCN",
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: objective(trial, args.data_name, args.data_type),
        n_trials=100,
        timeout=3600,
    )
