import argparse
import gc
import os
import sys

import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.model_selection import KFold
from tqdm import tqdm

current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from drGT import drGT
from drGT.load_data import load_data
from drGT.metrics import compute_metrics_stats
from drGT.myutils import get_all_edges_and_labels
from drGT.sampler import BalancedSampler

parser = argparse.ArgumentParser()
parser.add_argument(
    "--method", type=str, choices=["GAT", "GATv2", "Transformer"], default="GATv2"
)
parser.add_argument(
    "--data", type=str, choices=["gdsc1", "gdsc2", "ctrp", "nci"], default="nci"
)
args = parser.parse_args()

method = args.method
data = args.data


def suggest_hyperparams(trial, S_d, S_c, S_g):
    hidden1 = trial.suggest_int("hidden1", 256, 512)
    hidden2 = trial.suggest_int("hidden2", 64, min(256, hidden1))
    hidden3 = trial.suggest_int("hidden3", 32, min(128, hidden2))

    final_mlp_layers = trial.suggest_int("final_mlp_layers", 1, 3)

    params = {
        "n_drug": S_d.shape[0],
        "n_cell": S_c.shape[0],
        "n_gene": S_g.shape[0],
        "dropout1": trial.suggest_float("dropout1", 0.1, 0.5, step=0.1),
        "dropout2": trial.suggest_float("dropout2", 0.1, 0.5, step=0.1),
        "dropout3": (
            trial.suggest_float("dropout3", 0.1, 0.5, step=0.1)
            if final_mlp_layers >= 2
            else 0.0
        ),
        "hidden1": hidden1,
        "hidden2": hidden2,
        "hidden3": hidden3,
        "epochs": trial.suggest_int("epochs", 300, 1000, step=100),
        "heads": trial.suggest_int("heads", 2, 8),
        "activation": trial.suggest_categorical("activation", ["relu", "gelu"]),
        "optimizer": trial.suggest_categorical("optimizer", ["Adam", "AdamW"]),
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "scheduler": trial.suggest_categorical("scheduler", [None, "Cosine"]),
        "norm_type": trial.suggest_categorical(
            "norm_type", ["GraphNorm", "BatchNorm", "LayerNorm"]
        ),
        "n_layers": trial.suggest_int("n_layers", 2, 4),
        "gnn_layer": method,
        "final_mlp_layers": final_mlp_layers,
        # "residual": trial.suggest_categorical("residual", [True, False]),
        "attention_dropout": trial.suggest_float(
            "attention_dropout", 0.0, 0.4, step=0.1
        ),
    }

    if params["scheduler"] == "Cosine":
        min_epoch_div = max(1, params["epochs"] // 5)
        max_epoch_div = max(min_epoch_div + 1, params["epochs"] // 2)
        params["T_max"] = trial.suggest_int(
            "T_max", low=min_epoch_div, high=max_epoch_div
        )

        if params["T_max"] <= 0:
            raise optuna.TrialPruned(f"Invalid T_max: {params['T_max']}")

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


def objective(trial):
    try:
        is_zero_pad = trial.suggest_categorical("is_zero_pad", [True, False])
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
        ) = load_data(data, is_zero_pad=is_zero_pad)

        params = suggest_hyperparams(trial, S_d, S_c, S_g)

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
                _,
                _,
                _,
                true_data,
                predict_data,
                _,
                _,
                _,
                _,
            ) = drGT.train(sampler, params=params, device=device, verbose=False)

            true_datas = pd.concat(
                [true_datas, pd.DataFrame(true_data).T], ignore_index=True
            )
            predict_datas = pd.concat(
                [predict_datas, pd.DataFrame(predict_data).T], ignore_index=True
            )

        metrics_result = compute_metrics_stats(
            trial=trial,
            true=true_datas,
            pred=predict_datas,
            target_metrics=["AUROC", "AUPR", "F1", "ACC"],
        )
        return tuple(metrics_result["target_values"])

    except Exception as e:
        handle_optuna_errors(e, trial)


if __name__ == "__main__":
    study = optuna.create_study(
        directions=["maximize"] * 4,
        sampler=optuna.samplers.NSGAIISampler(),
        pruner=optuna.pruners.HyperbandPruner(),
        storage=f"sqlite:///{method}_{data}.sqlite3",
        study_name=method,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=1000)
