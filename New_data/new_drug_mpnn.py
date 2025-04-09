# mypy: ignore-errors
# ruff: noqa
import gc
import math
import os
import sys
import warnings

import numpy as np
import optuna
import pandas as pd
import torch
from tqdm import tqdm

warnings.simplefilter("ignore")
from sklearn.model_selection import KFold

current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from drGAT import No_atten_drGAT
from drGAT.load_data import load_data
from drGAT.sampler import NewSampler
from get_params import get_params
from metrics import compute_metrics_stats

name = "gdsc2"
if name == "nci":
    task = "cell"
else:
    task = "drug"


method = "MPNN"
PATH = f"../{name}_data/"

if name == "nci":
    target_dim = [
        0,  # Cell
        # 1  # Drug
    ]
else:
    target_dim = [
        1,  # Drug
        # 0  # Cell
    ]

(
    res,
    pos_num,
    null_mask,
    S_d,
    S_c,
    S_g,
    A_cg,
    A_dg,
    _,
    _,
    _,
) = load_data(name)
res = res.T
cell_sum = np.sum(res, axis=1)
drug_sum = np.sum(res, axis=0)


def drGAT_new(
    res_mat,
    null_mask,
    target_dim,
    target_index,
    S_d,
    S_c,
    S_g,
    A_cg,
    A_dg,
    PATH,
    params,
    device,
    seed,
):
    sampler = NewSampler(
        res_mat,
        null_mask,
        target_dim,
        target_index,
        S_d,
        S_c,
        S_g,
        A_cg,
        A_dg,
        PATH,
        seed,
    )

    (_, best_val_labels, best_val_prob, best_metrics, _, _, _) = No_atten_drGAT.train(
        sampler, params=params, device=device, verbose=False
    )

    return best_val_labels, best_val_prob


def objective(trial):
    params = {
        "n_drug": S_d.shape[0],
        "n_cell": S_c.shape[0],
        "n_gene": S_g.shape[0],
        "dropout1": trial.suggest_float("dropout1", 0.1, 0.5, step=0.05),
        "dropout2": trial.suggest_float("dropout2", 0.1, 0.5, step=0.05),
        "hidden1": trial.suggest_int("hidden1", 256, 1024),
        "hidden2": trial.suggest_int("hidden2", 64, min(512, trial.params["hidden1"])),
        "hidden3": trial.suggest_int("hidden3", 32, min(256, trial.params["hidden2"])),
        "epochs": 100,
        # trial.suggest_int("epochs", 100, 10000, step=100),
        "heads": trial.suggest_int("heads", 2, 8),
        "activation": trial.suggest_categorical("activation", ["relu", "gelu"]),
        "optimizer": trial.suggest_categorical("optimizer", ["Adam", "AdamW"]),
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "scheduler": trial.suggest_categorical("scheduler", [None, "Cosine"]),
        "gnn_layer": method,
    }

    # スケジューラ関連パラメータの条件付き追加
    if params["scheduler"] == "Cosine":
        # T_maxの最小値を1以上に保証
        min_epoch_div = max(1, params["epochs"] // 5)  # 最小値1を強制
        max_epoch_div = max(
            min_epoch_div + 1, params["epochs"] // 2
        )  # low < highを保証

        params["T_max"] = trial.suggest_int(
            "T_max", low=min_epoch_div, high=max_epoch_div
        )

        # 追加のチェック（防御的プログラミング）
        if params["T_max"] <= 0:
            raise optuna.TrialPruned(f"Invalid T_max: {params['T_max']}")

    try:
        n_kfold = 1
        true_datas = pd.DataFrame()
        predict_datas = pd.DataFrame()
        for dim in target_dim:
            for seed, target_index in tqdm(enumerate(np.arange(res.shape[dim]))):
                if dim:
                    if drug_sum[target_index] < 10:
                        continue
                else:
                    if cell_sum[target_index] < 10:
                        continue

                true_datas = pd.DataFrame()
                true_datas = pd.DataFrame()
                for fold in range(n_kfold):
                    true_data, predict_data = drGAT_new(
                        res_mat=res,
                        null_mask=null_mask.T.values,
                        target_dim=dim,
                        target_index=target_index,
                        S_d=S_d,
                        S_c=S_c,
                        S_g=S_g,
                        A_cg=A_cg,
                        A_dg=A_dg,
                        PATH=PATH,
                        params=params,
                        device=device,
                        seed=seed,
                    )

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

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(f"Pruned trial {trial.number}: CUDA OOM")

            with torch.cuda.device("cuda"):
                torch.cuda.empty_cache()
            gc.collect()

            raise optuna.TrialPruned(f"OOM at trial {trial.number}")
        else:
            print(f"RuntimeError in trial {trial.number}: {str(e)}")
            raise e

    except ValueError as e:
        if "Input contains NaN" in str(e):
            print(f"Pruned trial {trial.number}: Input contains NaN")
            raise optuna.TrialPruned(f"NaN input at trial {trial.number}")
        else:
            print(f"ValueError in trial {trial.number}: {str(e)}")
            raise e

    except ZeroDivisionError:
        print(f"Pruned trial {trial.number}: ZeroDivisionError in CosineAnnealingLR")
        raise optuna.TrialPruned("ZeroDivisionError in CosineAnnealingLR")
    except Exception as e:
        print(f"Unexpected error in trial {trial.number}: {str(e)}")
        raise e


study = optuna.create_study(
    directions=["maximize"] * 4,
    sampler=optuna.samplers.NSGAIISampler(),
    pruner=optuna.pruners.HyperbandPruner(),
    storage=f"sqlite:///{method}_{task}_small.sqlite3",
    study_name=method,
    load_if_exists=True,
)
study.optimize(objective, n_trials=1000)
