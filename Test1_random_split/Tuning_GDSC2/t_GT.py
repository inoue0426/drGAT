import gc
import os
import sys
import warnings

import numpy as np
import optuna
import pandas as pd
import torch

warnings.simplefilter("ignore")
from sklearn.model_selection import KFold

current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from metrics import compute_metrics_stats

from drGAT import drGAT
from drGAT.load_data import load_data
from drGAT.sampler import RandomSampler

name = "gdsc2"
PATH = f"../{name}_data/"
method = "Transformer"

drugAct, pos_num, null_mask, S_d, S_c, S_g, A_cg, A_dg = load_data(name)


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
        "epochs": trial.suggest_int("epochs", 100, 10000, step=100),
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
        k = 5
        kfold = KFold(n_splits=k, shuffle=True, random_state=42)

        res = pd.DataFrame()

        metrics_collector = {"acc": [], "f1": [], "auroc": [], "aupr": []}

        true_datas = pd.DataFrame()
        predict_datas = pd.DataFrame()

        for train_index, test_index in kfold.split(np.arange(pos_num)):
            sampler = RandomSampler(
                drugAct,
                train_index,
                test_index,
                null_mask,
                S_d,
                S_c,
                S_g,
                A_cg,
                A_dg,
                PATH,
            )
            (_, _, _, best_val_labels, best_val_prob, best_metrics, _, _, _) = (
                drGAT.train(sampler, params=params, device=device, verbose=False)
            )

            true_datas = pd.concat(
                [true_datas, pd.DataFrame(best_val_labels)],
                ignore_index=True,
                axis=1,
            )
            predict_datas = pd.concat(
                [predict_datas, pd.DataFrame(best_val_prob)],
                ignore_index=True,
                axis=1,
            )

            del sampler, best_val_labels, best_val_prob
            torch.cuda.empty_cache()
            gc.collect()

        true_datas = true_datas.T
        predict_datas = predict_datas.T

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
    storage=f"sqlite:///{method}.sqlite3",
    study_name=method,
    load_if_exists=True,
)
study.optimize(objective, n_trials=1000)
