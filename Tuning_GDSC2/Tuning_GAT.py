# type: ignore
# ruff: noqa
import gc
import os
import sys

import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.model_selection import KFold

current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")
from drGAT import drGAT
from drGAT.load_data import load_data
from drGAT.sampler import RandomSampler

drugAct, pos_num, null_mask, S_d, S_c, S_g, A_cg, A_dg = load_data("gdsc2")

PATH = "../gdsc2_data/"


def objective(trial):
    params = {
        "n_drug": S_d.shape[0],
        "n_cell": S_c.shape[0],
        "n_gene": S_g.shape[0],
        "dropout1": trial.suggest_categorical("dropout1", [0.1, 0.2, 0.3, 0.4, 0.5]),
        "dropout2": trial.suggest_categorical("dropout2", [0.1, 0.2, 0.3, 0.4, 0.5]),
        "hidden1": trial.suggest_categorical(
            "hidden1",
            [256, 512, 1028],
        ),
        "hidden2": trial.suggest_categorical(
            "hidden2",
            [
                128,
                256,
                512,
            ],
        ),
        "hidden3": trial.suggest_categorical(
            "hidden3",
            [
                64,
                128,
                256,
            ],
        ),
        "epochs": trial.suggest_categorical("epochs", [10, 50, 100, 200, 500]),
        "heads": trial.suggest_categorical("heads", [1, 2, 3, 4, 5]),
        "activation": trial.suggest_categorical(
            "activation", ["relu", "gelu", "swish"]
        ),
        "optimizer": trial.suggest_categorical("optimizer", ["Adam", "AdamW", "SGD"]),
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "scheduler": trial.suggest_categorical(
            "scheduler", [None, "Cosine", "Step", "Plateau"]
        ),
        "gnn_layer": trial.suggest_categorical(
            "gnn_layer", ["GAT", "GATv2", "Transformer"]
        ),
    }

    # スケジューラ関連パラメータの条件付き追加
    if params["scheduler"] == "Cosine":
        params["T_max"] = trial.suggest_int("T_max", 20, 50)
    elif params["scheduler"] == "Step":
        params["scheduler_gamma"] = trial.suggest_float("gamma_step", 0.1, 0.95)
        params["step_size"] = trial.suggest_int("step_size", 10, 30)
    elif params["scheduler"] == "Plateau":
        params["patience"] = trial.suggest_int("patience_plateau", 3, 10)
        params["threshold"] = trial.suggest_float(
            "thresh_plateau", 1e-4, 1e-2, log=True
        )

    if params["hidden1"] < params["hidden2"] or params["hidden2"] < params["hidden3"]:
        raise optuna.TrialPruned("Invalid layer size configuration")

    if params["optimizer"] in ["Adam", "AdamW"]:
        params["amsgrad"] = trial.suggest_categorical("amsgrad", [True, False])

    if params["optimizer"] == "SGD":
        params["momentum"] = trial.suggest_float("momentum", 0.8, 0.95)
        params["nesterov"] = trial.suggest_categorical("nesterov", [True, False])

    # 隠れ層サイズとバッチサイズの関係を制約
    if (params["hidden1"] > 512) and (params["hidden2"] > 256):
        raise optuna.TrialPruned("Memory intensive configuration")

    try:
        k = 5
        kfold = KFold(n_splits=k, shuffle=True, random_state=42)

        res = pd.DataFrame()
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
            (_, _, _, best_metrics, _, _, _) = drGAT.train(
                sampler, params=params, device=device, verbose=False
            )

            res = pd.concat(
                [
                    res,
                    pd.DataFrame(best_metrics, index=["acc", "f1", "auroc", "aupr"]).T,
                ]
            )

        return [float(i) for i in res.mean()]

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(f"Pruned trial {trial.number}: CUDA OOM")

            # メモリ解放処理
            with torch.cuda.device("cuda"):
                torch.cuda.empty_cache()
            gc.collect()

            # Pruning通知
            raise optuna.TrialPruned(f"OOM at trial {trial.number}")

        else:
            raise e


name = "GDSC2_GAT"
study = optuna.create_study(
    directions=["maximize"] * 4,
    sampler=optuna.samplers.TPESampler(),
    pruner=optuna.pruners.HyperbandPruner(),
    storage="sqlite:///{}.sqlite3".format(name),
    study_name=name,
    load_if_exists=True,
)
study.optimize(objective, n_trials=100)
