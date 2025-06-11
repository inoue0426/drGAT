import argparse
import gc
import os
import sys
import time
from pathlib import Path

from joblib import Parallel, delayed

MAX_TRIAL_DURATION = 172800  # 48時間 = 2日（秒単位）

import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.model_selection import KFold
from tqdm import tqdm

# Add path to drGAT package (Singularity/Local 両対応)
if os.path.exists("/workspace/drGAT"):
    sys.path.append("/workspace")
else:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
    sys.path.append(parent_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from drGAT import drGAT
from drGAT.load_data import load_data
from drGAT.metrics import compute_metrics_stats
from drGAT.sampler import NewSampler
from drGAT.utility import filter_target

parser = argparse.ArgumentParser()
parser.add_argument(
    "--method", type=str, choices=["GAT", "GATv2", "Transformer"], default="GATv2"
)
parser.add_argument(
    "--data", type=str, choices=["gdsc1", "gdsc2", "ctrp", "nci"], default="nci"
)
parser.add_argument("--target_dim", type=int, choices=[0, 1], default=0)

parser.add_argument(
    "--n_jobs",
    type=int,
    default=3,
    help="Number of parallel jobs for target processing",
)

args = parser.parse_args()

method = args.method
data = args.data
target_dim = args.target_dim
n_jobs = args.n_jobs


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

    (_, _, _, best_val_labels, best_val_prob, best_metrics, _, _, _) = drGAT.train(
        sampler, params=params, device=device, verbose=False
    )

    return best_val_labels, best_val_prob


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
        # Data Load
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
        ) = load_data(data, is_zero_pad=is_zero_pad)

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

        # Compute Final Metrics
        metrics_result = compute_metrics_stats(
            trial=trial,
            true=true_datas,
            pred=predict_datas,
            target_metrics=["AUROC", "AUPR", "F1", "ACC"],
        )
        return tuple(metrics_result["target_values"])

    except Exception as e:
        handle_optuna_errors(e, trial)


# Create and run Optuna study
optuna_db_path = f"/workspace/Test2_leave_X_out/{method}_{data}_{'cell' if target_dim == 0 else 'drug'}.sqlite3"

study = optuna.create_study(
    directions=["maximize"] * 4,
    sampler=optuna.samplers.NSGAIISampler(),
    pruner=optuna.pruners.HyperbandPruner(),
    storage=f"sqlite:///{optuna_db_path}",
    study_name=method,
    load_if_exists=True,
)

study.optimize(objective, n_trials=1000)
