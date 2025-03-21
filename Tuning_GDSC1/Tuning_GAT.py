import gc
import os
import sys

import numpy as np
import optuna
import pandas as pd
import torch

current_dir = os.getcwd()  # noqa: E402
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))  # noqa: E402
sys.path.append(parent_dir)  # noqa: E402

from drGAT import drGAT  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data = pd.read_csv("../GDSC1_data/train.csv")
val_data = pd.read_csv("../GDSC1_data/val.csv")
test_data = pd.read_csv("../GDSC1_data/test.csv")

idxs = np.load("../GDSC1_data/idxs.npy", allow_pickle=True)
converter = {idxs[1, i]: int(idxs[0, i]) for i in range(idxs.shape[1])}

edge_index = np.load("../GDSC1_data/edge_idxs.npy")
edge_attr = np.load("../GDSC1_data/edge_attr.npy")


def get_idx(X):
    X["Drug"] = [converter[(i)] for i in X["Drug"]]
    X["Cell"] = [converter[(i)] for i in X["Cell"]]
    return X


train_data = get_idx(train_data)
val_data = get_idx(val_data)
test_data = get_idx(test_data)

edge_index = torch.tensor(edge_index).int()
edge_index = edge_index.type(torch.int64)

edge_attr = torch.tensor(edge_attr).float()

train_drug = train_data["Drug"].values
train_cell = train_data["Cell"].values
val_drug = val_data["Drug"].values
val_cell = val_data["Cell"].values

train_labels = np.load("../GDSC1_data/train_labels.npy")
val_labels = np.load("../GDSC1_data/val_labels.npy")

train_labels = torch.tensor(train_labels).float()
val_labels = torch.tensor(val_labels).float()

drug = pd.read_csv("../GDSC1_data/drug_sim.csv", index_col=0)
cell = pd.read_csv("../GDSC1_data/cell_sim.csv", index_col=0)
gene = pd.read_csv("../GDSC1_data/gene_sim.csv", index_col=0)

drug = torch.tensor(drug.values).float()
cell = torch.tensor(cell.values).float()
gene = torch.tensor(gene.values).float()

data = [
    drug,
    cell,
    gene,
    edge_index,
    edge_attr,
    train_drug,
    train_cell,
    val_drug,
    val_cell,
    train_labels,
    val_labels,
]


def objective(trial):
    params = {
        "n_drug": drug.shape[0],
        "n_cell": cell.shape[0],
        "n_gene": gene.shape[0],
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
        "epochs": trial.suggest_int("epochs", 10, 200, step=50),
        "heads": trial.suggest_categorical("heads", [1, 2, 3]),
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

    if (params["hidden1"] > 512) and (params["hidden2"] > 256):
        raise optuna.TrialPruned("Memory intensive configuration")

    try:
        _, _, _, best_metrics, early_stopping_epoch = drGAT.train(
            data, params=params, device=device, verbose=False
        )
        print("#####")
        print(best_metrics)
        print("#####")

        early_stop_threshold = trial.suggest_float("early_stop_threshold", 0.3, 0.7)
        if (
            early_stopping_epoch is not None
            and early_stopping_epoch < params["epochs"] * early_stop_threshold
        ):
            raise optuna.TrialPruned("Early stopping occurred too early")

        trial.set_user_attr("early_stopping_epoch", early_stopping_epoch)
        return best_metrics

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("CUDA out of memory")
            trial.set_user_attr("status", "CUDA OOM")

            torch.cuda.empty_cache()
            gc.collect()

            return [float("-inf")] * 4
        else:
            raise e


name = "GDSC1_GAT"
study = optuna.create_study(
    directions=["maximize"] * 4,
    sampler=optuna.samplers.TPESampler(),
    pruner=optuna.pruners.HyperbandPruner(),
    storage="sqlite:///{}.sqlite3".format(name),
    study_name=name,
    load_if_exists=True,
)
study.optimize(objective, n_trials=200, n_jobs=5)
