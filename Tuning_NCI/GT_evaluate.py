import gc
import math
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

from drGAT import drGAT
from drGAT.load_data import load_data
from drGAT.sampler import RandomSampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create and load optuna study
name = "NCI"
method = "Transformer"  # 任意のstudy名
storage_path = f"./{method}.sqlite3"  # "./subdir"は1階層下のディレクトリ
study = optuna.create_study(
    storage=f"sqlite:///{storage_path}",
    study_name=method,
    load_if_exists=True,
)
df = study.trials_dataframe()
df = df.dropna(subset=[i for i in df.columns if "values" in i])
tmp = df.loc[
    df[["values_0", "values_1", "values_2", "values_3"]]
    .max(axis=1)
    .sort_values(ascending=False)
    .index
]
params = tmp[tmp["params_gnn_layer"] == method].head().iloc[0]
print(params)
params = {
    i.replace("params_", ""): j
    for i, j in zip(pd.DataFrame(params).index, params)
    if "params" in i
}

# Load data
drugAct, pos_num, null_mask, S_d, S_c, S_g, A_cg, A_dg = load_data(data)


def auto_convert_params(params, nan_replace=None):
    """Convert parameter types automatically

    Args:
        params (dict): Parameter dictionary before conversion
        nan_replace: Replacement value for NaN (default None)

    Returns:
        dict: Parameter dictionary after type conversion
    """
    converted = {}
    for k, v in params.items():
        if isinstance(v, float) and math.isnan(v):
            converted[k] = nan_replace
        elif isinstance(v, float) and v.is_integer():
            converted[k] = int(v)
        else:
            converted[k] = v
    return converted


params = auto_convert_params(params, nan_replace=0)

params.update(
    {
        "n_drug": S_d.shape[0],
        "n_cell": S_c.shape[0],
        "n_gene": S_g.shape[0],
        "epochs": 1000,
    }
)

print(params)

PATH = f"../{data}_data/"

# K-fold cross validation
k = 5
kfold = KFold(n_splits=k, shuffle=True, random_state=42)

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
    (_, _, _, best_val_labels, best_val_prob, best_metrics, _, _, _) = drGAT.train(
        sampler, params=params, device=device, verbose=False
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

true_datas.to_csv(f"true_{data}_{method}_2.csv")
predict_datas.to_csv(f"pred_{data}_{method}_2.csv")
