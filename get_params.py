import math

import optuna
import pandas as pd


def get_params(method, name):
    method = "GAT"
    name = "NCI_GAT_New"
    study = optuna.create_study(
        storage="sqlite:///../Tuning_nci/{}.sqlite3".format(name),
        study_name=name,
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
    return params
