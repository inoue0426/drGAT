import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, brier_score_loss,
                             cohen_kappa_score, f1_score, fbeta_score,
                             log_loss, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)


def evaluate_predictions(true_labels, pred_probs, threshold=0.5):
    pred_labels = (pred_probs >= threshold).astype(int)

    metrics = {
        "accuracy": round(accuracy_score(true_labels, pred_labels), 4),
        "f1_score": round(f1_score(true_labels, pred_labels), 4),
        "auroc": round(roc_auc_score(true_labels, pred_probs), 4),
        "aupr": round(average_precision_score(true_labels, pred_probs), 4),
    }

    # Format and display
    df = pd.DataFrame(metrics, index=["Score"]).T
    df.index.name = "Metric"
    display(df)


def get_result(true, pred, data):
    res = pd.DataFrame()
    for i in range(true.shape[0]):
        true_labels = true.loc[i].dropna()
        pred_values = pred.loc[i].dropna()

        pred_labels = np.round(pred_values)

        # 主要メトリクスの計算
        metrics = {
            # 基本指標
            "ACC": accuracy_score(true_labels, pred_labels),
            "Precision": precision_score(true_labels, pred_labels, zero_division=0),
            "Recall": recall_score(true_labels, pred_labels, zero_division=0),
            "Specificity": recall_score(
                true_labels, pred_labels, pos_label=0, zero_division=0
            ),
            "F1": f1_score(true_labels, pred_labels, zero_division=0),
            "F2": fbeta_score(true_labels, pred_labels, beta=2, zero_division=0),
            # 不均衡データ向け
            "Balanced ACC": balanced_accuracy_score(true_labels, pred_labels),
            "G-Mean": np.sqrt(
                recall_score(true_labels, pred_labels, zero_division=0)
                * recall_score(true_labels, pred_labels, pos_label=0, zero_division=0)
            ),
            # 確率ベース
            "AUROC": roc_auc_score(true_labels, pred_values),
            "AUPR": average_precision_score(true_labels, pred_values),
            "LogLoss": log_loss(true_labels, pred_values),
            "Brier": brier_score_loss(true_labels, pred_values),
            # 一致度指標
            "MCC": matthews_corrcoef(true_labels, pred_labels),
            "Kappa": cohen_kappa_score(true_labels, pred_labels),
            # コスト考慮型
            "Youden J": roc_auc_score(true_labels, pred_values) * 2 - 1,
            "Cost Ratio": (
                precision_score(true_labels, pred_labels, zero_division=0)
                / (1 - recall_score(true_labels, pred_labels, zero_division=0))
            ),
        }
        res = pd.concat([res, pd.DataFrame([metrics])])

    # 統計量の計算
    means = res.mean()
    stds = res.std()

    # フォーマット整形
    formatted = means.map("{:.3f}".format) + " (± " + stds.map("{:.3f}".format) + ")"

    result_table = pd.DataFrame({data.upper(): formatted.values})
    result_table.index = means.index

    return result_table


def compute_metrics_stats(true, pred, trial=None, data=None, target_metrics=None):
    """
    Optuna統合用メトリクス計算関数

    Parameters:
    trial (optuna.Trial): Optunaトライアルオブジェクト
    true (pd.DataFrame): 正解ラベルのDataFrame
    pred (pd.DataFrame): 予測値のDataFrame
    data (str): データセット識別子（任意）
    target_metrics (list): 最適化対象のメトリクスリスト

    Returns:
    dict: メトリクス統計量の辞書
    """
    # デフォルト対象メトリクス
    if target_metrics is None:
        target_metrics = ["ACC", "F1", "AUROC", "AUPR", "MCC"]

    res = pd.DataFrame()
    for i in range(true.shape[0]):
        # データ前処理
        true_labels = true.loc[i].dropna()
        pred_values = pred.loc[i].dropna()

        assert len(true_labels) == len(
            pred_values
        ), f"Mismatch: {len(true_labels)} vs {len(pred_values)}"

        pred_labels = np.round(pred_values).astype(int)

        # メトリクス計算
        metrics = {
            "ACC": accuracy_score(true_labels, pred_labels),
            "Precision": precision_score(true_labels, pred_labels, zero_division=0),
            "Recall": recall_score(true_labels, pred_labels, zero_division=0),
            "F1": f1_score(true_labels, pred_labels, zero_division=0),
            "AUROC": roc_auc_score(true_labels, pred_values),
            "AUPR": average_precision_score(true_labels, pred_values),
            "MCC": matthews_corrcoef(true_labels, pred_labels),
            "Specificity": recall_score(
                true_labels, pred_labels, pos_label=0, zero_division=0
            ),
            "Balanced_ACC": balanced_accuracy_score(true_labels, pred_labels),
            "LogLoss": log_loss(true_labels, pred_values),
            "Brier": brier_score_loss(true_labels, pred_values),
        }
        res = pd.concat([res, pd.DataFrame([metrics])], ignore_index=True)

    # 統計量計算
    stats = {"means": res.mean().to_dict(), "stds": res.std().to_dict()}

    if trial is not None:
        # Optunaへの結果保存
        for metric in stats["means"]:
            # ユーザー属性として保存
            trial.set_user_attr(f"{metric}_mean", float(stats["means"][metric]))
            trial.set_user_attr(f"{metric}_std", float(stats["stds"][metric]))

            if metric in target_metrics:
                print(f"{metric}_mean: {stats['means'][metric]:.4f}")
                print(f"{metric}_std: {stats['stds'][metric]:.4f}")

    # フォーマット済み結果（オプション）
    formatted_stats = {
        metric: f"{stats['means'][metric]:.3f} (±{stats['stds'][metric]:.3f})"
        for metric in stats["means"]
    }

    return {
        "raw": stats,
        "formatted": formatted_stats,
        "target_values": [stats["means"][m] for m in target_metrics],
    }


def get_parsed_df(df):
    parsed_data = {}
    for column in df.columns:
        if "_mean" in column:
            base_name = column.replace("_mean", "")
            std_column = base_name + "_std"
            if std_column in df.columns:
                parsed_data[base_name] = (
                    round(df[column], 3).astype(str)
                    + " (± "
                    + round(df[std_column], 3).astype(str)
                    + ")"
                )

    return pd.DataFrame(parsed_data)[
        ["ACC", "AUPR", "AUROC", "Balanced_ACC", "F1", "Recall", "Precision"]
    ]
