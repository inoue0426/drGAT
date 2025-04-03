import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score, log_loss,
    brier_score_loss, matthews_corrcoef, cohen_kappa_score,
    balanced_accuracy_score, fbeta_score
)

def get_result(true, pred, data):
    res = pd.DataFrame()
    for i in range(5):
        true_labels = true.loc[i].dropna()
        pred_values = pred.loc[i].dropna()
        pred_labels = np.round(pred_values)
        
        # 主要メトリクスの計算
        metrics = {
            # 基本指標
            "ACC": accuracy_score(true_labels, pred_labels),
            "Precision": precision_score(true_labels, pred_labels, zero_division=0),
            "Recall": recall_score(true_labels, pred_labels, zero_division=0),
            "Specificity": recall_score(true_labels, pred_labels, pos_label=0, zero_division=0),
            "F1": f1_score(true_labels, pred_labels, zero_division=0),
            "F2": fbeta_score(true_labels, pred_labels, beta=2, zero_division=0),
            
            # 不均衡データ向け
            "Balanced ACC": balanced_accuracy_score(true_labels, pred_labels),
            "G-Mean": np.sqrt(recall_score(true_labels, pred_labels, zero_division=0) *
                            recall_score(true_labels, pred_labels, pos_label=0, zero_division=0)),
            
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
            "Cost Ratio": (precision_score(true_labels, pred_labels, zero_division=0) /
                         (1 - recall_score(true_labels, pred_labels, zero_division=0)))
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