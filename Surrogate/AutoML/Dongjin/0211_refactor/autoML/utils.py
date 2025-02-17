import math
import pandas as pd
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    mean_squared_log_error,
    explained_variance_score
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc



def evaluate_regression(y_true, y_pred, dataset_name="Dataset"):
    dicts = {'R2': r2_score(y_true, y_pred),
             'MAE': mean_absolute_error(y_true, y_pred),
             'RMSE': math.sqrt(mean_squared_error(y_true, y_pred))}

    print(f"\nEvaluation for {dataset_name}:")
    print(f"R2 Score: {dicts['R2']:.4f}")
    print(f"Mean Absolute Error (MAE): {dicts['MAE']:.4f}")
    print(f"Root Mean Squared Error (RMSE): {dicts['RMSE']:.4f}")

    return dicts

def evaluate_classification(y_true, y_pred, dataset_name="Dataset"):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)

    dicts = {
        'ROC-AUC' : roc_auc_score(y_true, y_pred),
        'PR_AUC': pr_auc,
        'F1 Score': f1_score(y_true, y_pred),
        'F1 Score_Weighted': f1_score(y_true, y_pred, average='weighted'),
        'Precision': precision_score(y_true, y_pred),
        'Accuracy': accuracy_score(y_true, y_pred)
    }
    print(f"\nEvaluation for {dataset_name}:")
    print(f"F1 Score: {dicts['F1 Score']:.4f}")
    print(f"F1 Score_Weighted: {dicts['F1 Score_Weighted']:.4f}")
    print(f"ROC-AUC: {dicts['ROC-AUC']:.4f}")
    print(f"Precision: {dicts['Precision']:.4f}")
    print(f"Accuracy: {dicts['Accuracy']:.4f}")

    return dicts