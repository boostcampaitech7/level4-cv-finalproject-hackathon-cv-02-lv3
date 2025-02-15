import math
import pandas as pd
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    accuracy_score
)


def data_preparation(data_path, verbose=False):
    """
    데이터 전처리 및 학습/테스트 데이터 분리.
    
    Parameters:
        data_path (str): CSV 데이터 파일 경로
        verbose (bool): 데이터 로드 및 전처리 상태 출력 여부
        
    Returns:
        tuple: X_train, y_train, X_test, y_test
    """
    drop_columns = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]
    
    df = pd.read_csv(data_path).drop(columns=drop_columns).dropna()

    # 데이터셋 분리
    train_data = df[df['Split'] == 'Train'].drop(['Split'], axis=1)
    test_data = df[df['Split'] == 'Test'].drop(['Split'], axis=1)

    # 특성과 타겟 분리
    y_train = train_data.pop('Attrition')
    y_test = test_data.pop('Attrition')
    X_train, X_test = train_data, test_data

    if verbose:
        print(f"Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")
        print(f"Missing Values in X_train: {X_train.isnull().sum().sum()}")
        print(f"Missing Values in y_train: {y_train.isnull().sum()}")
    
    return X_train, y_train, X_test, y_test


def evaluate_regression(y_true, y_pred, dataset_name="Dataset", verbose=True):
    """
    회귀 모델 평가.
    
    Parameters:
        y_true (array-like): 실제 값
        y_pred (array-like): 예측 값
        dataset_name (str): 데이터셋 이름
        verbose (bool): 출력 여부
        
    Returns:
        dict: 평가 지표 (R2, MAE, RMSE)
    """
    metrics = {
        'R2': r2_score(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': math.sqrt(mean_squared_error(y_true, y_pred))
    }

    if verbose:
        print(f"\n[Regression Evaluation - {dataset_name}]")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
    
    return metrics


def evaluate_classification(y_true, y_pred, dataset_name="Dataset", verbose=True):
    """
    분류 모델 평가.
    
    Parameters:
        y_true (array-like): 실제 값
        y_pred (array-like): 예측 값
        dataset_name (str): 데이터셋 이름
        verbose (bool): 출력 여부
        
    Returns:
        dict: 평가 지표 (F1 Score, AUC, Precision, Accuracy)
    """
    metrics = {
        'F1 Score': f1_score(y_true, y_pred),
        'F1 Score (Weighted)': f1_score(y_true, y_pred, average='weighted'),
        'Precision': precision_score(y_true, y_pred),
        'Accuracy': accuracy_score(y_true, y_pred)
    }

    # AUC 계산 시 예외 처리 (이진 분류가 아닌 경우)
    try:
        metrics['AUC'] = roc_auc_score(y_true, y_pred)
    except ValueError:
        metrics['AUC'] = None

    if verbose:
        print(f"\n[Classification Evaluation - {dataset_name}]")
        for metric, value in metrics.items():
            if value is not None:
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: N/A")
    
    return metrics
