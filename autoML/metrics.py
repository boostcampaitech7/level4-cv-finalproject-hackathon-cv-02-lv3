from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, f1_score, mean_absolute_error
from collections import defaultdict
import statistics
import math


def evaluate_regression(X, y_true, y_pred, verbose=False):
    """
    회귀 문제의 성능을 평가하는 함수로, r2, adjusted_r2, MAE, RMSE를 계산

    Args:
        y_true: 실제 값
        y_pred: 예측된 값

    Returns:
        dict: r2, adjusted_r2, MAE, RMSE를 포함하는 딕셔너리
    """
    n = len(y_pred)
    k = X.shape[1]
    r2 = r2_score(y_true, y_pred)
    
    if k+1 < n: # sample 수가 부족하면 adjusted r2 계산 X
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
    else:
        adjusted_r2 = -100

    dicts = {'r2': r2,
             'adjusted_r2': adjusted_r2,
             'MAE': mean_absolute_error(y_true, y_pred),
             'RMSE': math.sqrt(mean_squared_error(y_true, y_pred))}
    return dicts

def evaluate_classification(y_true, y_pred):
    """
    분류 문제의 성능을 평가하는 함수로, F1 점수와 정확도(accuracy)를 계산

    Args:
        y_true: 실제 값
        y_pred: 예측된 값

    Returns:
        dict: F1 점수아 정확도를 포함하는 딕셔너리
    """
    
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    dicts = {'f1': f1, 'accuracy': accuracy}
    return dicts

def compute_metrics_statistics(metrics):
    """
    각 메트릭의 평균과 표준 편차를 계산

    Args:
        metrics (List[Dict[str, float]]): 여러 번의 측정에서 얻은 메트릭 딕셔너리들의 리스트.
                                          각 딕셔너리는 메트릭 이름을 키로 하며, 그 값은 측정된 값.

    Returns:
        Dict[str, float]: 각 메트릭의 평균과 표준 편차를 포함하는 딕셔너리.
                          메트릭 이름으로 평균 값이, '메트릭_std' 이름으로 표준 편차를 저장.
    """
    
    # 모든 메트릭 딕셔너리를 순회하며 값을 그룹화
    grouped_metrics = defaultdict(list)
    for metric in metrics:
        for key, value in metric.items():
            grouped_metrics[key].append(value)
    
    metric_statistics = {} # 평균과 표준 편차를 저장할 딕셔너리
    for key, values in grouped_metrics.items():
        metric_statistics[key] = statistics.mean(values)
        
        # 표준 편차 계산: 값이 하나일 경우 0 처리
        metric_statistics[f'{key}_std'] = statistics.stdev(values) if len(values) > 1 else 0

    return metric_statistics
