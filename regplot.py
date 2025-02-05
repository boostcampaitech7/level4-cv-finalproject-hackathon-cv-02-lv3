import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# PDP + 신뢰구간을 계산하는 함수
def partial_dependence_with_confidence(model, X, feature_name, grid_points=50):
    feature_idx = X.columns.get_loc(feature_name)  # 선택한 feature의 index
    feature_values = np.linspace(X[feature_name].min(), X[feature_name].max(), grid_points)  # Feature 값을 일정 간격으로 나눔
    
    avg_predictions = []
    lower_bounds = []
    upper_bounds = []
    
    for val in feature_values:
        X_temp = X.copy()
        X_temp.iloc[:, feature_idx] = val  # 특정 Feature만 변경
        y_pred = model.predict(X_temp)  # 모델 예측
        
        avg_predictions.append(y_pred.mean())  # 평균 예측값
        lower_bounds.append(np.percentile(y_pred, 5))  # 5% 신뢰구간 (하한)
        upper_bounds.append(np.percentile(y_pred, 95))  # 95% 신뢰구간 (상한)
    
    return feature_values, avg_predictions, lower_bounds, upper_bounds

