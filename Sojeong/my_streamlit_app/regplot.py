import numpy as np
import pandas as pd

def partial_dependence_with_error(model, X, feature_name, grid_points=50):
    feature_idx = X.columns.get_loc(feature_name)  # 선택한 feature의 index
    feature_values = np.linspace(X[feature_name].min(), X[feature_name].max(), grid_points)  # Feature 값을 일정 간격으로 나눔
    
    avg_predictions = []
    errors = []  # 표준편차(오차) 저장

    for val in feature_values:
        X_temp = X.copy()
        X_temp.iloc[:, feature_idx] = val  # 특정 Feature만 변경
        y_pred = model.predict(X_temp)  # 모델 예측
        
        mean_prediction = y_pred.mean()  # 평균 예측값
        std_dev = y_pred.std()  # 표준편차(오차)

        avg_predictions.append(mean_prediction)
        errors.append(std_dev)  # 오차 저장
    
    return feature_values, avg_predictions, errors