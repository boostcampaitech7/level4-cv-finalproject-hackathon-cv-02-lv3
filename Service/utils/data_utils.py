import numpy as np
import pandas as pd


def remove_outliers_iqr(df, option, method):
    """
    IQR(Interquartile Range)를 이용하여 이상치를 제거하는 함수.

    Args:
        df (pd.DataFrame): 이상치를 제거할 데이터프레임.
        option (str): 이상치 제거를 적용할 열 이름.
        method (str): "제거하기"로 설정하면 이상치를 제거하고, 그 외의 값이면 원본 데이터 반환.

    Returns:
        pd.DataFrame: 이상치가 제거된 데이터프레임 (method가 "제거하기"인 경우) 또는 원본 데이터프레임.
    """
    if method == "제거하기":
        Q1 = df[option].quantile(0.25)  # 1사분위수 (Q1)
        Q3 = df[option].quantile(0.75)  # 3사분위수 (Q3)
        IQR = Q3 - Q1  # IQR 계산
        lower_bound = Q1 - 1.5 * IQR  # 이상치 하한값
        upper_bound = Q3 + 1.5 * IQR  # 이상치 상한값

        # 이상치가 아닌 데이터만 선택
        filtered_df = df[(df[option] >= lower_bound) & (df[option] <= upper_bound)].reset_index(drop=True)
        return filtered_df
    else:
        return df


def remove_na(df, option, method):
    """
    결측치를 처리하는 함수.

    Args:
        df (pd.DataFrame): 결측치를 처리할 데이터프레임.
        option (str): 결측치를 처리할 열 이름.
        method (str): 결측치 처리 방법. 
            - "관련 행 제거하기": 결측치가 있는 행을 제거.
            - "평균으로 채우기": 해당 열의 평균값으로 채움.
            - "0으로 채우기": 해당 열의 결측치를 0으로 채움.
            - "최빈값으로 채우기": 해당 열의 최빈값으로 채움.

    Returns:
        pd.DataFrame: 결측치가 처리된 데이터프레임.
    """
    if method == "관련 행 제거하기":
        return df.dropna(subset=[option]).reset_index(drop=True)

    elif method == "평균으로 채우기":
        mean_value = df[option].mean()
        return df.fillna({option: mean_value})

    elif method == "0으로 채우기":
        return df.fillna({option: 0})

    elif method == "최빈값으로 채우기":
        mode_value = df[option].mode()[0]
        return df.fillna({option: mode_value})

    else:
        return df


def one_hot(df):
    """
    범주형 변수를 원-핫 인코딩하는 함수.

    Args:
        df (pd.DataFrame): 원-핫 인코딩을 적용할 데이터프레임.

    Returns:
        pd.DataFrame: 원-핫 인코딩이 적용된 데이터프레임.
    """
    for i in df.columns:
        if pd.api.types.is_string_dtype(df[i]) or pd.api.types.is_object_dtype(df[i]):
            if len(df[i]) != len(df[i].unique()):
                df = pd.get_dummies(df, columns=[i], drop_first=True)

    return df


def partial_dependence_with_error(model, X, feature_name, grid_points=50):
    """
    부분 의존도(Partial Dependence) 그래프를 생성하고, 예측값의 평균과 표준편차를 계산하는 함수.

    Args:
        model: 예측 모델 (사전 학습된 모델).
        X (pd.DataFrame): 예측에 사용할 데이터프레임.
        feature_name (str): 부분 의존도를 계산할 특성(Feature) 이름.
        grid_points (int, optional): 부분 의존도를 계산할 구간 개수. 기본값은 50.

    Returns:
        tuple: (feature_values, avg_predictions, errors)
            - feature_values (np.array): 특성 값의 범위.
            - avg_predictions (list): 각 특성 값에서의 평균 예측값.
            - errors (list): 예측값의 표준편차.
    """
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
