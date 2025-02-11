import numpy as np
import pandas as pd


# IQR을 이용한 이상치 제거 함수
def remove_outliers_iqr(df, option, method):
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

# 결측치 처리 함수
# def remove_na(df, option, method):
#     df = df.copy()  # 원본을 보존하기 위해 복사 (중요!)

#     if method == "관련 행 제거하기":
#         return df.dropna(subset=[option]).reset_index(drop=True)  # 해당 열에서 결측치가 있는 행 제거

#     elif method == "평균으로 채우기":
#         mean_value = df[option].mean()  # 평균값 계산
#         return df.fillna({option: mean_value})  # 특정 열만 채우기

#     elif method == "0으로 채우기":
#         return df.fillna({option: 0})  # 특정 열만 0으로 채우기

#     elif method == "최빈값으로 채우기":
#         mode_value = df[option].mode()[0]  # 최빈값 추출
#         return df.fillna({option: mode_value})  # 특정 열만 채우기 (inplace=False)

#     else:
#         return df
# 결측치 처리 함수
def remove_na(df, option, method):
    #df = df.copy()  # ✅ 원본을 보존하기 위해 복사 (중요!)

    if method == "관련 행 제거하기":
        return df.dropna(subset=[option]).reset_index(drop=True)  # ✅ 해당 열에서 결측치가 있는 행 제거

    elif method == "평균으로 채우기":
        mean_value = df[option].mean()  # 평균값 계산
        return df.fillna({option: mean_value})  # ✅ 특정 열만 채우기

    elif method == "0으로 채우기":
        return df.fillna({option: 0})  # ✅ 특정 열만 0으로 채우기

    elif method == "최빈값으로 채우기":
        mode_value = df[option].mode()[0]  # ✅ 최빈값 추출
        return df.fillna({option: mode_value})  # ✅ 특정 열만 채우기 (inplace=False)

    else:
        return df

# 범주형 변수 원-핫 인코딩 함수
def one_hot(df):
    for i in df.columns:
        if pd.api.types.is_string_dtype(df[i]) or pd.api.types.is_object_dtype(df[i]):
            if len(df[i])!=len(df[i].unique()):
                df = pd.get_dummies(df, columns=[i], drop_first=True)

    return df


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