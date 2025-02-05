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


def data_preparation(data_path, verbose=False):
    drop_tables = ['Suburb', 'Address', 'Rooms', 'Method', 'SellerG', 'Date', 'Distance', 'Postcode',
               'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'YearBuilt', 'CouncilArea',
               'Regionname', 'Propertycount']

    # df 불러오기 및 column 제거
    df = pd.read_csv(data_path)
    
    print(df.shape)
    print(df.head(10))
    df = df.drop(drop_tables, axis=1)
    df = df.dropna(axis=0)

    index = 0.1 < df['BuildingArea'] # BuildingArea 0값 제거
    df = df.loc[index]

    print(df.shape)
    print(df.head(10))
    # 데이터셋 분리
    train_data = df[df['Split'] == 'Train']
    train_data = train_data.drop(['Split'], axis=1)
    train_data = pd.get_dummies(train_data, dtype='float')

    test_data = df[df['Split'] == 'Test']
    test_data = test_data.drop(['Split'], axis=1)
    test_data = pd.get_dummies(test_data, dtype='float')

    # 타겟 변수와 특성 분리
    y_train = train_data['Price']
    X_train = train_data.drop(['Price'], axis=1)
    y_test = test_data['Price']
    X_test = test_data.drop(['Price'], axis=1)

    if verbose:
        # 결과 확인
        print("X_train.shape, y_train.shape, X_test.shape, y_test.shape: ", X_train.shape, y_train.shape, X_test.shape, y_test.shape)

        # na값 통계
        print("X_train, y_train, X_test, y_test null")
        print(X_train.isnull().sum())
        print(y_train.isnull().sum())
        print(X_test.isnull().sum())
        print(y_test.isnull().sum())

    return X_train, y_train, X_test, y_test



def evaluate_regression(y_true, y_pred, dataset_name="Dataset"):
    dicts = {'R2': r2_score(y_true, y_pred),
             'MAE': mean_absolute_error(y_true, y_pred),
             'RMSE': math.sqrt(mean_squared_error(y_true, y_pred))}

    print(f"\nEvaluation for {dataset_name}:")
    print(f"R2 Score: {dicts['R2']:.4f}")
    print(f"Mean Absolute Error (MAE): {dicts['MAE']:.4f}")
    print(f"Root Mean Squared Error (RMSE): {dicts['RMSE']:.4f}")

    return dicts