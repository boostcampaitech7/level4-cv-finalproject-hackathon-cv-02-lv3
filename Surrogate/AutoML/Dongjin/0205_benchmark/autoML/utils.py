import math
import pandas as pd
from datetime import datetime
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os

def data_preparation(data_path, verbose=False):
    drop_tables = ['Suburb', 'Address', 'Rooms', 'Method', 'SellerG', 'Date', 'Distance', 'Postcode',
               'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'YearBuilt', 'CouncilArea',
               'Regionname', 'Propertycount']

    # df 불러오기 및 column 제거
    df = pd.read_csv(data_path)
    df = df.drop(drop_tables, axis=1)
    df = df.dropna(axis=0)

    index = 0.1 < df['BuildingArea'] # BuildingArea가 0인 값 제거
    df = df.loc[index]

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

class Log:
    def __init__(self, logger_name=None):
        now = datetime.now()
        time_string = now.strftime("%y%m%d_%H%M%S")
        py_dir_path = os.path.dirname(os.path.abspath(__file__))

        if logger_name is None:
            self.log_dir_path = os.path.join(py_dir_path, 'log')
        else:
            self.log_dir_path = os.path.join(py_dir_path, 'log', logger_name)
        
        self.log_path = os.path.join(self.log_dir_path, f"{time_string}.txt")
        os.makedirs(self.log_dir_path, exist_ok=True)
        

    def log_dicts(self, dicts, message=""):
        log = []
        for k, v in dicts.items():
            if isinstance(v, float):
                log.append(f'{k}: {v:.4f}')
            else:
                log.append(f'{k}: {v}')
        
        log = ', '.join(log)
        if len(message):
            log = f'{message} - {log}'
        self.log(log)
    
    def log(self, message):
        """
        log 기록

        Args:
            message (str): log 메세지
        """
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{now}] {message}"
        print(log_message) # 로그 출력 

        # 로그 저장
        with open(self.log_path, 'a') as file:
            file.write(log_message + "\n") 
            file.flush()