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
    drop_tables = [ "Employee_ID",
    "Gender", "Marital_Status", "Department", "Job_Role", "Monthly_Income", 
    "Hourly_Rate", "Years_at_Company", "Years_in_Current_Role", 
    "Work_Environment_Satisfaction", "Performance_Rating", "Training_Hours_Last_Year", 
    "Overtime", "Project_Count", "Average_Hours_Worked_Per_Week", "Absenteeism", 
    "Relationship_with_Manager", "Job_Involvement", "Distance_From_Home", 
    "Number_of_Companies_Worked"
    ]   

    # df 불러오기
    df = pd.read_csv(data_path)
    df["Attrition"] = df["Attrition"].map({"No": 0, "Yes": 1})

    
    # 8:2 비율로 train/test 분리
    split_index = int(10000 * 0.8)  # 8000번째 행까지 Train, 나머지 Test

    # Split 컬럼 추가
    df["Split"] = ["Train"] * split_index + ["Test"] * (10000 - split_index)
    
    #column 제거
    df = df.drop(drop_tables, axis=1)
    df = df.dropna(axis=0)

    
    # 데이터셋 분리
    train_data = df[df['Split'] == 'Train']
    train_data = train_data.drop(['Split'], axis=1)

    test_data = df[df['Split'] == 'Test']
    test_data = test_data.drop(['Split'], axis=1)
    

    # 타겟 변수와 특성 분리
    y_train = train_data['Attrition']
    X_train = train_data.drop(['Attrition'], axis=1)
    y_test = test_data['Attrition']
    X_test = test_data.drop(['Attrition'], axis=1)
    
    print(y_train.head(10))
    print(X_test.head(10))

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