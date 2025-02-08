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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split



def data_preparation(data_path, verbose=False):
    # drop_tables = [
    # "BusinessTravel", "DailyRate", "Department", "DistanceFromHome", "Education", 
    # "EducationField", "EmployeeCount", "EmployeeNumber", "EnvironmentSatisfaction", 
    # "Gender", "HourlyRate", "JobInvolvement", "JobRole", "MaritalStatus", 
    # "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked", "Over18", "OverTime", 
    # "PercentSalaryHike", "PerformanceRating", "RelationshipSatisfaction", 
    # "StandardHours", "StockOptionLevel", "TotalWorkingYears", "TrainingTimesLastYear", 
    # "YearsAtCompany", "YearsInCurrentRole", "YearsWithCurrManager"]
    drop_tables = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]


    # df 불러오기 및 column 제거
    df = pd.read_csv(data_path)
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

def evaluate_classification(y_true, y_pred, dataset_name="Dataset"):
    dicts = {
        'F1 Score': f1_score(y_true, y_pred),
        'F1 Score_Weighted': f1_score(y_true, y_pred, average='weighted'),
        'auc' : roc_auc_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),      
        'Accuracy': accuracy_score(y_true, y_pred)
    }
    print(f"\nEvaluation for {dataset_name}:")
    print(f"F1 Score: {dicts['F1 Score']:.4f}")
    print(f"F1 Score_Weighted: {dicts['F1 Score_Weighted']:.4f}")
    print(f"AUC: {dicts['auc']:.4f}")
    print(f"Precision: {dicts['Precision']:.4f}")
    print(f"Recall: {dicts['Recall']:.4f}")
    print(f"Accuracy: {dicts['Accuracy']:.4f}")

    return dicts