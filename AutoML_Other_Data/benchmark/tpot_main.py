from tpot import TPOTClassifier
from sklearn.datasets import load_digits
import pandas as pd
import os

py_dir_path = os.path.dirname(os.path.abspath(__file__)) # 현재 파이썬 스크립트 디렉토리
data_path = os.path.join(py_dir_path, '../data/IBM_employee_attrition_encoding_fin.csv') 

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

digits = load_digits()


pipeline_optimizer = TPOTClassifier(generations=5, population_size=30, cv=5, random_state=42, verbosity=2, scoring="f1_weighted", n_jobs=-1)
pipeline_optimizer.fit(X_train, y_train)
print("test score : ", pipeline_optimizer.score(X_test, y_test)) # 0.7821675437298992
# 0.727601223014917 - f1_macro
# 0.5348837209302325 - f1
# 0.861947558602798 - ROC AUC
# 0.7906976744186046 - recall 
# 0.8938052143934497 - f1_weignted
pipeline_optimizer.export('tpot_exported_pipeline_2.py')