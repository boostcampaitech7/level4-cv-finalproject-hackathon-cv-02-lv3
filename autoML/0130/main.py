import pandas as pd
from autoML import AutoML
import os
from utils import evaluate_regression, data_preparation
import time

py_dir_path = os.path.dirname(os.path.abspath(__file__)) # 현재 파이썬 스크립트 디렉토리
data_path = os.path.join(py_dir_path, 'melb_split.csv') 
X_train, y_train, X_test, y_test = data_preparation(data_path) # 데이터 준비

start = time.time()
autoML = AutoML(n_population=30, n_generation=1, n_parent=2, prob_mutation=0.1, use_joblib=True, n_jobs=-1)
autoML.fit(X_train, y_train, timeout=30)
end = time.time()

y_test_pred = autoML.predict(X_test)
y_train_pred = autoML.predict(X_train)

print(autoML.best_structure)
evaluate_regression(y_train, y_train_pred, 'train') 
evaluate_regression(y_test, y_test_pred, 'test')
print(f'{end-start:.1f} s')


# with open(os.path.join(py_dir_path, "autoML.pkl"), "wb") as file:
#     pickle.dump(autoML, file)
