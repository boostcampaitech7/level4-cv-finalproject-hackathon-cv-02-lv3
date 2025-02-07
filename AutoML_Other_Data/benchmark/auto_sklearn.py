import pandas as pd
import os
import time
import autosklearn.regression
from log import do_log
import sys

py_dir_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(py_dir_path + f'/..')
from utils import evaluate_regression, data_preparation


def main():
    data_path = os.path.join(py_dir_path, '../../data/employee_attrition_dataset_encoding.csv') 
    X_train, y_train, X_test, y_test = data_preparation(data_path) # 데이터 준비

    log = do_log()

    log.log('Start')
    start = time.time()
    
    automl = autosklearn.regression.AutoSklearnRegressor(
    time_left_for_this_task=200,
    per_run_time_limit=30,
    tmp_folder="/data/ephemeral/home/Dongjin/tmp",
    n_jobs=-1)
    automl.fit(X_train, y_train)

    end = time.time()
    end = time.time()

    y_test_pred = automl.predict(X_test)
    y_train_pred = automl.predict(X_train)

    train_score = evaluate_regression(y_train, y_train_pred, 'train') 
    test_score = evaluate_regression(y_test, y_test_pred, 'test')
    log.log_dicts(train_score, 'Evaluation - Train')
    log.log_dicts(test_score, 'Evaluation - Test')

    elapsed_time = end-start

    log.log(automl.leaderboard())
    log.log(f'Autosklearn.regression init to training finished in: {elapsed_time:.1f} s')

if __name__ == '__main__':
    main()



