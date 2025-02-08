import pandas as pd
from autoML import AutoML
import os
from utils import evaluate_regression, data_preparation, evaluate_classification
import time


def aisolution(n_population=30, n_generation=5, n_parent=5, prob_mutations=[0.2, 0.5], use_joblib=True, n_jobs=-1, use_kfold=True, kfold=5, timeout=30, seed=42,\
        X_train=None, y_train=None, X_test=None, y_test=None,task_type='regression'):
    #py_dir_path = os.path.dirname(os.path.abspath(__file__)) # 현재 파이썬 스크립트 디렉토리
    #data_path = os.path.join(py_dir_path, '../data/melb_split.csv') 
    #X_train, y_train, X_test, y_test = data_preparation(data_path) # 데이터 준비

    start = time.time()
    
    autoML = AutoML(n_population=n_population, n_generation=n_generation,
                    n_parent=n_parent, prob_mutations=prob_mutations,
                    use_joblib=use_joblib, n_jobs=n_jobs)
    autoML.fit(X_train, y_train, use_kfold=use_kfold,
               kfold=kfold, timeout=timeout, seed=seed)
    
    end = time.time()

    y_test_pred = autoML.predict(X_test)
    y_train_pred = autoML.predict(X_train)

    if task_type == 'regression':
        train_score = evaluate_regression(X_train,y_train, y_train_pred, 'train') 
        test_score = evaluate_regression(X_test,y_test, y_test_pred, 'test')
    else:
        y_test_pred = (y_test_pred >= 0.5).astype(int)
        y_train_pred = (y_train_pred >= 0.5).astype(int)
        train_score = evaluate_classification(y_train, y_train_pred, 'train') 
        test_score = evaluate_classification(y_test, y_test_pred, 'test')        

    autoML.log_dicts(train_score, 'Evaluation - Train')
    autoML.log_dicts(test_score, 'Evaluation - Test')

    elapsed_time = end-start
    autoML.log(f'AutoML init to training finished in: {elapsed_time:.1f} s')

    return train_score, test_score, elapsed_time, autoML



