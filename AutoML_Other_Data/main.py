import pandas as pd
from autoML import AutoML
import os
from utils import evaluate_regression, data_preparation, evaluate_classification
import time
import cloudpickle
from joblib import dump



def main(task_type='regression', n_population=30, n_generation=5, n_parent=5, prob_mutations=[0.2, 0.5], use_joblib=True, n_jobs=-1, use_kfold=True, kfold=5, timeout=30, seed=42):
    py_dir_path = os.path.dirname(os.path.abspath(__file__)) # 현재 파이썬 스크립트 디렉토리
    data_path = os.path.join(py_dir_path, 'data/IBM_employee_attrition_encoding_4.csv') 
    X_train, y_train, X_test, y_test = data_preparation(data_path) # 데이터 준비

    start = time.time()
    
    autoML = AutoML(task_type=task_type, n_population=n_population, n_generation=n_generation,
                    n_parent=n_parent, prob_mutations=prob_mutations,
                    use_joblib=use_joblib, n_jobs=n_jobs)
    autoML.fit(X_train, y_train, use_kfold=use_kfold,
               kfold=kfold, timeout=timeout, seed=seed, task_type=task_type)
    
    end = time.time()

    y_test_pred = autoML.predict(X_test)
    y_train_pred = autoML.predict(X_train)

    train_score = evaluate_classification(y_train, y_train_pred, 'train') 
    test_score = evaluate_classification(y_test, y_test_pred, 'test')

    autoML.log_dicts(train_score, 'Evaluation - Train')
    autoML.log_dicts(test_score, 'Evaluation - Test')

    elapsed_time = end-start
    autoML.log(f'AutoML init to training finished in: {elapsed_time:.1f} s')
    
    py_dir_path = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(py_dir_path, "autoML_attrition_classification_SMOTE.pkl"), "wb") as file:
        cloudpickle.dump(autoML, file)
    # dump(autoML, os.path.join(py_dir_path, "autoML.joblib"))
    print(f"Model saved in '{os.path.join(py_dir_path, 'autoML_attrition_classification_SMOTE.pkl')}'.")

if __name__ == '__main__':
    # main()
    main(task_type='classification')
    # main(use_joblib=False)
    # main(use_kfold=False)
    # main(prob_mutations=[0.2, -1]) # hyperparameter mutation X


