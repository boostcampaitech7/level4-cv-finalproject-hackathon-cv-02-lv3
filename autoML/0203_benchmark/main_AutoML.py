import pandas as pd
from autoML import AutoML
import os
from utils import evaluate_regression, data_preparation, Log
import time
import pandas as pd


def evaluate_AutoML(X_train, y_train, n_population=30, n_generation=5, n_parent=5, prob_mutations=[0.2, 0.5], use_joblib=True, n_jobs=-1, use_kfold=True, kfold=5, timeout=30, seed=42):
    start = time.time()
    
    autoML = AutoML(n_population=n_population, n_generation=n_generation,
                    n_parent=n_parent, prob_mutations=prob_mutations,
                    use_joblib=use_joblib, n_jobs=n_jobs)
    autoML.fit(X_train, y_train, use_kfold=use_kfold,
               kfold=kfold, timeout=timeout, seed=seed)
    
    end = time.time()

    y_test_pred = autoML.predict(X_test)
    y_train_pred = autoML.predict(X_train)

    train_score = evaluate_regression(y_train, y_train_pred, 'train') 
    test_score = evaluate_regression(y_test, y_test_pred, 'test')

    autoML.log_dicts(train_score, 'Evaluation - Train')
    autoML.log_dicts(test_score, 'Evaluation - Test')

    elapsed_time = end-start
    autoML.log(f'AutoML init to training finished in: {elapsed_time:.1f} s')

    return test_score['R2'], elapsed_time

if __name__ == '__main__':
    seed_list = list(range(1, 4))
    n_jobs_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1] # list(range(1, 9))
    rel_data_path = '../data/melb_split.csv'
    rel_save_path = 'result/AutoML/parallel.csv'

    py_dir_path = os.path.dirname(os.path.abspath(__file__)) # 현재 파이썬 스크립트 디렉토리
    data_path = os.path.join(py_dir_path, rel_data_path) 
    save_path = os.path.join(py_dir_path, rel_save_path)
    save_raw_path = save_path.replace(".csv", "_raw.csv")

    X_train, y_train, X_test, y_test = data_preparation(data_path) # 데이터 준비
    df = pd.DataFrame(columns=['n_jobs', 'seed', 'test_score', 'elapsed_time'])

    for n_jobs in n_jobs_list:
        for seed in seed_list:
            test_score, elapsed_time = evaluate_AutoML(X_train, y_train, seed=seed, n_jobs=n_jobs, n_generation=5)
            df.loc[len(df)] = [n_jobs, seed, test_score, elapsed_time]

    df_analysis = df.groupby('n_jobs')[['test_score', 'elapsed_time']].agg(['mean', 'std'])
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_raw_path)
    df_analysis.to_csv(save_path)

    
