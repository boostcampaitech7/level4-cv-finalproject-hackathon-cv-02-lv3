import autosklearn.regression
from autoML.utils import evaluate_regression, data_preparation, Log
import time
import shutil
import os

def run_autosklearn(X_train, y_train, X_test, y_test, run_time, seed):
    tmp_folder = '/data/ephemeral/home/Dongjin/temp'
    if os.path.exists(tmp_folder) and os.path.isdir(tmp_folder):
        shutil.rmtree(tmp_folder)

    log = Log(logger_name="autosklearn")
    log.log("autosklearn - start")
    
    start = time.time()
    automl = autosklearn.regression.AutoSklearnRegressor(
        time_left_for_this_task=run_time,
        per_run_time_limit=30,
        tmp_folder=tmp_folder,
        n_jobs=-1,
        seed=seed)

    automl.fit(X_train, y_train)
    end = time.time()

    y_train_pred = automl.predict(X_train)
    y_test_pred = automl.predict(X_test)

    train_score = evaluate_regression(y_train, y_train_pred, 'train') 
    test_score = evaluate_regression(y_test, y_test_pred, 'test')
    scores = {'train': train_score, 'test': test_score}
    elapsed_time = end-start

    log.log(automl.leaderboard())
    log.log(f'Autosklearn.regression init to training finished in: {elapsed_time:.1f} s')
    log.log_dicts(train_score)
    log.log_dicts(test_score)

    return scores, elapsed_time


if __name__ == "__main__":
    run_times = list(range(60, 610, 60))
    run_times.insert(0, 30)
    seeds = [1, 2, 3]
    data_path = '/data/ephemeral/home/Dongjin/data/melbourne/melb_split.csv'

    X_train, y_train, X_test, y_test = data_preparation(data_path) 

    for run_time in run_times:
        for seed in seeds:
            scores, elapsed_time = run_autosklearn(X_train, y_train, X_test, y_test, run_time, seed)
    

    