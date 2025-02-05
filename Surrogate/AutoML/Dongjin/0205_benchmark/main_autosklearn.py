import autosklearn.regression
from autoML.utils import evaluate_regression, data_preparation, Log
import time
import shutil
import os
import pandas as pd 

def flat_dicts(dicts):
    flat_dicts = {}
    for phase, metrics in dicts.items():
        for metric_name, value in metrics.items():
            flat_dicts[f"{phase}_{metric_name}"] = value

    return flat_dicts


def run_autosklearn(X_train, y_train, X_test, y_test, target_time, seed, n_jobs):
    tmp_folder = '/data/ephemeral/home/Dongjin/temp'
    if os.path.exists(tmp_folder) and os.path.isdir(tmp_folder):
        shutil.rmtree(tmp_folder)

    log = Log(logger_name="autosklearn")
    log.log("autosklearn - start")
    log.log(f"target_time: {target_time}, seed: {seed}, n_jobs: {n_jobs}")
    
    start = time.time()
    automl = autosklearn.regression.AutoSklearnRegressor(
        time_left_for_this_task=target_time,
        per_run_time_limit=30,
        tmp_folder=tmp_folder,
        n_jobs=n_jobs,
        seed=seed)

    automl.fit(X_train, y_train)
    end = time.time()

    y_train_pred = automl.predict(X_train)
    y_test_pred = automl.predict(X_test)

    train_score = evaluate_regression(y_train, y_train_pred, 'train') 
    test_score = evaluate_regression(y_test, y_test_pred, 'test')
    scores = flat_dicts({'train': train_score, 'test': test_score})
    elapsed_time = end-start

    log.log(automl.leaderboard())
    log.log(f'Autosklearn.regression init to training finished in: {elapsed_time:.1f} s')
    log.log_dicts(train_score, message="train")
    log.log_dicts(test_score, message="test")

    return scores, elapsed_time


if __name__ == "__main__":
    save_name = 'autosklearn'
    target_times = list(range(60, 610, 60))
    target_times.insert(0, 30)
    seeds = [1, 2, 3]
    n_jobs = -1
    data_path = '/data/ephemeral/home/Dongjin/data/melbourne/melb_split.csv'


    py_dir_path = os.path.dirname(os.path.abspath(__file__))
    raw_save_path = os.path.join(py_dir_path, f'result/{save_name}_raw.csv')
    save_path = os.path.join(py_dir_path, f'result/{save_name}.csv')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    X_train, y_train, X_test, y_test = data_preparation(data_path) 
    df = None

    for target_time in target_times:
        for seed in seeds:
            scores, elapsed_time = run_autosklearn(X_train, y_train, X_test, y_test, 
                                                target_time=target_time, seed=seed, n_jobs=n_jobs)
            result = {'target_time': target_time, 'seed': seed, 'n_jobs': n_jobs, 'elapsed_time': elapsed_time}
            result.update(scores)

            if df is None:
                df = pd.DataFrame([result])
            else:
                df.loc[len(df)] = result

    df_summary = df.groupby('target_time').agg(['mean', 'std'])
    df.to_csv(raw_save_path)
    df_summary.to_csv(save_path)