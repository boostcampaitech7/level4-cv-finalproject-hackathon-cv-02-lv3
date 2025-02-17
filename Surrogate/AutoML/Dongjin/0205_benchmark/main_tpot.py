from tpot import TPOTRegressor
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


def run_TPOT(X_train, y_train, X_test, y_test, generations, seed, n_jobs):
    log = Log(logger_name="TPOT")
    log.log("TPOT - start")
    log.log(f"generation: {generations}, seed: {seed}, n_jobs: {n_jobs}")
    
    start = time.time()
    automl = TPOTRegressor(generations=generations, population_size=30, cv=5,
                            random_state=seed, verbosity=2, n_jobs=n_jobs)

    automl.fit(X_train, y_train)

    end = time.time()

    y_train_pred = automl.predict(X_train)
    y_test_pred = automl.predict(X_test)

    train_score = evaluate_regression(y_train, y_train_pred, 'train') 
    test_score = evaluate_regression(y_test, y_test_pred, 'test')
    scores = flat_dicts({'train': train_score, 'test': test_score})
    elapsed_time = end - start

    log.log(f'Autosklearn.regression init to training finished in: {elapsed_time:.1f} s')
    log.log_dicts(train_score, message="train")
    log.log_dicts(test_score, message="test")

    return scores, elapsed_time


if __name__ == "__main__":
    save_name = 'TPOT'
    generations = list(range(1, 11, 1))
    seeds = [1, 2, 3]
    n_jobs = -1
    data_path = '/data/ephemeral/home/Dongjin/data/melbourne/melb_split.csv'


    py_dir_path = os.path.dirname(os.path.abspath(__file__))
    raw_save_path = os.path.join(py_dir_path, f'result/{save_name}_raw.csv')
    save_path = os.path.join(py_dir_path, f'result/{save_name}.csv')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    X_train, y_train, X_test, y_test = data_preparation(data_path) 
    df = None

    for generation in generations:
        for seed in seeds:
            scores, elapsed_time = run_TPOT(X_train, y_train, X_test, y_test, 
                                                generations=generation, seed=seed, n_jobs=n_jobs)
            result = {'generation': generation, 'seed': seed, 'n_jobs': n_jobs, 'elapsed_time': elapsed_time}
            result.update(scores)

            if df is None:
                df = pd.DataFrame([result])
            else:
                df.loc[len(df)] = result

    df_summary = df.groupby('generation').agg(['mean', 'std'])
    df.to_csv(raw_save_path)
    df_summary.to_csv(save_path)