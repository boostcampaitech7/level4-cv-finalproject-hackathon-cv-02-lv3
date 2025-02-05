import pandas as pd
from autoML.autoML import AutoML
import os
from autoML.utils import evaluate_regression, data_preparation
import time


def run_autoML(X_train, y_train, X_test, y_test, generations, seed, n_jobs):

    start = time.time()
    
    autoML = AutoML(n_population=30, n_generation=generations, n_jobs=n_jobs)
    autoML.fit(X_train, y_train, seed=seed, use_kfold=True, kfold=5)
    
    end = time.time()

    y_test_pred = autoML.predict(X_test)
    y_train_pred = autoML.predict(X_train)

    train_score = evaluate_regression(y_train, y_train_pred, 'train') 
    test_score = evaluate_regression(y_test, y_test_pred, 'test')
    scores = flat_dicts({'train': train_score, 'test': test_score})

    elapsed_time = end - start

    autoML.log(f'AutoML init to training finished in: {elapsed_time:.1f} s')
    autoML.log_dicts(train_score, 'Evaluation - Train')
    autoML.log_dicts(test_score, 'Evaluation - Test')

    return scores, elapsed_time



def flat_dicts(dicts):
    flat_dicts = {}
    for phase, metrics in dicts.items():
        for metric_name, value in metrics.items():
            flat_dicts[f"{phase}_{metric_name}"] = value

    return flat_dicts



if __name__ == "__main__":
    save_name = 'autoML_njobs'
    generation = 5
    seeds = [1, 2, 3]
    n_jobs = list(range(1, 11, 1))
    n_jobs.append(-1)
    data_path = '/data/ephemeral/home/Dongjin/data/melbourne/melb_split.csv'


    py_dir_path = os.path.dirname(os.path.abspath(__file__))
    raw_save_path = os.path.join(py_dir_path, f'result/{save_name}_raw.csv')
    save_path = os.path.join(py_dir_path, f'result/{save_name}.csv')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    X_train, y_train, X_test, y_test = data_preparation(data_path) 
    df = None

    for n_job in n_jobs:
        for seed in seeds:
            scores, elapsed_time = run_autoML(X_train, y_train, X_test, y_test, 
                                                generations=generation, seed=seed, n_jobs=n_job)
            result = {'n_jobs': n_job, 'generation': generation, 'seed': seed, 'elapsed_time': elapsed_time}
            result.update(scores)

            if df is None:
                df = pd.DataFrame([result])
            else:
                df.loc[len(df)] = result

    df_summary = df.groupby('n_jobs').agg(['mean', 'std'])
    df.to_csv(raw_save_path)
    df_summary.to_csv(save_path)
