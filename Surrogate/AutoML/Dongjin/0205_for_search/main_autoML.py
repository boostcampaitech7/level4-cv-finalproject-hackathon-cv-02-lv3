from autoML.autoML import AutoML
import os
from autoML.utils import evaluate_regression, data_preparation
import time
import dill 

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

    return autoML, scores, elapsed_time


def flat_dicts(dicts):
    flat_dicts = {}
    for phase, metrics in dicts.items():
        for metric_name, value in metrics.items():
            flat_dicts[f"{phase}_{metric_name}"] = value

    return flat_dicts


def train_autoML(data_path, save_pkl_path):
    generations = 6
    seed = 3
    n_jobs = -1

    X_train, y_train, X_test, y_test = data_preparation(data_path) 
    autoML, scores, elapsed_time = run_autoML(X_train, y_train, X_test, y_test, 
                                              generations=generations, seed=seed, n_jobs=n_jobs)
    with open(file=save_pkl_path, mode='wb') as f:
        dill.dump(autoML, f)


def load_autoML(pkl_path):
    if not os.path.exists(pkl_path):
        raise(Exception("pkl 파일을 찾을 수 없습니다."))
    with open(file=pkl_path, mode='rb') as f:
        autoML = dill.load(f)
    return autoML



if __name__ == "__main__":
    mode = 'load' # 'train' or 'load'

    data_path = '/data/ephemeral/home/Dongjin/data/melbourne/melb_split.csv'
    py_dir_path = os.path.dirname(os.path.abspath(__file__))
    pkl_path = os.path.join(py_dir_path, 'autoML/autoML.pkl')

    if mode == 'train':
        train_autoML(data_path, pkl_path)
    
    elif mode == 'load':
        X_train, y_train, X_test, y_test = data_preparation(data_path) 
        autoML = load_autoML(pkl_path)
        
        y_train_pred = autoML.predict(X_train)
        y_test_pred = autoML.predict(X_test)
        train_score = evaluate_regression(y_train, y_train_pred, 'train') 
        test_score = evaluate_regression(y_test, y_test_pred, 'test')
    
    else:
        raise(Exception("Invalid mode: {mode}"))