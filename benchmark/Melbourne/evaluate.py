import time
import os
import sys
py_dir_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(py_dir_path + '/../../autoML')

from metrics import evaluate_classification, evaluate_regression

def flat_dicts(dicts):
    flat_dicts = {}
    for phase, metrics in dicts.items():
        for metric_name, value in metrics.items():
            flat_dicts[f"{phase}_{metric_name}"] = value

    return flat_dicts

def evaluate_autoML(X_train, y_train, X_test, y_test, n_generation, seed):
    from autoML import AutoML

    start = time.time()
    autoML = AutoML(n_population=30, n_generation=n_generation, n_jobs=-1, task_type='regression')
    autoML.fit(X_train, y_train, seed=seed, use_kfold=True, kfold=5)

    end = time.time()

    y_test_pred = autoML.predict(X_test)
    y_train_pred = autoML.predict(X_train)

    train_score = evaluate_regression(y_train, y_train_pred) 
    test_score = evaluate_regression(y_test, y_test_pred)
    scores = flat_dicts({'train': train_score, 'test': test_score})
    elapsed_time = end - start

    autoML.log(f'AutoML init to training finished in: {elapsed_time:.1f} s')
    autoML.log_dicts(train_score, 'Evaluation - Train')
    autoML.log_dicts(test_score, 'Evaluation - Test')
    scores['elapsed_time'] = elapsed_time
    return scores, autoML

def evaluate_auto_scikit(X_train, y_train, X_test, y_test, target_time, seed):
    import autosklearn.classification, shutil

    tmp_folder = '/data/ephemeral/home/Dongjin/temp'
    if os.path.exists(tmp_folder) and os.path.isdir(tmp_folder):
        shutil.rmtree(tmp_folder)

    start = time.time()
    autoML = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=target_time,
        per_run_time_limit=30,
        tmp_folder=tmp_folder,
        n_jobs=-1,
        seed=seed)

    autoML.fit(X_train, y_train)

    end = time.time()

    y_test_pred = autoML.predict(X_test)
    y_train_pred = autoML.predict(X_train)

    train_score = evaluate_classification(y_train, y_train_pred, 'train') 
    test_score = evaluate_classification(y_test, y_test_pred, 'test')
    scores = flat_dicts({'train': train_score, 'test': test_score})
    elapsed_time = end - start
    scores['elapsed_time'] = elapsed_time
    print(f'Auto-scikitlearn init to training finished in: {elapsed_time:.1f} s')
    return scores, autoML


def evaluate_tpot(X_train, y_train, X_test, y_test, generations, seed):
    from tpot import TPOTClassifier

    start = time.time()
    autoML = TPOTClassifier(generations=generations, population_size=30, cv=5,
                            random_state=seed, verbosity=2, n_jobs=-1)


    autoML.fit(X_train, y_train)

    end = time.time()

    y_test_pred = autoML.predict(X_test)
    y_train_pred = autoML.predict(X_train)

    train_score = evaluate_classification(y_train, y_train_pred, 'train') 
    test_score = evaluate_classification(y_test, y_test_pred, 'test')
    scores = flat_dicts({'train': train_score, 'test': test_score})
    elapsed_time = end - start
    scores['elapsed_time'] = elapsed_time
    print(f'TPOT init to training finished in: {elapsed_time:.1f} s')
    return scores, autoML
