from evaluate import evaluate_autoML, evaluate_auto_scikit, evaluate_tpot
import pandas as pd
import os

def rename_index_if_exists(df, index):
    if not index in df.index:
        return index

    num = 0
    while (num < 1000):
        new_index = f'{index}_{num}'
        if not new_index in df.index:
            break
        num += 1
    return new_index
    

if __name__ == "__main__":
    mode = 'autoML'
    rel_data_dir_path = 'data'
    rel_save_path = 'result/result.csv'

    py_dir_path = os.path.dirname(os.path.abspath(__file__))
    data_dir_path = os.path.join(py_dir_path, rel_data_dir_path)
    save_path = os.path.join(py_dir_path, rel_save_path)

    X_train = pd.read_csv(data_dir_path + '/X_train.csv')
    y_train = pd.read_csv(data_dir_path + '/y_train.csv')['strength']
    X_test = pd.read_csv(data_dir_path + '/X_test.csv')
    y_test = pd.read_csv(data_dir_path + '/y_test.csv')['strength']

    func_dicts = {'autoML': {'func': evaluate_autoML, 'args': {'n_generation': 5}},
                'auto-scikitlearn': {'func': evaluate_auto_scikit, 'args': {'target_time': 60}},
                'tpot': {'func': evaluate_tpot, 'args': {'generations': 3}}}


    func = func_dicts[mode]['func']
    args = func_dicts[mode]['args']
    scores = func(X_train, y_train, X_test, y_test, **args)
    scores['args'] = args

    if os.path.exists(save_path):
        df = pd.read_csv(save_path)
    else:
        df = pd.DataFrame(columns=scores.keys())
        df.index.name = 'func'

    index = rename_index_if_exists(df, mode)
    df.loc[index] = scores

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path)
