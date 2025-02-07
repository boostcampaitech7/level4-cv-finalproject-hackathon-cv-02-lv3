from evaluate import evaluate_autoML, evaluate_auto_scikit, evaluate_tpot
import pandas as pd
import os
import sys

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


def load_data(drop_categorical, target):
    # drop_categorical = True
    # target = 'Annual_revenue'
    
    data_dir_path = "data"
    train = pd.read_csv(data_dir_path + '/train.csv')
    test = pd.read_csv(data_dir_path + '/test.csv')
    n_train = len(train)
    combined = pd.concat([train, test], axis=0)

    if drop_categorical:
        string_cols = combined.select_dtypes(include=['object']).columns
        combined = combined.drop(columns=string_cols)
    else:
        combined = pd.get_dummies(combined)

    train = combined.iloc[:n_train, :]
    test = combined.iloc[n_train:, :]

    X_train = train.drop(columns=[target])
    y_train = train[target]
    X_test = test.drop(columns=[target])
    y_test = test[target]

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    drop_categorical = False
    target = 'Annual_revenue'
    
    rel_data_dir_path = 'data'
    rel_save_path = 'result/DVM-CAR.csv'
    

    if len(sys.argv) == 2:
        mode = sys.argv[1]
    else:
        mode = 'autoML'  

    if mode not in ['autoML', 'auto-scikitlearn', 'tpot']:
        print(f"Invalid mode: {mode}")
        sys.exit()

    py_dir_path = os.path.dirname(os.path.abspath(__file__))
    data_dir_path = os.path.join(py_dir_path, rel_data_dir_path)
    save_path = os.path.join(py_dir_path, rel_save_path)

    X_train, y_train, X_test, y_test = load_data(drop_categorical, target)

    func_dicts = {'autoML': {'func': evaluate_autoML, 'args': {'n_generation': 6}},
                'auto-scikitlearn': {'func': evaluate_auto_scikit, 'args': {'target_time': 400}},
                'tpot': {'func': evaluate_tpot, 'args': {'generations': 3}}}

    func = func_dicts[mode]['func']
    args = func_dicts[mode]['args']

    result = func(X_train, y_train, X_test, y_test, **args)
    result['args'] = args
    result['drop_categorical'] = drop_categorical
    result['target'] = target

    if os.path.exists(save_path):
        df = pd.read_csv(save_path, index_col='func')
    else:
        df = pd.DataFrame(columns=result.keys())
        df.index.name = 'func'

    index = rename_index_if_exists(df, mode)
    df.loc[index] = result

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path)


