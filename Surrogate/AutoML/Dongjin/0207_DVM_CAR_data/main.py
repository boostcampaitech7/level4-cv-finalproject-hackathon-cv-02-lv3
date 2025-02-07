from evaluate import evaluate_autoML, evaluate_auto_scikit, evaluate_tpot
import pandas as pd
import os
import sys



def load_data(data_dir_path, drop_categorical, drop_stylish, target):
    # drop_categorical = True, False
    # drop_stylish = True, False
    # target = 'Annual_revenue'
    
    train = pd.read_csv(data_dir_path + '/train.csv')
    test = pd.read_csv(data_dir_path + '/test.csv')
    n_train = len(train)
    combined = pd.concat([train, test], axis=0)

    if drop_categorical:
        string_cols = combined.select_dtypes(include=['object']).columns
        combined = combined.drop(columns=string_cols)
    else:
        # categorical data를 남기면 one-hot encoding 수행 
        combined = pd.get_dummies(combined)
        bool_cols = combined.select_dtypes(include=['bool']).columns
        combined[bool_cols] = combined[bool_cols].astype(int)

    if drop_stylish:
        combined = combined.drop(columns=['stylish'])

    train = combined.iloc[:n_train, :]
    test = combined.iloc[n_train:, :]

    X_train = train.drop(columns=[target])
    y_train = train[target]
    X_test = test.drop(columns=[target])
    y_test = test[target]

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    seeds = [1]
    drop_categorical = False
    drop_stylishs = [True]
    target = 'Annual_revenue'

    rel_data_dir_path = 'data'
    rel_save_path = 'result/DVM-CAR.csv'

    if len(sys.argv) == 2:
        mode = sys.argv[1]
    else:
        mode = 'auto-scikitlearn'  

    if mode not in ['autoML', 'auto-scikitlearn', 'tpot']:
        print(f"Invalid mode: {mode}")
        sys.exit()

    py_dir_path = os.path.dirname(os.path.abspath(__file__))
    data_dir_path = os.path.join(py_dir_path, rel_data_dir_path)
    save_path = os.path.join(py_dir_path, rel_save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    func_dicts = {'autoML': {'func': evaluate_autoML, 'args': {'n_generation': 6}},
                'auto-scikitlearn': {'func': evaluate_auto_scikit, 'args': {'target_time': 460}}, # target_time': 400
                'tpot': {'func': evaluate_tpot, 'args': {'generations': 6}}} # generations': 1

    func = func_dicts[mode]['func']
    args = func_dicts[mode]['args']

    for seed in seeds:
        for drop_stylish in drop_stylishs:   
            X_train, y_train, X_test, y_test = load_data(data_dir_path, drop_categorical, drop_stylish, target)
            result = {'mode': mode, 'drop_categorical': drop_categorical, 'drop_stylish': drop_stylish,
                      'target': target, 'args': args, 'seed': seed}
            scores = func(X_train, y_train, X_test, y_test, **args, seed=seed)
            result.update(scores)

            if os.path.exists(save_path):
                df = pd.read_csv(save_path)
            else:
                df = pd.DataFrame(columns=result.keys())
            
            df.loc[len(df)] = result
            df.to_csv(save_path, index=False)


