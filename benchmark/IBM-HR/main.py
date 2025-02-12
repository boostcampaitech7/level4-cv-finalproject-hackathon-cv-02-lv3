from evaluate import evaluate_autoML, evaluate_auto_scikit, evaluate_tpot
import pandas as pd
import os
import sys
from datetime import datetime
from sklearn.model_selection import train_test_split

import sys
py_dir_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(py_dir_path + '/../../autoML')


def load_data(data_path):
    target = 'Attrition'    
    df = pd.read_csv(data_path)

    df.loc[df[target] == 'Yes', target] = 1
    df.loc[df[target] == 'No', target] = 0
    df[target] = df[target].astype(int)
    df = pd.get_dummies(df)

    train, test = train_test_split(
        df, test_size=0.2, random_state=42)

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    X_train = train.drop(target, axis=1)
    y_train = train[target]
    X_test = test.drop(target, axis=1)
    y_test = test[target]

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    seeds = [1]
    rel_data_dir_path = 'data'
    rel_save_path = 'result/IBM-HR.csv'

    if len(sys.argv) == 2:
        mode = sys.argv[1]
    else:
        mode = 'autoML'  

    if mode not in ['autoML', 'auto-sklearn', 'tpot']:
        print(f"Invalid mode: {mode}")
        sys.exit()

    py_dir_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(py_dir_path, 'WA_Fn-UseC_-HR-Employee-Attrition.csv')
    save_path = os.path.join(py_dir_path, rel_save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    func_dicts = {'autoML': {'func': evaluate_autoML, 'args': {'n_generation': 6}}, # 6 
                  'auto-sklearn': {'func': evaluate_auto_scikit, 'args': {'target_time': 70}}, # target_time': 400
                  'tpot': {'func': evaluate_tpot, 'args': {'generations': 5}}} # generations': 1

    func = func_dicts[mode]['func']
    args = func_dicts[mode]['args']

    for seed in seeds:
        now = datetime.now()
        timestamp = now.strftime("%y%m%d_%H%M%S")
        pkl_path = os.path.join(py_dir_path, f'pkl/{mode}_{timestamp}.pkl')

        X_train, y_train, X_test, y_test = load_data(data_path)
        result = {'mode': mode, 'args': args, 'seed': seed, 'timestamp': timestamp}
        scores, autoML = func(X_train, y_train, X_test, y_test, **args, seed=seed)
        result.update(scores)

        if os.path.exists(save_path):
            df = pd.read_csv(save_path)
        else:
            df = pd.DataFrame(columns=result.keys())
        
        df.loc[len(df)] = result
        df.to_csv(save_path, index=False)


