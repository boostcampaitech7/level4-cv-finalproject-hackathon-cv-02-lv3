from evaluate import evaluate_autoML, evaluate_auto_scikit, evaluate_tpot
import pandas as pd
import os
import sys
from datetime import datetime
import dill



def load_data(data_dir_path):
    target = 'sales_volume'    
    X_train = pd.read_csv(data_dir_path + '/X_train.csv', index_col='datetime')
    y_train = pd.read_csv(data_dir_path + '/y_train.csv', index_col='datetime')[target]
    X_test = pd.read_csv(data_dir_path + '/X_test.csv', index_col='datetime')
    y_test = pd.read_csv(data_dir_path + '/y_test.csv', index_col='datetime')[target]

    for df in [X_train, y_train, X_test, y_test]:
        df.index = pd.to_datetime(df.index)
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    seeds = [1]
    rel_data_dir_path = 'data'
    rel_save_path = 'result/eCommerce.csv'

    if len(sys.argv) == 2:
        mode = sys.argv[1]
    else:
        mode = 'autoML'  

    if mode not in ['autoML', 'auto-sklearn', 'tpot']:
        print(f"Invalid mode: {mode}")
        sys.exit()

    py_dir_path = os.path.dirname(os.path.abspath(__file__))
    data_dir_path = os.path.join(py_dir_path, rel_data_dir_path)
    save_path = os.path.join(py_dir_path, rel_save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    func_dicts = {'autoML': {'func': evaluate_autoML, 'args': {'n_generation': 6}}, # 6 
                  'auto-sklearn': {'func': evaluate_auto_scikit, 'args': {'target_time': 400}}, # target_time': 400
                  'tpot': {'func': evaluate_tpot, 'args': {'generations': 6}}} # generations': 1

    func = func_dicts[mode]['func']
    args = func_dicts[mode]['args']

    for seed in seeds:
        now = datetime.now()
        timestamp = now.strftime("%y%m%d_%H%M%S")
        pkl_path = os.path.join(py_dir_path, f'pkl/{mode}_{timestamp}.pkl')

        X_train, y_train, X_test, y_test = load_data(data_dir_path,)
        result = {'mode': mode, 'args': args, 'seed': seed, 'timestamp': timestamp}
        scores, autoML = func(X_train, y_train, X_test, y_test, **args, seed=seed)
        result.update(scores)

        if os.path.exists(save_path):
            df = pd.read_csv(save_path)
        else:
            df = pd.DataFrame(columns=result.keys())
        
        df.loc[len(df)] = result
        df.to_csv(save_path, index=False)

        os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
        with open(pkl_path, 'wb') as f:
            dill.dump(autoML, f) 

