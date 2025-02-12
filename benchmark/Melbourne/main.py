from evaluate import evaluate_autoML, evaluate_auto_scikit, evaluate_tpot
import pandas as pd
import os
import sys
from datetime import datetime
from sklearn.model_selection import train_test_split

import sys
py_dir_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(py_dir_path + '/../../autoML')


def load_data(data_path, verbose=False):
    drop_tables = ['Suburb', 'Address', 'Rooms', 'Method', 'SellerG', 'Date', 'Distance', 'Postcode',
               'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'YearBuilt', 'CouncilArea',
               'Regionname', 'Propertycount']

    # df 불러오기 및 column 제거
    df = pd.read_csv(data_path)
    df = df.drop(drop_tables, axis=1)
    df = df.dropna(axis=0)

    index = 0.1 < df['BuildingArea'] # BuildingArea가 0인 값 제거
    df = df.loc[index]

    # 데이터셋 분리
    train_data = df[df['Split'] == 'Train']
    train_data = train_data.drop(['Split'], axis=1)
    train_data = pd.get_dummies(train_data, dtype='float')

    test_data = df[df['Split'] == 'Test']
    test_data = test_data.drop(['Split'], axis=1)
    test_data = pd.get_dummies(test_data, dtype='float')

    # 타겟 변수와 특성 분리
    y_train = train_data['Price']
    X_train = train_data.drop(['Price'], axis=1)
    y_test = test_data['Price']
    X_test = test_data.drop(['Price'], axis=1)

    if verbose:
        # 결과 확인
        print("X_train.shape, y_train.shape, X_test.shape, y_test.shape: ", X_train.shape, y_train.shape, X_test.shape, y_test.shape)

        # na값 통계
        print("X_train, y_train, X_test, y_test null")
        print(X_train.isnull().sum())
        print(y_train.isnull().sum())
        print(X_test.isnull().sum())
        print(y_test.isnull().sum())

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    seeds = [1]
    rel_data_dir_path = 'data'
    rel_save_path = 'result/melbourne.csv'

    if len(sys.argv) == 2:
        mode = sys.argv[1]
    else:
        mode = 'autoML'  

    if mode not in ['autoML', 'auto-sklearn', 'tpot']:
        print(f"Invalid mode: {mode}")
        sys.exit()

    py_dir_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(py_dir_path, 'melb_split1.csv')
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


