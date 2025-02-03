from xgboost import XGBRegressor
from utils import data_preparation, evaluate_regression
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def evaluate(X_train, y_train, X_test, y_test, model_name, seed, use_yscale_log):
    if use_yscale_log:
        y_train_trans = np.log10(y_train)
    else:
        y_train_trans = y_train

    if model_name == 'XGB':
        model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, n_jobs=-1, random_state=seed)
    elif model_name == "RandomForest":
        model = RandomForestRegressor(n_estimators=100, random_state=seed)


    model.fit(X_train, y_train_trans)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    

    if use_yscale_log:
        y_train_pred = np.power(10, y_train_pred)
        y_test_pred = np.power(10, y_test_pred)

    print(f'model: {model_name}, seed: {seed}, use_yscale_log: {use_yscale_log}')
    test_score = evaluate_regression(y_test, y_test_pred, 'test')
    
    return test_score


if __name__ == '__main__':
    data_path = '/data/ephemeral/home/Dongjin/level4-cv-finalproject-hackathon-cv-02-lv3/autoML/data/melb_split.csv'
    model_name = 'RandomForest'
    X_train, y_train, X_test, y_test = data_preparation(data_path) # 데이터 준비
    
    seeds = [1]
    use_yscale_logs = [False, True]
    
    for use_yscale_log in use_yscale_logs:
        for seed in seeds:
            test_score = evaluate(X_train, y_train, X_test, y_test, model_name, seed, use_yscale_log)
            print("")
