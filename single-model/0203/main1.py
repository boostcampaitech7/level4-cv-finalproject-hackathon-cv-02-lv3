from xgboost import XGBRegressor
from utils import data_preparation, evaluate_regression
import numpy as np
from copy import deepcopy


if __name__ == '__main__':
    data_path = '/data/ephemeral/home/Dongjin/level4-cv-finalproject-hackathon-cv-02-lv3/autoML/data/melb_split.csv'
    seed = 15
    use_yscale_log = False
    X_train, y_train, X_test, y_test = data_preparation(data_path) # 데이터 준비

    if use_yscale_log:
        y_train_trans = np.log10(y_train)
    else:
        y_train_trans = y_train

    model = XGBRegressor(
        n_estimators=100,      
        max_depth=4,           
        learning_rate=0.1,     
        n_jobs=-1,             
        random_state=seed
    )
    
    model.fit(X_train, y_train_trans)
    y_test_pred = model.predict(X_test)

    if use_yscale_log:
        y_test_pred = np.power(10, y_test_pred)

    test_score = evaluate_regression(y_test, y_test_pred, 'test')

