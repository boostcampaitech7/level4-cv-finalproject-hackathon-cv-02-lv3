import pandas as pd
from autoML.autoML import AutoML
import os
from autoML.metrics import evaluate_regression, evaluate_classification
import time


def aisolution(n_population=30, n_generation=5, n_parent=5, prob_mutations=[0.2, 0.5], use_joblib=True, n_jobs=-1, use_kfold=True, kfold=5, timeout=30, seed=42,\
        X_train=None, y_train=None, X_test=None, y_test=None,task_type='regression'):

    """
    AutoML을 사용하여 최적의 머신러닝 모델을 탐색하고 평가하는 함수.

    Args:
        n_population (int, optional): 유전 알고리즘에서 개체군의 크기 기본값은 30
        n_generation (int, optional): 유전 알고리즘에서 세대 수 기본값은 5
        n_parent (int, optional): 다음 세대를 생성하는 부모 개체의 수 기본값은 5
        prob_mutations (list, optional): 돌연변이 확률의 범위 기본값은 [0.2, 0.5]
        use_joblib (bool, optional): 병렬 처리를 사용할지 여부 기본값은 True
        n_jobs (int, optional): 병렬 작업을 수행할 CPU 코어 수 (-1은 모든 가용 코어 사용) 기본값은 -1
        use_kfold (bool, optional): K-Fold 교차 검증을 사용할지 여부 기본값은 True
        kfold (int, optional): K-Fold의 분할 개수 기본값은 5
        timeout (int, optional): 모델 학습 제한 시간 (초 단위), 기본값은 30
        seed (int, optional): 난수 시드 값, 기본값은 42
        X_train (pd.DataFrame, optional): 훈련 데이터의 특징(Feature) 행렬, 기본값은 None
        y_train (pd.Series or np.array, optional): 훈련 데이터의 타겟 값, 기본값은 None
        X_test (pd.DataFrame, optional): 테스트 데이터의 특징(Feature) 행렬, 기본값은 None
        y_test (pd.Series or np.array, optional): 테스트 데이터의 타겟 값, 기본값은 None
        task_type (str, optional): 'regression' 또는 'classification' 중 하나로 설정, 기본값은 'regression'

    Returns:
        tuple: (train_score, test_score, elapsed_time, autoML)
            - train_score (dict): 훈련 데이터에 대한 성능 평가 결과
            - test_score (dict): 테스트 데이터에 대한 성능 평가 결과
            - elapsed_time (float): 전체 AutoML 수행 시간 (초 단위)
            - autoML (AutoML): 학습된 AutoML 객체
    """

    start = time.time()
    
    autoML = AutoML(n_population=n_population, n_generation=n_generation,
                    n_parent=n_parent, prob_mutations=prob_mutations,
                    use_joblib=use_joblib, n_jobs=n_jobs, task_type=task_type)
    autoML.fit(X_train, y_train, use_kfold=use_kfold,
               kfold=kfold, timeout=timeout, seed=seed)
    
    end = time.time()

    y_test_pred = autoML.predict(X_test)
    y_train_pred = autoML.predict(X_train)

    if task_type == 'regression':
        train_score = evaluate_regression(X_train,y_train, y_train_pred) 
        test_score = evaluate_regression(X_test, y_test, y_test_pred)
    else:
        train_score = evaluate_classification(y_train, y_train_pred) 
        test_score = evaluate_classification(y_test, y_test_pred)        

    autoML.log_dicts(train_score, 'Evaluation - Train')
    autoML.log_dicts(test_score, 'Evaluation - Test')

    elapsed_time = end-start
    autoML.log(f'AutoML init to training finished in: {elapsed_time:.1f} s')

    return train_score, test_score, elapsed_time, autoML



