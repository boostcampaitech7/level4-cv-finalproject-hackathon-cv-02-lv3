# 본 코드는 TPOT (Evaluation of a Tree-based Pipeline Optimization Tool
# for Automating Data Science, GECCO '16)에서 아이디어를 얻어 구현했습니다.

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, SelectPercentile, VarianceThreshold
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV

import random
from datetime import datetime
import math
import signal
import os
import numpy as np


preprocessors = {'StandardScaler': StandardScaler(), 'RobustScaler': RobustScaler(), 
                 'PolynomialFeatures': PolynomialFeatures(), 'PCA': PCA()}

feature_selections = {'SelectKBest': SelectKBest(), 'SelectPercentile': SelectPercentile(),
                      'VarianceThreshold': VarianceThreshold()} # RFE

models = {'DecisionTreeRegressor': DecisionTreeRegressor(), 'RandomForestRegressor': RandomForestRegressor(), 
          'GradientBoostingRegressor': GradientBoostingRegressor(), 'LogisticRegression': LogisticRegression(),
          'KNeighborsRegressor': KNeighborsRegressor()}

pipeline_components = {'preprocessors': preprocessors, 'feature_selections': feature_selections, 'models': models}
choose_random_key = lambda dictionary: random.choice(list(dictionary.values()))


class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Timeout: pipeline.fit did not complete in given time.")

def evaluate_regression(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    RMSE = math.sqrt(mean_squared_error(y_true, y_pred))    
    dicts = {'r2': r2, 'RMSE': RMSE}
    return dicts


def build_pipeline(structure):
    """ ML 계산을 위한 scikit pipeline 생성

    Args:
        structure (dict): pipeline을 생성하기 위한 구조체

    Returns:
        Pipeline: structure로 생성한 Pipeline
    """
    
    _pipeline = []
    for k, v in structure.items():
        _pipeline.append((k, clone(v)))
    return Pipeline(_pipeline)


def get_random_structures(n):
    """n개의 임의의 structure 생성

    Args:
        n (int): 임의로 생성할 structure 개수

    Returns:
        random_structures (list): 임의로 생성한 structure
    """
    random_structures = []
    for _ in range(n):
        random_structure = {'preprocessors': choose_random_key(preprocessors),
                            'feature_selections': choose_random_key(feature_selections),
                            'models': choose_random_key(models)}
        
        random_structure['pipeline'] = build_pipeline(random_structure)
        random_structures.append(random_structure)
        
    return random_structures

def fit_pipeline(pipeline, X_train, y_train):
    pipeline.fit(X_train, y_train)    

def sort(structures):
    """ structures를 평가지표를 기준으로 정렬

    Args:
        structures (list): 정렬되지 않은 구조

    Returns:
        structures (list): ['valid_metric']['r2']를 기준으로 정렬
    """
    return sorted(structures, key=lambda x: x['valid_metric']['r2'], reverse=True)


def is_same_structure(structure1, structure2):
    """ structure1과 structure2가 동일한지 확인

    Args:
        structure1 (dict): 구조1
        structure2 (dict): 구조2

    Returns:
        Bool: 동일하면 True, 다르면 False 반환
    """
    keys = list(pipeline_components.keys())
    for k in keys:
        if structure1[k] != structure2[k]:
            return False
    return True


def is_in_structures(structure, structures):
    """structures 내 동일한 structure가 있는지 확인

    Args:
        structure (dict): 구조
        structures (list): 구조 집합

    Returns:
        Bool: 동일하면 True, 다르면 False 반환
    """
    for s in structures:
        if is_same_structure(structure, s):
            return True
    return False


def crossover(structure1, structure2):
    """유전적 교차를 이용한 새로운 구조 생성

    Args:
        structure1 (dict): 부모 구조1
        structure2 (dict): 부모 구조2

    Returns:
        structure (dict): 유전적 교차로 생성한 구조
    """
    prob = 0.5
    new_structure = {}
    keys = list(pipeline_components.keys())

    for k in keys:
        rand = random.random()
        if rand < prob:
            new_structure[k] = structure1[k]
        else:
            new_structure[k] = structure2[k]
    return new_structure


def mutation(structure, prob_mutation):
    """돌연변이 구조 생성

    Args:
        structure (dict): 입력 구조
        prob_mutation (float): 각 요소의 변이확률

    Returns:
        structure (dict): 돌연변이 구조
    """
    keys = list(pipeline_components.keys())
    
    for k in keys:
        rand = random.random()
        if rand < prob_mutation:
            element = choose_random_key(pipeline_components[k])
            if structure[k] != element:
                structure[k] = element
    
    return structure

def tune_hyperparameters(pipeline, param_grid, X_train, y_train, n_iter=50):
    """하이퍼파라미터 튜닝을 위한 Randomized Search 수행

    Args:
        pipeline (Pipeline): 모델 학습을 위한 파이프라인
        param_grid (dict): 하이퍼파라미터 검색 공간
        X_train (DataFrame): 학습 데이터의 특성(feature) 값
        y_train (Series): 학습 데이터의 타겟(target) 값
        n_iter (int, optional): 탐색할 하이퍼파라미터 조합 개수 (기본값: 50)

    Returns:
        best_estimator_ (Pipeline): 최적 하이퍼파라미터로 학습된 모델
        best_params_ (dict): 최적 하이퍼파라미터 값
        best_score_ (float): 검증 데이터에서의 최고 R2 점수
    """
    search = RandomizedSearchCV(
        pipeline, param_distributions=param_grid, n_iter=n_iter, 
        cv=3, scoring='r2', n_jobs=-1, verbose=1, random_state=42
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_

def get_param_grid(model_name):
    """모델별 하이퍼파라미터 검색 공간 정의

    Args:
        model_name (str): 사용할 모델의 이름

    Returns:
        param_grid (dict): 모델에 해당하는 하이퍼파라미터 검색 공간

    Raises:
        ValueError: 지원하지 않는 모델명이 입력된 경우
    """
    if model_name == 'DecisionTreeRegressor':
        return {
            'models__max_depth': range(1, 11),
            'models__min_samples_split': range(2, 21),
            'models__min_samples_leaf': range(1, 21)
        }
    elif model_name == 'RandomForestRegressor':
        return {
            'models__n_estimators': [100],
            'models__max_features': np.arange(0.05, 1.01, 0.05),
            'models__min_samples_split': range(2, 21),
            'models__min_samples_leaf': range(1, 21),
            'models__bootstrap': [True, False]
        }
    elif model_name == 'GradientBoostingRegressor':
        return{
            'models__n_estimators': [100],
            'models__loss': ["ls", "lad", "huber", "quantile"],
            'models__learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
            'models__max_depth': range(1, 11),
            'models__min_samples_split': range(2, 21),
            'models__min_samples_leaf': range(1, 21),
            'models__subsample': np.arange(0.05, 1.01, 0.05),
            'models__max_features': np.arange(0.05, 1.01, 0.05),
            'models__alpha': [0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
            
        }
    elif model_name == 'LogisticRegression':
        return{
            'models__penalty': ["l1", "l2"],
            'models__C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
            'models__dual': [True, False]
        }
    elif model_name == 'KNeighborsRegressor':
        return{
            'models__n_neighbors': range(1, 101),
            'models__weights': ["uniform", "distance"],
            'models__p': [1, 2]
        }
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

##################################

class AutoML:
    """
    유전 알고리즘을 이용한 ML pipeline 최적화 수행
    """
    def __init__(self, n_population=20, n_generation=50, n_parent=5, prob_mutation=0.1):
        self.n_population = n_population
        self.n_generation = n_generation
        self.n_parent = n_parent
        self.prob_mutation = prob_mutation
        self.n_child = n_population - n_parent

        py_dir_path = os.path.dirname(os.path.abspath(__file__))
        self.log_path = os.path.join(py_dir_path, "log.txt")


    def fit_structures(self, timeout=30):
        """ self.structures에 fitting 및 evaluation 수행

        Args:
            timeout (int, optional): Pipeline 별 최대 실행시간. 기본값 30초.
        """
        for i, structure in enumerate(self.structures):
            if 'valid_metric' in structure: # fitting 결과가 있으면 skip
                continue
            
            structure['valid_metric'] = {'r2': -100} # valid_metric 초기화
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)  # timeout 초 후에 알람 발생

            try:
                pipeline = structure['pipeline']
                pipeline.fit(self.X_train, self.y_train)
                y_train_pred = pipeline.predict(self.X_train)
                y_valid_pred = pipeline.predict(self.X_valid)
                structure['train_metric'] = evaluate_regression(self.y_train, y_train_pred)
                structure['valid_metric'] = evaluate_regression(self.y_valid, y_valid_pred)
                print(f"{i+1} structure - r2: {structure['valid_metric']['r2']}") # 결과 출력

            except TimeoutException as e:
                pipeline = []
                print(e)

            finally:
                signal.alarm(0)
    
    def log(self, message):
        """ log 기록

        Args:
            message (str): log 메세지
        """
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{now}] {message}"

        with open(self.log_path, 'a') as file:
            file.write(log_message + "\n") # 로그 저장
            file.flush()
            print(log_message) # 로그 출력 


    def predict(self, X):
        """얻은 최적 구조를 이용한 예측 수행

        Args:
            X (DataFrame): 예측할 X값

        Returns:
            y_pred (DataFrame): 예측된 y값
        """
        y_pred = self.best_structure['pipeline'].predict(X)
        return y_pred


    def fit(self, X_train, y_train, valid_size=0.2, seed=42, max_n_try=1000, timeout=30):
        """ 유전 알고리즘을 이용한 최적 모델 탐색

        Args:
            X_train (DataFrame): X_train
            y_train (DataFrame): y_train
            valid_size (float, optional): train data로 나눌 valid 비율. 기본값 0.2.
            seed (int, optional): 동일한 실험결과를 위한 시드 설정. 기본값 42.
            max_n_try (int, optional): 최대 새 구조 생성횟수. 기본값 1000.
            timeout (int, optional): Pipeline 별 최대 실행시간. 기본값 30.
        """

        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(X_train, y_train, 
                                                                                    test_size=valid_size, random_state=seed)
        self.structures = get_random_structures(self.n_population) # 임의 구조 생성
        keys = list(pipeline_components.keys())

        for generation in range(self.n_generation):
            self.fit_structures(timeout) # 형질 별 피팅 및 점수 계산
            self.structures = sort(self.structures) # 점수 높은 순으로 정렬
            self.best_structure = self.structures[0]
            self.best_score = self.best_structure['valid_metric']['r2']
            
            self.log(f"{generation+1} - best R2: {self.best_score:.3f}") # 최적값 기록
            self.log(" - ".join([str(self.best_structure[k]) for k in keys])) # 구조 기록

            if (generation+1 == self.n_generation):
                # 하이퍼파라미터 튜닝 수행
                model_name = self.best_structure['models'].__class__.__name__  # 모델 이름 가져오기
                print('model_name :', model_name)
                param_grid = get_param_grid(model_name)  # 모델에 맞는 파라미터 그리드 가져오기
                tuned_pipeline, best_params, best_score = tune_hyperparameters(
                    self.best_structure['pipeline'], param_grid, self.X_train, self.y_train
                )
                self.best_structure['pipeline'] = tuned_pipeline
                self.best_structure['tuned_params'] = best_params
                self.best_structure['tuned_score'] = best_score
                self.log(f"Tuned Best R2: {best_score:.3f}")
                break
            
            del self.structures[self.n_parent:] # 점수가 낮은 형질 제거
            
            n_success = 0
            n_try = 0

            while (n_success < self.n_child and n_try < max_n_try):
                n_try += 1
                structure1, structure2 = random.sample(self.structures, 2) # 임의로 형질 2개 고르기
                new_structure = crossover(structure1, structure2) # 형질 교차
                new_structure = mutation(new_structure, self.prob_mutation) # 형질 변형

                if is_in_structures(new_structure, self.structures): # 이미 존재하는 형질이면 재생성
                    continue

                new_structure['pipeline'] = build_pipeline(new_structure) # pipeline 생성
                self.structures.append(new_structure) 
                n_success += 1

            if not(n_try < max_n_try):
                print("Warning: max_n_try <= n_try")



def tune_hyperparameters(pipeline, param_grid, X_train, y_train, n_iter=50):
    """하이퍼파라미터 튜닝을 위한 Randomized Search 수행

    Args:
        pipeline (Pipeline): 모델 학습을 위한 파이프라인
        param_grid (dict): 하이퍼파라미터 검색 공간
        X_train (DataFrame): 학습 데이터의 특성(feature) 값
        y_train (Series): 학습 데이터의 타겟(target) 값
        n_iter (int, optional): 탐색할 하이퍼파라미터 조합 개수 (기본값: 50)

    Returns:
        best_estimator_ (Pipeline): 최적 하이퍼파라미터로 학습된 모델
        best_params_ (dict): 최적 하이퍼파라미터 값
        best_score_ (float): 검증 데이터에서의 최고 R2 점수
    """
    search = RandomizedSearchCV(
        pipeline, param_distributions=param_grid, n_iter=n_iter, 
        cv=3, scoring='r2', n_jobs=-1, verbose=1, random_state=42, error_score='raise')
    
    import pandas as pd

    X_train = pd.concat(X_train, ignore_index=True)  # 리스트 내부 DataFrame을 하나로 합침
    X_train = X_train.to_numpy()  # numpy 배열로 변환
    y_train = pd.concat(y_train, ignore_index=True)  # 리스트 내부 DataFrame을 하나로 합침
    y_train = y_train.to_numpy()  # numpy 배열로 변환
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_

def get_param_grid(model_name):
    """모델별 하이퍼파라미터 검색 공간 정의

    Args:
        model_name (str): 사용할 모델의 이름

    Returns:
        param_grid (dict): 모델에 해당하는 하이퍼파라미터 검색 공간

    Raises:
        ValueError: 지원하지 않는 모델명이 입력된 경우
    """
    if model_name == 'DecisionTreeClassifier':
        return {
            'models__max_depth': range(1, 11),
            'models__min_samples_split': range(2, 21),
            'models__min_samples_leaf': range(1, 21),
            'models__class_weight': ['balanced', None]
        }
    elif model_name == 'RandomForestClassifier':
        return {
            'models__n_estimators': [100],
            'models__max_features': np.arange(0.05, 1.01, 0.05),
            'models__min_samples_split': range(2, 21),
            'models__min_samples_leaf': range(1, 21),
            'models__bootstrap': [True, False],
            'models__class_weight': ['balanced', None]
        }
    elif model_name == 'GradientBoostingClassifier':
        return {
            'models__n_estimators': [100],
            'models__learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
            'models__max_depth': range(1, 11),
            'models__min_samples_split': range(2, 21),
            'models__min_samples_leaf': range(1, 21),
            'models__subsample': np.arange(0.05, 1.01, 0.05),
            'models__max_features': np.arange(0.05, 1.01, 0.05),
        }
    elif model_name == 'LogisticRegression':
        return {
            'models__C': np.arange(0., 5.),
        }
    elif model_name == 'KNeighborsClassifier':
        return {
            'models__n_neighbors': range(1, 101),
            'models__weights': ["uniform", "distance"],
            'models__p': [1, 2],
            'models__leaf_size': [30, 40, 50],
            'models__metric': ['euclidean', 'manhattan']
        }
    elif model_name == 'SGDClassifier':
        return {
            'models__loss': ['hinge', 'log', 'modified_huber', 'squared_hinge'],
            'models__alpha': [0.0001, 0.001, 0.01, 0.1],
            'models__penalty': ['l2', 'l1', 'elasticnet'],
            'models__class_weight': ['balanced', None],
            'models__max_iter': [1000, 2000, 3000],
            'models__learning_rate': ['constant', 'optimal', 'invscaling']
        }
    elif model_name == 'XGBClassifier':
        return {
            'models__n_estimators': [100],
            'models__learning_rate': [1e-2, 0.05, 0.1, 0.3],
            'models__max_depth': range(1, 11),
            'models__min_child_weight': [1, 2, 3],
            'models__subsample': np.arange(0.5, 1.1, 0.1),
            'models__colsample_bytree': np.arange(0.5, 1.1, 0.1),
            'models__scale_pos_weight': [1, 2, 3],
            'models__gamma': [0, 0.1, 0.2]
        }
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

##################################