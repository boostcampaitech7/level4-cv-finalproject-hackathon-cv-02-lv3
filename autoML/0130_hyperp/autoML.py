# 본 코드는 TPOT (Evaluation of a Tree-based Pipeline Optimization Tool
# for Automating Data Science, GECCO '16)에서 아이디어를 얻어 구현했습니다.

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, SelectPercentile, VarianceThreshold, f_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.base import clone
from xgboost import XGBRegressor

import random
from datetime import datetime
import math
import signal
import os
from joblib import Parallel, delayed
import numpy as np
import statistics
from collections import defaultdict


preprocessors = {'StandardScaler': StandardScaler(), 'RobustScaler': RobustScaler(), 
                 'PolynomialFeatures': PolynomialFeatures(), 'passthrough': 'passthrough'}

feature_selections = {'SelectKBest': SelectKBest(score_func=f_regression), 
                      'SelectPercentile': SelectPercentile(score_func=f_regression),
                      'VarianceThreshold': VarianceThreshold(),
                      'passthrough': 'passthrough'} 


models = {'DecisionTreeRegressor': DecisionTreeRegressor(), 'RandomForestRegressor': RandomForestRegressor(), 
          'GradientBoostingRegressor': GradientBoostingRegressor(), 'LogisticRegression': LogisticRegression(),
          'KNeighborsRegressor': KNeighborsRegressor(), 'XGBRegressor': XGBRegressor()} 

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
    """
    ML 계산을 위한 scikit pipeline 생성

    Args:
        structure (dict): pipeline을 생성하기 위한 구조체

    Returns:
        Pipeline: structure로 생성한 Pipeline
    """
    
    _pipeline = []
    for k, v in structure.items():
        if isinstance(v, str):
            _pipeline.append((k, v))
        else:
            _pipeline.append((k, clone(v)))
    return Pipeline(_pipeline)


def get_random_structures(n):
    """
    n개의 임의의 structure 생성

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
    """
    structures를 평가지표를 기준으로 정렬

    Args:
        structures (list): 정렬되지 않은 구조

    Returns:
        structures (list): ['valid_metric']['r2']를 기준으로 정렬
    """
    return sorted(structures, key=lambda x: x['valid_metric']['r2'], reverse=True)


def is_same_structure(structure1, structure2):
    """
    structure1과 structure2가 동일한지 확인

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
    """
    structures 내 동일한 structure가 있는지 확인

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
    """
    유전적 교차를 이용한 새로운 구조 생성

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
    """
    돌연변이 구조 생성

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

def average_metrics(metrics):
    """
    각 메트릭의 평균과 표준 편차를 계산

    Parameters:
    metrics (List[Dict[str, float]]): 여러 번의 측정에서 얻은 메트릭 딕셔너리들의 리스트.
                                      각 딕셔너리는 메트릭 이름을 키로 하며, 그 값은 측정된 값.

    Returns:
    Dict[str, float]: 각 메트릭의 평균과 표준 편차를 포함하는 딕셔너리.
                      메트릭 이름으로 평균 값이, '메트릭_std' 형태로 표준 편차가 저장.
    """

    # 모든 메트릭 딕셔너리를 순회하며 값을 그룹화
    grouped_metrics = defaultdict(list)
    for metric in metrics:
        for key, value in metric.items():
            grouped_metrics[key].append(value)
    
    # 평균과 표준 편차를 저장할 딕셔너리
    avg_metric = {}
    for key, values in grouped_metrics.items():
        avg_metric[key] = statistics.mean(values)
        avg_metric[f'{key}_std'] = statistics.stdev(values)

    return avg_metric


class AutoML:
    """
    유전 알고리즘을 이용한 ML pipeline 최적화 수행
    """
    def __init__(self, n_population=20, n_generation=50, n_parent=5, prob_mutation=0.1, use_joblib=True, n_jobs=-1):
        self.n_population = n_population
        self.n_generation = n_generation
        self.n_parent = n_parent
        self.prob_mutation = prob_mutation
        self.use_joblib = use_joblib
        self.n_jobs = n_jobs
        self.n_child = n_population - n_parent

        py_dir_path = os.path.dirname(os.path.abspath(__file__))
        self.log_path = os.path.join(py_dir_path, "log.txt")


    def fit_structures(self, timeout=30):
        """
        self.structures에 fitting 및 evaluation 수행

        Args:
            timeout (int, optional): Pipeline 별 최대 실행시간. 기본값 30초.
        """
        
        if self.use_joblib:
            self.structures = Parallel(n_jobs=self.n_jobs)(
                              delayed(self.fit_structure)(structure, timeout)
                              for structure in self.structures
                              )
            
        else:
            self.structures = [self.fit_structure(structure, timeout) for structure in self.structures]


    def fit_structure(self, structure, timeout=30):
        """
        입력 structure에 fitting 및 evaluation 수행

        Args:
            structure (dict): 입력 구조
            timeout (int, optional): Pipeline 별 최대 실행시간. 기본값 30초.
        """
        if 'valid_metric' in structure: # fitting 결과가 있으면 skip
            return structure
        
        structure['valid_metric'] = {'r2': -100} # valid_metric 초기화
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)  # timeout 초 후에 알람 발생
        pipeline = structure['pipeline']
        train_metrics = []
        valid_metrics = []

        try:
            for i in range(len(self.X_trains)):
                clone_pipeline = clone(pipeline)
                X_train, y_train = self.X_trains[i], self.y_trains[i]
                X_valid, y_valid = self.X_valids[i], self.y_valids[i]

                clone_pipeline.fit(X_train, y_train)
                y_train_pred = pipeline.predict(X_train)
                y_valid_pred = pipeline.predict(X_valid)

                train_metric = evaluate_regression(y_train, y_train_pred)
                valid_metric = evaluate_regression(y_valid, y_valid_pred)
                train_metrics.append(train_metric)
                valid_metrics.append(valid_metric)


            # print(f"structure - r2: {structure['valid_metric']['r2']}") # 결과 출력

        except TimeoutException as e:
            print(e)

        finally:
            signal.alarm(0) # alarm 초기화
        
        pipeline = clone_pipeline
        structure['train_metric'] = average_metrics(train_metrics)
        structure['valid_metric'] = average_metrics(valid_metrics)
        valid_r2 = structure['valid_metric']['r2']
        valid_r2_std = structure['valid_metric']['r2_std']
        
        print(f"structure - valid r2: {valid_r2:.4f}±{valid_r2_std:.4f}") # 결과 출력
        return structure


    
    def log(self, message):
        """
        log 기록

        Args:
            message (str): log 메세지
        """
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{now}] {message}"
        print(log_message) # 로그 출력 

        # 로그 저장
        with open(self.log_path, 'a') as file:
            file.write(log_message + "\n") 
            file.flush()


    def predict(self, X):
        """
        얻은 최적 구조를 이용한 예측 수행

        Args:
            X (DataFrame): 예측할 X값

        Returns:
            y_pred (DataFrame): 예측된 y값
        """
        y_pred = self.best_structure['pipeline'].predict(X)
        return y_pred


    def fit(self, X_train, y_train, use_kfold=True, kfold=5, valid_size=0.2, seed=42, max_n_try=1000, timeout=30):
        """
        유전 알고리즘을 이용한 최적 모델 탐색

        Args:
            X_train (DataFrame): X_train
            y_train (DataFrame): y_train
            use_kfold (bool, optional): k-fold validation 사용 여부. 기본값 True.
            kfold (int, optional): k-fold 수. 기본값 5.
            valid_size (float, optional): k-fold validation을 사용하지 않을 때 train와 valid 비율. 기본값 0.2.
            seed (int, optional): 동일한 실험결과를 위한 시드 설정. 기본값 42.
            max_n_try (int, optional): 최대 새 구조 생성 시도횟수. 기본값 1000.
            timeout (int, optional): Pipeline 별 최대 실행시간. 기본값 30.
        """

        random.seed(seed)
        np.random.seed(seed)

        if use_kfold: # k-fold validation으로 모델 평가
            kf = KFold(n_splits=kfold, shuffle=True, random_state=seed)
            self.X_trains, self.X_valids, self.y_trains, self.y_valids = [], [], [], [] # 초기화
            
            for train_index, valid_index in kf.split(X_train):
                self.X_trains.append(X_train[train_index])
                self.X_valids.append(X_train[valid_index])
                self.y_trains.append(y_train[train_index])
                self.y_valids.append(y_train[valid_index])
 
        else: # single-fold validation으로 모델 평가
            X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, 
                                                                    test_size=valid_size, random_state=seed)
            self.X_trains = [X_train] # List로 변환
            self.X_valids = [X_valid]
            self.y_trains = [y_train]
            self.y_valids = [y_valid]


            
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
