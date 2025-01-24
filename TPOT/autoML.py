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

import random
from datetime import datetime
import math
import signal
import os


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