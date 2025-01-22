# 본 코드는 TPOT (Evaluation of a Tree-based Pipeline Optimization Tool for Automating Data Science, GECCO '16)에서
# 아이디어를 얻어 구현을 수행했습니다.

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, SelectPercentile, VarianceThreshold
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from tqdm import tqdm
import multiprocessing

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    mean_squared_log_error,
    explained_variance_score
)

import random
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError, ProcessPoolExecutor
import math
import signal


preprocessors = {'StandardScaler': StandardScaler(), 'RobustScaler': RobustScaler(), 
                 'PolynomialFeatures': PolynomialFeatures(), 'PCA': PCA()}

feature_selections = {'SelectKBest': SelectKBest(), 'SelectPercentile': SelectPercentile(),
                      'VarianceThreshold': VarianceThreshold()} # RFE()

models = {'DecisionTreeRegressor': DecisionTreeRegressor(), 'RandomForestRegressor': RandomForestRegressor(), 
          'GradientBoostingRegressor': GradientBoostingRegressor(), 'LogisticRegression': LogisticRegression(),
          'KNeighborsRegressor': KNeighborsRegressor()}


def evaluate_regression(y_true, y_pred, dataset_name="Dataset"):
    r2 = r2_score(y_true, y_pred)
    RMSE = math.sqrt(mean_squared_error(y_true, y_pred))
    
    dicts = {'r2': r2, 'RMSE': RMSE}
    return dicts


def build_pipeline(structure):
    _pipeline = []
    for k, v in structure.items():
        _pipeline.append((k, v))
    return Pipeline(_pipeline)

choose_random_key = lambda dictionary: random.choice(list(dictionary.values()))

def get_random_structures(n):
    random_structures = []
    for _ in range(n):
        random_structure = {'preprocessor': choose_random_key(preprocessors),
                            'feature_selection': choose_random_key(feature_selections),
                            'model': choose_random_key(models)}
        
        random_structure['pipeline'] = build_pipeline(random_structure)
        random_structures.append(random_structure)
        
    return random_structures

def fit_pipeline(pipeline, X_train, y_train):
    pipeline.fit(X_train, y_train)


class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Timeout: pipeline.fit did not complete in given time.")


def fit_structures(structures, timeout=30):
    structures = [structures] if isinstance(structures, dict) else structures
    for i, structure in enumerate(structures):
        if 'train_metric' in structure: # fitting 결과가 있으면 skip
            continue
        
        structure['test_metric'] = {'r2': -100}

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)  # timeout 초 후에 알람 발생

        try:
            pipeline = structure['pipeline']
            pipeline.fit(X_train, y_train)
            y_train_pred = pipeline.predict(X_train)
            y_test_pred = pipeline.predict(X_test)
            structure['train_metric'] = evaluate_regression(y_train, y_train_pred)
            structure['test_metric'] = evaluate_regression(y_test, y_test_pred)
        except TimeoutException as e:
            print(e)
        finally:
            signal.alarm(0)
        
        print(f"{i+1} structure - r2: {structure['test_metric']['r2']}")
    

def sort(structures):    
    r2s = [structure['test_metric']['r2'] for structure in structures]
    r2s.sort(reverse=True)
    sorted_structures = []

    for r2 in r2s:
        for structure in structures:
            if r2 == structure['test_metric']['r2']:
                sorted_structures.append(structure)
 
    return sorted_structures


def is_same_structure(structure1, structure2):
    keys = ['preprocessor', 'feature_selection', 'model']
    for k in keys:
        if structure1[k] != structure2[k]:
            return False
    return True


def is_in_structures(structure, structures):
    for s in structures:
        if is_same_structure(structure, s):
            return True
    return False


def crossover(structure1, structure2):    
    keys = ['preprocessor', 'feature_selection', 'model']
    new_structure = {}
    for k in keys:
        rand = random.random()
        if rand < 0.5:
            new_structure[k] = structure1[k]
        else:
            new_structure[k] = structure2[k]

    return new_structure


def mutation(structure, prob_mutation):
    keys = ['preprocessor', 'feature_selection', 'model']
    
    for k in keys:
        rand = random.random()
        if rand < prob_mutation:
            if (k == 'preprocessor'): 
                structure[k] = choose_random_key(preprocessors)
            elif (k == 'feature_selection'): 
                structure[k] = choose_random_key(feature_selections)
            elif (k == 'model'):
                structure[k] = choose_random_key(models)
    
    return structure


class AutoML:
    def __init__(self, n_population=20, n_generation=50, n_parent=5, prob_mutation=0.1):
        self.n_population = n_population
        self.n_generation = n_generation
        self.n_parent = n_parent
        self.prob_mutation = prob_mutation
        self.n_child = n_population - n_parent

    def fit(self, X_train, y_train):
            self.structures = get_random_structures(self.n_population)


if __name__ == "__main__":

    structures = get_random_structures(n_population)

    for generation in range(n_generation):
        fit_structures(structures)
        structures = sort(structures) # 점수가 높은 순으로 정렬

        print(f"{generation+1} generation - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(structures)

        if (generation+1 == n_generation):
            break 

        del structures[n_parent:] # 점수가 낮은 형질 제거

        for _ in range(n_child):
            structure1, structure2 = random.sample(structures, 2)
            new_structure = crossover(structure1, structure2)
            new_structure = mutation(new_structure, prob_mutation)
            new_structure['pipeline'] = build_pipeline(new_structure)
            structures.append(new_structure)


