from sklearn.pipeline import Pipeline
from sklearn.base import clone
from copy import deepcopy
import random

choose_random_key = lambda dictionary: random.choice(list(dictionary.keys()))

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Timeout: pipeline.fit did not complete in given time.")

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
        cl = v['class']
        if 'params' in v: # params 값이 v에 있으면 instance 생성
            _pipeline.append((k, clone(cl(**v['params']))))
        else:
            _pipeline.append((k, clone(cl))) # instance 복사
                                     
    return Pipeline(_pipeline)

def fit_pipeline(pipeline, X_train, y_train):
    pipeline.fit(X_train, y_train)    

def crossover(structure1, structure2, pipeline_components):
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
    
    return deepcopy(new_structure)


def mutation(pipeline_components, structure, prob_mutations, hyperparam_bound=[0.5, 2.0]):
    """
    돌연변이 구조 생성

    Args:
        structure (dict): 입력 구조
        prob_mutations (List): 각 요소의 변이확률 (첫 번째: 구조변이, 두 번째: hyperparameter 변이)

    Returns:
        structure (dict): 돌연변이 구조
    """
    
    keys = list(pipeline_components.keys())
    
    # 구조 변이
    for k in keys:
        rand = random.random()
        if rand < prob_mutations[0]: 
            element = choose_random_key(pipeline_components[k])
            if structure[k]['class_name'] != element:
                structure[k] = deepcopy(pipeline_components[k][element])
                structure[k]['class_name'] = element

    # 하이퍼 파라미터 변이
    for k, v in structure.items():
        if 'params' not in v: # 파라미터가 없으면 continue
            continue

        params = v['params']

        for param_name, param_value in params.items():
            rand = random.random()
            if prob_mutations[1] <= rand:
                continue
                   
            origin_type = type(param_value)
            
            # param_value가 숫자일 때만 곱셈을 수행
            if isinstance(param_value, (int, float)):
                rand = random.uniform(hyperparam_bound[0], hyperparam_bound[1])
                params[param_name] = origin_type(rand * param_value)
            else:
                # 그 외의 경우에는 변이 X
                print(f"Skipping parameter {param_name} with value type {origin_type}.")
            
    return structure

def sort(structures, task_type):
    """
    structures를 평가지표를 기준으로 정렬

    Args:
        structures (list): 정렬되지 않은 구조

    Returns:
        structures (list): ['valid_metric']['r2']를 기준으로 정렬
    """
    
    if task_type == 'regression':
        # 회귀 문제에서는 'r2'를 기준으로 정렬
        return sorted(structures, key=lambda x: x['valid_metric']['r2'], reverse=True)
    elif task_type == 'classification':
        # 분류 문제에서는 'f1'를 기준으로 정렬
        return sorted(structures, key=lambda x: x['valid_metric']['f1'], reverse=True)

def is_in_structures(structure, structures, pipeline_components):
    """
    structures 내 동일한 structure가 있는지 확인

    Args:
        structure (dict): 구조
        structures (list): 구조 집합

    Returns:
        Bool: 동일하면 True, 다르면 False 반환
    """
    
    for s in structures:
        if is_same_structure(structure, s, pipeline_components):
            return True
    return False

def is_same_structure(structure1, structure2, pipeline_components):
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