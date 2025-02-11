from sklearn.pipeline import Pipeline
from sklearn.base import clone
from joblib import Parallel, delayed
from copy import deepcopy
from metrics import evaluate_regression, evaluate_classification, compute_metrics_statistics
import signal
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

def fit_structures(self, timeout=30):
    """
    self.structures에 fitting 및 evaluation 수행

    Args:
        timeout (int, optional): Pipeline 별 최대 실행시간. 기본값 30초.
    """

    if self.use_joblib:
        self.structures = Parallel(n_jobs=self.n_jobs)(
                            delayed(fit_structure)(self, structure, timeout, i)
                            for i, structure in enumerate(self.structures)
                            )
        
    else:
        self.structures = [fit_structure(self, structure, timeout, i) for i, structure in enumerate(self.structures)]


def fit_structure(self, structure, timeout=30, order=0):
    """
    입력 structure에 fitting 및 evaluation 수행

    Args:
        structure (dict): 입력 구조
        timeout (int, optional): Pipeline 별 최대 실행시간. 기본값 30초.
    """
    if 'valid_metric' in structure: # fitting 결과가 있으면 skip
        return structure
    
    # valid_metric 초기화
    structure['valid_metric'] = {'r2': -100, 'r2_std': -100} if self.task_type == 'regression' else {'accuracy': -100, 'f1': -100}

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)  # timeout 초 후에 알람 발생
    pipeline = structure['pipeline']
    train_metrics = []
    valid_metrics = []

    try:
        for i in range(len(self.X_trains)):
            pipeline_clone = clone(pipeline)
            X_train, y_train = self.X_trains[i], self.y_trains[i]
            X_valid, y_valid = self.X_valids[i], self.y_valids[i]

            pipeline_clone.fit(X_train, y_train)
            y_train_pred = pipeline_clone.predict(X_train)
            y_valid_pred = pipeline_clone.predict(X_valid)

            if self.task_type == 'regression':
                train_metric = evaluate_regression(y_train, y_train_pred)
                valid_metric = evaluate_regression(y_valid, y_valid_pred)
                
            else:
                train_metric = evaluate_classification(y_train, y_train_pred)
                valid_metric = evaluate_classification(y_valid, y_valid_pred)
                
            train_metrics.append(train_metric)
            valid_metrics.append(valid_metric)

        structure['pipeline'] = pipeline_clone
        structure['train_metric'] = compute_metrics_statistics(train_metrics)
        structure['valid_metric'] = compute_metrics_statistics(valid_metrics)
        structure['train_metrics'] = train_metrics
        structure['valid_metrics'] = valid_metrics

    except TimeoutException as e:
        print(e)

    finally:
        signal.alarm(0) # alarm 초기화
        
    if self.task_type == 'classification':
        valid_f1 = structure['valid_metric']['f1']
        valid_accuracy = structure['valid_metric']['accuracy']
        print(f"Structure-{order} - valid_f1: {valid_f1:.4f}")
        print(f"Structure-{order} - valid_accuracy: {valid_accuracy:.4f}")
    else:
        valid_r2 = structure['valid_metric']['r2']
        valid_r2_std = structure['valid_metric']['r2_std']
        print(f"Structure-{order} - valid r2: {valid_r2:.4f}±{valid_r2_std:.4f}")
        
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
    
def generate_structure_summary(self, structure):
    """
    주어진 구조체에 대한 요약 정보를 생성하여 반환하는 함수

    Args:
        structure (dict): 구조체 정보 딕셔너리, 각 구성 요소와 그에 대한 파라미터 및 메트릭 포함

    Returns:
        str: 구조체의 요약 정보를 포함하는 문자열
    """
    structure_summary = [] # 요약된 정보를 저장할 리스트
    keys = list(self.pipeline_components.keys())
    
    # 파이프라인에 대한 정보를 요약
    for k in keys:
        class_info = structure[k]
        if isinstance(class_info, str):
            s = class_info
        else:
            class_name = class_info['class_name']
            if 'params' in class_info:
                params = ', '.join([f'{k}: {v}' if not isinstance(v, float) else f'{k}: {v:.4f}'
                                    for k, v in class_info['params'].items()])
                s = f"({class_name}: {params})"
            else:
                s = f"({class_name})"  

        structure_summary.append(s)

    # valid_metric이 존재하는 경우, 해당 메트릭 정보를 추가
    if 'valid_metric' in structure:
        if self.task_type == 'classification':
            f1 = structure['valid_metric']['f1']
            accuracy = structure['valid_metric']['accuracy']
            structure_summary.append(f'f1:{f1:.4f}, accuracy:{accuracy:.4f}')
        else:
            r2 = structure['valid_metric']['r2']
            r2_std = structure['valid_metric']['r2_std']
            structure_summary.append(f'{r2:.4f}±{r2_std:.4f}')

    s = ' - '.join(structure_summary)
    return s