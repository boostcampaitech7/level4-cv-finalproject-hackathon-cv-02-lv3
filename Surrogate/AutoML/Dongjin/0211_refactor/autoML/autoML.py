# 본 코드는 TPOT (Evaluation of a Tree-based Pipeline Optimization Tool
# for Automating Data Science, GECCO '16)에서 아이디어를 얻어 구현했습니다.

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures, FunctionTransformer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.feature_selection import SelectKBest, SelectPercentile, VarianceThreshold, f_regression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, f1_score
from sklearn.base import clone
from xgboost import XGBRegressor, XGBClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.inspection import permutation_importance
import random
from datetime import datetime
import math
import signal
import os
from joblib import Parallel, delayed
import numpy as np
import statistics
from collections import defaultdict
from copy import deepcopy
import warnings
from typing import Dict, Sequence, Any, List, Tuple


# feature_selection, ConvergenceWarning warning 무시
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.feature_selection")
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# 사용할 유전 알고리즘에서 preprocessors, feature_selections, models_regression, models_classification 정의
preprocessors = {'StandardScaler': {'class': StandardScaler()},
                 'RobustScaler': {'class': RobustScaler()}, 
                 'PolynomialFeatures': {'class': PolynomialFeatures()},
                 'passthrough': {'class': FunctionTransformer(func=lambda X: X)}}

feature_selections = {'SelectKBest': {'class': SelectKBest(score_func=f_regression)}, 
                      'SelectPercentile': {'class': SelectPercentile(score_func=f_regression)},
                      'VarianceThreshold': {'class': VarianceThreshold()},
                      'passthrough': {'class': FunctionTransformer(func=lambda X: X)}}

models_regression = {
    'DecisionTreeRegressor': {'class': DecisionTreeRegressor()},
    'RandomForestRegressor': {'class': RandomForestRegressor,'params': {'n_estimators': 100}}, 
    'GradientBoostingRegressor': {'class': GradientBoostingRegressor, 'params': {'n_estimators': 100, 'learning_rate': 0.1}},
    'LogisticRegression': {'class': LogisticRegression, 'params': {'C': 1.0}},
    'KNeighborsRegressor': {'class': KNeighborsRegressor, 'params': {'n_neighbors': 5}},
    'XGBRegressor': {'class': XGBRegressor, 'params': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6}}}

models_classification = {
    'DecisionTreeClassifier': {'class': DecisionTreeClassifier, 'params': {'max_depth': 6, 'class_weight': 'balanced'}},
    'RandomForestClassifier': {'class': RandomForestClassifier, 'params': {'n_estimators': 100, 'class_weight': 'balanced'}},
    'GradientBoostingClassifier': {'class': GradientBoostingClassifier, 'params': {'max_depth': 10, 'n_estimators': 100, 'learning_rate': 0.1}},
    'LogisticRegression': {'class': LogisticRegression, 'params': {'C': 1.0, 'class_weight': 'balanced'}},
    'KNeighborsClassifier': {'class': KNeighborsClassifier, 'params': {'n_neighbors': 5}},
    'BernoulliNB': {'class': BernoulliNB, 'params': {'alpha':0.01}},
    'SGDClassifier': {'class': SGDClassifier, 'params': {'class_weight': 'balanced', 'alpha': 0.001, 'power_t': 0.5}},
    'XGBClassifier': {'class': XGBClassifier, 'params': {'n_estimators': 100, 'learning_rate': 1.0, 'max_depth':3, 'scale_pos_weight': 1}}}

# dicts에서 임의로 key값 선택
choose_random_key = lambda dictionary: random.choice(list(dictionary.keys()))


class TimeoutException(Exception):
    """
    파이프라인 탐색 시간초과 기능 구현을 위한 TimeoutException.
    """
    pass

def timeout_handler(signum, frame):
    """
    파이프라인 탐색 시간초과 기능 구현을 위한 timeout_handler.
    """
    raise TimeoutException("Timeout: pipeline.fit did not complete in given time.")


def evaluate_regression(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    """
    회귀 모델 평가: R2 스코어와 RMSE를 계산.

    Args:
        y_true (Sequence[float]): 실제 타겟 값.
        y_pred (Sequence[float]): 예측한 타겟 값.

    Returns:
        Dict[str, float]: 'r2'와 'RMSE' 키를 가진 평가 지표 딕셔너리.
    """
    r2 = r2_score(y_true, y_pred)
    RMSE = math.sqrt(mean_squared_error(y_true, y_pred))    
    metrics = {'r2': r2, 'RMSE': RMSE}
    return metrics


def evaluate_classification(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    """
    분류 모델 평가: 정확도와 F1 스코어를 계산.

    Args:
        y_true (Sequence[int]): 실제 레이블.
        y_pred (Sequence[int]): 예측한 레이블.

    Returns:
        Dict[str, float]: 'accuracy'와 'f1' 키를 가진 평가 지표 덕셔너리.
    """
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    metrics = {'accuracy': accuracy, 'f1': f1}
    return metrics


def build_pipeline(structure: dict) -> Pipeline:
    """
    ML 계산을 위한 scikit-learn Pipeline 생성

    Args:
        structure (dict): pipeline을 생성을 위한 구조체.

    Returns:
        Pipeline: structure로 생성한 Pipeline
    """
    
    _pipeline = []
    for step, config in structure.items():
        if 'params' in config:
            clas = config['class']
            params = config['params']
            _pipeline.append((step, clone(clas(**params)))) # params를 인자로 인스턴스 생성 
        else: 
            instance = config['class']
            _pipeline.append((step, clone(instance))) # 생성되었던 인스턴스 복사

    return Pipeline(_pipeline)


def get_random_structures(n: int, task_type: str) -> list:
    """
    n개의 임의의 structure 생성

    Args:
        n (int): 임의로 생성할 structure 개수
        task_type (str): 'regression' 혹은 'classification'

    Returns:
        random_structures (list): 임의로 생성한 structure
    """
    random_structures = []
    
    for _ in range(n):
        if task_type == 'regression':
            random_structure = deepcopy({'preprocessors': preprocessors,
                                         'feature_selections': feature_selections,
                                         'models': models_regression})
            
        elif task_type == 'classification':
            random_structure = deepcopy({'preprocessors': preprocessors,
                                         'feature_selections': feature_selections,
                                         'models': models_classification})
        
        for step, operators in random_structure.items():
            operator_name = choose_random_key(operators)
            random_structure[step] = operators[operator_name]
            if isinstance(random_structure[step], dict):
                random_structure[step]['class_name'] = operator_name
        
        random_structure['pipeline'] = build_pipeline(random_structure)
        random_structures.append(random_structure)
        
    return random_structures


def sort_structures(structures: List, task_type: str) -> List:
    """
    structures를 평가지표를 기준으로 정렬

    Args:
        structures (List): 정렬되지 않은 structure 리스트.
        task_type (str): 'regression'이면 'valid_metric'의 r2 기준,
                         'classification'이면 'valid_metric'의 'f1' 기준으로 정렬.

    Returns:
        structures (List): 평가지표 기준으로 정렬된 structure 리스트.
    """
    if task_type == 'regression':
        key_metric = 'r2'
    elif task_type == 'classification':
        key_metric = 'f1'

    return sorted(structures, key=lambda x: x['valid_metric'][key_metric], reverse=True)


def is_same_structure(structure1: Dict, structure2: Dict, pipeline_components: Dict):
    """
    structure1과 structure2가 동일한지 확인

    Args:
        structure1 (Dict): 첫 번째 구조체
        structure2 (Dict): 두 번째 구조체
        pipeline_components (Dict): 비교에 사용될 pipeline 구성 요소

    Returns:
        Bool: 모든 구조가 동일하면 True, 그렇지 않으면 False 반환
    """
    keys = list(pipeline_components.keys())
    for k in keys:
        if structure1[k] != structure2[k]:
            return False
    return True


def is_in_structures(structure: Dict, structures:List[Dict], pipeline_components: Dict) -> bool:
    """
    structure와 동일한 구조체가 structures 내에 존재하는지 확인 

    Args:
        structure (Dict): 비교 대상 구조체
        structures (List): 구조체들의 리스트
        pipeline_components (Dict): 비교에 사용될 pipeline 구성 요소

    Returns:
        Bool: 동일한 구조체가 하나라도 있으면 True, 없으면 False 반환
    """
    for s in structures:
        if is_same_structure(structure, s, pipeline_components):
            return True
    return False


def crossover(structure1: Dict, structure2: Dict, pipeline_components: Dict) -> Dict:
    """
    유전적 교차를 이용한 새로운 구조 생성
    각 pipeline 구성요소에 대해 50% 확률로 두 부모 구조 중 하나의 값을 선택

    Args:
        structure1 (Dict): 부모 구조 1
        structure2 (Dict): 부모 구조 2
        pipeline_components (Dict): 교차할 pipeline 구성 요소들을 나타내는 딕셔너리


    Returns:
        structure (Dict): 유전적 교차로 생성한 구조
    """
    new_structure = {
        k: structure1[k] if random.random() < 0.5 else structure2[k]
        for k in pipeline_components
    }
    return deepcopy(new_structure)


def mutation(pipeline_components: Dict,
             structure: Dict,
             prob_mutations: List,
             hyperparam_bound: Tuple[float, float] = (0.5, 2.0)
             ) -> Dict:
    """
    돌연변이 구조 생성

    각 piepline 구성요소에 대해,
    - 구조 변이: prob_mutations[0] 확률로 해당 구성요소를 다른 옵션으로 교체
    - 하이퍼 파라미터 변이: prob_mutations[1] 확률로 숫자형 하이퍼파라미터 값을 변형

    Args:
        pipeline_components (Dict): pipeline 구성 요소들
        structure (Dict): 입력 구조
        prob_mutations (List):
            [구조 변이 확률, 하이퍼 파라미터 변이 확률]
        hyperparams_bound (Tuple[float, float], optional):
            하이퍼 파라미터 변이 시 곱해질 계수의 범위. Default to (0.5, 2.0).

    Returns:
        structure (Dict): 돌연변이가 적용된 구조체
    """

    
    # 구조 변이: 각 구성 요소에 대해 확률(prob_mutations[0])에 따라 다른 옵션으로 교체
    for k in pipeline_components:
        if random.random() < prob_mutations[0]: 
            element = choose_random_key(pipeline_components[k])
            # 현재 구성 요소의 'class_name'과 선택된 옵션이 다른 경우에만 교체
            if structure[k]['class_name'] != element:
                structure[k] = deepcopy(pipeline_components[k][element])
                structure[k]['class_name'] = element

    # 하이퍼 파라미터 변이: 각 구성 요소의 파라미터에 대해 확률(prob_mutations[1])에 따라 값 변형
    for k, v in structure.items():
        if 'params' not in v: # 하이퍼 파라미터가 없으면 continue
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
                params[param_name] = origin_type(rand * param_value) # 원래 type으로 casting
            else:
                # 그 외의 경우에는 변이 X
                print(f"Skipping parameter {param_name} with value type {origin_type}.")
            
    return structure


def average_metrics(metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """
    여러 번 측정에서 얻은 메트릭 딕셔너리들의 리스트로 
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

        if 1 < len(values):
            avg_metric[f'{key}_std'] = statistics.stdev(values)
        else: 
            avg_metric[f'{key}_std'] = 0

    return avg_metric


class AutoML:
    """
    유전 알고리즘을 이용한 ML pipeline 최적화 수행
    """
    def __init__(self, task_type='regression', n_population=30, n_generation=50, n_parent=5, prob_mutations=[0.2, 0.5], use_joblib=True, n_jobs=-1):
        assert task_type in ['regression', 'classification'], "task_type은 'regression' 또는 'classification' 이어야 합니다."

        self.task_type = task_type  # task type 설정
        self.n_population = n_population
        self.n_generation = n_generation
        self.n_parent = n_parent
        self.prob_mutations = prob_mutations
        self.use_joblib = use_joblib
        self.n_jobs = n_jobs
        self.n_child = n_population - n_parent
        
        if self.task_type == "classification":
            self.pipeline_components = {'preprocessors': preprocessors, 'feature_selections': feature_selections, 'models': models_classification}
        else:
            self.pipeline_components = {'preprocessors': preprocessors, 'feature_selections': feature_selections, 'models': models_regression}
        

        dicts = {'task_type' : task_type, 'n_population': n_population, 'n_generation': n_generation, 'n_parent': n_parent,
                 'prob_mutations': prob_mutations, 'use_joblib': use_joblib, 'n_jobs': n_jobs}
        
        now = datetime.now()
        time_string = now.strftime("%y%m%d_%H%M%S")
        py_dir_path = os.path.dirname(os.path.abspath(__file__))

        self.log_dir_path = os.path.join(py_dir_path, 'log')
        self.log_path = os.path.join(self.log_dir_path, f"{time_string}.txt")
        os.makedirs(self.log_dir_path, exist_ok=True)

        self.log_dicts(dicts, 'AutoML.__init__()')


    def fit_structures(self, timeout=30):
        """
        self.structures에 fitting 및 evaluation 수행

        Args:
            timeout (int, optional): Pipeline 별 최대 실행시간. 기본값 30초.
        """
        
        if self.use_joblib:
            self.structures = Parallel(n_jobs=self.n_jobs)(
                              delayed(self.fit_structure)(structure, timeout, i)
                              for i, structure in enumerate(self.structures)
                              )
            
        else:
            self.structures = [self.fit_structure(structure, timeout, i) for i, structure in enumerate(self.structures)]


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
            structure['train_metric'] = average_metrics(train_metrics)
            structure['valid_metric'] = average_metrics(valid_metrics)
            structure['train_metrics'] = train_metrics
            structure['valid_metrics'] = valid_metrics

        except TimeoutException as e:
            print(e)

        finally:
            signal.alarm(0) # alarm 초기화
            
        if self.task_type == 'regression':
            valid_r2 = structure['valid_metric']['r2']
            valid_r2_std = structure['valid_metric']['r2_std']
            print(f"Structure-{order} - valid r2: {valid_r2:.4f}±{valid_r2_std:.4f}") # 결과 출력
        else:
            valid_f1 = structure['valid_metric']['f1']
            valid_accuracy = structure['valid_metric']['accuracy']
            print(f"Structure-{order} - valid_f1: {valid_f1:.4f}, valid_accuracy: {valid_accuracy:.4f}") # 결과 출력
        return structure

    def log_dicts(self, dicts, message=""):
        log = []
        for k, v in dicts.items():
            if isinstance(v, float):
                log.append(f'{k}: {v:.4f}')
            else:
                log.append(f'{k}: {v}')
        
        log = ', '.join(log)
        if len(message):
            log = f'{message} - {log}'
        self.log(log)
    
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
        print("predict:", self.best_structure['pipeline'])
        y_pred = self.best_structure['pipeline'].predict(X)
        return y_pred

    def get_feature_importance(self):
        feature_names = self.X_trains[0].columns
        model = self.best_structure['pipeline']['models']

        try:
            feature_importances = model.feature_importances_
        except:
            result = permutation_importance(model, self.X_valids[0], self.y_valids[0], n_repeats=10)
            feature_importances = result.importances_mean 

        importance_dict = {name: importance for name, importance in zip(feature_names, feature_importances)}
        sorted_importances = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

        return sorted_importances

    def report_structure(self, structure):
        arr = []
        keys = list(self.pipeline_components.keys())
        
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

            arr.append(s)

        if 'valid_metric' in structure:
            if self.task_type == 'regression':
                r2 = structure['valid_metric']['r2']
                r2_std = structure['valid_metric']['r2_std']
                arr.append(f'{r2:.4f}±{r2_std:.4f}')
            else:
                f1 = structure['valid_metric']['f1']
                accuracy = structure['valid_metric']['accuracy']
                arr.append(f'f1:{f1:.4f}, accuracy:{accuracy:.4f}')

        s = ' - '.join(arr)
        return s

    def report(self):
        log = []
        self.structures = sort_structures(self.structures, self.task_type)
        for i, structure in enumerate(self.structures):
            log.append(self.report_structure(structure))
        
        log = '\n' + '\n'.join(log)
        self.log(log)


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
        
        dicts = {'use_kfold': use_kfold, 'kfold': kfold, 'valid_size': valid_size,
                 'seed': seed, 'max_n_try': max_n_try, 'timeout': timeout, 'task_type': self.task_type}

        random.seed(seed)
        np.random.seed(seed)
        self.log_dicts(dicts, 'AutoML.fit()')

        if use_kfold: # k-fold validation으로 모델 평가
            kf = KFold(n_splits=kfold, shuffle=True, random_state=seed)
            self.X_trains, self.X_valids, self.y_trains, self.y_valids = [], [], [], [] # 초기화

            for train_index, valid_index in kf.split(X_train):
                self.X_trains.append(X_train.iloc[train_index, :])
                self.X_valids.append(X_train.iloc[valid_index, :])
                self.y_trains.append(y_train.iloc[train_index])
                self.y_valids.append(y_train.iloc[valid_index])
                

 
        else: # single-fold validation으로 모델 평가
            X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, 
                                                                    test_size=valid_size, random_state=seed)
            self.X_trains = [X_train] # List로 변환
            self.X_valids = [X_valid]
            self.y_trains = [y_train]
            self.y_valids = [y_valid]


            
        self.structures = get_random_structures(self.n_population, self.task_type) # 임의 구조 생성

        for generation in range(self.n_generation):
            self.fit_structures(timeout) # 형질 별 피팅 및 점수 계산
            self.structures = sort_structures(self.structures, self.task_type) # 점수 높은 순으로 정렬
            self.best_structure = self.structures[0] # 최적 구조 선택
            if self.task_type == 'regression':
                self.best_score = self.best_structure['valid_metric']['r2']
            else:
                self.best_score = self.best_structure['valid_metric']['f1']
            
            if self.task_type == 'regression':
                self.log(f"{generation+1} - best R2: {self.best_score:.3f}") # 최적값 기록
                self.report()
            else:
                self.log(f"{generation+1} - best f1: {self.best_score:.3f}") # 최적값 기록
                self.report()

            if (generation+1 == self.n_generation):
                break
                
            
            del self.structures[self.n_parent:] # 점수가 낮은 형질 제거
            
            n_success = 0
            n_try = 0

            while (n_success < self.n_child and n_try < max_n_try):
                n_try += 1
                structure1, structure2 = random.sample(self.structures, 2) # 임의로 형질 2개 고르기
                new_structure = crossover(structure1, structure2, self.pipeline_components) # 형질 교차
                new_structure = mutation(self.pipeline_components, new_structure, self.prob_mutations) # 형질 변형

                if is_in_structures(new_structure, self.structures, self.pipeline_components): # 이미 존재하는 형질이면 재생성
                    continue

                new_structure['pipeline'] = build_pipeline(new_structure) # pipeline 생성
                self.structures.append(new_structure) 
                n_success += 1

            if not(n_try < max_n_try):
                print("Warning: max_n_try <= n_try")