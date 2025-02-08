# # 본 코드는 TPOT (Evaluation of a Tree-based Pipeline Optimization Tool
# # for Automating Data Science, GECCO '16)에서 아이디어를 얻어 구현했습니다.

# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures, FunctionTransformer
# from sklearn.decomposition import PCA
# from sklearn.linear_model import LogisticRegression
# from sklearn.feature_selection import SelectKBest, SelectPercentile, VarianceThreshold, f_regression
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.model_selection import train_test_split, KFold
# from sklearn.metrics import r2_score, mean_squared_error
# from sklearn.base import clone
# from xgboost import XGBRegressor
# from sklearn.exceptions import ConvergenceWarning
# from sklearn.inspection import permutation_importance

# import random
# from datetime import datetime
# import math
# import signal
# import os
# from joblib import Parallel, delayed
# import numpy as np
# import statistics
# from collections import defaultdict
# from copy import deepcopy
# import warnings
# from datetime import datetime

# warnings.filterwarnings(
#     "ignore",
#     message="k=10 is greater than n_features=6. All the features will be returned."
# )

# warnings.filterwarnings("ignore", category=ConvergenceWarning)



# preprocessors = {'StandardScaler': {'class': StandardScaler()},
#                  'RobustScaler': {'class': RobustScaler()}, 
#                  'PolynomialFeatures': {'class': PolynomialFeatures()},
#                  'passthrough': {'class': FunctionTransformer(func=lambda X: X)}}

# feature_selections = {'SelectKBest': {'class': SelectKBest(score_func=f_regression)}, 
#                       'SelectPercentile': {'class': SelectPercentile(score_func=f_regression)},
#                       'VarianceThreshold': {'class': VarianceThreshold()},
#                       'passthrough': {'class': FunctionTransformer(func=lambda X: X)}}


# models = {'DecisionTreeRegressor': {'class': DecisionTreeRegressor()},
#           'RandomForestRegressor': {'class': RandomForestRegressor,
#                                     'params': {'n_estimators': 100}}, 
#           'GradientBoostingRegressor': {'class': GradientBoostingRegressor,
#                                         'params': {'n_estimators': 100, 'learning_rate': 0.1}},
#           'KNeighborsRegressor': {'class': KNeighborsRegressor,
#                                   'params': {'n_neighbors': 5}},
#           'XGBRegressor': {'class': XGBRegressor,
#                            'params': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6}}}


# pipeline_components = {'preprocessors': preprocessors, 'feature_selections': feature_selections, 'models': models}
# choose_random_key = lambda dictionary: random.choice(list(dictionary.keys()))


# class TimeoutException(Exception):
#     pass

# def timeout_handler(signum, frame):
#     raise TimeoutException("Timeout: pipeline.fit did not complete in given time.")

# def evaluate_regression(y_true, y_pred):
#     r2 = r2_score(y_true, y_pred)
#     RMSE = math.sqrt(mean_squared_error(y_true, y_pred))    
#     dicts = {'r2': r2, 'RMSE': RMSE}
#     return dicts


# def build_pipeline(structure):
#     """
#     ML 계산을 위한 scikit pipeline 생성

#     Args:
#         structure (dict): pipeline을 생성하기 위한 구조체

#     Returns:
#         Pipeline: structure로 생성한 Pipeline
#     """
    
#     _pipeline = []
#     for k, v in structure.items():
#         cl = v['class']
#         if 'params' in v: # params 값이 v에 있으면 instance 생성
#             _pipeline.append((k, clone(cl(**v['params']))))
#         else:
#             _pipeline.append((k, clone(cl))) # instance 복사
                                     
#     return Pipeline(_pipeline)


# def get_random_structures(n):
#     """
#     n개의 임의의 structure 생성

#     Args:
#         n (int): 임의로 생성할 structure 개수

#     Returns:
#         random_structures (list): 임의로 생성한 structure
#     """

#     random_structures = []
    
#     for _ in range(n):
#         random_structure = deepcopy({'preprocessors': preprocessors,
#                                      'feature_selections': feature_selections,
#                                      'models': models})
        
#         for category, options in random_structure.items():
#             class_name = choose_random_key(options)
#             random_structure[category] = options[class_name]
#             if isinstance(random_structure[category], dict):
#                 random_structure[category]['class_name'] = class_name
        
#         random_structure['pipeline'] = build_pipeline(random_structure)
#         random_structures.append(random_structure)
        
#     return random_structures


# def fit_pipeline(pipeline, X_train, y_train):
#     pipeline.fit(X_train, y_train)    


# def sort(structures):
#     """
#     structures를 평가지표를 기준으로 정렬

#     Args:
#         structures (list): 정렬되지 않은 구조

#     Returns:
#         structures (list): ['valid_metric']['r2']를 기준으로 정렬
#     """
#     return sorted(structures, key=lambda x: x['valid_metric']['r2'], reverse=True)


# def is_same_structure(structure1, structure2):
#     """
#     structure1과 structure2가 동일한지 확인

#     Args:
#         structure1 (dict): 구조1
#         structure2 (dict): 구조2

#     Returns:
#         Bool: 동일하면 True, 다르면 False 반환
#     """
#     keys = list(pipeline_components.keys())
#     for k in keys:
#         if structure1[k] != structure2[k]:
#             return False
#     return True


# def is_in_structures(structure, structures):
#     """
#     structures 내 동일한 structure가 있는지 확인

#     Args:
#         structure (dict): 구조
#         structures (list): 구조 집합

#     Returns:
#         Bool: 동일하면 True, 다르면 False 반환
#     """
#     for s in structures:
#         if is_same_structure(structure, s):
#             return True
#     return False


# def crossover(structure1, structure2):
#     """
#     유전적 교차를 이용한 새로운 구조 생성

#     Args:
#         structure1 (dict): 부모 구조1
#         structure2 (dict): 부모 구조2

#     Returns:
#         structure (dict): 유전적 교차로 생성한 구조
#     """
#     prob = 0.5
#     new_structure = {}
#     keys = list(pipeline_components.keys())

#     for k in keys:
#         rand = random.random()
#         if rand < prob:
#             new_structure[k] = structure1[k]
#         else:
#             new_structure[k] = structure2[k]
    
#     return deepcopy(new_structure)


# def mutation(structure, prob_mutations, hyperparam_bound=[0.5, 2.0]):
#     """
#     돌연변이 구조 생성

#     Args:
#         structure (dict): 입력 구조
#         prob_mutations (List): 각 요소의 변이확률 (첫 번째: 구조변이, 두 번째: hyperparameter 변이)

#     Returns:
#         structure (dict): 돌연변이 구조
#     """
#     keys = list(pipeline_components.keys())
    
#     # 구조 변이
#     for k in keys:
#         rand = random.random()
#         if rand < prob_mutations[0]: 
#             element = choose_random_key(pipeline_components[k])
#             if structure[k]['class_name'] != element:
#                 structure[k] = deepcopy(pipeline_components[k][element])
#                 structure[k]['class_name'] = element

#     # 하이퍼 파라미터 변이
#     for k, v in structure.items():
#         if 'params' not in v: # 파라미터가 없으면 continue
#             continue

#         params = v['params']

#         for param_name, param_value in params.items():
#             rand = random.random()
#             if prob_mutations[1] <= rand:
#                 continue
                   
#             origin_type = type(param_value)
#             rand = random.uniform(hyperparam_bound[0], hyperparam_bound[1])
#             params[param_name] = origin_type(rand * param_value)

#     return structure

# def average_metrics(metrics):
#     """
#     각 메트릭의 평균과 표준 편차를 계산

#     Parameters:
#     metrics (List[Dict[str, float]]): 여러 번의 측정에서 얻은 메트릭 딕셔너리들의 리스트.
#                                       각 딕셔너리는 메트릭 이름을 키로 하며, 그 값은 측정된 값.

#     Returns:
#     Dict[str, float]: 각 메트릭의 평균과 표준 편차를 포함하는 딕셔너리.
#                       메트릭 이름으로 평균 값이, '메트릭_std' 형태로 표준 편차가 저장.
#     """

#     # 모든 메트릭 딕셔너리를 순회하며 값을 그룹화
#     grouped_metrics = defaultdict(list)
#     for metric in metrics:
#         for key, value in metric.items():
#             grouped_metrics[key].append(value)
    
#     # 평균과 표준 편차를 저장할 딕셔너리
#     avg_metric = {}
#     for key, values in grouped_metrics.items():
#         avg_metric[key] = statistics.mean(values)

#         if 1 < len(values):
#             avg_metric[f'{key}_std'] = statistics.stdev(values)
#         else: 
#             avg_metric[f'{key}_std'] = 0

#     return avg_metric


# class AutoML:
#     """
#     유전 알고리즘을 이용한 ML pipeline 최적화 수행
#     """
#     def __init__(self, n_population=20, n_generation=50, n_parent=5, prob_mutations=[0.2, 0.5], use_joblib=True, n_jobs=-1):
#         self.n_population = n_population
#         self.n_generation = n_generation
#         self.n_parent = n_parent
#         self.prob_mutations = prob_mutations
#         self.use_joblib = use_joblib
#         self.n_jobs = n_jobs
#         self.n_child = n_population - n_parent

#         dicts = {'n_population': n_population, 'n_generation': n_generation, 'n_parent': n_parent,
#                  'prob_mutations': prob_mutations, 'use_joblib': use_joblib, 'n_jobs': n_jobs}
        
#         now = datetime.now()
#         time_string = now.strftime("%y%m%d_%H%M%S")
#         py_dir_path = os.path.dirname(os.path.abspath(__file__))

#         self.log_dir_path = os.path.join(py_dir_path, 'log')
#         self.log_path = os.path.join(self.log_dir_path, f"{time_string}.txt")
#         os.makedirs(self.log_dir_path, exist_ok=True)

#         self.log_dicts(dicts, 'AutoML.__init__()')


#     def fit_structures(self, timeout=30):
#         """
#         self.structures에 fitting 및 evaluation 수행

#         Args:
#             timeout (int, optional): Pipeline 별 최대 실행시간. 기본값 30초.
#         """
        
#         if self.use_joblib:
#             self.structures = Parallel(n_jobs=self.n_jobs)(
#                               delayed(self.fit_structure)(structure, timeout, i)
#                               for i, structure in enumerate(self.structures)
#                               )
            
#         else:
#             self.structures = [self.fit_structure(structure, timeout, i) for i, structure in enumerate(self.structures)]


#     def fit_structure(self, structure, timeout=30, order=0):
#         """
#         입력 structure에 fitting 및 evaluation 수행

#         Args:
#             structure (dict): 입력 구조
#             timeout (int, optional): Pipeline 별 최대 실행시간. 기본값 30초.
#         """
#         if 'valid_metric' in structure: # fitting 결과가 있으면 skip
#             return structure
        
#         structure['valid_metric'] = {'r2': -100, 'r2_std': -100} # valid_metric 초기화
#         signal.signal(signal.SIGALRM, timeout_handler)
#         signal.alarm(timeout)  # timeout 초 후에 알람 발생
#         pipeline = structure['pipeline']
#         train_metrics = []
#         valid_metrics = []

#         try:
#             for i in range(len(self.X_trains)):
#                 pipeline_clone = clone(pipeline)
#                 X_train, y_train = self.X_trains[i], self.y_trains[i]
#                 X_valid, y_valid = self.X_valids[i], self.y_valids[i]

#                 pipeline_clone.fit(X_train, y_train)
#                 y_train_pred = pipeline_clone.predict(X_train)
#                 y_valid_pred = pipeline_clone.predict(X_valid)

#                 train_metric = evaluate_regression(y_train, y_train_pred)
#                 valid_metric = evaluate_regression(y_valid, y_valid_pred)
#                 train_metrics.append(train_metric)
#                 valid_metrics.append(valid_metric)

#             structure['pipeline'] = pipeline_clone
#             structure['train_metric'] = average_metrics(train_metrics)
#             structure['valid_metric'] = average_metrics(valid_metrics)
#             structure['train_metrics'] = train_metrics
#             structure['valid_metrics'] = valid_metrics

#         except TimeoutException as e:
#             print(e)

#         finally:
#             signal.alarm(0) # alarm 초기화

#         valid_r2 = structure['valid_metric']['r2']
#         valid_r2_std = structure['valid_metric']['r2_std']
#         print(f"Structure-{order} - valid r2: {valid_r2:.4f}±{valid_r2_std:.4f}") # 결과 출력
#         return structure

#     def log_dicts(self, dicts, message=""):
#         log = []
#         for k, v in dicts.items():
#             if isinstance(v, float):
#                 log.append(f'{k}: {v:.4f}')
#             else:
#                 log.append(f'{k}: {v}')
        
#         log = ', '.join(log)
#         if len(message):
#             log = f'{message} - {log}'
#         self.log(log)
    
#     def log(self, message):
#         """
#         log 기록

#         Args:
#             message (str): log 메세지
#         """
#         now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#         log_message = f"[{now}] {message}"
#         print(log_message) # 로그 출력 

#         # 로그 저장
#         with open(self.log_path, 'a') as file:
#             file.write(log_message + "\n") 
#             file.flush()


#     def predict(self, X):
#         """
#         얻은 최적 구조를 이용한 예측 수행

#         Args:
#             X (DataFrame): 예측할 X값

#         Returns:
#             y_pred (DataFrame): 예측된 y값
#         """
#         y_pred = self.best_structure['pipeline'].predict(X)
#         return y_pred
    

#     def get_feature_importance(self):
#         feature_names = self.X_trains[0].columns
#         model = self.best_structure['pipeline']['models']

#         try:
#             feature_importances = model.feature_importances_
#         except:
#             result = permutation_importance(model, self.X_valids[0], self.y_valids[0], n_repeats=10)
#             feature_importances = result.importances_mean 

#         importance_dict = {name: importance for name, importance in zip(feature_names, feature_importances)}
#         sorted_importances = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

#         return sorted_importances


#     def report_structure(self, structure):
#         arr = []
#         keys = list(pipeline_components.keys())
        
#         for k in keys:
#             class_info = structure[k]
#             if isinstance(class_info, str):
#                 s = class_info
#             else:
#                 class_name = class_info['class_name']
#                 if 'params' in class_info:
#                     params = ', '.join([f'{k}: {v}' if not isinstance(v, float) else f'{k}: {v:.4f}'
#                                         for k, v in class_info['params'].items()])
#                     s = f"({class_name}: {params})"
#                 else:
#                     s = f"({class_name})"  

#             arr.append(s)

#         if 'valid_metric' in structure:
#             r2 = structure['valid_metric']['r2']
#             r2_std = structure['valid_metric']['r2_std']
#             arr.append(f'{r2:.4f}±{r2_std:.4f}')

#         s = ' - '.join(arr)
#         return s

#     def report(self):
#         log = []
#         self.structures = sort(self.structures)
#         for i, structure in enumerate(self.structures):
#             log.append(self.report_structure(structure))
        
#         log = '\n' + '\n'.join(log)
#         self.log(log)

#     def report_best_structure(self):
#         self.log("Best structure")
#         log = self.report_structure(self.best_structure)
#         self.log(log)


#     def fit(self, X_train, y_train, use_kfold=True, kfold=5, valid_size=0.2, seed=42, max_n_try=1000, timeout=30):
#         """
#         유전 알고리즘을 이용한 최적 모델 탐색

#         Args:
#             X_train (DataFrame): X_train
#             y_train (DataFrame): y_train
#             use_kfold (bool, optional): k-fold validation 사용 여부. 기본값 True.
#             kfold (int, optional): k-fold 수. 기본값 5.
#             valid_size (float, optional): k-fold validation을 사용하지 않을 때 train와 valid 비율. 기본값 0.2.
#             seed (int, optional): 동일한 실험결과를 위한 시드 설정. 기본값 42.
#             max_n_try (int, optional): 최대 새 구조 생성 시도횟수. 기본값 1000.
#             timeout (int, optional): Pipeline 별 최대 실행시간. 기본값 30.
#         """
        
#         dicts = {'use_kfold': use_kfold, 'kfold': kfold, 'valid_size': valid_size,
#                  'seed': seed, 'max_n_try': max_n_try, 'timeout': timeout}

#         random.seed(seed)
#         np.random.seed(seed)
#         self.log_dicts(dicts, 'AutoML.fit()')

#         if use_kfold: # k-fold validation으로 모델 평가
#             kf = KFold(n_splits=kfold, shuffle=True, random_state=seed)
#             self.X_trains, self.X_valids, self.y_trains, self.y_valids = [], [], [], [] # 초기화
            
#             for train_index, valid_index in kf.split(X_train):
#                 self.X_trains.append(X_train.iloc[train_index, :])
#                 self.X_valids.append(X_train.iloc[valid_index, :])
#                 self.y_trains.append(y_train.iloc[train_index])
#                 self.y_valids.append(y_train.iloc[valid_index])
 
#         else: # single-fold validation으로 모델 평가
#             X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, 
#                                                                     test_size=valid_size, random_state=seed)
#             self.X_trains = [X_train] # List로 변환
#             self.X_valids = [X_valid]
#             self.y_trains = [y_train]
#             self.y_valids = [y_valid]


            
#         self.structures = get_random_structures(self.n_population) # 임의 구조 생성

#         for generation in range(self.n_generation):
#             self.fit_structures(timeout) # 형질 별 피팅 및 점수 계산
#             self.structures = sort(self.structures) # 점수 높은 순으로 정렬
#             self.best_structure = self.structures[0]
#             self.best_score = self.best_structure['valid_metric']['r2']
            
            
#             self.log(f"{generation+1} - best R2: {self.best_score:.3f}") # 최적값 기록
#             self.report()

#             if (generation+1 == self.n_generation):
#                 self.report_best_structure()
#                 break
            
#             del self.structures[self.n_parent:] # 점수가 낮은 형질 제거
            
#             n_success = 0
#             n_try = 0

#             while (n_success < self.n_child and n_try < max_n_try):
#                 n_try += 1
#                 structure1, structure2 = random.sample(self.structures, 2) # 임의로 형질 2개 고르기
#                 new_structure = crossover(structure1, structure2) # 형질 교차
#                 new_structure = mutation(new_structure, self.prob_mutations) # 형질 변형

#                 if is_in_structures(new_structure, self.structures): # 이미 존재하는 형질이면 재생성
#                     continue

#                 new_structure['pipeline'] = build_pipeline(new_structure) # pipeline 생성
#                 self.structures.append(new_structure) 
#                 n_success += 1

#             if not(n_try < max_n_try):
#                 print("Warning: max_n_try <= n_try")




# 본 코드는 TPOT (Evaluation of a Tree-based Pipeline Optimization Tool
# for Automating Data Science, GECCO '16)에서 아이디어를 얻어 구현했습니다.

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures, FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, SelectPercentile, VarianceThreshold, f_regression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, f1_score
from sklearn.base import clone
from xgboost import XGBRegressor, XGBClassifier
from sklearn.exceptions import ConvergenceWarning
from imblearn.over_sampling import SMOTE
import random
from datetime import datetime
import math
import signal
import os
from joblib import Parallel, delayed
import numpy as np
import statistics
from collections import defaultdict, Counter
from copy import deepcopy
import warnings

warnings.filterwarnings(
    "ignore",
    message="k=10 is greater than n_features=6. All the features will be returned."
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)



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

# models_classification = {
#     'DecisionTreeClassifier': {'class': DecisionTreeClassifier, 'params': {'max_depth': 6, 'class_weight':{0:0.5,1:2}}},
#     'RandomForestClassifier': {'class': RandomForestClassifier, 'params': {'n_estimators': 100,'class_weight':{0:0.5,1:2}}},
#     'GradientBoostingClassifier': {'class': GradientBoostingClassifier, 'params': {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.1}},
#     'LogisticRegression': {'class': LogisticRegression, 'params': {'C': 1.0,'class_weight':{0:0.5,1:2}}},
#     'KNeighborsClassifier': {'class': KNeighborsClassifier, 'params': {'n_neighbors': 5}},
#     'XGBClassifier': {'class': XGBClassifier, 'params': {'n_estimators': 100, 'learning_rate': 0.1, 'scale_pos_weight': 1}}}
models_classification = {
    'DecisionTreeClassifier': {'class': DecisionTreeClassifier, 'params': {'max_depth': 6, 'class_weight': 'balanced'}},
    'RandomForestClassifier': {'class': RandomForestClassifier, 'params': {'n_estimators': 100, 'class_weight': 'balanced'}},
    'GradientBoostingClassifier': {'class': GradientBoostingClassifier, 'params': {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.1}},
    'LogisticRegression': {'class': LogisticRegression, 'params': {'C': 1.0, 'class_weight': 'balanced'}},
    'KNeighborsClassifier': {'class': KNeighborsClassifier, 'params': {'n_neighbors': 5}},
    'XGBClassifier': {'class': XGBClassifier, 'params': {'n_estimators': 100, 'learning_rate': 0.1, 'scale_pos_weight': 1}}}
    

choose_random_key = lambda dictionary: random.choice(list(dictionary.keys()))


class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Timeout: pipeline.fit did not complete in given time.")

def evaluate_regression(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    RMSE = math.sqrt(mean_squared_error(y_true, y_pred))    
    dicts = {'r2': r2, 'RMSE': RMSE}
    return dicts

def evaluate_classification(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    dicts = {'accuracy': accuracy, 'f1': f1}
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
        cl = v['class']
        if 'params' in v: # params 값이 v에 있으면 instance 생성
            _pipeline.append((k, clone(cl(**v['params']))))
        else:
            _pipeline.append((k, clone(cl))) # instance 복사
                                     
    return Pipeline(_pipeline)


def get_random_structures(n, task_type):
    """
    n개의 임의의 structure 생성

    Args:
        n (int): 임의로 생성할 structure 개수

    Returns:
        random_structures (list): 임의로 생성한 structure
    """

    random_structures = []
    
    for _ in range(n):
        if task_type == 'regression':
            random_structure = deepcopy({'preprocessors': preprocessors,
                                     'feature_selections': feature_selections,
                                     'models': models_regression})
        else:
            random_structure = deepcopy({'preprocessors': preprocessors,
                                     'feature_selections': feature_selections,
                                     'models': models_classification})
        
        for category, options in random_structure.items():
            class_name = choose_random_key(options)
            random_structure[category] = options[class_name]
            if isinstance(random_structure[category], dict):
                random_structure[category]['class_name'] = class_name
        
        random_structure['pipeline'] = build_pipeline(random_structure)
        random_structures.append(random_structure)
        
    return random_structures


def fit_pipeline(pipeline, X_train, y_train):
    pipeline.fit(X_train, y_train)    


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

        if 1 < len(values):
            avg_metric[f'{key}_std'] = statistics.stdev(values)
        else: 
            avg_metric[f'{key}_std'] = 0

    return avg_metric


class AutoML:
    """
    유전 알고리즘을 이용한 ML pipeline 최적화 수행
    """
    def __init__(self, task_type='regression', n_population=20, n_generation=50, n_parent=5, prob_mutations=[0.2, 0.5], use_joblib=True, n_jobs=-1):
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
            print(f"Structure-{order} - valid_f1: {valid_f1:.4f}") # 결과 출력
            print(f"Structure-{order} - valid_accuracy: {valid_accuracy:.4f}") # 결과 출력
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
        y_pred = self.best_structure['pipeline'].predict(X)
        return y_pred

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
        self.structures = sort(self.structures, self.task_type)
        for i, structure in enumerate(self.structures):
            log.append(self.report_structure(structure))
        
        log = '\n' + '\n'.join(log)
        self.log(log)

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

    def fit(self, X_train, y_train, use_kfold=True, kfold=5, valid_size=0.2, seed=42, max_n_try=1000, timeout=30, task_type='regression'):
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
        self.task_type = task_type
        
        dicts = {'use_kfold': use_kfold, 'kfold': kfold, 'valid_size': valid_size,
                 'seed': seed, 'max_n_try': max_n_try, 'timeout': timeout, 'task_type': task_type}

        random.seed(seed)
        np.random.seed(seed)
        self.log_dicts(dicts, 'AutoML.fit()')

        if use_kfold: # k-fold validation으로 모델 평가
            kf = KFold(n_splits=kfold, shuffle=True, random_state=seed)
            self.X_trains, self.X_valids, self.y_trains, self.y_valids = [], [], [], [] # 초기화
            smote = SMOTE(sampling_strategy='auto', random_state=42)

            for train_index, valid_index in kf.split(X_train):
                X_train_fold, y_train_fold = X_train.iloc[train_index, :], y_train.iloc[train_index]
                X_valid_fold, y_valid_fold = X_train.iloc[valid_index, :], y_train.iloc[valid_index]

                # SMOTE 적용 (train set에만)
                X_train_resampled, y_train_resampled = smote.fit_resample(X_train_fold, y_train_fold)
                
                print("Before SMOTE:", Counter(y_train_fold))  
                print("After SMOTE:", Counter(y_train_resampled))

                # 리스트에 추가
                self.X_trains.append(X_train_resampled)
                self.y_trains.append(y_train_resampled)
                self.X_valids.append(X_valid_fold)
                self.y_valids.append(y_valid_fold)
                
                # self.X_trains.append(X_train.iloc[train_index, :])
                # self.X_valids.append(X_train.iloc[valid_index, :])
                # self.y_trains.append(y_train.iloc[train_index])
                # self.y_valids.append(y_train.iloc[valid_index])

 
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
            self.structures = sort(self.structures, self.task_type) # 점수 높은 순으로 정렬
            self.best_structure = self.structures[0]
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