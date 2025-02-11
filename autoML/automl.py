from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures, FunctionTransformer
from sklearn.feature_selection import SelectKBest, SelectPercentile, VarianceThreshold, f_regression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.exceptions import ConvergenceWarning
from pipeline_utils import crossover, mutation, build_pipeline, fit_structures, sort, is_in_structures, generate_structure_summary
from datetime import datetime
from copy import deepcopy
import numpy as np
import warnings
import random
import os

warnings.filterwarnings("ignore",message="k=10 is greater than n_features=6. All the features will be returned.")
warnings.filterwarnings("ignore", category=ConvergenceWarning)

choose_random_key = lambda dictionary: random.choice(list(dictionary.keys()))

preprocessors = {
    'StandardScaler': {'class': StandardScaler()},
    'RobustScaler': {'class': RobustScaler()},
    'PolynomialFeatures': {'class': PolynomialFeatures()},
    'passthrough': {'class': FunctionTransformer(func=lambda X: X)}
    }

feature_selections = {
    'SelectKBest': {'class': SelectKBest(score_func=f_regression)},
    'SelectPercentile': {'class': SelectPercentile(score_func=f_regression)},
    'VarianceThreshold': {'class': VarianceThreshold()},
    'passthrough': {'class': FunctionTransformer(func=lambda X: X)}
    }

models_regression = {
    'DecisionTreeRegressor': {'class': DecisionTreeRegressor()},
    'RandomForestRegressor': {'class': RandomForestRegressor,'params': {'n_estimators': 100}}, 
    'GradientBoostingRegressor': {'class': GradientBoostingRegressor, 'params': {'n_estimators': 100, 'learning_rate': 0.1}},
    'LogisticRegression': {'class': LogisticRegression, 'params': {'C': 1.0}},
    'KNeighborsRegressor': {'class': KNeighborsRegressor, 'params': {'n_neighbors': 5}},
    'XGBRegressor': {'class': XGBRegressor, 'params': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6}}
    }

models_classification = {
    'DecisionTreeClassifier': {'class': DecisionTreeClassifier, 'params': {'max_depth': 6, 'class_weight': 'balanced'}},
    'RandomForestClassifier': {'class': RandomForestClassifier, 'params': {'n_estimators': 100, 'class_weight': 'balanced'}},
    'GradientBoostingClassifier': {'class': GradientBoostingClassifier, 'params': {'max_depth': 10, 'n_estimators': 100, 'learning_rate': 0.1}},
    'LogisticRegression': {'class': LogisticRegression, 'params': {'C': 1.0, 'class_weight': 'balanced'}},
    'KNeighborsClassifier': {'class': KNeighborsClassifier, 'params': {'n_neighbors': 5}},
    'SGDClassifier': {'class': SGDClassifier, 'params': {'class_weight': 'balanced', 'alpha': 0.001, 'power_t': 0.5}},
    'XGBClassifier': {'class': XGBClassifier, 'params': {'n_estimators': 100, 'learning_rate': 1.0, 'max_depth':3, 'scale_pos_weight': 1}}
    }

class AutoML:
    """
    유전 알고리즘을 이용한 ML pipeline 최적화 수행
    """
    def __init__(self, task_type='regression', n_population=20, n_generation=50, n_parent=5, prob_mutations=[0.2, 0.5], use_joblib=True, n_jobs=-1):
        
        self.task_type = task_type
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
    
        self._create_log_dir()

        params_dict = {
            'task_type' : task_type, 
            'n_population': n_population, 
            'n_generation': n_generation, 
            'n_parent': n_parent,
            'prob_mutations': prob_mutations, 
            'use_joblib': use_joblib, 
            'n_jobs': n_jobs
        }
        self.log_dicts(params_dict, 'AutoML.__init__()')
    
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
        
        params_dict = {
            'use_kfold': use_kfold, 
            'kfold': kfold, 
            'valid_size': valid_size,
            'seed': seed, 
            'max_n_try': max_n_try,
            'timeout': timeout, 
            'task_type': self.task_type
        }
        self.log_dicts(params_dict, 'AutoML.fit()')
        
        # 데이터 분할
        self.X_trains, self.X_valids, self.y_trains, self.y_valids = self._split_data(X_train, y_train, use_kfold, kfold, valid_size, seed)

        # 랜덤 구조 생성
        self.structures = self._generate_random_structures(self.n_population, self.task_type)

        for generation in range(self.n_generation):
            self._evaluate_current_generation_and_log(generation, timeout)

            if (generation+1 == self.n_generation):
                break
            
            self._reproduce_structures(max_n_try)
        
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

    def log_dicts(self, dicts, message=""):
        """
        딕셔너리 형태의 로그 정보를 기록하는 메서드
        
        Args:
            dicts (dict): 로그에 기록할 딕셔너리
            message (str, optional): 로그 메시지의 앞부분에 추가할 메시지. 기본값은 빈 문자열.
        """
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
        로그 메시지를 기록하고 출력하는 메서드
        
        Args:
            message (str): 기록할 로그 메시지
        """
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{now}] {message}"
        print(log_message)

        # 로그 저장
        with open(self.log_path, 'a') as file:
            file.write(log_message + "\n") 
            file.flush()
            
    def _log_structures(self):
        """
        현재 structure의 요약 정보를 로그에 기록하는 함수.

        함수는 현재 저장된 구조체들을 정렬하고 각 구조체에 대해 `generate_structure_summary` 함수를 호출하여
        그 요약 정보를 생성한 후, 이를 로그에 기록.

        Returns:
            None
        """
        log = []
        self.structures = sort(self.structures, self.task_type)
        for _, structure in enumerate(self.structures):
            log.append(generate_structure_summary(structure))
        
        log = '\n' + '\n'.join(log)
        self.log(log)
    
    def _create_log_dir(self):
        """
        로그 디렉토리 및 로그 파일 경로를 생성하는 메서드        
        로그 파일은 실행 시각을 기준으로 파일명을 생성하여 저장된다.
        """
        now = datetime.now()
        time_string = now.strftime("%y%m%d_%H%M%S")
        py_dir_path = os.path.dirname(os.path.abspath(__file__))
        self.log_dir_path = os.path.join(py_dir_path, 'log')
        self.log_path = os.path.join(self.log_dir_path, f"{time_string}.txt")
        os.makedirs(self.log_dir_path, exist_ok=True)
    
    def _split_data(X_train, y_train, use_kfold=True, kfold=5, valid_size=0.2, seed=42):
        """
        데이터를 k-fold 또는 단일 validation 세트로 분할하는 함수

        Args:
            X_train (DataFrame): X_train 데이터
            y_train (DataFrame): y_train 데이터
            use_kfold (bool, optional): k-fold validation 사용 여부. 기본값 True.
            kfold (int, optional): k-fold 수. 기본값 5.
            valid_size (float, optional): k-fold validation을 사용하지 않을 때 train과 valid 비율. 기본값 0.2.
            seed (int, optional): 동일한 실험결과를 위한 시드 설정. 기본값 42.

        Returns:
            X_trains (list): 분할된 X_train 데이터 리스트
            X_valids (list): 분할된 X_valid 데이터 리스트
            y_trains (list): 분할된 y_train 데이터 리스트
            y_valids (list): 분할된 y_valid 데이터 리스트
        """
        X_trains, X_valids, y_trains, y_valids = [], [], [], []

        if use_kfold:   # k-fold validation으로 모델 평가
            kf = KFold(n_splits=kfold, shuffle=True, random_state=seed)
            for train_index, valid_index in kf.split(X_train):
                X_trains.append(X_train.iloc[train_index, :])
                X_valids.append(X_train.iloc[valid_index, :])
                y_trains.append(y_train.iloc[train_index])
                y_valids.append(y_train.iloc[valid_index])
        else:   # single-fold validation으로 모델 평가
            X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=valid_size, random_state=seed)
            X_trains = [X_train]
            X_valids = [X_valid]
            y_trains = [y_train]
            y_valids = [y_valid]

        return X_trains, X_valids, y_trains, y_valids
    
    def _evaluate_current_generation_and_log(self, generation, timeout):
        """
        현재 세대에 대해 평가 및 로그 기록을 수행하는 메서드
        
        Args:
            generation (int): 현재 세대 번호
            timeout (int): 각 pipeline 별 최대 실행 시간
        """
        fit_structures(self, timeout)
        self.structures = sort(self.structures, self.task_type)
        self.best_structure = self.structures[0]
        self.best_score = self.best_structure['valid_metric']['f1'] if self.task_type == 'classification' else self.best_structure['valid_metric']['r2']
        self.log(f"{generation+1} - best {'F1' if self.task_type == 'classification' else 'R2'}: {self.best_score:.3f}")
        self._log_structures(self)
    
    def _generate_random_structures(n, task_type):
        """
        n개의 임의의 structure 생성

        Args:
            n (int): 임의로 생성할 structure 개수

        Returns:
            random_structures (list): 임의로 생성한 structure
        """
        
        # task_type에 맞는 모델 카테고리 선택
        if task_type == 'regression':
            model_options = models_regression
        else:
            model_options = models_classification

        random_structures = []
        
        for _ in range(n):
            # 기본 파이프라인 구조 생성
            random_structure = deepcopy({
                'preprocessors': preprocessors,
                'feature_selections': feature_selections,
                'models': model_options
            })
            
            # 각 카테고리에서 임의의 옵션 선택
            for category, options in random_structure.items():
                class_name = choose_random_key(options)
                random_structure[category] = options[class_name]
                if isinstance(random_structure[category], dict):
                    random_structure[category]['class_name'] = class_name
            
            # 파이프라인 생성
            random_structure['pipeline'] = build_pipeline(random_structure)
            random_structures.append(random_structure)
            
        return random_structures
    
    def _reproduce_structures(self, max_n_try):
        """
        유전 알고리즘을 통해 새 구조를 생성하고 기존 구조들을 재구성하는 메서드
        
        이 함수는 부모 구조들 간의 교차(crossover)와 변형(mutation)을 통해 새로운 자식 구조를 생성하며,
        새로 생성된 구조가 기존 구조와 중복되지 않도록 확인한다.

        Args:
            max_n_try (int): 새 구조를 생성할 최대 시도 횟수
        """
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