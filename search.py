from bayes_opt import BayesianOptimization
from bayes_opt.util import UtilityFunction
import numpy as np
import joblib
import pandas as pd
import argparse  # 실행 모드 선택을 위해 추가
import time  # 실행 시간 측정을 위한 모듈
from joblib import Parallel, delayed
from shapely.geometry import Point, Polygon
import alphashape
import random

def single_search(X_train, y_train, model, search_x,search_y):

    start_time = time.time()  # ⏱️ 최적화 시작 시간 기록

    min_price = y_train.min() #y_train의 최소값
    x_list=search_x.keys()
    def optimize_row(index, row):

        # `index`는 Pandas의 실제 DataFrame 인덱스이므로, `y_train`에서 위치를 찾을 때 `.index.get_loc()` 사용
        idx_loc = y_train.index.get_loc(index)
        initial_y = y_train.iloc[idx_loc]  # 🚀 에러 해결


        def objective_function(**kwargs):
            # 🔹 Concave Hull 내부인지 확인 (아닐 경우, 큰 패널티 값 반환)
            # if not concave_hull_polygon.contains(Point(Lattitude, Longtitude)):
            #     return -1e9  # 패널티 값 반환

            X_simulation = row.copy()  # 현재 행을 복사하여 사용

            for key, value in kwargs.items():
                X_simulation[key] = value  # ✅ 각 Feature의 값을 업데이트
            
            # 데이터를 모델에 맞는 포맷으로 변환
            input_df = pd.DataFrame([X_simulation])
            prediction = model.predict(input_df)[0]  # 1개의 값 예측
            
            return prediction  # 최적화된 가격 반환

        # Bayesian Optimization 실행
        optimizer = BayesianOptimization(
            f=objective_function,
            pbounds={
                i : (X_train[i].min(), X_train[i].max()) for i in search_x.keys()
            },
            random_state=42,
            allow_duplicate_points=True
        )

        # 기존 X_simulation의 값을 초기 값으로 설정
        optimizer.register(
            params={i : row[i] for i in search_x.keys()}, 
            target=initial_y  # ✅ 초기 가격 값을 Bayesian Optimization에 등록
        )

        utility = UtilityFunction(kind="ei", xi=0.1)
        optimizer.maximize(init_points=0, n_iter=30, acquisition_function=utility) #acquisition_function=utility

        # 최적의 결과 저장
        best_solution = optimizer.max['params']


        dict1={i : best_solution[i] for i in search_x.keys()}
        dict2={
            'index': index, 
            'Predicted_Price': optimizer.max['target']  # 최적화된 가격도 함께 저장
        }
        return {**dict1,**dict2}

    # 병렬 처리 실행 (n_jobs=-1: 모든 CPU 코어 사용)
    optimal_solutions = Parallel(n_jobs=-1, backend="loky")(
        delayed(optimize_row)(index, row) for index, row in X_train.iterrows()
    )

    optimal_solutions = [sol for sol in optimal_solutions]

    # ✅ 최적 결과를 CSV 파일로 저장
    optimal_solutions_df = pd.DataFrame(optimal_solutions)

    # ⏱️ 종료 시간 기록 및 출력
    end_time = time.time()
    elapsed_time = end_time - start_time


    return elapsed_time, optimal_solutions_df

