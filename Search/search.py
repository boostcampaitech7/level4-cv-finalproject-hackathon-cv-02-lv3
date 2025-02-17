from bayes_opt import BayesianOptimization
from bayes_opt.util import UtilityFunction
import numpy as np
import pandas as pd
import argparse  
import time 
from joblib import Parallel, delayed
import random
from Search.custom_bayes import CustomBayesianOptimization

def calculate(row, pl, prediction, search_y, y):
    target = 0 
    for i in row.index:
        if i in pl.keys():
            if pl[i][1]=='최대화하기':
                target+= (1/(pl[i][0]))**2 * row[i]                                 # 우선순위에 따라서 target에 적게 반영되는것을 의도
            else:
                target-= (1/(pl[i][0]))**2 * row[i]

    if pl[y][1] == "최대화하기":
        target+= (1/(pl[y][0]))**2 * prediction
    elif pl[y][1] == "최소화하기":
        target-= (1/(pl[y][0]))**2 * prediction
        
    elif pl[y][1] == "목표값에 맞추기":
        target-= (1/(pl[y][0]))*abs(prediction - search_y[y]["목표값"])
    else:
        search_range = search_y[y]['범위 설정'][0] + search_y[y]['범위 설정'][1]
        target-= (1/(pl[y][0]))*abs(search_range - 2*prediction)
    
    return target
    

def optimize_row(index, row, X_train, y_train, model, search_x, search_y):
        """주어진 행(row)에 대해 최적화를 수행하는 함수"""
        
        idx_loc = y_train.index.get_loc(index)                                      # index는 Pandas의 실제 DataFrame 인덱스이므로, y_train에서 위치를 찾을 때 .index.get_loc() 사용
        initial_y = y_train.iloc[idx_loc]  
        y = list(search_y.keys())[0]

        if "순위" not in search_y[y].keys():                                         # pl = priority_list
            pl = { y : search_y[y]["목표"]}    
        else:
            pl = { y : [search_y[y]["순위"], search_y[y]["목표"]]}
            

        search_x_keys = sorted(list(search_x.keys()))                               # 고정된 순서 유지

        for i in search_x_keys:
            if search_x[i]["목표"] != "최적화하지 않기":
                pl[i] = [search_x[i]["순위"], search_x[i]['목표']]
            
        range_dict = {i: search_x[i]["범위 설정"] for i in search_x.keys()}

        
        if len(pl)==1:                                                              # y만 최적화 하면 되는 single objective 상황
            target = initial_y
            def objective_function(**kwargs):
                X_simulation = row.copy()                                           # 현재 행을 복사하여 사용
                for key, value in kwargs.items():
                    X_simulation[key] = value                                       # 각 Feature의 값을 업데이트
                
                
                input_df = pd.DataFrame([X_simulation])                             # 데이터를 모델에 맞는 포맷으로 변환
                prediction = model.predict(input_df)[0]                             # 1개의 값 예측
                
                if pl[y] == "최대화하기":
                    return prediction                                               # 최적화된 가격 반환
                
                elif pl[y] == "최소화하기":
                    return - prediction
                    
                elif pl[y] == "목표값에 맞추기":
                    return - abs(prediction - search_y[y]["목표값"])
                
                else:
                    return - abs(search_y[y]['범위 설정'].sum()-2* prediction)
                
        
        else:                                                                       # 0~1 정규화 

            y_min, y_max = y_train.min(), y_train.max()

            
            feature_ranges = {                                                      # 각 feature의 최소, 최대 값 저장
                col: (X_train[col].min(), X_train[col].max()) for col in X_train.columns
            }

            
            def objective_function(**kwargs):

                X_simulation = row.copy()                                           # 현재 행을 복사하여 사용
                X_simulation_scaled = X_simulation.copy()                           # 마찬가지로 복사하는데 정규화된 값 저장용

                for key, value in kwargs.items(): 
                    if key in feature_ranges:
                        X_min, X_max = feature_ranges[key]
                        if (X_max - X_min) > 0:                                     # 분모가 0이 되는 경우 방지
                            X_scaled = (value - X_min) / (X_max - X_min)            # X를 0~1로 정규화
                            X_simulation_scaled[key] = X_scaled 
                        else:
                            X_simulation_scaled[key] = value                        # 변화가 없는 경우 그대로 사용
                        X_simulation[key] = value
                
                
                input_df = pd.DataFrame([X_simulation])                             # 데이터를 모델에 맞는 포맷으로 변환
                prediction = model.predict(input_df)[0]                             # 1개의 값 예측

                prediction = (prediction - y_min) / (y_max - y_min)                 # y를 0~1 정규화               

                target = calculate(X_simulation_scaled, pl, prediction, search_y, y)
                
                return target
                

        
        optimizer = CustomBayesianOptimization(                                     # Bayesian Optimization 실행
            f=objective_function,
            pbounds={
                i : (range_dict[i][0], range_dict[i][1]) for i in range_dict.keys()
            },
            random_state=42,
            allow_duplicate_points=True
        )

        #candidate_acq = ["ei", "ucb", "poi"]                                       # acquisition_function 랜덤 설정
        #selected_acq = random.choice(candidate_acq)

        #utility = UtilityFunction(kind="ei", xi=0.1)
        optimizer.maximize(init_points=10, n_iter=50)                               # acquisition_function=utility
        
        best_solution = optimizer.max['params']                                     # 최적의 결과 저장

        
        best_x_df = row.copy()                                                      # 원본 데이터(row) 기반으로 새로운 DataFrame 생성 (최적화하지 않은 변수 포함)                                                      # 기존 row의 모든 feature 포함
        for key, value in best_solution.items():
            best_x_df[key] = value                                                  # 최적화된 feature 값 업데이트
        
        best_x_df = pd.DataFrame([best_x_df])                                       # DataFrame으로 변환 후, feature 순서 맞추기
        best_x_df = best_x_df.reindex(columns=X_train.columns, fill_value=0)        # 누락된 feature는 0으로 채움
        
        y_pred = model.predict(best_x_df)[0]                                        # 모델 예측
        
        dict1 = {'index': index, **{i: best_solution[i] for i in search_x.keys()}}
        dict2 = {
            'target': optimizer.max['target'],                                      # 기존 최적화된 target 값
            'y': y_pred                                                             # 모델이 최적 x 값에서 예측한 y 값 추가
        }
        return {**dict1,**dict2}


def search(X_train, y_train, model, search_x, search_y):
    start_time = time.time()                                                        # 최적화 시작 시간 기록

    """
    single + multi 합치기 목표
    multi의 경우에 search_x에서 다 최적화 않기만 있을때는 y만 target -> single
    이중반복문 search_x를 최적화 하지 않을때, search_y의 목표가 범위에 맞추기인지, 최소환지 최대환지, 목표값 만주긴지
    x는 우선순위 받으면서 target값도 받고 최적화 인지 (+ , -) 
    """
    
    
    optimal_solutions = Parallel(n_jobs=-1, backend="loky")(                        # 병렬 처리 실행 (n_jobs=-1: 모든 CPU 코어 사용)
        delayed(optimize_row)(index, row, X_train, y_train, model, search_x, search_y) for index, row in X_train.iterrows()
    )
    optimal_solutions = [sol for sol in optimal_solutions]
    
    optimal_solutions_df = pd.DataFrame(optimal_solutions)                          # 최적 결과를 CSV 파일로 저장

    
    csv_filename = "optimized_solutions.csv"                                        # CSV 파일로 저장 (index 제외)
    optimal_solutions_df.to_csv(csv_filename, index=False)

    
    end_time = time.time()                                                          # 종료 시간 기록 및 출력
    elapsed_time = end_time - start_time
    return elapsed_time, optimal_solutions_df

