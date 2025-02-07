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
from sklearn.preprocessing import MinMaxScaler



# 데이터 불러오기
df = pd.read_csv('melb_split.csv')
drop_tables = ['Suburb', 'Address', 'Rooms', 'Method', 'SellerG', 'Date', 'Distance', 'Postcode',
               'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'YearBuilt', 'CouncilArea',
               'Regionname', 'Propertycount', 'Split']
df = df.drop(drop_tables, axis=1)
df = df.dropna(axis=0)

index = 0.1 < df['BuildingArea']  # BuildingArea 0값 제거
df = df.loc[index]

# 베이지안 최적화에는 train/test 나눌 필요 없음
train_data = pd.get_dummies(df, dtype='float')

# 타겟 변수와 특성 분리
y_train = train_data['Price']
X_train = train_data.drop(['Price'], axis=1)



# 데이터 정의
search_y = {
    "price": {
        "목표": "최대화하기",
        "순위": 1
    }
}

search_x = {
    "lattitude": {
        "목표": "최소화하기",
        "범위 설정": [0, 1],
        "순위": 4
    },
    "longitude": {
        "목표": "최소화하기",
        "범위 설정": [0, 1],
        "순위": 3
    }
}



def calculate(row, priority_list, max_num, initial_y, search_y, y): # 우선순위와 방향을 고려하여 objective를 정함
    target = 0 

    for i in row.index:
        if i in priority_list.keys():
            if priority_list[i][1]=='최대화하기':
                target+= (max_num-priority_list[i][0]+1)*10*row[i]
        
            else:
                target-= (max_num-priority_list[i][0]+1)*10*row[i]

        else:
            target+=row[i]

    if priority_list[y][1] == "최대화하기":
        target+= (max_num-priority_list[y][0]+1)*10*initial_y

    elif priority_list[y][1] == "최소화하기":
        target-= (max_num-priority_list[y][0]+1)*10*initial_y
        
    elif priority_list[y][1] == "목표값에 맞추기":
        target-= (max_num-priority_list[y][0]+1)*10*abs(initial_y - search_y[y]["목표값"])

    else:
        target-= (max_num-priority_list[y][0]+1)*10*abs(search_y[y]['범위 설정'].sum()-2*initial_y)

    
    return target


def search(X_train, y_train, model, search_x, search_y):

    start_time = time.time()  # ⏱️ 최적화 시작 시간 기록

    y = list(search_y.keys())[0]
    if "순위" not in search_y[y].keys():
        priority_list = { y : search_y[y]["목표"]}
    
    else:
        priority_list = { y : [search_y[y]["순위"], search_y[y]["목표"]]}

    range_dict = {}

    for i in search_x.keys():
        if search_x[i]["목표"]=="최적화하지 않기":
            pass
        else:    
            priority_list[i]=[search_x[i]["순위"], search_x[i]['목표']]
        
        range_dict[i]=search_x[i]["범위 설정"]

    
    # single + multi 합치기 목표
    # multi의 경우에 search_x에서 다 최적화 않기만 있을때는 y만 target -> single
    # 이중반복문 search_x를 최적화 하지 않을때, search_y의 목표가 범위에 맞추기인지, 최소환지 최대환지, 목표값 만주긴지
    # x는 우선순위 받으면서 target값도 받고 최적화 인지 ( + , -) , 

    def optimize_row(index, row):

        # `index`는 Pandas의 실제 DataFrame 인덱스이므로, `y_train`에서 위치를 찾을 때 `.index.get_loc()` 사용
        idx_loc = y_train.index.get_loc(index)
        initial_y = y_train.iloc[idx_loc]  # 🚀 에러 해결

        # y만 최적화 하면 되는 single objective 상황
        if len(priority_list)==1:

            target = initial_y

            def objective_function(**kwargs):

                X_simulation = row.copy()  # 현재 행을 복사하여 사용

                for key, value in kwargs.items():
                    X_simulation[key] = value  # ✅ 각 Feature의 값을 업데이트
                
                # 데이터를 모델에 맞는 포맷으로 변환
                input_df = pd.DataFrame([X_simulation])
                prediction = model.predict(input_df)[0]  # 1개의 값 예측
                
                if priority_list[y] == "최대화하기":
                    return prediction  # 최적화된 가격 반환
                
                elif priority_list[y] == "최소화하기":
                    return - prediction
                    
                elif priority_list[y] == "목표값에 맞추기":
                    return - abs(prediction - search_y[y]["목표값"])
                
                else:
                    return - abs(search_y[y]['범위 설정'].sum()-2* prediction)

                
        else:


            # 🔹 feature별 MinMaxScaler 생성
            feature_scalers = {col: MinMaxScaler() for col in X_train.columns}
            prediction_scaler = MinMaxScaler()

            # 🔹 각 feature별 정규화 범위 학습
            for col in X_train.columns:
                feature_scalers[col].fit(X_train[[col]])  # 각 변수에 대해 fit

            # 🔹 target(y_train) 정규화 학습
            prediction_scaler.fit(y_train.values.reshape(-1, 1))

            
            max_num = max(map(lambda x: x[0], priority_list.values()))
            target = calculate(row, priority_list, max_num, initial_y, search_y, y)
            
            def objective_function(**kwargs):

                X_simulation = row.copy()  # 현재 행을 복사하여 사용
                X_simulation_normalized = X_simulation.copy()  # 마찬가지로 복사하는데 정규화된 값 저장용

                # 🔹 kwargs 내 각 변수별 정규화 적용 
                for key, value in kwargs.items(): 
                    if key in feature_scalers: # # kwargs 조작변수를 원래 x_train scaler와 매칭
                        X_simulation_normalized[key] = feature_scalers[key].transform(np.array([[value]]))[0][0]  # 조작변수 개별 정규화
                    X_simulation[key] = value  # 원본 값 유지 (model.predict에 사용)
                
                # 데이터를 모델에 맞는 포맷으로 변환
                input_df = pd.DataFrame([X_simulation])
                prediction = model.predict(input_df)[0]  # 1개의 값 예측
                prediction_normalized = prediction_scaler.transform(np.array([[prediction]]))[0][0]


                target = calculate(X_simulation_normalized, priority_list, max_num, prediction_normalized, search_y, y)
                
                return target





        # Bayesian Optimization 실행
        optimizer = BayesianOptimization(
            f=objective_function,
            pbounds={
                i : (range_dict[i][0], range_dict[i][1]) for i in range_dict.keys()
            },
            random_state=42,
            allow_duplicate_points=True
        )

        # 기존 X_simulation의 값을 초기 값으로 설정
        optimizer.register(
            params={i : row[i] for i in search_x.keys()}, 
            target=target  # ✅ 초기 가격 값을 Bayesian Optimization에 등록  ## 이렇게 해도 되나?
        )

        utility = UtilityFunction(kind="ei", xi=0.1)
        optimizer.maximize(init_points=10, n_iter=30, acquisition_function=utility) #acquisition_function=utility

        # 최적의 결과 저장
        best_solution = optimizer.max['params']

        # ✅ 원본 데이터(row) 기반으로 새로운 DataFrame 생성 (최적화하지 않은 변수 포함)
        best_x_df = row.copy()  # 기존 row의 모든 feature 포함
        for key, value in best_solution.items():
            best_x_df[key] = value  # ✅ 최적화된 feature 값 업데이트

        # ✅ DataFrame으로 변환 후, feature 순서 맞추기
        best_x_df = pd.DataFrame([best_x_df])
        best_x_df = best_x_df.reindex(columns=X_train.columns, fill_value=0)  # 누락된 feature는 0으로 채움

        # ✅ 모델 예측
        y_pred = model.predict(best_x_df)[0]

        dict1 = {i: best_solution[i] for i in search_x.keys()}
        dict2 = {
            'index': index,
            'target': optimizer.max['target'],  # 기존 최적화된 target 값
            'y': y_pred  # 🚀 모델이 최적 x 값에서 예측한 y 값 추가
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

