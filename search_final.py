from bayes_opt import BayesianOptimization
from bayes_opt.util import UtilityFunction
import numpy as np
import joblib
import pandas as pd
import argparse  # 실행 모드 선택을 위해 추가
import time  # 실행 시간 측정을 위한 모듈
from joblib import Parallel, delayed
from shapely.geometry import Point, Polygon
from sklearn.preprocessing import MinMaxScaler


# 데이터 불러오기
df = pd.read_csv('/data/ephemeral/home/jihwan/level4-cv-finalproject-hackathon-cv-02-lv3/melb_split_test.csv')
drop_tables = ['Suburb', 'Address', 'Rooms', 'Method', 'SellerG', 'Date', 'Distance', 'Postcode',
               'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'YearBuilt', 'CouncilArea',
               'Regionname', 'Propertycount', 'Split']
df = df.drop(drop_tables, axis=1)
df = df.dropna(axis=0)

# index = 0.1 < df['BuildingArea']  # BuildingArea 0값 제거
# df = df.loc[index]

# 베이지안 최적화에는 train/test 나눌 필요 없음
train_data = pd.get_dummies(df, dtype='float')

# 타겟 변수와 특성 분리
y_train = train_data['Price']
X_train = train_data.drop(['Price'], axis=1)

# 학습된 모델 로드
model_path = "autoML.pkl"  # 저장된 pkl 파일 경로
loaded_model = joblib.load(model_path)



# 데이터 정의
search_y = {
    "Price": {
        "목표": "최대화하기",
        "순위": 1
    }
}

search_x = {
    "Lattitude": {
        "목표": "최소화하기",
        "범위 설정": [-38.18255, -37.40853],
        "순위": 4
    },
    "Longtitude": {
        "목표": "최소화하기",
        "범위 설정": [144.43181, 145.52635],
        "순위": 3
    }
}



# def calculate(row, priority_list, max_num, initial_y, search_y, y):
#     target = 0 
#     for i in row.index:
#         if i in priority_list.keys():
#             if priority_list[i][1]=='최대화하기':
#                 target+= (max_num-priority_list[i][0]+1)*10*row[i]
        
#             else:
#                 target-= (max_num-priority_list[i][0]+1)*10*row[i]
#         else:
#             target+=row[i]
#     if priority_list[y][1] == "최대화하기":
#         target+= (max_num-priority_list[y][0]+1)*10*initial_y
#     elif priority_list[y][1] == "최소화하기":
#         target-= (max_num-priority_list[y][0]+1)*10*initial_y
        
#     elif priority_list[y][1] == "목표값에 맞추기":
#         target-= (max_num-priority_list[y][0]+1)*10*abs(initial_y - search_y[y]["목표값"])
#     else:
#         target-= (max_num-priority_list[y][0]+1)*10*abs(search_y[y]['범위 설정'].sum()-2*initial_y)
    
#     return target


def calculate(row, priority_list, max_num, prediction, search_y, y):
    target = 0 
    for i in row.index:
        if i in priority_list.keys():
            # priority_list
            if priority_list[i][1]=='최대화하기':
                target+= (1/(priority_list[i][0]+1))**2 * row[i] # 우선순위에 따라서 target에 적게 반영되는것을 의도
            else:
                target-= (1/(priority_list[i][0]+1))**2 * row[i]

    if priority_list[y][1] == "최대화하기":
        target+= (1/(priority_list[y][0]+1))**2 * prediction
    elif priority_list[y][1] == "최소화하기":
        target-= (1/(priority_list[y][0]+1))**2 * prediction
        
    elif priority_list[y][1] == "목표값에 맞추기":
        target-= (max_num-priority_list[y][0]+1)*10*abs(prediction - search_y[y]["목표값"])
    else:
        target-= (max_num-priority_list[y][0]+1)*10*abs(search_y[y]['범위 설정'].sum()-2*prediction)
    
    return target





def search(X_train, y_train, model, search_x, search_y):
    start_time = time.time()  # 최적화 시작 시간 기록
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
        initial_y = y_train.iloc[idx_loc]  # 
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


            y_min, y_max = y_train.min(), y_train.max()

            # 각 feature의 최소, 최대 값 저장
            feature_ranges = {
                col: (X_train[col].min(), X_train[col].max()) for col in X_train.columns
            }

            
            max_num = max(map(lambda x: x[0], priority_list.values()))
            target = calculate(row, priority_list, max_num, initial_y, search_y, y)
            
            def objective_function(**kwargs):

                X_simulation = row.copy()  # 현재 행을 복사하여 사용
                X_simulation_scaled = X_simulation.copy()  # 마찬가지로 복사하는데 정규화된 값 저장용

                for key, value in kwargs.items(): 
                    if key in feature_ranges:
                        X_min, X_max = feature_ranges[key]
                        if (X_max - X_min) > 0:  # 분모가 0이 되는 경우 방지
                            X_scaled = (value - X_min) / (X_max - X_min) # X를 0~1로 정규화
                            X_simulation_scaled[key] = X_scaled * (y_max - y_min) + y_min # y 범위로 변환
                        else:
                            X_simulation_scaled[key] = value  # 변화가 없는 경우 그대로 사용
                        X_simulation[key] = value
                
                # 데이터를 모델에 맞는 포맷으로 변환
                input_df = pd.DataFrame([X_simulation])
                prediction = model.predict(input_df)[0]  # 1개의 값 예측
                

                target = calculate(X_simulation_scaled, priority_list, max_num, prediction, search_y, y)
                
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
            target=target  # 초기 가격 값을 Bayesian Optimization에 등록  ## 이렇게 해도 되나?
        )

        utility = UtilityFunction(kind="ei", xi=0.1)
        optimizer.maximize(init_points=10, n_iter=50, acquisition_function=utility) #acquisition_function=utility
        # 최적의 결과 저장
        best_solution = optimizer.max['params']
        # 원본 데이터(row) 기반으로 새로운 DataFrame 생성 (최적화하지 않은 변수 포함)
        best_x_df = row.copy()  # 기존 row의 모든 feature 포함
        for key, value in best_solution.items():
            best_x_df[key] = value  # 최적화된 feature 값 업데이트
        # DataFrame으로 변환 후, feature 순서 맞추기
        best_x_df = pd.DataFrame([best_x_df])
        best_x_df = best_x_df.reindex(columns=X_train.columns, fill_value=0)  # 누락된 feature는 0으로 채움
        # 모델 예측
        y_pred = model.predict(best_x_df)[0]
        dict1 = {i: best_solution[i] for i in search_x.keys()}
        dict2 = {
            'index': index,
            'target': optimizer.max['target'],  # 기존 최적화된 target 값
            'y': y_pred  # 모델이 최적 x 값에서 예측한 y 값 추가
        }
        return {**dict1,**dict2}
    # 병렬 처리 실행 (n_jobs=-1: 모든 CPU 코어 사용)
    optimal_solutions = Parallel(n_jobs=-1, backend="loky")(
        delayed(optimize_row)(index, row) for index, row in X_train.iterrows()
    )
    optimal_solutions = [sol for sol in optimal_solutions]
    # 최적 결과를 CSV 파일로 저장
    optimal_solutions_df = pd.DataFrame(optimal_solutions)

    # CSV 파일로 저장 (index 제외)
    csv_filename = "optimized_solutions_minmax_역수.csv"
    optimal_solutions_df.to_csv(csv_filename, index=False)

    
    # 종료 시간 기록 및 출력
    end_time = time.time()
    elapsed_time = end_time - start_time
    return elapsed_time, optimal_solutions_df


# def search(X_train, y_train, model, search_x, search_y):
search(X_train, y_train, loaded_model, search_x, search_y)