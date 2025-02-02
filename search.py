from bayes_opt import BayesianOptimization
import numpy as np
import joblib
import pandas as pd
import argparse  # 실행 모드 선택을 위해 추가
import time  # 실행 시간 측정을 위한 모듈
from joblib import Parallel, delayed
from bayes_opt.util import UtilityFunction

# ⏱️ 시작 시간 기록
overall_start_time = time.time()

# 학습된 모델 로드
model_path = "autoML.pkl"  # 저장된 pkl 파일 경로
loaded_model = joblib.load(model_path)

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

print("Model loaded successfully!")

# 실행 모드 설정
parser = argparse.ArgumentParser(description="Bayesian Optimization for Lattitude and Longtitude")
parser.add_argument('--mode', type=str, choices=['all', 'row'], default='all', help="Optimization mode: 'all' or 'row'")
args = parser.parse_args()
mode = args.mode  # 'all' or 'row'

# --------------------- #
# 📌 전체 최적화 (mode='all') + CSV 저장
# --------------------- #
if mode == 'all':
    start_time = time.time()  # ⏱️ 최적화 시작 시간 기록
    
    def objective_function(Lattitude, Longtitude):
        X_simulation = X_train.copy()
        X_simulation["Lattitude"] = Lattitude
        X_simulation["Longtitude"] = Longtitude
        predictions = loaded_model.predict(X_simulation)
        return predictions.mean()

    # Bayesian Optimization 초기화
    optimizer = BayesianOptimization(
        f=objective_function,
        pbounds={"Lattitude": (-38.18255, -37.40853), "Longtitude": (144.43181, 145.5264)},
        random_state=40,
    )

    optimizer.maximize(init_points=5, n_iter=50)  # 전체 데이터셋 기반 최적화

    # 최적의 결과 저장
    best_solution = optimizer.max['params']
    optimal_all_df = pd.DataFrame([{
        'Lattitude': best_solution['Lattitude'],
        'Longtitude': best_solution['Longtitude'],
        'Predicted_Price': optimizer.max['target']
    }])

    csv_filename = "optimized_all.csv"
    optimal_all_df.to_csv(csv_filename, index=False)

    # ⏱️ 종료 시간 기록 및 출력
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"✅ 전체 최적화 완료! ({elapsed_time:.2f} 초 소요)")
    print(f"✅ 전체 최적화 결과가 {csv_filename} 파일로 저장되었습니다!")

# --------------------- #
# 📌 각 행별 최적화 (mode='row') + CSV 저장
# --------------------- #

if mode == 'row':
    start_time = time.time()  # ⏱️ 최적화 시작 시간 기록

    def optimize_row(index, row):
        initial_lattitude = row["Lattitude"]
        initial_longtitude = row["Longtitude"]
        # initial_buildingarea = row["BuildingArea"]
        # initial_price = y_train.iloc[index]

        def objective_function(Lattitude, Longtitude):
            X_simulation = row.copy()  # 현재 행을 복사하여 사용
            X_simulation["Lattitude"] = Lattitude
            X_simulation["Longtitude"] = Longtitude
            # X_simulation["BuildingArea"] = BuildingArea
            
            # 데이터를 모델에 맞는 포맷으로 변환
            input_df = pd.DataFrame([X_simulation])
            prediction = loaded_model.predict(input_df)[0]  # 1개의 값 예측
            
            return prediction  # 최적화된 가격 반환

        # Bayesian Optimization 실행
        optimizer = BayesianOptimization(
            f=objective_function,
            pbounds={
                "Lattitude": (X_train['Lattitude'].min(), X_train['Lattitude'].max()),
                "Longtitude": (X_train['Longtitude'].min(), X_train['Longtitude'].max()),
                }, # "BuildingArea": (X_train['BuildingArea'].min(), X_train['BuildingArea'].max())
            random_state=42,
        )

        # 기존 X_simulation의 값을 초기 값으로 설정
        optimizer.probe(
            params={"Lattitude": initial_lattitude, "Longtitude": initial_longtitude }, # "BuildingArea": initial_buildingarea
            lazy=True
        )


        utility = UtilityFunction(kind="ei", xi=0.1)
        optimizer.maximize(init_points=5, n_iter=10, acquisition_function=utility)

        # 최적의 결과 저장
        best_solution = optimizer.max['params']
        return {
            'index': index, 
            'Lattitude_optimized': best_solution['Lattitude'], 
            'Longtitude_optimized': best_solution['Longtitude'],
            # 'BuildingArea': best_solution['BuildingArea'],
            'Predicted_Price': optimizer.max['target']  # 최적화된 가격도 함께 저장
        }

    # 병렬 처리 실행 (n_jobs=-1: 모든 CPU 코어 사용)
    optimal_solutions = Parallel(n_jobs=-1, backend="loky")(
        delayed(optimize_row)(index, row) for index, row in X_train.iterrows()
    )

    # ✅ 최적 결과를 CSV 파일로 저장
    optimal_solutions_df = pd.DataFrame(optimal_solutions)
    print(optimal_solutions_df.head())
    
    csv_filename = "optimized_solutions_5_ei.csv"
    optimal_solutions_df.to_csv(csv_filename, index=False)

    # ⏱️ 종료 시간 기록 및 출력
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"✅ 행별 최적화 완료! ({elapsed_time:.2f} 초 소요)")
    print(f"✅ 행별 최적화 결과가 {csv_filename} 파일로 저장되었습니다!")


# 전체 실행 종료 시간 출력
overall_end_time = time.time()
overall_elapsed_time = overall_end_time - overall_start_time
print(f"🚀 전체 실행 완료! 총 실행 시간: {overall_elapsed_time:.2f} 초")
