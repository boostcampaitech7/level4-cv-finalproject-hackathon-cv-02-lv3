import pandas as pd
import autosklearn.regression as automl
from bayes_opt import BayesianOptimization


def main():
    
    # -------------------------------
    # 1. 데이터 불러오기 및 전처리
    # -------------------------------
    df = pd.read_csv('melb_split.csv')

    # 데이터셋 분리 (Train/Test)
    train_data = df[df['Split'] == 'Train']
    test_data = df[df['Split'] == 'Test']

    # 타겟 변수와 특성 분리
    y_train = train_data['Price']
    X_train = train_data.drop(['Price', 'Split', 'Address', 'BuildingArea', 'YearBuilt'], axis=1)

    y_test = test_data['Price']
    X_test = test_data.drop(['Price', 'Split', 'Address', 'BuildingArea', 'YearBuilt'], axis=1)

    # -------------------------------
    # 2. AutoML 모델 학습 (서로게이트 모델)
    # -------------------------------
    automl_model = automl.AutoSklearnRegressor(
        time_left_for_this_task=30,  # 전체 수행 시간 (10분)
        per_run_time_limit=10,        # 개별 모델당 최대 실행 시간 (30초)
        n_jobs=-1                     # 병렬 처리 활성화
    )

    automl_model.fit(X_train, y_train)

    # -------------------------------
    # 3. 최적화 목표 함수 정의
    # -------------------------------
    def objective_function(distance, landsize):
        X_simulation = X_train.copy()
        X_simulation["Distance"] = distance
        X_simulation["Landsize"] = landsize
        predictions = automl_model.predict(X_simulation)

        # 범위 제약 적용
        price_penalty = 0 if (1000000 <= predictions.mean() <= 1100000) else -abs(predictions.mean() - 1050000)

        # 최적화 목표: Distance 최소화, Landsize 최대화
        objective_value = -distance * 0.6 + landsize * 0.4 + price_penalty
        print("objective_value",objective_value )
        return objective_value


    # -------------------------------
    # 4. 베이지안 최적화 실행
    # -------------------------------
    param_bounds = {
        'distance': (X_train['Distance'].min(), X_train['Distance'].max()),
        'landsize': (X_train['Landsize'].min(), X_train['Landsize'].max())
    }

    optimizer = BayesianOptimization(f=objective_function, pbounds=param_bounds, random_state=42)
    optimizer.maximize(init_points=5, n_iter=20)

    # -------------------------------
    # 5. 최적의 해 찾기 및 출력
    # -------------------------------
    best_solution = optimizer.max['params']
    print("Best Solution:", best_solution)

if __name__ == "__main__":
    main()