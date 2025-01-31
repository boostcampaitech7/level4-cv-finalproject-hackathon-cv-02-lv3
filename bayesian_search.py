from bayes_opt import BayesianOptimization
from sklearn.model_selection import ParameterGrid
import numpy as np
import joblib
import pandas as pd

# 학습된 모델 로드
model_path = "autosklearn_model.pkl"  # 저장된 pkl 파일 경로
loaded_model = joblib.load(model_path)

# 데이터 불러오기
df = pd.read_csv('melb_split.csv')

# 데이터셋 분리
train_data = df[df['Split'] == 'Train']
test_data = df[df['Split'] == 'Test']

# 타겟 변수와 특성 분리
y_train = train_data['Price']
X_train = train_data.drop(['Price', 'Split'], axis=1)

y_test = test_data['Price']
X_test = test_data.drop(['Price', 'Split'], axis=1)

print("Model loaded successfully!")

# 목적 함수 정의
def objective_function(Distance, Landsize):
    X_simulation = X_train.copy()
    X_simulation["Distance"] = Distance
    X_simulation["Landsize"] = Landsize
    predictions = loaded_model.predict(X_simulation)
    print(X_simulation["Distance"], X_simulation["Landsize"])
    return predictions.mean()

# Bayesian Optimization 초기화
optimizer = BayesianOptimization(
    f=objective_function,
    pbounds={"Distance": (0, 38), "Landsize": (0, 4584)},
    random_state=40,
)

# 최적화를 실행하기 전에 초기 샘플링 단계 수행
optimizer.maximize(init_points=5, n_iter=5)  # `n_iter=0`로 설정하면 초기 좌표만 생성됩니다.

# 초기 샘플링 단계의 좌표 출력
print("Initial points (init_points):")
for params, target in zip(optimizer._space.params, optimizer._space.target):
    print(f"Coordinates: {params}, Objective Value: {target}")