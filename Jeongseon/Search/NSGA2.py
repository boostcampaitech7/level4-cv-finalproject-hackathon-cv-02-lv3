import pandas as pd
import autosklearn.regression as automl
from deap import base, creator, tools, algorithms
import random

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
        time_left_for_this_task=600,  # 전체 수행 시간 (10분)
        per_run_time_limit=30,        # 개별 모델당 최대 실행 시간 (30초)
        n_jobs=-1                     # 병렬 처리 활성화
    )

    automl_model.fit(X_train, y_train)

    # -------------------------------
    # 3. NSGA-II 최적화 설정
    # -------------------------------

    # 다목적 최적화를 위한 Fitness 설정 (Distance 최소화, Landsize 최대화)
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))  # Distance 최소화, Landsize 최대화
    creator.create("Individual", list, fitness=creator.FitnessMulti)

    # 랜덤 유전자 초기화 함수 정의
    def random_distance():
        return random.uniform(X_train['Distance'].min(), X_train['Distance'].max())

    def random_landsize():
        return random.uniform(X_train['Landsize'].min(), X_train['Landsize'].max())

    toolbox = base.Toolbox()
    toolbox.register("attr_distance", random_distance)
    toolbox.register("attr_landsize", random_landsize)
    toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.attr_distance, toolbox.attr_landsize))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # -------------------------------
    # 4. 목적 함수 정의
    # -------------------------------
    def evaluate(individual):
        distance, landsize = individual
        X_simulation = X_train.copy()
        X_simulation["Distance"] = distance
        X_simulation["Landsize"] = landsize
        predictions = automl_model.predict(X_simulation)

        # 범위 제약 적용 (가격이 1,000,000 ~ 1,100,000 사이인지 확인)
        price_penalty = 0 if (1000000 <= predictions.mean() <= 1100000) else -abs(predictions.mean() - 1050000)

        # 목표 1: Distance 최소화 (값이 작을수록 좋음)
        # 목표 2: Landsize 최대화 (값이 클수록 좋음)
        return distance + price_penalty, landsize

    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=[X_train['Distance'].min(), X_train['Landsize'].min()], 
                     up=[X_train['Distance'].max(), X_train['Landsize'].max()], eta=20.0)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=[X_train['Distance'].min(), X_train['Landsize'].min()], 
                     up=[X_train['Distance'].max(), X_train['Landsize'].max()], eta=20.0, indpb=0.2)
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("evaluate", evaluate)

    # -------------------------------
    # 5. NSGA-II 최적화 실행
    # -------------------------------
    population_size = 5
    num_generations = 4
    crossover_prob = 0.9
    mutation_prob = 0.2

    pop = toolbox.population(n=population_size)

    # 초기 적합도 평가
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # 진화 실행
    for gen in range(num_generations):
        offspring = algorithms.varAnd(pop, toolbox, cxpb=crossover_prob, mutpb=mutation_prob)
        fitnesses = list(map(toolbox.evaluate, offspring))
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit
        pop = toolbox.select(offspring + pop, population_size)

    # -------------------------------
    # 6. 최적의 해 찾기 및 출력
    # -------------------------------
    pareto_front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
    
    print("\nPareto-Optimal Solutions:")
    for solution in pareto_front:
        print(f"Distance: {solution[0]:.2f}, Landsize: {solution[1]:.2f}")

    # Pareto Front 시각화
    import matplotlib.pyplot as plt

    distances = [ind[0] for ind in pareto_front]
    landsizes = [ind[1] for ind in pareto_front]

    plt.scatter(distances, landsizes, color='red', label='Pareto Front')
    plt.xlabel('Distance (Minimize)')
    plt.ylabel('Landsize (Maximize)')
    plt.title('NSGA-II Pareto Front')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
