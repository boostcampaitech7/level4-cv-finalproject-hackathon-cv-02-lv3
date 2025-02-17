## AutoML을 활용한 Prescriptive AI 솔루션 개발

![poster](assets/img1.png)


### 1. model architecture
우리는 유전 알고리즘을 활용한 우리만의 autoML 라이브러리를 구축하여 가장 잘 예측하는 surrogate 모델을 찾았으며,

고객의 목적에 맞게 개발한 objective function을 이용하여 bayesian optimization을 진행하였다.

전체적인 service는 streamlit을 통해 구현해보았다.

![poster](assets/img2.png)

### 1-1. surrogate model
![poster](assets/img3.png)

데이터 전처리, 피처 선택, 모델 선택, 하이퍼 파라미터 최적화를 자동으로 수행하는 AutoML 방법론 도입


### 1-2. search model
![poster](assets/img4.png)
Gaussian Process Regression : 주어진 데이터로부터 확률적 예측 모델을 만든다
Acquisition Function : 최적의 x를 찾기 위해 새로운 x값을 탐색할 때 어떤 점을 평가해야 하는지 결정하는 함수



### 2. AI solution 실행

- 필요한 라이브러리 설치
pip install -r requirements.txt

- streamlit 실행 
streamlit run ../Service/Home.py

#### 2-1. Demo
영상