import pandas as pd
from autoML import AutoML

data_path = '/data/ephemeral/home/Dongjin/level4-cv-finalproject-hackathon-cv-02-lv3/autoML/melb_split.csv'
drop_tables = ['Address', 'BuildingArea', 'YearBuilt',
               'Suburb', 'Address', 'Type', 'Method', 'SellerG', 'Date', 'CouncilArea', 'Regionname']

# df 불러오기 및 column 제거
df = pd.read_csv(data_path)
df = df.drop(drop_tables, axis=1)
df = df.dropna(axis=0)

# 데이터셋 분리
train_data = df[df['Split'] == 'Train']
test_data = df[df['Split'] == 'Test']

# 타겟 변수와 특성 분리
y_train = train_data['Price']
X_train = train_data.drop(['Price', 'Split'], axis=1)
y_test = test_data['Price']
X_test = test_data.drop(['Price', 'Split'], axis=1)

# # shape 확인
# print("X_train.shape, y_train.shape, X_test.shape, y_test.shape: ", X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# # na 통계
# print("X_train, y_train, X_test, y_test null")
# print(X_train.isnull().sum())
# print(y_train.isnull().sum())
# print(X_test.isnull().sum())
# print(y_test.isnull().sum())

autoML = AutoML(n_population=20, n_generation=1, n_parent=5, prob_mutation=0.1)
autoML.fit(X_train, y_train, timeout=3)