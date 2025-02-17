from autoML.autoML import AutoML
import pandas as pd 

def data_preparation(data_path, verbose=False):
    drop_tables = ['Suburb', 'Address', 'Rooms', 'Method', 'SellerG', 'Date', 'Distance', 'Postcode',
               'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'YearBuilt', 'CouncilArea',
               'Regionname', 'Propertycount']

    # df 불러오기 및 column 제거
    df = pd.read_csv(data_path)
    df = df.drop(drop_tables, axis=1)
    df = df.dropna(axis=0)

    index = 0.1 < df['BuildingArea'] # BuildingArea가 0인 값 제거
    df = df.loc[index]

    # 데이터셋 분리
    train_data = df[df['Split'] == 'Train']
    train_data = train_data.drop(['Split'], axis=1)
    train_data = pd.get_dummies(train_data, dtype='float')

    test_data = df[df['Split'] == 'Test']
    test_data = test_data.drop(['Split'], axis=1)
    test_data = pd.get_dummies(test_data, dtype='float')

    # 타겟 변수와 특성 분리
    y_train = train_data['Price']
    X_train = train_data.drop(['Price'], axis=1)
    y_test = test_data['Price']
    X_test = test_data.drop(['Price'], axis=1)

    if verbose:
        # 결과 확인
        print("X_train.shape, y_train.shape, X_test.shape, y_test.shape: ", X_train.shape, y_train.shape, X_test.shape, y_test.shape)

        # na값 통계
        print("X_train, y_train, X_test, y_test null")
        print(X_train.isnull().sum())
        print(y_train.isnull().sum())
        print(X_test.isnull().sum())
        print(y_test.isnull().sum())

    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    data_path = '/data/ephemeral/home/Dongjin/level4-cv-finalproject-hackathon-cv-02-lv3/Surrogate/AutoML/Dongjin/0211_refactor/melb_split1.csv'
    X_train, y_train, X_test, y_test = data_preparation(data_path, verbose=False)
    autoML = AutoML()
    autoML.fit(X_train, y_train)