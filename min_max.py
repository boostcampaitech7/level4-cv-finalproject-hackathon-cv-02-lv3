import pandas as pd

# CSV 파일 불러오기
df = pd.read_csv("/data/ephemeral/home/jihwan/level4-cv-finalproject-hackathon-cv-02-lv3/melb_split.csv")  # CSV 파일 경로를 입력하세요.

# 숫자형 데이터만 선택하여 최소/최대값 계산
numeric_cols = df.select_dtypes(include=['number'])  # 숫자형 데이터만 필터링

index = 0.1 < df['BuildingArea']  # BuildingArea 0값 제거
df = df.loc[index]

min_values = df['BuildingArea'].min()
max_values = df['BuildingArea'].max()

# 결과를 DataFrame으로 정리
summary_df = pd.DataFrame({'Min': min_values, 'Max': max_values})

# 결과 출력
print(summary_df)