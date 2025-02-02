import pandas as pd

df = pd.read_csv('/data/ephemeral/home/Jeongseon/melb_split.csv')

unique_values = df['Suburb'].unique()
unique_count = df['Suburb'].nunique()
print("유니크 갯수 : ",unique_count )
print(unique_values)