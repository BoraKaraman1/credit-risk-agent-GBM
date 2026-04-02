import pandas as pd

df = pd.read_csv('data/accepted_2007_to_2018Q4.csv.gz', nrows=1000)
print(df.shape)
print(df.columns.tolist())
print(df.head())