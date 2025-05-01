import pandas as pd

parquet_file = 'part-00000-3a2049b8-c272-44ab-b838-fe1989e9ce19.c000.snappy.parquet'
csv_file = 'output.csv'

df = pd.read_parquet(parquet_file)

chunksize = 100000

for i in range(0, len(df), chunksize):
    chunk = df[i:i+chunksize]
    chunk.to_csv(csv_file, mode='a', header= i == 0, index=False)