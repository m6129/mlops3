import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/m6129/UrFU_2022_python/main/all_xackatons/train.csv')
df.to_csv('/home/anton/mlops3/datasets/df.csv')