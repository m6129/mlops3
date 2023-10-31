import pandas as pd
import numpy as np
df = pd.read_csv('/home/anton/mlops3/datasets/df.csv')
#print(df)
mean_values = df[df.columns[:-1]].kurtosis(axis=1, skipna=True)
X = np.reshape(np.array(mean_values), (-1, 1))
y = df["label"]
X = pd.DataFrame(X)

X.to_csv('/home/anton/mlops3/datasets/X.csv', index=False)
y.to_csv('/home/anton/mlops3/datasets/y.csv', index =False)

print(X)