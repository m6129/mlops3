import pandas as pd
from sklearn.model_selection import train_test_split

X = pd.read_csv('/home/anton/mlops3/datasets/X.csv')
y = pd.read_csv('/home/anton/mlops3/datasets/y.csv')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=133)

X_train.to_csv('/home/anton/mlops3/datasets/X_train.csv',index=False)
X_test.to_csv('/home/anton/mlops3/datasets/X_test.csv',index=False)
y_train.to_csv('/home/anton/mlops3/datasets/y_train.csv',index=False)
y_test.to_csv('/home/anton/mlops3/datasets/y_test.csv',index=False)

print(X)