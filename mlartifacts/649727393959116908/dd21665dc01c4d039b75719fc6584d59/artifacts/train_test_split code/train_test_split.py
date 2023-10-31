import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://0.0.0.0:5000")
mlflow.set_experiment("train_test_split")

with mlflow.start_run():
    X = pd.read_csv('/home/anton/mlops3/datasets/X.csv')
    y = pd.read_csv('/home/anton/mlops3/datasets/y.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=133)

    X_train.to_csv('/home/anton/mlops3/datasets/X_train.csv',index=False)
    X_test.to_csv('/home/anton/mlops3/datasets/X_test.csv',index=False)
    y_train.to_csv('/home/anton/mlops3/datasets/y_train.csv',index=False)
    y_test.to_csv('/home/anton/mlops3/datasets/y_test.csv',index=False)
    mlflow.log_artifact(local_path="/home/anton/mlops3/scripts/train_test_split.py",artifact_path="train_test_split code")
    mlflow.end_run()