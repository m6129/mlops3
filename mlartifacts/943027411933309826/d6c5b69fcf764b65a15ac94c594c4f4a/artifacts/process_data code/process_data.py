import pandas as pd
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://0.0.0.0:5000")
mlflow.set_experiment("process_data")
with mlflow.start_run():
    df = pd.read_csv('/home/anton/mlops3/datasets/df.csv')

    mean_values = df[df.columns[:-1]].kurtosis(axis=1, skipna=True)
    X = np.reshape(np.array(mean_values), (-1, 1))
    y = df["label"]
    X = pd.DataFrame(X)

    X.to_csv('/home/anton/mlops3/datasets/X.csv', index=False)
    y.to_csv('/home/anton/mlops3/datasets/y.csv', index=False)
    mlflow.log_artifact(local_path="/home/anton/mlops3/scripts/process_data.py",artifact_path="process_data code")
    mlflow.end_run()
