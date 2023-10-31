import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
import requests

mlflow.set_tracking_uri("http://0.0.0.0:5000")
mlflow.set_experiment("get_model")
with mlflow.start_run():
    df = requests.get("https://raw.githubusercontent.com/m6129/UrFU_2022_python/main/all_xackatons/train.csv")
    
    with open("/home/anton/mlops3/datasets/df.csv", "w") as f:
        f.write(df.text)
        mlflow.log_artifact(local_path="/home/anton/mlops3/scripts/get_data.py",artifact_path="get_data code")
        mlflow.end_run()

