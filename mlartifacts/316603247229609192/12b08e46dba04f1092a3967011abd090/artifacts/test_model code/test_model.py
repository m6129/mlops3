import pandas as pd
from sklearn.metrics import f1_score
import pickle
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://0.0.0.0:5000")
mlflow.set_experiment("test_model")

with mlflow.start_run():
    X_test = pd.read_csv('/home/anton/mlops3/datasets/X_test.csv')
    y_test = pd.read_csv('/home/anton/mlops3/datasets/y_test.csv')
    with open('/home/anton/mlops3/models/ada1.pickle', 'rb') as model_file:
        clf = pickle.load(model_file)
        
        predicted_label_y = clf.predict(X_test)
        score = f1_score(predicted_label_y, y_test, average=None)
        mlflow.log_artifact(local_path='/home/anton/mlops3/scripts/test_model.py',artifact_path="test_model code")
        mlflow.log_metric("score", score)
        mlflow.end_run()