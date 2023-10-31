import pandas as pd
from sklearn.metrics import f1_score
import pickle

X_test = pd.read_csv('/home/anton/mlops3/datasets/X_test.csv')
y_test = pd.read_csv('/home/anton/mlops3/datasets/y_test.csv')
with open('/home/anton/mlops3/models/ada1.pickle', 'rb') as model_file:
    clf = pickle.load(model_file)
    
predicted_label_y = clf.predict(X_test)
print(f1_score(predicted_label_y, y_test, average=None))