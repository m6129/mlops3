import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
import pickle

X_train = pd.read_csv('/home/anton/mlops3/datasets/X_train.csv')
y_train = pd.read_csv('/home/anton/mlops3/datasets/y_train.csv')

clf = AdaBoostClassifier(n_estimators=2, learning_rate=1, random_state=0)
clf.fit(X_train, y_train)
# Save the model using pickle
with open('/home/anton/mlops3/models/ada1.pickle', 'wb') as model_file:
    pickle.dump(clf, model_file)

