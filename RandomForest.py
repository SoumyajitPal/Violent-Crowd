from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.externals import joblib
import data_load

#Random Forrest 85.52 accuracy

def RandomForest():

    train_features, train_labels, test_features, test_labels = data_load.get_data_matrix()

    clf = RandomForestClassifier(n_estimators=600, n_jobs=2)
    clf.fit(train_features, train_labels)

    print('Model Accuracy: ', clf.score(train_features, train_labels))
    print('Test Accuracy: ', clf.score(test_features, test_labels))


if __name__ == '__main__':

    RandomForest()