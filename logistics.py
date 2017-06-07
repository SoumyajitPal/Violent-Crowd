import numpy as np
from sklearn import linear_model
import data_load


def logistic_regression():

    train_features, train_labels, test_features, test_labels = data_load.get_data_matrix()
    log_reg = linear_model.LogisticRegression(C=1e5)
    log_reg.fit(train_features, train_labels)

    print('Model Accuracy: ', log_reg.score(train_features, train_labels))
    print('Test Accuracy: ', log_reg.score(test_features, test_labels))

if __name__ == '__main__':

    logistic_regression()