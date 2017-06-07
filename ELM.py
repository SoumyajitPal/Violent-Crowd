import numpy as np
from sklearn.linear_model import LogisticRegression

from sklearn_extensions.extreme_learning_machines.elm import GenELMClassifier
from sklearn_extensions.extreme_learning_machines.random_layer import RBFRandomLayer, MLPRandomLayer
from sklearn.externals import joblib

import data_load


def make_classifiers():

    names = ["ELM(10,tanh)", "ELM(10,tanh,LR)", "ELM(10,sinsq)", "ELM(10,tribas)", "ELM(hardlim)", "ELM(20,rbf(0.1))"]

    nh = 1000

    # pass user defined transfer func
    sinsq = (lambda x: np.power(np.sin(x), 2.0))
    srhl_sinsq = MLPRandomLayer(n_hidden=nh, activation_func=sinsq)

    # use internal transfer funcs
    srhl_tanh = MLPRandomLayer(n_hidden=nh, activation_func='tanh')
    srhl_tribas = MLPRandomLayer(n_hidden=nh, activation_func='tribas')
    srhl_hardlim = MLPRandomLayer(n_hidden=nh, activation_func='hardlim')

    # use gaussian RBF
    srhl_rbf = RBFRandomLayer(n_hidden=nh*2, rbf_width=0.1, random_state=0)
    log_reg = LogisticRegression()

    classifiers = [GenELMClassifier(hidden_layer=srhl_tanh),
                  GenELMClassifier(hidden_layer=srhl_sinsq),
                  GenELMClassifier(hidden_layer=srhl_tribas),
                  GenELMClassifier(hidden_layer=srhl_hardlim),
                  GenELMClassifier(hidden_layer=srhl_rbf)]
    
    # classifiers = [GenELMClassifier(hidden_layer=srhl_tanh)]

    return names, classifiers


def getData():

    train_features, train_labels, test_features, test_labels = data_load.get_data_matrix()
    return train_features, train_labels, test_features, test_labels


if __name__ == '__main__':
    # generate some datasets
    # datasets = make_datasets()
    names, classifiers = make_classifiers()
    train_features, train_labels, test_features, test_labels = getData()

    for name, clf in zip(names, classifiers):
            clf.fit(train_features, train_labels)
            score = clf.score(test_features, test_labels)
            print('Model %s score: %s' % (name, score))
    # joblib.dump(clf,'modelELM.pkl')


    