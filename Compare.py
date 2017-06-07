import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
import data_load

def getData():

    train_features, train_labels, test_features, test_labels = data_load.get_data_matrix()
    return train_features, train_labels, test_features, test_labels


def Ensemble(train_features, train_labels, test_features, test_labels):
    random_forest = RandomForestClassifier(n_estimators=600, n_jobs=2)
    knn = KNeighborsClassifier(n_neighbors=50, weights='distance', algorithm='kd_tree', n_jobs=2)
    lsvm = svm.SVC(kernel='rbf', gamma=0.3, probability=True)
    mlp = MLPClassifier(hidden_layer_sizes=(1000, 50))
    log_reg = linear_model.LogisticRegression(C=1e5)

    ensembleCLF = VotingClassifier(estimators=[('rf', random_forest),
                                               ('knn', knn),
                                               ('svm', lsvm),
                                               ('mlp', mlp),
                                               ('logistics',log_reg)], voting='hard')
    # ensembleCLF = VotingClassifier(estimators=[
    #    ('rf',rForest),  ('mlp', mlp)], voting='soft')
    # ensembleCLF = VotingClassifier(estimators=[
    #     ('rf',rForest), ('svm',lsvm), ('mlp', mlp), ('elm',elm)], voting='hard', weights=[6,2,4,2])

    ensembleCLF.fit(train_features, train_labels)


    #ada = AdaBoostClassifier(base_estimator='ensambleCLF',algorithm='SAMME')
    #ada.fit(train_features,train_labels)
    #adaP = ada.predict(test_features)
    acc = ensembleCLF.predict(test_features)
    non_zero = (np.count_nonzero(np.fabs(acc-test_labels)))/len(acc)
    print(1 - non_zero)
    #scores=cross_val_score(ensembleCLF,all_features,all_labels, cv=5)
    #print (scores.mean())
    #non_zero1 = (np.count_nonzero(np.fabs(adaP-test_labels)))/len(adaP)
    #print(1 - non_zero1)


if __name__ == '__main__':
    train_features, train_labels, test_features, test_labels = getData()
    Ensemble(train_features, train_labels, test_features, test_labels)
