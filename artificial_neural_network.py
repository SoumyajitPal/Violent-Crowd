import numpy as np
from sklearn.neural_network import MLPClassifier
import data_load


def MLP():

    train_features, train_labels, test_features, test_labels = data_load.get_data_matrix()

    mlp = MLPClassifier(hidden_layer_sizes=(1000, 500, 100, 50), activation='tanh', solver='adam', alpha=1e-5, random_state=0, verbose=True)
    print('data loaded...')
    print('Train data : ', np.shape(train_features))
    print('Test data :', np.shape(test_features))
    mlp.fit(train_features, train_labels)
    print('Model Accuracy: ', mlp.score(train_features, train_labels))
    print('Test Accuracy: ', mlp.score(test_features, test_labels))


if __name__ == '__main__':
    MLP()