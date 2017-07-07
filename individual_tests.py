import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib

def load_test_data(file, label):
    X = np.loadtxt(file, delimiter=' ')
    test_data = np.mean(X, axis=0)
    test_data = test_data.reshape(1, -1)

    mlp = joblib.load('averahe_nn.pkl')
    print(mlp.predict(test_data))


if __name__ == '__main__':
    array = [30, 70, 44, 5, 100, 9, 74, 90, 97, 39, 46, 112, 42, 36, 43, 63, 93, 24, 114, 15, 68]
    for n in array:
        load_test_data('D:/Data/Violent Crowd/Special Cases/NonViolent_features_improved/video (' + str(n) + ')features.dat', 0)