import numpy as np
import os
import constants
from sklearn.model_selection import ShuffleSplit

def load_data(folder, label):

    data = None
    for f in os.listdir(folder):
        print(str(f))
        if str(f).endswith('.dat'):
            # X = np.loadtxt(folder+str(f), delimiter=' ')
            try:
                X = np.genfromtxt(folder+str(f), delimiter=' ')
            except ValueError:
                continue
            if data is None:
                data = X
            else:
                data = np.vstack((data, X))

    np.savetxt(folder+'all_data.dat', data, fmt='%6.5f', delimiter=' ')
    data_size = np.shape(data)[0]
    # print(data_size)
    # return data[:(data_size - constants.TEST_SIZE)], data[(data_size - constants.TEST_SIZE):], np.ones(shape=(data_size-constants.TEST_SIZE, ))*label, np.ones(shape=(constants.TEST_SIZE, ))*label
    return data, np.ones(shape=(data_size, ))*label


def shuffle_and_split(data, labels, ratio):
    splitter = ShuffleSplit(n_splits=1, test_size=ratio, random_state=0)
    for train_index, test_index in splitter.split(data):
        # print(train_index)
        # print(test_index)
        train_data = data[train_index]
        tran_labels = labels[train_index]
        test_data = data[test_index]
        test_labels = labels[test_index]
    return train_data, tran_labels, test_data, test_labels


def get_from_cache(folder, label):
    data = np.loadtxt(folder + 'all_data.dat', delimiter=' ')
    data_size = np.shape(data)[0]
    # print(data_size)
    # return data[:(data_size - constants.TEST_SIZE)], data[(data_size - constants.TEST_SIZE):], np.ones(shape=(data_size - constants.TEST_SIZE,)) * label, np.ones(shape=(constants.TEST_SIZE,)) * label
    return data, np.ones(shape=(data_size,)) * label


def get_data_matrix():
    # non_violent_train_data, non_violent_test_data, non_violent_train_labels, non_violent_test_labels = load_data('E:/Data/Violent Crowd/HockeyFights/NonViolent_features/', label=0)
    # violent_train_data, violent_test_data, violent_train_labels, violent_test_labels = load_data('E:/Data/Violent Crowd/HockeyFights/Violent_features/', label=1)

    # non_violent_data, non_violent_labels = get_from_cache('D:/Data/Violent Crowd/HockeyFights/NonViolent_features/', label=0)
    # violent_data, violent_labels = get_from_cache('D:/Data/Violent Crowd/HockeyFights/Violent_features/', label=1)

    non_violent_data, non_violent_labels = get_from_cache('E:/Data/Violent Crowd/Crowd_violence_dataset/NonViolent_features_improved/', label=0)
    violent_data, violent_labels = get_from_cache('E:/Data/Violent Crowd/Crowd_violence_dataset/Violent_features_improved/', label=1)

    # train_data = np.vstack((non_violent_train_data, violent_train_data))
    # train_labels = np.hstack((non_violent_train_labels, violent_train_labels))
    # test_data = np.vstack((non_violent_test_data, violent_test_data))
    # test_labels = np.hstack((non_violent_test_labels, violent_test_labels))
    #
    # print(train_labels.shape, test_labels.shape, train_data.shape, test_data.shape)

    data = np.vstack((non_violent_data, violent_data))
    labels = np.hstack((non_violent_labels, violent_labels))

    train_data, train_labels, test_data, test_labels = shuffle_and_split(data, labels, ratio=0.15)

    return train_data, train_labels, test_data, test_labels

if __name__ =='__main__':

    non_violence_data, non_violence_labels = load_data('D:/Data/Violent Crowd/Crowd_violence_dataset/NonViolent_features_improved/', label=0)
    violence_data, violence_labels = load_data('D:/Data/Violent Crowd/Crowd_violence_dataset/Violent_features_improved/', label=1)
