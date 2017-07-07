import numpy as np
import os
import constants
from sklearn.model_selection import ShuffleSplit
import k_means
import preprocess_data
import per_video_test
import pickle


def load_average_violent_data(folder, label):
    data = None
    # outliers = [0, 7, 42, 43, 56, 57, 95, 118]
    outliers = []
    names = []
    array = [31, 120, 60, 58, 77, 48, 12, 68, 64, 97, 56, 44, 61, 70, 111, 45, 59, 55, 9]
    # array = []
    for index, f in enumerate(os.listdir(folder)):

        if index in outliers:
            continue
        
        if index+1 in array:
            X = np.genfromtxt('E:/Data/Violent Crowd/Special Cases/Violent_features_improved/video (' + str(index+1) + ')features.dat', delimiter= ' ')

        elif str(f).endswith('.dat') and not str(f).startswith('all_data'):
            # X = np.loadtxt(folder+str(f), delimiter=' ')
            # print(str(f))
            try:
                X = np.genfromtxt(folder+str(f), delimiter=' ')
            except ValueError:
                continue
        else:
            continue
        if data is None:
            data = np.mean(X, axis=0)
        else:
            data = np.vstack((data, np.mean(X, axis=0)))
        names.append('vi' + str(index))

    # np.savetxt(folder+'all_data.dat', data, fmt='%6.5f', delimiter=' ')
    data_size = np.shape(data)[0]
    return data, np.ones(shape=(data_size, ))*label, names


def load_average_non_violent_data(folder, label):
    data = None
    outliers = [3, 28, 46, 68, 98, 112, 117, 118]
    # outliers = []
    array = [30, 70, 44, 5, 100, 9, 74, 90, 97, 39, 46, 112, 42, 36, 43, 63, 93, 24, 114, 15, 68] #, 4, 29, 47, 69, 99, 113, 118, 119]
    # array = []

    names = []
    for index, f in enumerate(os.listdir(folder)):

        if index in outliers:
            continue
        
        if index+1 in array:
            X = np.genfromtxt('E:/Data/Violent Crowd/Special Cases/NonViolent_features_improved/video (' + str(index+1) + ')features.dat', delimiter= ' ')


        elif str(f).endswith('.dat') and not str(f).startswith('all_data'):
            # X = np.loadtxt(folder+str(f), delimiter=' ')
            # print(str(f))
            try:
                X = np.genfromtxt(folder+str(f), delimiter=' ')
            except ValueError:
                continue
        else:
            continue
        if data is None:
            data = np.mean(X, axis=0)
        else:
            data = np.vstack((data, np.mean(X, axis=0)))
        names.append('nv' + str(index))
        

    # np.savetxt(folder+'all_data.dat', data, fmt='%6.5f', delimiter=' ')
    data_size = np.shape(data)[0]
    return data, np.ones(shape=(data_size, ))*label, names


def load_non_violent_angles(folder, label):
    data = None
    outliers = [3, 28, 46, 68, 98, 112, 117, 118]
    # outliers = []

    for index, f in enumerate(os.listdir(folder)):

        if index in outliers:
            continue
        
        if str(f).endswith('.dat') and not str(f).startswith('all_data'):
            # X = np.loadtxt(folder+str(f), delimiter=' ')
            # print(str(f))
            try:
                X = np.genfromtxt(folder+str(f), delimiter=' ')
            except ValueError:
                continue
        else:
            continue
        if data is None:
            data = np.mean(X, axis=0)
        else:
            data = np.vstack((data, np.mean(X, axis=0)))
        

    # np.savetxt(folder+'all_data.dat', data, fmt='%6.5f', delimiter=' ')
    data_size = np.shape(data)[0]
    return data, np.ones(shape=(data_size, ))*label

def load_violent_angles(folder, label):
    data = None
    outliers = []

    for index, f in enumerate(os.listdir(folder)):

        if index in outliers:
            continue
        
        if str(f).endswith('.dat') and not str(f).startswith('all_data'):
            # X = np.loadtxt(folder+str(f), delimiter=' ')
            # print(str(f))
            try:
                X = np.genfromtxt(folder+str(f), delimiter=' ')
            except ValueError:
                continue
        else:
            continue
        if data is None:
            data = np.mean(X, axis=0)
        else:
            data = np.vstack((data, np.mean(X, axis=0)))
        

    # np.savetxt(folder+'all_data.dat', data, fmt='%6.5f', delimiter=' ')
    data_size = np.shape(data)[0]
    return data, np.ones(shape=(data_size, ))*label


def load_sounak_descriptor():

    violent_data = np.load('E:/Data/Violent Crowd/Crowd_violence_dataset/Sounak_descriptor/Violence_descriptor.dat')
    non_violent_data = np.load('E:/Data/Violent Crowd/Crowd_violence_dataset/Sounak_descriptor/NonViolence_descriptor.dat')

    # non_violent_data = pickle.load(open('E:/Data/Violent Crowd/Crowd_violence_dataset/Sounak_descriptor/NonViolence_descriptor.dat', 'rb'))
   
    print(len(violent_data))
    print(len(non_violent_data))

    stacked_data = None

    outliers = [3, 28, 46, 68, 98, 112, 117, 118]
    # outliers = []
    for index, instance in enumerate(non_violent_data):
        if index in outliers:
            continue
        X = np.array(instance)
        X = np.mean(X, axis=0)
        # print(X)
        if stacked_data is None:
            stacked_data = X
        else:
            stacked_data = np.vstack((stacked_data, X))

    for instance in violent_data:
        X = np.array(instance)
        X = np.mean(X, axis=0)
        # print(X)        
        stacked_data = np.vstack((stacked_data, X))

    return stacked_data            



def load_data(folder, label):

    data = None
    for f in os.listdir(folder):

        if str(f).endswith('.dat') and not str(f).startswith('all_data'):
            # X = np.loadtxt(folder+str(f), delimiter=' ')
            print(str(f))
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


def shuffle_and_split(data, labels, names, ratio):
    splitter = ShuffleSplit(n_splits=1, test_size=ratio, random_state=0)
    for train_index, test_index in splitter.split(data):
        # print(train_index)
        # print(test_index)
        train_data = data[train_index]
        train_labels = labels[train_index]
        train_names = [names[i] for i in train_index]
        test_data = data[test_index]
        test_labels = labels[test_index]
        test_names = [names[i] for i in test_index]
    return train_data, train_labels, test_data, test_labels, train_names, test_names


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

    estimated_labels, non_violent_label, violent_label = preprocess_data.refine_raw_data()

    train_data, train_labels, test_data, test_labels = shuffle_and_split(data, estimated_labels, ratio=0.15)

    return train_data, train_labels, test_data, test_labels

if __name__ =='__main__':

    # non_violence_data, non_violence_labels = load_data('D:/Data/Violent Crowd/Crowd_violence_dataset/NonViolent_features_improved/', label=0)
    # violence_data, violence_labels = load_data('D:/Data/Violent Crowd/Crowd_violence_dataset/Violent_features_improved/', label=1)

    # non_violence_data, non_violence_labels = load_average_data('E:/Data/Violent Crowd/Crowd_violence_dataset/NonViolent_features_improved/', label=0)
    # violence_data, violence_labels = load_average_data('E:/Data/Violent Crowd/Crowd_violence_dataset/Violent_features_improved/', label=1)

    # print(non_violence_data)
    # print(non_violence_data.shape, non_violence_labels.shape)

    # print(violence_data)
    # print(violence_data.shape, violence_labels.shape)

    load_sounak_descriptor()
