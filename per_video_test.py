import numpy as np
import csv
import os
import data_load
import preprocess_data
import constants
import random


def get_start_and_end_indices(folder):

    start_index = 0
    terminal_indices = []

    for file in os.listdir(folder):

        if str(file).startswith('all_data'):
            continue

        with open(folder + str(file), 'r') as f:
            reader = csv.reader(f, delimiter=' ')
            data = list(reader)
            length_of_video = len(data)

        terminal_indices.append((start_index, start_index + length_of_video))
        start_index = start_index + length_of_video

    return terminal_indices


def load_train_and_test_data(ratio):

    non_violent_data, non_violent_labels = data_load.get_from_cache('E:/Data/Violent Crowd/Crowd_violence_dataset/NonViolent_features_improved/', label=0)
    violent_data, violent_labels = data_load.get_from_cache('E:/Data/Violent Crowd/Crowd_violence_dataset/Violent_features_improved/', label=1)

    # non_violent_angle, dummy_label = data_load.get_from_cache('E:/Data/Violent Crowd/Crowd_violence_dataset/Non_violent_angle/', label=0)
    # violent_angle, dummy_label = data_load.get_from_cache('E:/Data/Violent Crowd/Crowd_violence_dataset/Violent_angle/', label=0)

    # non_violent_data = np.hstack((non_violent_data, non_violent_angle))
    # violent_data = np.hstack((violent_data, violent_angle))

    estimated_labels, non_violent_outlier_indices, violent_outlier_indices, non_violent_label, violent_label  = preprocess_data.refine_raw_data()
    non_violent_labels = estimated_labels[:non_violent_data.shape[0]]
    violent_labels = estimated_labels[non_violent_data.shape[0]:]

    non_violent_terminal_indices = get_start_and_end_indices('E:/Data/Violent Crowd/Crowd_violence_dataset/NonViolent_features_improved/')
    violent_terminal_indices = get_start_and_end_indices('E:/Data/Violent Crowd/Crowd_violence_dataset/Violent_features_improved/')

    number_of_videos_from_each_set = np.floor(ratio * constants.TOTAL_NUMBER_OF_VIDEOS / 2)

    non_violent_allowed_range = list(range(0, len(non_violent_terminal_indices)))
    violent_allowed_range = list(range(0, len(violent_terminal_indices)))

    for v in non_violent_outlier_indices:
        non_violent_allowed_range.remove(v)
    
    for v in violent_outlier_indices:
        violent_allowed_range.remove(v)
    
    non_violent_indices = [random.choice(non_violent_allowed_range) for p in range(0, int(number_of_videos_from_each_set))]
    violent_indices = [random.choice(violent_allowed_range) for p in range(0, int(number_of_videos_from_each_set))]

    non_violent_train_indices = [index for index in range(0, int(constants.TOTAL_NUMBER_OF_VIDEOS / 2)) if index not in non_violent_indices]
    violent_train_indices = [index for index in range(0, int(constants.TOTAL_NUMBER_OF_VIDEOS / 2)) if index not in violent_indices]

    non_violent_train_indices = list(set(non_violent_train_indices) - set(non_violent_outlier_indices))    
    violent_train_indices = list(set(violent_train_indices) - set(violent_outlier_indices))

    print('Number of non-violent train videos ', len(non_violent_train_indices))
    print('Number of non-violent test videos ', len(non_violent_indices))
    print('Number of violent train videos ', len(violent_train_indices))
    print('Number of violent test videos ', len(violent_indices))


    non_violent_indices = [non_violent_terminal_indices[index] for index in non_violent_indices]
    violent_indices = [violent_terminal_indices[index] for index in violent_indices]

    non_violent_train_indices = [non_violent_terminal_indices[index] for index in non_violent_train_indices]
    violent_train_indices = [violent_terminal_indices[index] for index in violent_train_indices]

    non_violent_test_set = []
    violent_test_set = []

    for index in non_violent_indices:
        start, end = index
        non_violent_test_set.append((non_violent_data[start:end], non_violent_labels[start:end]))

    for index in violent_indices:
        start, end = index
        violent_test_set.append((violent_data[start:end], violent_labels[start:end]))

    train_data = None
    train_labels = None

    for index in non_violent_train_indices:
        start, end = index
        if train_data is None:
            train_data = non_violent_data[start:end]
            train_labels = non_violent_labels[start:end]

        else:
            data = non_violent_data[start:end]
            labels = non_violent_labels[start:end]
            train_data = np.vstack((train_data, data))
            train_labels = np.hstack((train_labels, labels))

    for index in violent_train_indices:
        start, end = index

        data = violent_data[start:end]
        labels = violent_labels[start:end]
        train_data = np.vstack((train_data, data))
        train_labels = np.hstack((train_labels, labels))

    # print(train_data.shape, train_labels.shape)
    #
    # for instance in non_violent_test_set:
    #     features, lbels = instance
    #     print(features)
    #     print(labels)

    return train_data, train_labels, non_violent_test_set, violent_test_set, non_violent_label, violent_label


if __name__ == '__main__':
    load_train_and_test_data(0.15)