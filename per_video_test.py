import numpy as np
import csv
import os
import data_load
import preprocess_data
import constants
from random import randint


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

    estimated_labels, non_violent_label, violent_label = preprocess_data.refine_raw_data()
    non_violent_labels = estimated_labels[:non_violent_data.shape[0]]
    violent_labels = estimated_labels[non_violent_data.shape[0]:]

    non_violent_terminal_indices = get_start_and_end_indices('E:/Data/Violent Crowd/Crowd_violence_dataset/NonViolent_features_improved/')
    violent_terminal_indices = get_start_and_end_indices('E:/Data/Violent Crowd/Crowd_violence_dataset/Violent_features_improved/')

    number_of_videos_from_each_set = np.floor(ratio * constants.TOTAL_NUMBER_OF_VIDEOS / 2)

    non_violent_indices = [randint(0, len(non_violent_terminal_indices)) for p in range(0, int(number_of_videos_from_each_set))]
    violent_indices = [randint(0, len(violent_terminal_indices)) for p in range(0, int(number_of_videos_from_each_set))]

    non_violent_train_indices = [index for index in range(0, int(constants.TOTAL_NUMBER_OF_VIDEOS / 2)) if index not in non_violent_indices]
    violent_train_indices = [index for index in range(0, int(constants.TOTAL_NUMBER_OF_VIDEOS / 2)) if index not in violent_indices]

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