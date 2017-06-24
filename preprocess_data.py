import numpy as np
import data_load
import k_means
import matplotlib.pyplot as plt
import os
import csv


def plot_progress(estimated_labels, non_violent_label, violent_label):

    start_index = 0

    folder = 'E:/Data/Violent Crowd/Crowd_violence_dataset/NonViolent_features_improved/'
    savefolder = 'E:/Data/Violent Crowd/Crowd_violence_dataset/NonViolent_visualization/'

    for file in os.listdir(folder):

        if str(file).startswith('all_data'):
            continue

        with open(folder + str(file), 'r') as f:
            reader = csv.reader(f, delimiter=' ')
            data = list(reader)
            length_of_video = len(data)

        labels_of_the_day = estimated_labels[start_index:start_index+length_of_video]
        start_index = start_index + length_of_video

        fig = plt.figure(figsize=(8, 2))
        ax = fig.add_subplot(111)

        non_violent = [index for (index, point) in enumerate(labels_of_the_day) if point == non_violent_label]
        violent = [index for (index, point) in enumerate(labels_of_the_day) if point == violent_label]

        non_violent_axis = np.ones(shape=(len(non_violent), ))
        violent_axis = np.ones(shape=(len(violent),))
        ax.scatter(non_violent, non_violent_axis, c='green', lw=0, alpha=0.1)
        ax.scatter(violent, violent_axis, c='red', lw=0)

        # plt.show()
        plt.savefig(savefolder + str(file)[:-4] + '.jpg')
        plt.clf()


def refine_raw_data():
    non_violent_data, non_violent_labels = data_load.get_from_cache('E:/Data/Violent Crowd/Crowd_violence_dataset/NonViolent_features_improved/', label=0)
    violent_data, violent_labels = data_load.get_from_cache('E:/Data/Violent Crowd/Crowd_violence_dataset/Violent_features_improved/', label=1)

    data = np.vstack((non_violent_data, violent_data))
    labels = np.hstack((non_violent_labels, violent_labels))

    estimated_labels = k_means.K_Means(data, labels)

    non_violent_count = non_violent_labels.shape[0]
    violent_count = violent_labels.shape[0]

    estimated_non_violent_labels = estimated_labels[0:non_violent_count]
    estimated_violent_labels = estimated_labels[non_violent_count:]

    non_violent_label = 0
    violent_label = 1

    if np.count_nonzero(estimated_non_violent_labels) > (non_violent_count - np.count_nonzero(estimated_non_violent_labels)):
        non_violent_label = 1
        violent_label = 0


    actual_non_violent_labels = np.ones(shape=(non_violent_count,)) * non_violent_label
    actual_violent_labels = np.ones(shape=(violent_count,)) * violent_label

    print(np.count_nonzero(np.fabs(actual_non_violent_labels - estimated_non_violent_labels)))
    print(np.count_nonzero(np.fabs(actual_violent_labels - estimated_violent_labels)))

    estimated_labels = np.hstack((actual_non_violent_labels, estimated_violent_labels))

    plot_progress(estimated_non_violent_labels, non_violent_label, violent_label)

    return estimated_labels


if __name__ == '__main__':
    refine_raw_data()
