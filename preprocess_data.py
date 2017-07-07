import numpy as np
import data_load
import k_means
import matplotlib.pyplot as plt
import os
import csv
import per_video_test
import constants


def plot_progress(estimated_labels, non_violent_label, violent_label):

    start_index = 0

    folder = 'E:/Data/Violent Crowd/Crowd_violence_dataset/Violent_features_improved/'
    savefolder = 'E:/Data/Violent Crowd/Crowd_violence_dataset/NonViolent_visualization_including_angle/'

    if not os.path.exists(savefolder):
        os.makedirs(savefolder)

    ratio_array = []

    for file in os.listdir(folder):

        if str(file).startswith('all_data'):
            continue

        with open(folder + str(file), 'r') as f:
            reader = csv.reader(f, delimiter=' ')
            data = list(reader)
            length_of_video = len(data)

        labels_of_the_day = estimated_labels[start_index:start_index+length_of_video]
        start_index = start_index + length_of_video

        # fig = plt.figure(figsize=(8, 2))
        # ax = fig.add_subplot(111)

        non_violent = [index for (index, point) in enumerate(labels_of_the_day) if point == non_violent_label]
        violent = [index for (index, point) in enumerate(labels_of_the_day) if point == violent_label]

        ratio_array.append(len(violent)/(len(violent) + len(non_violent)))

        # non_violent_axis = np.ones(shape=(len(non_violent), ))
        # violent_axis = np.ones(shape=(len(violent),))
        # ax.scatter(non_violent, non_violent_axis, c='green', lw=0, alpha=0.1)
        # ax.scatter(violent, violent_axis, c='red', lw=0)

        # plt.show()
        # plt.savefig(savefolder + str(file)[:-4] + '.jpg')
        # plt.clf()

    print(np.min(ratio_array))
    print(np.max(ratio_array))
    print(np.mean(ratio_array))
    print(np.median(ratio_array))
    print(np.std(ratio_array))

    median = np.median(ratio_array)
    mean = np.mean(ratio_array)
    lowest_violence_point = mean - np.std(ratio_array)

    okay_violence = [video for video in ratio_array if video > mean]
    good_violence = [video for video in ratio_array if video > median]
    almost_non_violence = [video for video in ratio_array if video < lowest_violence_point]
    almost_non_violent_videos = [index for index, video in enumerate(ratio_array) if video < lowest_violence_point]

    print(len(okay_violence))
    print(len(good_violence))
    print(len(almost_non_violence))
    print(almost_non_violent_videos)


def preprocess_violent(estimated_violent_labels, non_violent_label, violent_label):
    violent_terminal_indices = per_video_test.get_start_and_end_indices('D:/Data/Violent Crowd/Crowd_violence_dataset/Violent_features_improved/')
    bad_video_count = 0
    outlier_indices = []
    for video_number, instance in enumerate(violent_terminal_indices):
        start, end = instance
        labels = estimated_violent_labels[start:end]
        violence = [index for index in labels if index == violent_label]
        non_violence = [index for index in labels if index == non_violent_label]

        if len(violence) < 0.05 * len(labels):
            bad_video_count += 1
            outlier_indices.append(video_number)

    constants.TOTAL_NUMBER_OF_VIDEOS -= bad_video_count
    print('Number of bad violent videos ', bad_video_count)
    print(outlier_indices)
    return outlier_indices


def preprocess_non_violent(estimated_non_violent_labels, non_violent_label, violent_label):
    non_violent_terminal_indices = per_video_test.get_start_and_end_indices('D:/Data/Violent Crowd/Crowd_violence_dataset/NonViolent_features_improved/')
    estimated_labels = None
    bad_video_count = 0
    outlier_indices = []
    for video_number, instance in enumerate(non_violent_terminal_indices):
        start, end = instance
        labels = estimated_non_violent_labels[start:end]
        violence = [index for index in labels if index == violent_label]
        non_violence = [index for index in labels if index == non_violent_label]

        if len(violence) > (20 * len(non_violence)):
            bad_video_count += 1
            outlier_indices.append(video_number)
        
        if len(violence) < (0.8 * len(non_violence)):
            if estimated_labels is None:
                estimated_labels = np.ones(shape=(len(labels),)) * non_violent_label
            else:
                estimated_labels = np.hstack((estimated_labels, np.ones(shape=(len(labels),)) * non_violent_label))
        else:
            if estimated_labels is None:
                estimated_labels = labels
            else:
                estimated_labels = np.hstack((estimated_labels, labels))

    constants.TOTAL_NUMBER_OF_VIDEOS -= bad_video_count
    print('Number of bad non-violent videos ', bad_video_count)
    print('Original non-violent labels size', estimated_non_violent_labels.shape)
    # print('Original non-violent data size', non_violent_data.shape)
    print('Estimated non-violent labels size', estimated_labels.shape)
    # print('Estimated non-violent data size', estimated_data.shape)
    print(outlier_indices)
    return estimated_labels, outlier_indices


def refine_raw_data():
    non_violent_data, non_violent_labels = data_load.get_from_cache('D:/Data/Violent Crowd/Crowd_violence_dataset/NonViolent_features_improved/', label=0)
    violent_data, violent_labels = data_load.get_from_cache('D:/Data/Violent Crowd/Crowd_violence_dataset/Violent_features_improved/', label=1)

    # non_violent_angle, dummy_label = data_load.get_from_cache('E:/Data/Violent Crowd/Crowd_violence_dataset/Non_violent_angle/', label=0)
    # violent_angle, dummy_label = data_load.get_from_cache('E:/Data/Violent Crowd/Crowd_violence_dataset/Violent_angle/',label=0)

    # non_violent_data = np.hstack((non_violent_data, non_violent_angle))
    # violent_data = np.hstack((violent_data, violent_angle))

    data = np.vstack((non_violent_data, violent_data))
    labels = np.hstack((non_violent_labels, violent_labels))

    # estimated_labels = k_means.K_Means(data, labels)

    # non_violent_count = non_violent_labels.shape[0]
    # violent_count = violent_labels.shape[0]

    # estimated_non_violent_labels = estimated_labels[0:non_violent_count]
    # estimated_violent_labels = estimated_labels[non_violent_count:]

    # non_violent_label = 0
    # violent_label = 1

    # if np.count_nonzero(estimated_non_violent_labels) > (non_violent_count - np.count_nonzero(estimated_non_violent_labels)):
    #     non_violent_label = 1
    #     violent_label = 0

    estimated_non_violent_labels = np.loadtxt('Non-violent-labels-3-7-17.txt', delimiter=' ')
    estimated_violent_labels = np.loadtxt('Violent-labels-3-7-17.txt', delimiter=' ')

    loaded_labels = np.loadtxt('labls-3-7-17.txt', delimiter=' ')
    non_violent_label = int(loaded_labels[0])
    violent_label = int(loaded_labels[1])
    # actual_non_violent_labels = np.ones(shape=(non_violent_count,)) * non_violent_label
    # actual_violent_labels = np.ones(shape=(violent_count,)) * violent_label

    # print(np.count_nonzero(np.fabs(actual_non_violent_labels - estimated_non_violent_labels)))
    # print(np.count_nonzero(np.fabs(actual_violent_labels - estimated_violent_labels)))
    estimated_non_violent_labels, non_violent_outlier_indices = preprocess_non_violent(estimated_non_violent_labels, non_violent_label, violent_label)
    violent_outlier_indices = preprocess_violent(estimated_violent_labels, non_violent_label, violent_label)

    estimated_labels = np.hstack((estimated_non_violent_labels, estimated_violent_labels))
    # plot_progress(estimated_violent_labels, non_violent_label, violent_label)

    # np.savetxt('Non-violent-outliers-3-7-17.txt', non_violent_outlier_indices, delimiter=' ', fmt='%d')
    # np.savetxt('Violent-outliers.txt-3-7-17', violent_outlier_indices, delimiter=' ', fmt='%d')

    # np.savetxt('labls.txt-3-7-17', np.array([non_violent_label, violent_label]), delimiter=' ', fmt='%d')

    # np.savetxt('Non-violent-labels.txt-3-7-17', estimated_non_violent_labels, delimiter=' ', fmt='%d')
    # np.savetxt('Violent-labels.txt-3-7-17', estimated_violent_labels, delimiter=' ', fmt='%d')

    return estimated_labels, non_violent_outlier_indices, violent_outlier_indices, non_violent_label, violent_label


if __name__ == '__main__':
    refine_raw_data()
