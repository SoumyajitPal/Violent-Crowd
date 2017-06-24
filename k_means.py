import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import data_load
from sklearn.cluster import KMeans


def plot3d(data, labels, axes):
    fig = plt.figure()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    ax1, ax2, ax3 = axes
    ax.scatter(data[:, ax1], data[:, ax2], data[:, ax3], c=labels.astype(np.float))
    plt.show()


def K_Means(features, actual_labels):

    estimator = KMeans(n_clusters=2)
    estimator.fit(features)
    labels = estimator.labels_
    # cluster_centers = estimator.cluster_centers_
    #
    # actual_violent_length = len([x for x in train_labels if x == 1])
    # actual_non_violent_length = len([x for x in train_labels if x == 0])
    #
    # estimated_violent_length = len([x for x in labels if x == 1])
    # estimated_non_violent_length = len([x for x in labels if x == 0])

    acc = 1 - (np.sum(np.fabs(labels - actual_labels))/len(actual_labels))
    print(acc)

    # print(cluster_centers)

    # plot3d(train_features, labels, axes=(4, 16, 29))

    # print(actual_non_violent_length, actual_violent_length)
    # print(estimated_non_violent_length, estimated_violent_length)

    return labels


def relabel_data(train_features, train_labels, test_features, test_labels ):

    kmeans_train_labels = K_Means(train_features, train_labels)
    kmeans_test_labels = K_Means(test_features, test_labels)

    return kmeans_train_labels, kmeans_test_labels


if __name__ =='__main__':
    train_features, train_labels, test_features, test_labels = data_load.get_data_matrix()
    K_Means(train_features, train_labels)
