import numpy as np
from sklearn import linear_model
from data_load import load_data


def model_fitting_ransac(non_violent_features, non_violent_labels, violent_features, violent_labels):

    model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
    model_ransac.fit(non_violent_features, non_violent_labels)

    inliner_mask = model_ransac.inlier_mask_
    outlier_mask = np.logical_not(inliner_mask)

    estimated_labels = model_ransac.predict(violent_features)

    for actual_label, estimated_label in zip(violent_labels, estimated_labels):
        print(actual_label, estimated_label)

    acc = 1 - ((np.sum(np.fabs(estimated_labels - violent_labels)))/len(violent_labels))
    print(acc)


if __name__ == '__main__':
    non_violence_data, non_violence_labels = load_data(
        'D:/Data/Violent Crowd/Crowd_violence_dataset/NonViolent_features_improved/', label=0)
    violence_data, violence_labels = load_data(
        'D:/Data/Violent Crowd/Crowd_violence_dataset/Violent_features_improved/', label=1)
    model_fitting_ransac(non_violence_data, non_violence_labels, violence_data, violence_labels)
