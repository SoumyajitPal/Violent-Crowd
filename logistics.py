import numpy as np
from sklearn import linear_model
from sklearn.externals import joblib
import constants
import train_test


def logistic_regression():
    log_reg = linear_model.LogisticRegression(C=1e5)
    per_video_accuracy, per_frame_accuracy, result_matrix = train_test.train_and_test(log_reg)
    return per_video_accuracy, per_frame_accuracy, result_matrix


def save_results(result_matrix):
    
    result_matrix = np.divide(result_matrix, 5)
    print(result_matrix)
    np.savetxt('Results-2-07-17-Log-reg.txt', result_matrix, delimiter=' ')


if __name__ == '__main__':

    result = np.zeros(shape=(15, 6))

    per_video = 0
    per_frame = 0

    for i in range(0, 5):
        constants.TOTAL_NUMBER_OF_VIDEOS = 246
        accuracy_per_video, accuracy_per_frame, result_matrix = logistic_regression()
        per_video += accuracy_per_video
        per_frame += accuracy_per_frame
        result = np.add(result, result_matrix)

        print('Average per video ', accuracy_per_video)
        print('Average per frame ', accuracy_per_frame)

    print('Five Fold per video = ', per_video / 5)
    print('Five Fold per frame = ', per_frame / 5)

    save_results(result)
