from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.externals import joblib
import constants
import train_test



def RandomForest():

    clf = RandomForestClassifier(n_estimators=600, n_jobs=2)
    per_video_accuracy, per_frame_accuracy, result_matrix = train_test.train_and_test(clf)
    return per_video_accuracy, per_frame_accuracy, result_matrix


def save_results(result_matrix):
    
    result_matrix = np.divide(result_matrix, 15)
    print(result_matrix)
    np.savetxt('Results-1-07-17-Random_forest.txt', result_matrix, delimiter=' ')


if __name__ == '__main__':

    result = np.zeros(shape=(20, 6))

    per_video = 0
    per_frame = 0

    for i in range(0, 5):
        constants.TOTAL_NUMBER_OF_VIDEOS = 246
        accuracy_per_video, accuracy_per_frame, result_matrix = RandomForest()
        per_video += accuracy_per_video
        per_frame += accuracy_per_frame
        result = np.add(result, result_matrix)

        print('Average per video ', accuracy_per_video)
        print('Average per frame ', accuracy_per_frame)

    print('Five Fold per video = ', per_video / 5)
    print('Five Fold per frame = ', per_frame / 5)

    save_results(result)
