import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
import constants
import train_test

result_matrix = np.zeros(shape=(20, 6))

def MLP():

    mlp = MLPClassifier(hidden_layer_sizes=(1000, 500, 100, 50), activation='tanh', solver='adam', alpha=1e-5, random_state=0, verbose=True)
    per_video_accuracy, per_frame_accuracy = train_test.train_and_test(mlp)
    return per_video_accuracy, per_frame_accuracy 
    

def save_results():
    global result_matrix    
    result_matrix = np.divide(result_matrix, 5)
    print(result_matrix)
    np.savetxt('Results-1-07-17.txt', result_matrix, delimiter=' ')


if __name__ == '__main__':

    per_video = 0
    per_frame = 0

    for i in range(0, 5):
        constants.TOTAL_NUMBER_OF_VIDEOS = 246
        accuracy_per_video, accuracy_per_frame = MLP()
        per_video += accuracy_per_video
        per_frame += accuracy_per_frame

        print('Average per video ', accuracy_per_video)
        print('Average per frame ', accuracy_per_frame)

    print('Five Fold per video = ', per_video / 5)
    print('Five Fold per frame = ', per_frame / 5)

    save_results()
