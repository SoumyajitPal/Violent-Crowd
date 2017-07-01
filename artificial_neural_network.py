import numpy as np
from sklearn.neural_network import MLPClassifier
import data_load
import per_video_test
from sklearn.externals import joblib
import constants


result_matrix = np.zeros(shape=(20, 6))

def MLP():

    global result_matrix
    # train_features, train_labels, test_features, test_labels = data_load.get_data_matrix()
    train_features, train_labels,  non_violent_test_set, violent_test_set, non_violent_label, violent_label = per_video_test.load_train_and_test_data(0.20)

    mlp = MLPClassifier(hidden_layer_sizes=(1000, 500, 100, 50), activation='tanh', solver='adam', alpha=1e-5, random_state=0, verbose=True)
    # mlp = joblib.load('current_model.pkl')
    print('data loaded...')
    print('Train data : ', np.shape(train_features))
    # print('Test data :', np.shape(test_features))
    mlp.fit(train_features, train_labels)
    print('Model Accuracy: ', mlp.score(train_features, train_labels))
    # print('Test Accuracy: ', mlp.score(test_features, test_labels))

    accuracy_per_video_array = []
    accuracy_per_frame_array = []
   
    violence_percentage = 0.05

    # print('Start Test')
    for iteration in range(0, 20):
        score = 0
        overall_accuracy = 0

        false_positive = 0
        false_negative = 0
        true_positive = 0
        true_negative = 0

        print('Violence Threshold = ', violence_percentage)
        # print('Starting Test on Non-Violent Set...')
        for instance in non_violent_test_set:
            features, labels = instance

            prediction = mlp.predict(features)
            predicted_violence = [x for x in prediction if x == violent_label]
            predicted_non_violnece = [x for x in prediction if x == non_violent_label]
            percentage_of_violence = len(predicted_violence) / len(prediction)
            # print('Percentage of violence predicted = ', percentage_of_violence)

            actual_violence = [x for x in labels if x == violent_label]
            actual_non_violence = [x for x in labels if x == non_violent_label]
            # print('Percentage of violence actually = ', len(actual_violence) / len(labels))

            accuracy = mlp.score(features, labels)
            overall_accuracy += accuracy
            # print('Accuracy = ', accuracy)

            if percentage_of_violence < violence_percentage:
                score += 1
                true_negative += 1
            else:
                false_positive += 1

        # print('Starting Test on Violent Set...')
        for instance in violent_test_set:
            features, labels = instance

            prediction = mlp.predict(features)
            predicted_violence = [x for x in prediction if x == violent_label]
            predicted_non_violnece = [x for x in prediction if x == non_violent_label]
            percentage_of_violence = len(predicted_violence) / len(prediction)
            # print('Percentage of violence predicted = ', percentage_of_violence)

            actual_violence = [x for x in labels if x == violent_label]
            actual_non_violence = [x for x in labels if x == non_violent_label]
            # print('Percentage of violence actually = ', len(actual_violence) / len(labels))

            accuracy = mlp.score(features, labels)
            overall_accuracy += accuracy
            # print('Accuracy = ', accuracy)

            if percentage_of_violence > violence_percentage:
                score += 1
                true_positive += 1
            else:
                false_negative += 1

        accuracy_per_video = score / (len(non_violent_test_set) + len(violent_test_set))
        accuracy_per_frame = overall_accuracy / (len(non_violent_test_set) + len(violent_test_set))

        accuracy_per_video_array.append(accuracy_per_video)
        accuracy_per_frame_array.append(accuracy_per_frame)
        print('Accuracy per video = ', accuracy_per_video)
        print('Accuracy per frame = ', accuracy_per_frame)

        # print('Confusion matrix: ')
        # print('True positive ', true_positive)
        # print('True negative ', true_negative)
        # print('False positive ', false_positive)
        # print('False negative ', false_negative)

        result_array = np.array([true_positive, true_negative, false_positive, false_negative, accuracy_per_video, accuracy_per_frame])
        result_matrix[iteration] += result_array

        violence_percentage += 0.05

    # filename = 'current_model.pkl'
    # _ = joblib.dump(mlp, filename, compress=9)


    return np.max(accuracy_per_video_array), np.max(accuracy_per_frame_array)
    

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
