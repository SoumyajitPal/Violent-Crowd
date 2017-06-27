import numpy as np
from sklearn.neural_network import MLPClassifier
import data_load
import per_video_test


def MLP():

    # train_features, train_labels, test_features, test_labels = data_load.get_data_matrix()
    train_features, train_labels,  non_violent_test_set, violent_test_set, non_violent_label, violent_label = per_video_test.load_train_and_test_data(0.15)

    mlp = MLPClassifier(hidden_layer_sizes=(1000, 500, 100, 50), activation='tanh', solver='adam', alpha=1e-5, random_state=0, verbose=True)
    print('data loaded...')
    print('Train data : ', np.shape(train_features))
    # print('Test data :', np.shape(test_features))
    mlp.fit(train_features, train_labels)
    print('Model Accuracy: ', mlp.score(train_features, train_labels))
    # print('Test Accuracy: ', mlp.score(test_features, test_labels))

    score = 0
    overall_accuracy = 0

    print('Starting Test on Non-Violent Set...')
    for instance in non_violent_test_set:
        features, labels = instance

        prediction = mlp.predict(features)
        predicted_violence = [x for x in prediction if x == violent_label]
        predicted_non_violnece = [x for x in prediction if x == non_violent_label]
        percentage_of_violence = len(predicted_violence) / len(prediction)
        print('Percentage of violence predicted = ', percentage_of_violence)

        actual_violence = [x for x in labels if x == violent_label]
        actual_non_violence = [x for x in labels if x == non_violent_label]
        print('Percentage of violence actually = ', len(actual_violence) / len(labels))

        accuracy = mlp.score(features, labels)
        overall_accuracy += accuracy
        print('Accuracy = ', accuracy)

        if percentage_of_violence < 0.3:
            score += 1

    print('Starting Test on Violent Set...')
    for instance in violent_test_set:
        features, labels = instance

        prediction = mlp.predict(features)
        predicted_violence = [x for x in prediction if x == violent_label]
        predicted_non_violnece = [x for x in prediction if x == non_violent_label]
        percentage_of_violence = len(predicted_violence) / len(prediction)
        print('Percentage of violence predicted = ', percentage_of_violence)

        actual_violence = [x for x in labels if x == violent_label]
        actual_non_violence = [x for x in labels if x == non_violent_label]
        print('Percentage of violence actually = ', len(actual_violence) / len(labels))

        accuracy = mlp.score(features, labels)
        overall_accuracy += accuracy
        print('Accuracy = ', accuracy)

        if percentage_of_violence > 0.3:
            score += 1

    accuracy_per_video = score / (len(non_violent_test_set) + len(violent_test_set))
    accuracy_per_frame = overall_accuracy / (len(non_violent_test_set) + len(violent_test_set))
    print('Accuracy per video = ', accuracy_per_video)
    print('Accuracy per frame = ', accuracy_per_frame)

    return accuracy_per_video, accuracy_per_frame


if __name__ == '__main__':

    per_video =0
    per_frame = 0

    for i in range(0, 5):
        accuracy_per_video, accuracy_per_frame = MLP()
        per_video += accuracy_per_video
        per_frame += accuracy_per_frame

    print('Five Fold per video = ', per_video / 5)
    print('Five Fold per frame = ', per_frame / 5)
