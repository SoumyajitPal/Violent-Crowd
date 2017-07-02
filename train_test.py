import numpy as np
from sklearn.neural_network import MLPClassifier
import data_load
import per_video_test
from sklearn.externals import joblib
import constants




def train_and_test(classifier):

    result_matrix = np.zeros(shape=(15, 6))
    # train_features, train_labels, test_features, test_labels = data_load.get_data_matrix()
    train_features, train_labels,  non_violent_test_set, violent_test_set, non_violent_label, violent_label = per_video_test.load_train_and_test_data(0.20)

    print('data loaded...')
    print('Train data : ', np.shape(train_features))
    # print('Test data :', np.shape(test_features))
    classifier.fit(train_features, train_labels)
    print('Model Accuracy: ', classifier.score(train_features, train_labels))
    # print('Test Accuracy: ', mlp.score(test_features, test_labels))

    accuracy_per_video_array = []
    overall_accuracy = 0

    non_violent_predictions = []
    for instance in non_violent_test_set:
        features, labels = instance

        prediction = classifier.predict(features)
        predicted_violence = [x for x in prediction if x == violent_label]
        predicted_non_violnece = [x for x in prediction if x == non_violent_label]
        percentage_of_violence = len(predicted_violence) / len(prediction)
        non_violent_predictions.append(percentage_of_violence)
        # print('Percentage of violence predicted = ', percentage_of_violence)

        actual_violence = [x for x in labels if x == violent_label]
        actual_non_violence = [x for x in labels if x == non_violent_label]
        # print('Percentage of violence actually = ', len(actual_violence) / len(labels))

        accuracy = classifier.score(features, labels)
        overall_accuracy += accuracy
        # print('Accuracy = ', accuracy)

        violent_predictions = []
    for instance in violent_test_set:
        features, labels = instance

        prediction = classifier.predict(features)
        predicted_violence = [x for x in prediction if x == violent_label]
        predicted_non_violnece = [x for x in prediction if x == non_violent_label]
        percentage_of_violence = len(predicted_violence) / len(prediction)
        violent_predictions.append(percentage_of_violence)        
        # print('Percentage of violence predicted = ', percentage_of_violence)

        actual_violence = [x for x in labels if x == violent_label]
        actual_non_violence = [x for x in labels if x == non_violent_label]
        # print('Percentage of violence actually = ', len(actual_violence) / len(labels))

        accuracy = classifier.score(features, labels)
        overall_accuracy += accuracy
        # print('Accuracy = ', accuracy)
    
    accuracy_per_frame = overall_accuracy / (len(non_violent_test_set) + len(violent_test_set))
    print('Accuracy per frame = ', accuracy_per_frame)
   
    violence_percentage = 0.05
    # print('Start Test')
    for iteration in range(0, 15):
        score = 0

        false_positive = 0
        false_negative = 0
        true_positive = 0
        true_negative = 0

        print('Violence Threshold = ', violence_percentage)
        # print('Starting Test on Non-Violent Set...')
        
        for percentage_of_violence in non_violent_predictions:
            if percentage_of_violence < violence_percentage:
                score += 1
                true_negative += 1
            else:
                false_positive += 1

        # print('Starting Test on Violent Set...')
        for percentage_of_violence in violent_predictions:
            if percentage_of_violence > violence_percentage:
                score += 1
                true_positive += 1
            else:
                false_negative += 1
        
        violence_percentage += 0.05

        accuracy_per_video = score / (len(non_violent_test_set) + len(violent_test_set))
        

        accuracy_per_video_array.append(accuracy_per_video)
        print('Accuracy per video = ', accuracy_per_video)
        

        # print('Confusion matrix: ')
        # print('True positive ', true_positive)
        # print('True negative ', true_negative)
        # print('False positive ', false_positive)
        # print('False negative ', false_negative)

        result_array = np.array([true_positive, true_negative, false_positive, false_negative, accuracy_per_video, accuracy_per_frame])
        result_matrix[iteration] += result_array

        

    # filename = 'current_model.pkl'
    # _ = joblib.dump(mlp, filename, compress=9)


    return np.max(accuracy_per_video_array), accuracy_per_frame, result_matrix
    
