import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
import constants
import data_load
from sklearn.metrics import confusion_matrix
import five_fold

def artificial_neural_network():

    non_violent_data, non_violent_labels, non_violent_names = data_load.load_average_non_violent_data('E:/Data/Violent Crowd/Crowd_violence_dataset/NonViolent_features_improved/', label=0)
    violent_data, violent_labels, violent_names = data_load.load_average_violent_data('E:/Data/Violent Crowd/Crowd_violence_dataset/Violent_features_improved/', label=1)

    # non_violent_angles, dummy_label = data_load.load_non_violent_angles('E:/Data/Violent Crowd/Crowd_violence_dataset/Non_violent_angle/', label=0)
    # violent_angles, dummy_label = data_load.load_violent_angles('E:/Data/Violent Crowd/Crowd_violence_dataset/Violent_angle/', label=1)

    # non_violent_data = np.hstack((non_violent_data, non_violent_angles))
    # violent_data = np.hstack((violent_data, violent_angles))

    data = np.vstack((non_violent_data, violent_data))

    sounak_descriptor = data_load.load_sounak_descriptor()

    data = np.hstack((data, sounak_descriptor))

    # data = np.vstack((non_violent_data, violent_data))
    labels = np.hstack((non_violent_labels, violent_labels))
    names = non_violent_names + violent_names

    # sounak_descriptor = data_load.load_sounak_descriptor()

    # data = np.hstack((data, sounak_descriptor))

    # indices = five_fold.k_fold(5, data, labels)

    train_data, train_labels, test_data, test_labels, train_names, test_names = data_load.shuffle_and_split(data, labels, names, ratio=0.2)
    print(train_data.shape, test_data.shape)

    # train_size = 190
    # train_data, train_labels, test_data, test_labels = data[:train_size], labels[:train_size], data[train_size:], labels[train_size:]
    # data = np.vstack((train_data, test_data))
    # labels = np.hstack((train_labels, test_labels))

    accuracy = 0
    c_matrix = np.zeros(shape=(2, 2))

    mlp = MLPClassifier(hidden_layer_sizes=(500, 200, 100, 50), activation='tanh', solver='adam', alpha=1e-5, random_state=0)

    mlp.fit(train_data, train_labels)
    print('Model Accuracy ', mlp.score(train_data, train_labels))
    acc = mlp.score(test_data, test_labels)
    accuracy += acc
    print('Test Accuracy ', acc)

    test_prediction = mlp.predict(test_data)
    train_prediction = mlp.predict(train_data)
    # print('Confusion matrix')
    c_mat_in_this_iteration = confusion_matrix(test_labels, test_prediction)
    # c_matrix += c_mat_in_this_iteration
    print(c_mat_in_this_iteration)

    train_incorrect = [index for index, p in enumerate(zip(train_prediction, train_labels)) if p[0] != p[1]]
    test_incorrect = [index for index, p in enumerate(zip(test_prediction, test_labels)) if p[0] != p[1]]

    names_train_incorrect = [train_names[i] for i in train_incorrect]
    names_test_incorrect = [test_names[i] for i in test_incorrect]    

    print(names_train_incorrect)
    print(names_test_incorrect)

    # _ = joblib.dump(mlp, 'averahe_nn.pkl', compress=9)
    

    # for train, test in indices:
    #     train_data = data[train]
    #     train_labels = labels[train]

    #     test_data = data[test]
    #     test_labels = labels[test]

    #     print(train_data.shape, test_data.shape)

    #     mlp = MLPClassifier(hidden_layer_sizes=(500, 200, 100, 50), activation='tanh', solver='adam', alpha=1e-5, random_state=0)

    #     mlp.fit(train_data, train_labels)
    #     print('Model Accuracy ', mlp.score(train_data, train_labels))
    #     acc = mlp.score(test_data, test_labels)
    #     accuracy += acc
    #     print('Test Accuracy ', acc)

    #     prediction = mlp.predict(test_data)
    #     # print('Confusion matrix')
    #     c_mat_in_this_iteration = confusion_matrix(test_labels, prediction)
    #     c_matrix += c_mat_in_this_iteration
    
    # print('Overall accuracy ', accuracy / 5)
    # print('Confusion Matrix ')
    # print(c_matrix / 5)
  


if __name__ == '__main__':

    artificial_neural_network()



