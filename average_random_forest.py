import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import constants
import data_load
from sklearn.metrics import confusion_matrix

def random_forest():

    non_violent_data, non_violent_labels = data_load.load_average_data('D:/Data/Violent Crowd/Crowd_violence_dataset/NonViolent_features_improved/', label=0)
    violent_data, violent_labels = data_load.load_average_data('D:/Data/Violent Crowd/Crowd_violence_dataset/Violent_features_improved/', label=1)

    data = np.vstack((non_violent_data, violent_data))
    labels = np.hstack((non_violent_labels, violent_labels))

    train_data, train_labels, test_data, test_labels = data_load.shuffle_and_split(data, labels, ratio=0.2)

    print(train_data.shape, test_data.shape)

    rnf = RandomForestClassifier(n_estimators=600, n_jobs=2)

    rnf.fit(train_data, train_labels)
    print('Model Accuracy ', rnf.score(train_data, train_labels))
    print('Test Accuracy ', rnf.score(test_data, test_labels))

    prediction = rnf.predict(test_data)
    print('Confusion matrix')
    print(confusion_matrix(test_labels, prediction))
  


if __name__ == '__main__':

    random_forest()



