from sklearn import svm
from sklearn.externals import joblib
import numpy as np

#Accuracy 71

train_labels = []
train_inps = []
for line in open('training_normalized.dat', 'r'):
    full_line = line[:-2].split(' ')
    lab = full_line.pop(0)
    train_labels.append(float(lab))
    temp_inp = np.array([float(x) for x in full_line])
    train_inps.append(temp_inp)

test_labels = []
test_inps = []
for line in open('testing_normalized.dat', 'r'):
    full_line = line[:-2].split(' ')
    lab = full_line.pop(0)
    test_labels.append(float(lab))
    temp_inp = np.array([float(x) for x in full_line])
    test_inps.append(temp_inp)

# print(len(train_inps[len(train_inps) - 1]),len(train_labels))
clf = svm.SVC(kernel='rbf',gamma=0.3,probability=True)
clf.fit(train_inps,train_labels)
joblib.dump(clf,'modelSVM.pkl')
answers = clf.predict(test_inps)
print(answers)

non_zero = np.count_nonzero(np.fabs(np.array(answers)-np.array(test_labels)))/len(answers)
print(1 - non_zero)
