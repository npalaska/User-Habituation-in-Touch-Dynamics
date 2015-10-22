import numpy as np
from numpy import *
import os
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import svm
from sklearn import metrics

currentdirectory = os.getcwd()  # get the directory.
parentdirectory = os.path.abspath(
    currentdirectory + "/../..")  # Get the parent directory(2 levels up)
path = parentdirectory + '\Output Files\One Class SVM\ROC Characteristics Random Forest'
if not os.path.exists(path):
    os.makedirs(path)

traindata = genfromtxt('../../Output Files/Experiment73/Genuine User/GenuineUser1/1-1-1-G-1.csv', dtype=float,
                       delimiter=',')
testdata = genfromtxt('../../Output Files/Experiment73/Genuine User/GenuineUser1/1-1-1-G-9.csv', dtype=float,
                      delimiter=',')

sample_train = traindata[:, 3:]
sample_test = testdata[:, 3:]

target_test = np.ones(len(testdata))
user = 1
row = 0
while row < len(testdata):
    if np.any(testdata[row, 0:3] != [1, user, 1]):
        target_test[row] = 0
    row += 1

# scale the data on all the features.
scaler = preprocessing.MinMaxScaler().fit(sample_train)
#scaler = preprocessing.StandardScaler().fit(sample_train)
sample_train_scaled = scaler.transform(sample_train)
sample_test_scaled = scaler.transform(sample_test)

clf = svm.OneClassSVM(nu=0.01, kernel="linear", gamma=0.01)
clf.fit(sample_train_scaled)

y_pred_train = clf.predict(sample_train_scaled)
y_pred_test = clf.predict(sample_test_scaled)

np.savetxt('../../Output Files/One Class SVM/train_pred_1-2.csv', y_pred_train, delimiter=',')
np.savetxt('../../Output Files/One Class SVM/test_pred_1-2.csv', y_pred_test, delimiter=',')

row = 0
while row < len(y_pred_train):
    if y_pred_train[row] == -1:
        y_pred_train[row] = 0
    row += 1

row = 0
while row < len(y_pred_test):
    if y_pred_test[row] == -1:
        y_pred_test[row] = 0
    row += 1

fpr1, tpr1, thresholds = metrics.roc_curve(target_test, y_pred_test)
print(tpr1)

#auc = metrics.roc_auc_score(target_train, y_pred_train)
#print(auc)