import numpy as np
from numpy import *
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_curve
from sklearn import metrics
from sklearn import preprocessing
from sklearn import cross_validation

a = np.arange(10).reshape((5, 2))
b = [1, 1, 1, 0, 0]
a_train, a_test, b_train, b_test = cross_validation.train_test_split \
    (a, b, test_size=0.33, random_state=42)
print(a_train, a_test)
print(b_train, b_test)
"""
traindata = genfromtxt(
    '../Output Files/Experiment100/GIUser/User2/1-2-1-6-3.csv', dtype=float,
    delimiter=',')
testdata = genfromtxt(
    '../Output Files/Experiment100/GIUser/User2/1-2-1-5-1.csv', dtype=float,
    delimiter=',')

target_train = np.ones(len(traindata))
target_test = np.ones(len(testdata))
row = 0
while row < len(traindata):
    if np.any(traindata[row, 0:3] != [1, 2, 1]):
        target_train[row] = 0
    row += 1

row = 0
while row < len(testdata):
    if np.any(testdata[row, 0:3] != [1, 2, 1]):
        target_test[row] = 0
    row += 1

sample_train = traindata[:, 3:]
sample_test = testdata[:, 3:]

scaler = preprocessing.StandardScaler().fit(sample_train)
sample_train_scaled = scaler.transform(sample_train)
sample_test_scaled = scaler.transform(sample_test)

clf = ExtraTreesClassifier(n_estimators=100)
# clf.fit(sample_train_scaled, target_train)

scores = cross_validation.cross_val_score(clf, sample_test_scaled, target_test,
                                          cv=5, scoring='roc_auc')
print(mean(scores))

clf.fit(sample_train_scaled, target_train)
target_test_prediction = clf.predict(sample_test_scaled)
print(metrics.roc_auc_score(target_test, target_test_prediction))
target_train_prediction = clf.predict(sample_train_scaled)

print(metrics.zero_one_loss(target_test, target_test_prediction))
print(metrics.zero_one_loss(target_train, target_train_prediction))

clf = AdaBoostClassifier(n_estimators=100)
clf.fit(sample_train_scaled, target_train)
target_test_prediction = clf.predict(sample_test_scaled)
print(metrics.roc_auc_score(target_test, target_test_prediction))
"""


def Make_Split(filename, totalsplits):
    arr_data = genfromtxt(filename, dtype=float, delimiter=',')
    lengthofarray = len(arr_data)

    lengthoffile = (lengthofarray - (lengthofarray % totalsplits)) / totalsplits
    fileno = 1
    while fileno <= totalsplits:
        x = arr_data[(fileno - 1) * lengthoffile: fileno * lengthoffile]




