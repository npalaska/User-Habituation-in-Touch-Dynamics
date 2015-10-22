import numpy as np
from numpy import *
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import cross_validation
from sklearn.metrics import roc_curve
from sklearn import metrics
from sklearn import preprocessing

arr = np.arange(186, dtype=float32).reshape(186, 1)
arr1 = np.transpose(arr.reshape(31, 6))
fpr = [1, 2, 3, 4]
tpr = [5, 6, 7, 8]
fpr_tpr = [9, 10, 11, 12]
relation = [13, 14, 15, 16]
block = 1
fold = 10
while block <= 6:
    while fold <= 10:
        traindata = genfromtxt('../Output Files/Experiment100/GIUser/User2/1-2-1-1-' + str(fold) + '.csv', dtype=float,
                               delimiter=',')
        testdata = genfromtxt('../Output Files/Experiment100/GIUser/User2/1-2-1-6-' + str(fold) + '.csv', dtype=float,
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
        clf.fit(sample_train_scaled, target_train)
        target_test_prediction = clf.predict(sample_test_scaled)
        # print(metrics.accuracy_score(target_test, target_test_prediction))
        print(metrics.roc_auc_score(target_test, target_test_prediction))
        #print(cross_validation.cross_val_score(clf, sample_train, target_train, cv=10))
        r2 = cross_validation.cross_val_score(clf, sample_test_scaled, target_test,
                                              cv=10)  #cv=cross_validation.KFold(target_test.size, 10))
        print(cross_validation.KFold(target_test.size, 10))
        #print(r2)
        print(mean(r2))
        print("-----------")
        fold += 1
    block += 1
    fold = 1
"""
target_train_prediction = clf.predict(sample_train)
target_test_prediction = clf.predict(sample_test)

recall = metrics.recall_score(target_test, target_test_prediction)
print(recall)
auc = metrics.roc_auc_score(target_test, target_test_prediction)
print(auc)
"""