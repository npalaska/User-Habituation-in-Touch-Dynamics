import numpy as np
from numpy import *
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn import metrics
from sklearn import preprocessing

currentdirectory = os.getcwd()  # get the directory.
parentdirectory = os.path.abspath(
    currentdirectory + "/../..")  # Get the parent directory(2 levels up)
path = parentdirectory + '\Output Files\Test set group size 5 for evaluation\slide1/6-6'
if not os.path.exists(path):
    os.makedirs(path)

# _Scaled indicate the data has been scaled.
# arr_auc, arr_auc_scaled are the temporary arrays to store auc values for real data and the scaled data respectively.
arr_eer = arange(1953, dtype=float32).reshape(1953, 1)
arr_auc = arange(1953, dtype=float32).reshape(1953, 1)

fold = 1
user = 1
block = 6
i = 0
while user <= 31:
    while block <= 12:
        while fold <= 9:
            # get the training data and testing data as an array.

            traindata = genfromtxt(
                '../../Output Files/Experiment100/12splits/GIUser/User' + str(
                    user) + '/1-' + str(user) + '-1-6-' + str(fold) + '.csv', dtype=float,
                delimiter=',')


            # traindata = np.vstack((traindata1))

            testdata = genfromtxt(
                '../../Output Files/Experiment100/12splits/GIUser/User' + str(
                    user) + '/1-' + str(user) + '-1-' + str(
                    block) + '-' + str(fold+1) + '.csv', dtype=float,
                delimiter=',')

            # Generate target_train and target_test files with 1 indicating genuine and 0 indicating imposture
            target_train = np.ones(len(traindata))
            target_test = np.ones(len(testdata))
            row = 0
            while row < len(traindata):
                if np.any(traindata[row, 0:3] != [1, user, 1]):
                    target_train[row] = 0
                row += 1

            row = 0
            while row < len(testdata):
                if np.any(testdata[row, 0:3] != [1, user, 1]):
                    target_test[row] = 0
                row += 1


            # get the sample training and testing data with all the features.
            sample_train = traindata[:, 3:]
            sample_test = testdata[:, 3:]
            score = np.arange(0, 1, 1 / len(target_test))

            # scale the data on all the features.
            scaler = preprocessing.StandardScaler().fit(sample_train)
            sample_train_scaled = scaler.transform(sample_train)
            sample_test_scaled = scaler.transform(sample_test)

            # clf_scaled = RandomForestClassifier(n_estimators=100)  # Classifier with data scaled
            clf_scaled = ExtraTreesClassifier(n_estimators=100)
            clf_scaled.fit(sample_train_scaled, target_train)
            clf = ExtraTreesClassifier(n_estimators=100)  # Classifier with actual data (without scaling)
            clf.fit(sample_train, target_train)

            # Predict the target values with the classifier for training and testing sets.
            target_train_prediction = clf_scaled.predict(sample_train_scaled)
            target_test_prediction = clf_scaled.predict(sample_test_scaled)

            # apply the group prediction strategy
            rowno = 0
            counter = 1
            rowcount = len(target_test_prediction)
            LastCounter = len(target_test_prediction)
            while rowno < len(target_test_prediction):
                G = 0
                I = 0
                while counter <= min(5, LastCounter):
                    if target_test_prediction[rowno] == 1:
                        G += 1
                    if target_test_prediction[rowno] == 0:
                        I += 1
                    counter += 1
                    rowno += 1
                    rowcount -= 1
                if G > I:
                    target_test_prediction[rowno-5:rowno] = np.ones(5)
                else:
                    target_test_prediction[rowno-5:rowno] = np.zeros(5)
                counter = 1
                LastCounter = rowcount
                rowno = rowno

            # get the confusion matrix and ROC characteristic for both scaled and without scaled data.
            confusion_metrics = metrics.confusion_matrix(target_test, target_test_prediction)
            fpr, tpr, thresholds = roc_curve(target_test_prediction, score, pos_label=0)

            row = 0
            fpr_tpr = np.arange(len(fpr), dtype=float32)  # A temporary array to hold the value of (1-fpr-tpr)
            while row <= len(fpr) - 1:
                x = fpr[row] + tpr[row]
                fpr_tpr[row] = 1 - x
                row += 1
            eerdata = np.transpose(np.vstack((fpr, tpr, fpr_tpr, thresholds)))
            # eer = np.array([])
            fpr_tpr = eerdata[:, 2]
            rowno = min(range(len(eerdata[:, 2])), key=lambda j:abs(fpr_tpr[j]-0))
            # eerr = np.append(eer, [eerdata[rowno, 0]], axis=0)
            meaneer = eerdata[rowno, 0]
            arr_eer[i, 0] = meaneer

            # Get the auc value based on the actual target and the predicted target.
            auc = metrics.roc_auc_score(target_test, target_test_prediction)
            arr_auc[i, 0] = auc

            fold += 1
            i += 1
        fold = 1
        block += 1
    block = 6
    print("#")
    user += 1


eervalues = np.transpose(arr_eer.reshape(217, 9))
np.savetxt(
    "../../Output Files/Test set group size 5 for evaluation/slide1/6-6/RandomForest_EER.csv",
    eervalues, delimiter=",")

aucvalues = np.transpose(arr_auc.reshape(217, 9))
np.savetxt(
    "../../Output Files/Test set group size 5 for evaluation/slide1/6-6/RandomForest_Auc.csv",
    aucvalues, delimiter=",")

eerdata = genfromtxt(
    '../../Output Files/Test set group size 5 for evaluation/slide1/6-6/RandomForest_EER.csv',
    dtype=float, delimiter=',')
arr_eer = arange(217, dtype=float32).reshape(217, 1)

aucdata = genfromtxt(
    '../../Output Files/Test set group size 5 for evaluation/slide1/6-6/RandomForest_Auc.csv',
    dtype=float, delimiter=',')
arr_auc = arange(217, dtype=float32).reshape(217, 1)

# find the mean of every 10 fold
i = 0
while i <= 216:
    arr_eer[i, 0] = np.mean(eerdata[:, i])
    arr_auc[i, 0] = np.mean(aucdata[:, i])
    i += 1

np.savetxt(
    "../../Output Files/Test set group size 5 for evaluation/slide1/6-6/RandomForest_MeanEER.csv",
    arr_eer.reshape(31, 7), delimiter=",")

np.savetxt(
    "../../Output Files/Test set group size 5 for evaluation/slide1/6-6/RandomForest_MeanAuc.csv",
    arr_auc.reshape(31, 7), delimiter=",")

