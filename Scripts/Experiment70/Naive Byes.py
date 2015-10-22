"""This will return ROC characteristics based on the decision tree model and
    Area Under the Curve for scaled data and without scaled data
    for all the Users with 10 folds and 1st and 3rd block of data."""

import numpy as np
from numpy import *
import os
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn import preprocessing

# Create the Folder 'ROC Characteristics Naive Bayes' to save the roc characteristic file
# for every User, Block and Fold combination.

currentdirectory = os.getcwd()  # get the directory.
parentdirectory = os.path.abspath(
    currentdirectory + "/../..")  # Get the parent directory(2 levels up)
path = parentdirectory + '/Output Files/Experiment70/Naive Bayes/ROC Characteristics Naive Bayes'
if not os.path.exists(path):
    os.makedirs(path)  # Crate the directory if one does not exist.

# arr_auc is the temporary arrays to store auc values for real data and the scaled data respectively.
arr_eer = arange(930, dtype=float32).reshape(930, 1)
arr_auc = arange(930, dtype=float32).reshape(930, 1)
fold = 1
user = 1
block = 1
i = 0
while user <= 31:
    while block <= 3:
        while fold <= 10:
            # get the training data and testing data as an array.
            traindata = genfromtxt(
                '../../Output Files/Experiment60/GIUser' + str(
                    user) + '/1-' + str(user) + '-1-Train-' + str(
                    block) + '-' + str(fold) + '.csv', dtype=float,
                delimiter=',')
            testdata = genfromtxt(
                '../../Output Files/Experiment60/GIUser' + str(
                    user) + '/1-' + str(user) + '-1-Test-' + str(
                    block) + '-' + str(fold) + '.csv', dtype=float,
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
            sample_train = scaler.transform(sample_train)
            sample_test = scaler.transform(sample_test)

            clf = GaussianNB()  # Using a gaussian Naive Bayes
            clf.fit(sample_train, target_train)

            # predict the target class based on our classifier
            target_train_prediction = clf.predict(sample_train)
            target_test_prediction = clf.predict(sample_test)

            # Confusion matrix will be stored in variable confusion.
            confusion = metrics.confusion_matrix(target_test,
                                                 target_test_prediction)
            # print(metrics.roc_auc_score(target_test, y_test_prediction))
            fpr, tpr, thresholds = metrics.roc_curve(target_test_prediction,
                                                     score, pos_label=0)

            row = 0
            fpr_tpr = np.arange(len(fpr),
                                dtype=float32)  # A temporary array to hold the value of (1-fpr-tpr)
            while row <= len(fpr) - 1:
                x = fpr[row] + tpr[row]
                fpr_tpr[row] = round((1 - x), 2)
                row += 1

            # declaring the variable fprtprrelation to hold the value of fpr, tpr, fpr_tpr, threshold in the table.
            fprtprrelation = np.vstack((fpr, tpr, fpr_tpr, thresholds))
            # save fprtprrelation table for every combination of training and testing file.
            np.savetxt(
                "../../Output Files/Experiment70/Naive Bayes/ROC Characteristics Naive Bayes/1-" + str(
                    user) + "-1-" + str(block) + "-" + str(
                    fold) + "-FPRTPR.csv", np.transpose(fprtprrelation),
                delimiter=",")

            eer = np.array([])
            eerdata = genfromtxt(
                '../../Output Files/Experiment70/Naive Bayes/ROC Characteristics Naive Bayes/1-' + str(
                    user) + '-1-' + str(block) + '-' + str(
                    fold) + '-FPRTPR.csv', dtype=float, delimiter=',')
            row = 0
            while row <= len(eerdata) - 1:
                if eerdata[row, 2] == 0:
                    eerr = np.append(eer, [eerdata[row, 0]], axis=0)
                row += 1
            meaneer = mean(eerr)
            arr_eer[i, 0] = meaneer

            auc = metrics.roc_auc_score(target_test, target_test_prediction)
            arr_auc[i, 0] = auc
            fold += 1
            i += 1
        fold = 1
        block += 1
    block = 1
    user += 1

eervalues = np.transpose(arr_eer.reshape(93, 10))
np.savetxt(
    "../../Output Files/Experiment70/Naive Bayes/NaiveBayesScaled_EER.csv",
    eervalues, delimiter=",")

aucvalues = np.transpose(arr_auc.reshape(93, 10))
np.savetxt(
    "../../Output Files/Experiment70/Naive Bayes/NaiveBayesScaled_AUC.csv",
    aucvalues, delimiter=",")

eerpreprocessdata = genfromtxt(
    '../../Output Files/Experiment70/Naive Bayes/NaiveBayesScaled_EER.csv',
    dtype=float, delimiter=',')
arr_eer = arange(93, dtype=float32).reshape(93, 1)

aucdata = genfromtxt(
    '../../Output Files/Experiment70/Naive Bayes/NaiveBayesScaled_AUC.csv',
    dtype=float, delimiter=',')
arr_auc = arange(93, dtype=float32).reshape(93, 1)

i = 0
while i <= 92:
    arr_eer[i, 0] = np.mean(eerpreprocessdata[:, i])
    arr_auc[i, 0] = np.mean(aucdata[:, i])
    i += 1

np.savetxt(
    "../../Output Files/Experiment70/Naive Bayes/NaiveBayesScaled_MeanEER.csv",
    arr_eer.reshape(31, 3), delimiter=",")
np.savetxt(
    "../../Output Files/Experiment70/Naive Bayes/NaiveBayesScaled_MeanAUC.csv",
    arr_auc.reshape(31, 3), delimiter=",")
