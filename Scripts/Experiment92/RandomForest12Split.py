"""This will return ROC characteristics based on the decision tree model and
    Area Under the Curve for scaled data and without scaled data
    for all the Users with 10 folds and 1st and 3rd block of data."""
import numpy as np
from numpy import *
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.metrics import roc_curve
from sklearn import metrics
from sklearn import preprocessing

# Create the Folder 'ROC Characteristics RandomForest' to save the roc characteristic file
# for every User, Block and Fold combination.

currentdirectory = os.getcwd()  # get the directory.
parentdirectory = os.path.abspath(
    currentdirectory + "/../..")  # Get the parent directory(2 levels up)
path = parentdirectory + '\Output Files\Experiment92\Random Forest-12Split\ROC Characteristics Random Forest'
if not os.path.exists(path):
    os.makedirs(path)  # Crate the directory if one does not exist.

# _Scaled indicate the data has been scaled.
# arr_auc, arr_auc_scaled are the temporary arrays to store auc values for real data and the scaled data respectively.
arr_auc_scaled = arange(3720, dtype=float32).reshape(3720, 1)
arr_auc = arange(3720, dtype=float32).reshape(3720, 1)
arr_eer = arange(3720, dtype=float32).reshape(3720, 1)

fold = 1
user = 1
block = 1
i = 0
while user <= 31:
    while block <= 12:
        while fold <= 10:
            # get the training data and testing data as an array.
            traindata = genfromtxt(
                '../../Output Files/Experiment73/GIUser/User' + str(
                    user) + '/1-' + str(user) + '-1-Train-' + str(
                    block) + '-' + str(fold) + '.csv', dtype=float,
                delimiter=',')
            testdata = genfromtxt(
                '../../Output Files/Experiment73/GIUser/User' + str(
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
            sample_train = traindata[:, [3, 4, 8, 9, 14, 19]]
            # sample_train = traindata[:, 3:]
            sample_test = testdata[:, [3, 4, 8, 9, 14, 19]]
            # sample_test = testdata[:, 3:]
            score = np.arange(0, 1, 1 / len(target_test))

            # scale the data on all the features.
            scaler = preprocessing.StandardScaler().fit(sample_train)
            sample_train_scaled = scaler.transform(sample_train)
            sample_test_scaled = scaler.transform(sample_test)

            clf_scaled = ExtraTreesClassifier(
                n_estimators=100)  # Classifier with data scaled
            clf_scaled.fit(sample_train_scaled, target_train)
            clf = ExtraTreesClassifier(
                n_estimators=100)  # Classifier with actual data (without scaling)
            clf.fit(sample_train, target_train)

            # Predict the target values with the classifier for training and testing sets.
            target_train_prediction_scaled = clf_scaled.predict(
                sample_train_scaled)
            target_test_prediction_scaled = clf_scaled.predict(
                sample_test_scaled)
            target_train_prediction = clf.predict(sample_train)
            target_test_prediction = clf.predict(sample_test)

            # get the confusion matrix and ROC characteristic for both scaled and without scaled data.
            confusion_scaled = metrics.confusion_matrix(target_test,
                                                        target_test_prediction_scaled)
            confusion = metrics.confusion_matrix(target_test,
                                                 target_test_prediction)
            fpr, tpr, thresholds = roc_curve(target_test_prediction_scaled,
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
                "../../Output Files/Experiment92/Random Forest-12Split/ROC Characteristics Random Forest/1-" + str(
                    user) + "-1-" + str(block) + "-" + str(
                    fold) + "-FPRTPR.csv", np.transpose(fprtprrelation),
                delimiter=",")

            eer = np.array([])
            eerdata = genfromtxt(
                '../../Output Files/Experiment92/Random Forest-12Split/ROC Characteristics Random Forest/1-' + str(
                    user) + '-1-' + str(block) + '-' + str(
                    fold) + '-FPRTPR.csv', dtype=float, delimiter=',')
            row = 0
            while row <= len(eerdata) - 1:
                if eerdata[row, 2] == 0:
                    eerr = np.append(eer, [eerdata[row, 0]], axis=0)
                row += 1
            meaneer = mean(eerr)
            arr_eer[i, 0] = meaneer

            # Get the auc value based on the actual target and the predicted target.
            auc_scaled = metrics.roc_auc_score(target_test,
                                               target_test_prediction_scaled)
            auc = metrics.roc_auc_score(target_test, target_test_prediction)
            arr_auc_scaled[i, 0] = auc_scaled
            arr_auc[i, 0] = auc
            fold += 1
            i += 1
        fold = 1
        block += 1
    block = 1
    user += 1

######################################################################################

eervalues = np.transpose(arr_eer.reshape(372, 10))
np.savetxt(
    "../../Output Files/Experiment92/Random Forest-12Split/RandomForestScaled_EER.csv",
    eervalues, delimiter=",")

aucvalues = np.transpose(arr_auc_scaled.reshape(372, 10))
np.savetxt(
    "../../Output Files/Experiment92/Random Forest-12Split/RandomForestScaled_AUC.csv",
    aucvalues, delimiter=",")
aucvalues = np.transpose(arr_auc.reshape(372, 10))
np.savetxt(
    "../../Output Files/Experiment92/Random Forest-12Split/RandomForest_AUC.csv",
    aucvalues, delimiter=",")

eerpreprocessdata = genfromtxt(
    '../../Output Files/Experiment92/Random Forest-12Split/RandomForestScaled_EER.csv',
    dtype=float, delimiter=',')
arr_eer = arange(372, dtype=float32).reshape(372, 1)

aucscaleddata = genfromtxt(
    '../../Output Files/Experiment92/Random Forest-12Split/RandomForestScaled_AUC.csv',
    dtype=float, delimiter=',')
arr_auc_scaled = arange(372, dtype=float32).reshape(372, 1)
aucdata = genfromtxt(
    '../../Output Files/Experiment92/Random Forest-12Split/RandomForest_AUC.csv',
    dtype=float, delimiter=',')
arr_auc = arange(372, dtype=float32).reshape(372, 1)

# find the mean of every 10 fold with 1st and 3rd block for every user so it will generate arrays of size 31*2.
i = 0
while i <= 371:
    arr_eer[i, 0] = np.mean(eerpreprocessdata[:, i])
    arr_auc[i, 0] = np.mean(aucdata[:, i])
    arr_auc_scaled[i, 0] = np.mean(aucscaleddata[:, i])
    i += 1

np.savetxt(
    "../../Output Files/Experiment92/Random Forest-12Split/RandomForestScaled_MeanEER.csv",
    arr_eer.reshape(31, 12), delimiter=",")
np.savetxt(
    "../../Output Files/Experiment92/Random Forest-12Split/RandomForestScaled_MeanAUC.csv",
    arr_auc_scaled.reshape(31, 12), delimiter=",")
np.savetxt(
    "../../Output Files/Experiment92/Random Forest-12Split/RandomForest_MeanAUC.csv",
    arr_auc.reshape(31, 12), delimiter=",")
