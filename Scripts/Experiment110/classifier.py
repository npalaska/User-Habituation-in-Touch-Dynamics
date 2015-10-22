import numpy as np
from numpy import *
import os
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_curve
from sklearn import metrics
from sklearn import preprocessing

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif
from sklearn.svm import LinearSVC

currentdirectory = os.getcwd()  # get the directory.
parentdirectory = os.path.abspath(
    currentdirectory + "/../..")  # Get the parent directory(2 levels up)
path = parentdirectory + '\Output Files\Experiment110\Random Forest\ROC Characteristics Random Forest'
if not os.path.exists(path):
    os.makedirs(path)

# _Scaled indicate the data has been scaled.
# arr_auc, arr_auc_scaled are the temporary arrays to store auc values for real data and the scaled data respectively.
arr_eer = arange(930, dtype=float32).reshape(930, 1)
arr_auc_scaled = arange(930, dtype=float32).reshape(930, 1)

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
            sample_train = traindata[:, [4, 8, 9, 11, 14, 15, 16, 17, 18, 19]]
            x = 0
            """
            for row in sample_train:
                while x < sample_train.size/len(sample_train):
                    if row[x]*(-1)>0:
                        row[x] = row[x]*(-1)
                    x += 1
                x = 0
            """

            sample_test = testdata[:, [4, 8, 9, 11, 14, 15, 16, 17, 18, 19]]
            # sample_test = testdata[:, [3, 4, 8, 9, 14, 19]]
            # sample_test = testdata[:, [3, 4, 5, 6, 7, 8, 9, 13, 14, 19]]
            # sample_test = testdata[:, 3:]
            """
            x = 0
            for row in sample_test:
                while x < sample_test.size/len(sample_test):
                    if row[x]*(-1)>0:
                        row[x] = row[x]*(-1)
                    x += 1
                x = 0
            """
            score = np.arange(0, 1, 1 / len(target_test))

            # scale the data on all the features.
            #scaler = preprocessing.StandardScaler().fit(sample_train)
            #sample_train_scaled = scaler.transform(sample_train)
            sample_train_scaled = preprocessing.normalize(sample_train,
                                                          norm='l1')
            #sample_test_scaled = scaler.transform(sample_test)
            sample_test_scaled = preprocessing.normalize(sample_test,
                                                         norm='l1')

            #clf_scaled = ExtraTreesClassifier(n_estimators=100)  # Classifier with data scaled
            #X_sample_train = SelectKBest(f_classif, k=10).fit_transform(sample_train_scaled, target_train)
            #X_sample_test = SelectKBest(f_classif, k=10).fit_transform(sample_test_scaled, target_test)
            clf_scaled = ExtraTreesClassifier(n_estimators=100)
            clf_scaled.fit(sample_train_scaled, target_train).transform(
                sample_train_scaled)

            # Predict the target values with the classifier for training and testing sets.
            target_train_prediction_scaled = clf_scaled.predict(
                sample_train_scaled)
            target_test_prediction_scaled = clf_scaled.predict(
                sample_test_scaled)

            # get the confusion matrix and ROC characteristic for both scaled and without scaled data.
            #confusion_scaled = metrics.confusion_matrix(target_test, target_test_prediction_scaled)

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
                "../../Output Files/Experiment110/Random Forest/ROC Characteristics Random Forest/1-" + str(
                    user) + "-1-" + str(block) + "-" + str(
                    fold) + "-FPRTPR.csv", np.transpose(fprtprrelation),
                delimiter=",")

            eer = np.array([])
            eerdata = genfromtxt(
                '../../Output Files/Experiment110/Random Forest/ROC Characteristics Random Forest/1-' + str(
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
            arr_auc_scaled[i, 0] = auc_scaled
            fold += 1
            i += 1
        fold = 1
        block += 1
    block = 1
    user += 1

######################################################################################

eervalues = np.transpose(arr_eer.reshape(93, 10))
np.savetxt(
    "../../Output Files/Experiment110/Random Forest/RandomForestScaled_EER.csv",
    eervalues, delimiter=",")

aucvalues = np.transpose(arr_auc_scaled.reshape(93, 10))
np.savetxt(
    "../../Output Files/Experiment110/Random Forest/RandomForestScaled_Auc.csv",
    aucvalues, delimiter=",")

eerpreprocessdata = genfromtxt(
    '../../Output Files/Experiment110/Random Forest/RandomForestScaled_EER.csv',
    dtype=float,
    delimiter=',')
arr_eer = arange(93, dtype=float32).reshape(93, 1)

aucscaleddata = genfromtxt(
    '../../Output Files/Experiment110/Random Forest/RandomForestScaled_Auc.csv',
    dtype=float,
    delimiter=',')
arr_auc_scaled = arange(93, dtype=float32).reshape(93, 1)

# find the mean of every 10 fold with 1st and 3rd block for every user so it will generate arrays of size 31*2.
i = 0
while i <= 92:
    arr_eer[i, 0] = np.mean(eerpreprocessdata[:, i])
    arr_auc_scaled[i, 0] = np.mean(aucscaleddata[:, i])
    i += 1

np.savetxt(
    "../../Output Files/Experiment110/Random Forest/RandomForestScaled_MeanEER.csv",
    arr_eer.reshape(31, 3),
    delimiter=",")
np.savetxt(
    "../../Output Files/Experiment110/Random Forest/RandomForestMeanScaled_Auc.csv",
    arr_auc_scaled.reshape(31, 3), delimiter=",")