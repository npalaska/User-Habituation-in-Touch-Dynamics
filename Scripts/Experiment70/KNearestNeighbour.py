import numpy as np
from numpy import *
import os
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn import preprocessing

currentdirectory = os.getcwd()  # get the directory.
parentdirectory = os.path.abspath(
    currentdirectory + "/../..")  # Get the parent directory(2 levels up)
path = parentdirectory + '\Output Files\Experiment70\KNN\ROC Characteristics KNN'
if not os.path.exists(path):
    os.makedirs(path)  # Crate the directory if one does not exist.

arr_eer = arange(930, dtype=float32).reshape(930, 1)
arr_auc = arange(930, dtype=float32).reshape(930, 1)
fold = 1
user = 1
block = 1
i = 0
while user <= 31:
    while block <= 3:
        while fold <= 10:
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

            X_train = traindata[:, 3:]
            X_test = testdata[:, 3:]

            scaler = preprocessing.StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

            clf = neighbors.KNeighborsClassifier(n_neighbors=10,
                                                 weights='uniform')

            clf.fit(X_train, target_train)

            target_train_prediction = clf.predict(X_train)
            target_test_prediction = clf.predict(X_test)
            score = np.arange(0, 1, 1 / len(target_test))
            fpr, tpr, thresholds = roc_curve(target_test_prediction, score,
                                             pos_label=0)

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
                "../../Output Files/Experiment70/KNN/ROC Characteristics KNN/1-" + str(
                    user) + "-1-" + str(block) + "-" + str(
                    fold) + "-FPRTPR.csv", np.transpose(fprtprrelation),
                delimiter=",")

            eer = np.array([])
            eerdata = genfromtxt(
                '../../Output Files/Experiment70/KNN/ROC Characteristics KNN/1-' + str(
                    user) + '-1-' + str(block) + '-' + str(
                    fold) + '-FPRTPR.csv', dtype=float, delimiter=',')
            row = 0
            while row <= len(eerdata) - 1:
                if eerdata[row, 2] == 0:
                    eerr = np.append(eer, [eerdata[row, 0]], axis=0)
                row += 1
            meaneer = mean(eerr)
            arr_eer[i, 0] = meaneer

            confusion = metrics.confusion_matrix(target_test,
                                                 target_test_prediction)

            auc = metrics.roc_auc_score(target_test, target_test_prediction)
            arr_auc[i, 0] = auc
            fold += 1
            i += 1
        fold = 1
        block += 1
    block = 1
    user += 1

#############################################################################

eervalues = np.transpose(arr_eer.reshape(93, 10))
np.savetxt("../../Output Files/Experiment70/KNN/KNNScaled_EER.csv", eervalues,
           delimiter=",")

aucvalues = np.transpose(arr_auc.reshape(93, 10))
np.savetxt("../../Output Files/Experiment70/KNN/KNNScaled_AUC.csv", aucvalues,
           delimiter=",")

eerpreprocessdata = genfromtxt(
    '../../Output Files/Experiment70/KNN/KNNScaled_EER.csv', dtype=float,
    delimiter=',')
arr_eer = arange(93, dtype=float32).reshape(93, 1)

aucdata = genfromtxt('../../Output Files/Experiment70/KNN/KNNScaled_AUC.csv',
                     dtype=float, delimiter=',')
arr_auc = arange(93, dtype=float32).reshape(93, 1)

i = 0
while i <= 92:
    arr_eer[i, 0] = np.mean(eerpreprocessdata[:, i])
    arr_auc[i, 0] = np.mean(aucdata[:, i])
    i += 1

np.savetxt("../../Output Files/Experiment70/KNN/KNNScaled_MeanEER.csv",
           arr_eer.reshape(31, 3), delimiter=",")
np.savetxt("../../Output Files/Experiment70/KNN/KNNScaled_MeanAUC.csv",
           arr_auc.reshape(31, 3), delimiter=",")