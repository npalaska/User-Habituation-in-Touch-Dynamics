import numpy as np
from numpy import *
import os
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_curve
from sklearn import metrics
from sklearn import preprocessing
from sklearn.cross_validation import ShuffleSplit


currentdirectory = os.getcwd()  # get the directory.
parentdirectory = os.path.abspath(
    currentdirectory + "/../..")  # Get the parent directory(2 levels up)
path = parentdirectory + '/Output Files/Experiment101/sameTrainTest/Random ' \
                         'Forest-8-9-10-11thTrain-12split/ROC Characteristics Random Forest'
if not os.path.exists(path):
    os.makedirs(path)  # Crate the directory if one does not exist.

arr_recall = arange(2232, dtype=float32).reshape(2232, 1)
arr_auc = arange(2232, dtype=float32).reshape(2232, 1)

fold = 1
user = 1
block = 1
i = 0
while user <= 31:
    while block <= 12:
        while fold <= 6:
            # We will keep the block = 1 constant in the training data file because we are training the classifier on the first block only and testing it on the remaining blocks
            """
            traindata1 = genfromtxt(
                '../../Output Files/Experiment100/GIUser/User' + str(
                    user) + '/1-' + str(user) + '-1-2-' + str(fold) + '.csv',
                dtype=float, delimiter=',')
                """
            traindata2 = genfromtxt(
                '../../Output Files/Experiment100/12splits/GIUser/User' + str(
                    user) + '/1-' + str(user) + '-1-10-' + str(fold) + '.csv',
                dtype=float, delimiter=',')
            traindata3 = genfromtxt(
                '../../Output Files/Experiment100/12splits/GIUser/User' + str(
                    user) + '/1-' + str(user) + '-1-11-' + str(fold+1) + '.csv',
                dtype=float, delimiter=',')
            traindata4 = genfromtxt(
                '../../Output Files/Experiment100/12splits/GIUser/User' + str(
                    user) + '/1-' + str(user) + '-1-9-' + str(fold+2) + '.csv',
                dtype=float, delimiter=',')
            traindata5 = genfromtxt(
                '../../Output Files/Experiment100/12splits/GIUser/User' + str(
                    user) + '/1-' + str(user) + '-1-8-' + str(fold+3) + '.csv',
                dtype=float, delimiter=',')


            traindata = np.vstack((traindata5, traindata4, traindata2, traindata3))
            target_train = np.ones(len(traindata))
            # Generate target_train and target_test files with 1 indicating genuine and 0 indicating impostor
            row = 0
            while row < len(traindata):
                if np.any(traindata[row, 0:3] != [1, user, 1]):
                    target_train[row] = 0
                row += 1
            sample_train = traindata[:, 3:]
            scaler = preprocessing.StandardScaler().fit(sample_train)
            sample_train_scaled = scaler.transform(sample_train)

            if block not in [8, 9, 10, 11]:
                testdata = genfromtxt(
                    '../../Output Files/Experiment100/12splits/GIUser/User' + str(
                        user) + '/1-' + str(user) + '-1-' + str(block) + '-' + str(
                        fold+4) + '.csv', dtype=float, delimiter=',')

                target_test = np.ones(len(testdata))
                row = 0
                while row < len(testdata):
                    if np.any(testdata[row, 0:3] != [1, user, 1]):
                        target_test[row] = 0
                    row += 1

                sample_test = testdata[:, 3:]

            # scale the data on all the features.
                sample_test_scaled = scaler.transform(sample_test)

            """
            clf = ExtraTreesClassifier(n_estimators=100, n_jobs=1)
            clf.fit(sample_train, target_train)

            target_train_prediction = clf.predict(sample_train)
            target_test_prediction = clf.predict(sample_test)

            recall = metrics.recall_score(target_test, target_test_prediction)
            arr_recall[i, 0] = recall
            """

            cv = ShuffleSplit(len(traindata), 10, test_size=0.25, train_size=0.25)
            arr_auc_test = np.array([])
            arr_recall_test = np.array([])
            for j, (train , test) in enumerate(cv):
                X_train, X_test, y_train, y_test = sample_train_scaled[train], sample_train_scaled[test], target_train[
                train], target_train[test]

                clf = ExtraTreesClassifier(n_estimators=100, n_jobs=1)
                clf.fit(X_train, y_train)

                #target_train_prediction = clf.predict(X_test)
                #auc_train = metrics.roc_auc_score(y_test, target_train_prediction)

                #arr_auc_train = np.append(arr_auc_train, [auc_train], axis=0)
                if block in [8, 9, 10, 11]:
                    target_test_prediction = clf.predict(X_test)
                    recall_split = metrics.recall_score(y_test, target_test_prediction)
                    auc_test_split = metrics.roc_auc_score(y_test, target_test_prediction)
                else:
                    target_test_prediction = clf.predict(sample_test_scaled)

                    recall_split = metrics.recall_score(target_test, target_test_prediction)
                    auc_test_split = metrics.roc_auc_score(target_test, target_test_prediction)

                arr_recall_test = np.append(arr_recall_test, [recall_split], axis=0)
                arr_auc_test = np.append(arr_auc_test, [auc_test_split], axis=0)

            """
            # Get the auc value based on the actual target and the predicted target.
            auc_scaled = metrics.roc_auc_score(target_test,
                                               target_test_prediction)
            auc = metrics.roc_auc_score(target_test, target_test_prediction)
            arr_auc[i, 0] = auc_scaled
            arr_auc[i, 0] = auc
            """
            auc_scaled = np.mean(arr_auc_test)
            arr_auc[i, 0] = auc_scaled
            recall = np.mean(arr_recall_test)
            arr_recall[i, 0] = recall
            #print(arr_auc[i, 0])
            #print(arr_recall[i, 0])
            #print("==================")
            arr_auc_test = np.array([])
            arr_recall_test = np.array([])

            i += 1
            fold += 1
        fold = 1
        block += 1
    block = 1
    user += 1
    print("x")

# for the consecutive group of 5 values of arr_recall you need to get the average
# so for total 60 values you will get 6 averaged values
# so consecutively this 6 values are of different users


aucvalues = np.transpose(arr_auc.reshape(372, 6))
np.savetxt(
    "../../Output Files/Experiment101/sameTrainTest/Random "
    "Forest-8-9-10-11thTrain-12split/RandomForest_AUC.csv",
    aucvalues, delimiter=",")

recallvalues = np.transpose(arr_recall.reshape(372, 6))
np.savetxt(
    "../../Output Files/Experiment101/sameTrainTest/Random "
    "Forest-8-9-10-11thTrain-12split/RandomForest_Recall.csv",
    recallvalues, delimiter=",")

aucdata = genfromtxt(
    '../../Output Files/Experiment101/sameTrainTest/Random '
    'Forest-8-9-10-11thTrain-12split/RandomForest_AUC.csv',
    dtype=float, delimiter=',')
arr_auc = arange(372, dtype=float32).reshape(372, 1)

recalldata = genfromtxt(
    "../../Output Files/Experiment101/sameTrainTest/Random "
    "Forest-8-9-10-11thTrain-12split/RandomForest_Recall.csv",
    dtype=float, delimiter=",")
arr_recall = arange(372, dtype=float32).reshape(372, 1)

# find the mean of every 5 fold with 1st and 3rd block for every user so it will generate arrays of size 31*2.
i = 0
while i <= 371:
    arr_auc[i, 0] = np.mean(aucdata[:, i])
    arr_recall[i, 0] = np.mean(recalldata[:, i])
    i += 1

np.savetxt(
    "../../Output Files/Experiment101/sameTrainTest/Random "
    "Forest-8-9-10-11thTrain-12split/RandomForestScaled_MeanRecall.csv",
    np.transpose(arr_recall.reshape(31, 12)), delimiter=",")

np.savetxt(
    "../../Output Files/Experiment101/sameTrainTest/Random "
    "Forest-8-9-10-11thTrain-12split/RandomForest_MeanAUC.csv",
    np.transpose(arr_auc.reshape(31, 12)), delimiter=",")
