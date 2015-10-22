import numpy as np
from numpy import *
import os
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_curve
from sklearn import metrics
from sklearn.cross_validation import ShuffleSplit
from sklearn import preprocessing
def model1():
    currentdirectory = os.getcwd()  # get the directory.
    parentdirectory = os.path.abspath(
        currentdirectory + "/../..")  # Get the parent directory(2 levels up)
    path = parentdirectory + '\Output Files\Experiment103\Random ' \
                             'Forest-2ndTrain\ROC Characteristics Random Forest'
    if not os.path.exists(path):
        os.makedirs(path)  # Crate the directory if one does not exist.

    arr_recall = arange(930, dtype=float32).reshape(930, 1)
    arr_auc = arange(930, dtype=float32).reshape(930, 1)
    arr_eer = arange(930, dtype=float32).reshape(930, 1)

    fold = 1
    user = 1
    block = 1
    i = 0
    while user <= 31:
        while block <= 3:
            while fold <= 10:
                # We will keep the block = 1 constant in the training data file because we are training the classifier on the first block only and testing it on the remaining blocks
                traindata = genfromtxt(
                    '../../Output Files/Experiment102/GIUser/User' + str(
                        user) + '/1-' + str(user) + '-1-2-' + str(fold) + '.csv',
                    dtype=float, delimiter=',')
                testdata = genfromtxt(
                    '../../Output Files/Experiment102/GIUser/User' + str(
                        user) + '/1-' + str(user) + '-1-' + str(block) + '-' + str(
                        fold) + '.csv', dtype=float, delimiter=',')

                # Generate target_train and target_test files with 1 indicating genuine and 0 indicating impostor
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

                sample_train = traindata[:, 3:]
                sample_test = testdata[:, 3:]
                score = np.arange(0, 1, 1 / len(target_test))

                # scale the data on all the features.
                scaler = preprocessing.StandardScaler().fit(sample_train)
                sample_train_scaled = scaler.transform(sample_train)
                sample_test_scaled = scaler.transform(sample_test)

                clf = ExtraTreesClassifier(n_estimators=100)
                clf.fit(sample_train, target_train)

                target_train_prediction = clf.predict(sample_train)
                target_test_prediction = clf.predict(sample_test)

                recall = metrics.recall_score(target_test, target_test_prediction)
                arr_recall[i, 0] = recall
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
                    "../../Output Files/Experiment103/Random Forest-2ndTrain/ROC "
                    "Characteristics Random Forest/1-" + str(
                        user) + "-1-" + str(block) + "-" + str(
                        fold) + "-FPRTPR.csv", np.transpose(fprtprrelation),
                    delimiter=",")

                eer = np.array([])
                eerdata = genfromtxt(
                    '../../Output Files/Experiment103/Random Forest-2ndTrain/ROC '
                    'Characteristics Random Forest/1-' + str(
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
                                                   target_test_prediction)
                auc = metrics.roc_auc_score(target_test, target_test_prediction)
                arr_auc[i, 0] = auc_scaled
                arr_auc[i, 0] = auc
                fold += 1
                i += 1
            fold = 1
            block += 1
        block = 1
        user += 1

    # for the consecutive group of 10 values of arr_recall you need to get the average
    # so for total 60 values you will get 6 averaged values
    # so consecutively this 6 values are of different users


    eervalues = np.transpose(arr_eer.reshape(93, 10))
    np.savetxt(
        "../../Output Files/Experiment103/Random "
        "Forest-2ndTrain/RandomForestScaled_EER.csv",
        eervalues, delimiter=",")

    aucvalues = np.transpose(arr_auc.reshape(93, 10))
    np.savetxt(
        "../../Output Files/Experiment103/Random "
        "Forest-2ndTrain/RandomForest_AUC.csv",
        aucvalues, delimiter=",")

    recallvalues = np.transpose(arr_recall.reshape(93, 10))
    np.savetxt(
        "../../Output Files/Experiment103/Random "
        "Forest-2ndTrain/RandomForest_Recall.csv",
        recallvalues, delimiter=",")

    eerpreprocessdata = genfromtxt(
        '../../Output Files/Experiment103/Random '
        'Forest-2ndTrain/RandomForestScaled_EER.csv',
        dtype=float, delimiter=',')
    arr_eer = arange(93, dtype=float32).reshape(93, 1)

    aucdata = genfromtxt(
        '../../Output Files/Experiment103/Random '
        'Forest-2ndTrain/RandomForest_AUC.csv',
        dtype=float, delimiter=',')
    arr_auc = arange(93, dtype=float32).reshape(93, 1)

    recalldata = genfromtxt(
        "../../Output Files/Experiment103/Random "
        "Forest-2ndTrain/RandomForest_Recall.csv",
        dtype=float, delimiter=",")
    arr_recall = arange(93, dtype=float32).reshape(93, 1)

    # find the mean of every 10 fold with 1st and 3rd block for every user so it will generate arrays of size 31*2.
    i = 0
    while i <= 92:
        arr_eer[i, 0] = np.mean(eerpreprocessdata[:, i])
        arr_auc[i, 0] = np.mean(aucdata[:, i])
        arr_recall[i, 0] = np.mean(recalldata[:, i])
        i += 1

    np.savetxt(
        "../../Output Files/Experiment103/Random "
        "Forest-2ndTrain/RandomForestScaled_MeanRecall.csv",
        np.transpose(arr_recall.reshape(31, 3)), delimiter=",")
    np.savetxt(
        "../../Output Files/Experiment103/Random "
        "Forest-2ndTrain/RandomForestScaled_MeanEER.csv",
        np.transpose(arr_eer.reshape(31, 3)), delimiter=",")
    np.savetxt(
        "../../Output Files/Experiment103/Random "
        "Forest-2ndTrain/RandomForest_MeanAUC.csv",
        np.transpose(arr_auc.reshape(31, 3)), delimiter=",")


def model2():
    currentdirectory = os.getcwd()  # get the directory.
    parentdirectory = os.path.abspath(
        currentdirectory + "/../..")  # Get the parent directory(2 levels up)
    path = parentdirectory + '\Output Files\Experiment103\Random ' \
                             'Forest-EachTrain\ROC Characteristics Random Forest'
    if not os.path.exists(path):
        os.makedirs(path)  # Crate the directory if one does not exist.

    arr_recall = arange(930, dtype=float32).reshape(930, 1)
    arr_auc = arange(930, dtype=float32).reshape(930, 1)

    fold = 1
    user = 1
    block = 1
    i = 0
    while user <= 31:
        while block <= 3:
            while fold <= 10:
                # We will keep the block = 1 constant in the training data file because we are training the classifier on the first block only and testing it on the remaining blocks
                traindata = genfromtxt(
                    '../../Output Files/Experiment102/GIUser/User' + str(
                        user) + '/1-' + str(user) + '-1-'+str(block)+'-' + str(fold) + '.csv',
                    dtype=float, delimiter=',')
                """
                testdata = genfromtxt(
                    '../../Output Files/Experiment102/GIUser/User' + str(
                        user) + '/1-' + str(user) + '-1-' + str(block) + '-' + str(
                        fold) + '.csv', dtype=float, delimiter=',')
                """
                # Generate target_train and target_test files with 1 indicating genuine and 0 indicating impostor
                target_train = np.ones(len(traindata))
                row = 0
                while row < len(traindata):
                    if np.any(traindata[row, 0:3] != [1, user, 1]):
                        target_train[row] = 0
                    row += 1

                sample_train = traindata[:, 3:]

                # scale the data on all the features.
                scaler = preprocessing.StandardScaler().fit(sample_train)
                sample_train_scaled = scaler.transform(sample_train)

                cv = ShuffleSplit(len(traindata), 10, test_size=0.5, train_size=0.5)
                arr_auc_test = np.array([])
                arr_recall_test = np.array([])
                for j, (train , test) in enumerate(cv):
                    X_train, X_test, y_train, y_test = sample_train_scaled[train], sample_train_scaled[test], target_train[
                    train], target_train[test]

                    clf = ExtraTreesClassifier(n_estimators=100, n_jobs=1)
                    clf.fit(X_train, y_train)

                    target_test_prediction = clf.predict(X_test)
                    recall_split = metrics.recall_score(y_test, target_test_prediction)
                    auc_test_split = metrics.roc_auc_score(y_test, target_test_prediction)

                    arr_recall_test = np.append(arr_recall_test, [recall_split], axis=0)
                    arr_auc_test = np.append(arr_auc_test, [auc_test_split], axis=0)

                # Get the auc value based on the actual target and the predicted target.
                auc_scaled = np.mean(arr_auc_test)
                arr_auc[i, 0] = auc_scaled
                recall_scaled = np.mean(arr_recall_test)
                arr_recall[i, 0] = recall_scaled
                fold += 1
                i += 1
            fold = 1
            block += 1
        block = 1
        user += 1
        print("x")

    # for the consecutive group of 10 values of arr_recall you need to get the average
    # so for total 60 values you will get 6 averaged values
    # so consecutively this 6 values are of different users


    aucvalues = np.transpose(arr_auc.reshape(93, 10))
    np.savetxt(
        "../../Output Files/Experiment103/Random "
        "Forest-EachTrain/RandomForest_AUC.csv",
        aucvalues, delimiter=",")

    recallvalues = np.transpose(arr_recall.reshape(93, 10))
    np.savetxt(
        "../../Output Files/Experiment103/Random "
        "Forest-EachTrain/RandomForest_Recall.csv",
        recallvalues, delimiter=",")

    aucdata = genfromtxt(
        '../../Output Files/Experiment103/Random '
        'Forest-EachTrain/RandomForest_AUC.csv',
        dtype=float, delimiter=',')
    arr_auc = arange(93, dtype=float32).reshape(93, 1)

    recalldata = genfromtxt(
        "../../Output Files/Experiment103/Random "
        "Forest-EachTrain/RandomForest_Recall.csv",
        dtype=float, delimiter=",")
    arr_recall = arange(93, dtype=float32).reshape(93, 1)

    # find the mean of every 10 fold with 1st and 3rd block for every user so it will generate arrays of size 31*2.
    i = 0
    while i <= 92:
        arr_auc[i, 0] = np.mean(aucdata[:, i])
        arr_recall[i, 0] = np.mean(recalldata[:, i])
        i += 1

    np.savetxt(
        "../../Output Files/Experiment103/Random "
        "Forest-EachTrain/RandomForestScaled_MeanRecall.csv",
        np.transpose(arr_recall.reshape(31, 3)), delimiter=",")
    np.savetxt(
        "../../Output Files/Experiment103/Random "
        "Forest-EachTrain/RandomForest_MeanAUC.csv",
        np.transpose(arr_auc.reshape(31, 3)), delimiter=",")

model2()
