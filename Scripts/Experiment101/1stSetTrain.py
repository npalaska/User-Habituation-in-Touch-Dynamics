import numpy as np
from numpy import *
import os
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_curve
from sklearn import metrics
from sklearn import preprocessing
import matplotlib.pyplot as plt

def sixSplitTrainTest():
    currentdirectory = os.getcwd()  # get the directory.
    parentdirectory = os.path.abspath(
        currentdirectory + "/../..")  # Get the parent directory(2 levels up)
    path = parentdirectory + '/Output Files/Experiment101/6Split/Random ' \
                             'Forest-4st-Train/ROC Characteristics Random Forest'
    if not os.path.exists(path):
        os.makedirs(path)  # Crate the directory if one does not exist.

    arr_recall = arange(930, dtype=float32).reshape(930, 1)
    arr_auc = arange(930, dtype=float32).reshape(930, 1)
    arr_eer = arange(930, dtype=float32).reshape(930, 1)

    fold = 1
    user = 3
    block = 1
    i = 0
    while user <= 31:
        while block <= 6:
            while fold <= 9:
                # We will keep the block = 1 constant in the training data file because we are training the classifier on the first block only and testing it on the remaining blocks

                traindata = genfromtxt(
                    '../../Output Files/Experiment100/GIUser/User' + str(
                        user) + '/1-' + str(user) + '-1-1-' + str(fold) + '.csv',
                    dtype=float, delimiter=',')


                """
                traindata2 = genfromtxt(
                    '../../Output Files/Experiment100/6splits/GIUser/User' + str(
                        user) + '/1-' + str(user) + '-1-5-' + str(fold+1) + '.csv',
                    dtype=float, delimiter=',')
                traindata3 = genfromtxt(
                    '../../Output Files/Experiment100/6splits_ordered/GIUser/User' + str(
                        user) + '/1-' + str(user) + '-1-3-' + str(fold+2) + '.csv',
                    dtype=float, delimiter=',')
                traindata4 = genfromtxt(
                    '../../Output Files/Experiment100/6splits_ordered/GIUser/User' + str(
                        user) + '/1-' + str(user) + '-1-2-' + str(fold+3) + '.csv',
                    dtype=float, delimiter=',')
                """

                # traindata = np.vstack((traindata1, traindata2, traindata3, traindata4))

                testdata = genfromtxt(
                    '../../Output Files/Experiment100/GIUser/User' + str(
                        user) + '/1-' + str(user) + '-1-'+str(block)+'-' + str(fold+1) + '.csv',
                    dtype=float, delimiter=',')

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

                recall = metrics.recall_score(target_test, target_test_prediction)
                arr_recall[i, 0] = recall
                fpr, tpr, thresholds = roc_curve(target_test_prediction, score,
                                                 pos_label=0)

                row = 0
                fpr_tpr = np.arange(len(fpr),
                                        dtype=float32)  # A temporary array to hold the value of (1-fpr-tpr)
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
                    "../../Output Files/Experiment101/12splits/Random Forest-10-11Th-Train/ROC "
                    "Characteristics Random Forest/1-" + str(
                        user) + "-1-" + str(block) + "-" + str(
                        fold) + "-FPRTPR.csv", np.transpose(fprtprrelation),
                    delimiter=",")

                eer = np.array([])
                eerdata = genfromtxt(
                    '../../Output Files/Experiment101/12splits/Random Forest-10-11Th-Train/ROC '
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
                # auc = metrics.roc_auc_score(target_test, target_test_prediction)
                arr_auc[i, 0] = auc_scaled
                # arr_auc[i, 0] = auc
                fold += 1
                i += 1
                probas_ = clf.predict_proba(sample_test_scaled)
                #fpr, tpr, thresholds = roc_curve(target_test, target_test_prediction)
                #roc_auc = auc(fpr[:], tpr[:])
                #plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
                plt.plot(fpr, tpr, lw=1, label='ROC fold', color='b')
                plt.show()
            fold = 1
            block += 1
        block = 1
        user += 1
        print("x")

    # for the consecutive group of 10 values of arr_recall you need to get the average
    # so for total 60 values you will get 6 averaged values
    # so consecutively this 6 values are of different users


    eervalues = np.transpose(arr_eer.reshape(186, 9))
    np.savetxt(
        "../../Output Files/Experiment101/6Split/Random "
        "Forest-4st-Train/RandomForestScaled_EER.csv",
        eervalues, delimiter=",")

    aucvalues = np.transpose(arr_auc.reshape(186, 9))
    np.savetxt(
        "../../Output Files/Experiment101/6Split/Random "
        "Forest-4st-Train/RandomForest_AUC.csv",
        aucvalues, delimiter=",")

    recallvalues = np.transpose(arr_recall.reshape(186, 9))
    np.savetxt(
        "../../Output Files/Experiment101/6Split/Random "
        "Forest-4st-Train/RandomForest_Recall.csv",
        recallvalues, delimiter=",")

    eerdata = genfromtxt(
        '../../Output Files/Experiment101/6Split/Random '
        'Forest-4st-Train/RandomForestScaled_EER.csv',
        dtype=float, delimiter=',')
    arr_eer = arange(186, dtype=float32).reshape(186, 1)


    aucdata = genfromtxt(
        '../../Output Files/Experiment101/6Split/Random '
        'Forest-4st-Train/RandomForest_AUC.csv',
        dtype=float, delimiter=',')
    arr_auc = arange(186, dtype=float32).reshape(186, 1)

    recalldata = genfromtxt(
        "../../Output Files/Experiment101/6Split/Random "
        "Forest-4st-Train/RandomForest_Recall.csv",
        dtype=float, delimiter=",")
    arr_recall = arange(186, dtype=float32).reshape(186, 1)

    # find the mean of every 10 fold with 1st and 3rd block for every user so it will generate arrays of size 31*2.
    i = 0
    while i <= 185:
        arr_eer[i, 0] = np.mean(eerdata[:, i])
        arr_auc[i, 0] = np.mean(aucdata[:, i])
        arr_recall[i, 0] = np.mean(recalldata[:, i])
        i += 1

    np.savetxt(
        "../../Output Files/Experiment101/6Split/Random "
        "Forest-4st-Train/RandomForestScaled_MeanRecall.csv",
        np.transpose(arr_recall.reshape(31, 6)), delimiter=",")
    np.savetxt(
        "../../Output Files/Experiment101/6Split/Random "
        "Forest-4st-Train/RandomForestScaled_MeanEER.csv",
        np.transpose(arr_eer.reshape(31, 6)), delimiter=",")

    np.savetxt(
        "../../Output Files/Experiment101/6Split/Random "
        "Forest-4st-Train/RandomForest_MeanAUC.csv",
        np.transpose(arr_auc.reshape(31, 6)), delimiter=",")

def twelveSplitTrainTest():
    trainingblocks = 1
    while trainingblocks <= 7:
        currentdirectory = os.getcwd()  # get the directory.
        parentdirectory = os.path.abspath(
            currentdirectory + "/../..")  # Get the parent directory(2 levels up)
        path = parentdirectory + '/Output Files/Experiment101/combined/12Splits Ordered/5 slide/Random ' \
                                 'Forest-'+str(trainingblocks)+'to'+str(trainingblocks+4)+'Th-Train/ROC ' \
                                                                                         'Characteristics Random Forest'
        if not os.path.exists(path):
            os.makedirs(path)  # Crate the directory if one does not exist.

        arr_recall = arange(1860, dtype=float32).reshape(1860, 1)
        arr_auc = arange(1860, dtype=float32).reshape(1860, 1)
        arr_eer = arange(1860, dtype=float32).reshape(1860, 1)

        fold = 1
        user = 1
        block = 1
        i = 0
        while user <= 31:
            while block <= 12:
                while fold <= 5:
                    # We will keep the block = 1 constant in the training data file because we are training the classifier on the first block only and testing it on the remaining blocks
                    traindata0 = genfromtxt(
                        '../../Output Files/Experiment100/12splits_up_ordered/GIUser/User' + str(
                            user) + '/1-' + str(user) + '-1-'+str(trainingblocks)+'-' + str(fold) + '.csv',
                        dtype=float, delimiter=',')

                    traindata1 = genfromtxt(
                        '../../Output Files/Experiment100/12splits_up_ordered/GIUser/User' + str(
                            user) + '/1-' + str(user) + '-1-'+str(trainingblocks+1)+'-' + str(fold+1) + '.csv',
                        dtype=float, delimiter=',')

                    traindata2 = genfromtxt(
                        '../../Output Files/Experiment100/12splits_up_ordered/GIUser/User' + str(
                            user) + '/1-' + str(user) + '-1-'+str(trainingblocks+2)+'-' + str(fold+2) + '.csv',
                        dtype=float, delimiter=',')

                    traindata3 = genfromtxt(
                        '../../Output Files/Experiment100/12splits_up_ordered/GIUser/User' + str(
                            user) + '/1-' + str(user) + '-1-'+str(trainingblocks+3)+'-' + str(fold+3) + '.csv',
                        dtype=float, delimiter=',')

                    traindata4 = genfromtxt(
                        '../../Output Files/Experiment100/12splits_up_ordered/GIUser/User' + str(
                            user) + '/1-' + str(user) + '-1-'+str(trainingblocks+4)+'-' + str(fold+4) + '.csv',
                        dtype=float, delimiter=',')

                    traindata = np.vstack((traindata0, traindata1, traindata2, traindata3, traindata4))

                    testdata = genfromtxt(
                        '../../Output Files/Experiment100/12splits_up_ordered/GIUser/User' + str(
                            user) + '/1-' + str(user) + '-1-' + str(block) + '-' + str(
                            fold+5) + '.csv', dtype=float, delimiter=',')

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

                    """
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
                    """
                    recall = metrics.recall_score(target_test, target_test_prediction)
                    arr_recall[i, 0] = recall

                    fpr, tpr, thresholds = roc_curve(target_test_prediction, score,
                                                     pos_label=0)
                    row = 0
                    fpr_tpr = np.arange(len(fpr),
                                        dtype=float32)  # A temporary array to hold the value of (1-fpr-tpr)
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
                    """
                    row = 0
                    while row <= len(eerdata) - 1:
                        if eerdata[row, 2] == 0:
                            # eer = np.append(eer, [eerdata[row, 0]], axis=0)
                            eerr = np.append(eer, [eerdata[row, 0]], axis=0)
                        else:
                            eerr = np.append(eer, [0], axis=0)
                        row += 1
                    if eerr == np.array([0]):
                        rowno = min(range(len(eerdata[:, 2])), key=lambda j:abs(j-0))
                        eerr = np.append(eer, [eerdata[rowno, 0]], axis=0)
                    meaneer = mean(eerr)
                    arr_eer[i, 0] = meaneer
                    # declaring the variable fprtprrelation to hold the value of fpr, tpr, fpr_tpr, threshold in the table.
                    fprtprrelation = np.vstack((fpr, tpr, fpr_tpr, thresholds))
                    # save fprtprrelation table for every combination of training and testing file.
                    np.savetxt(
                        "../../Output Files/Experiment101/combined/12Splits Ordered/5 slide/Random Forest-"+str(trainingblocks)+"to"+str(trainingblocks+4)+"Th-Train/ROC "
                        "Characteristics Random Forest/1-" + str(
                            user) + "-1-" + str(block) + "-" + str(
                            fold) + "-FPRTPR.csv", np.transpose(fprtprrelation),
                        delimiter=",")

                    eer = np.array([])
                    eerdata = genfromtxt(
                        '../../Output Files/Experiment101/combined/12Splits Ordered/5 slide/Random Forest-'+str(trainingblocks)+"to"+str(trainingblocks+4)+'Th-Train/ROC '
                        'Characteristics Random Forest/1-' + str(
                            user) + '-1-' + str(block) + '-' + str(
                            fold) + '-FPRTPR.csv', dtype=float, delimiter=',')
                    row = 0
                    while row <= len(eerdata) - 1:
                        if eerdata[row, 2] == 0:
                            eerr = np.append(eer, [eerdata[row, 0]], axis=0)
                        row += 1
                    meaneer = mean(eerr)
                    rowno = min(range(len(fpr_tpr)), key=lambda x:abs(x-0))
                    arr_eer[i, 0] = fpr[rowno]
                    """

                    # Get the auc value based on the actual target and the predicted target.
                    auc_scaled = metrics.roc_auc_score(target_test,
                                                       target_test_prediction)
                    # auc = metrics.roc_auc_score(target_test, target_test_prediction)
                    arr_auc[i, 0] = auc_scaled
                    # arr_auc[i, 0] = auc
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


        eervalues = np.transpose(arr_eer.reshape(372, 5))
        np.savetxt(
            "../../Output Files/Experiment101/combined/12Splits Ordered/5 slide/Random "
            "Forest-"+str(trainingblocks)+"to"+str(trainingblocks+4)+"Th-Train/RandomForest_EER.csv",
            eervalues, delimiter=",")

        aucvalues = np.transpose(arr_auc.reshape(372, 5))
        np.savetxt(
            "../../Output Files/Experiment101/combined/12Splits Ordered/5 slide/Random "
            "Forest-"+str(trainingblocks)+"to"+str(trainingblocks+4)+"Th-Train/RandomForest_AUC.csv",
            aucvalues, delimiter=",")

        recallvalues = np.transpose(arr_recall.reshape(372, 5))
        np.savetxt(
            "../../Output Files/Experiment101/combined/12Splits Ordered/5 slide/Random "
            "Forest-"+str(trainingblocks)+"to"+str(trainingblocks+4)+"Th-Train/RandomForest_Recall.csv",
            recallvalues, delimiter=",")


        eerpreprocessdata = genfromtxt(
            '../../Output Files/Experiment101/combined/12Splits Ordered/5 slide/Random '
            'Forest-'+str(trainingblocks)+'to'+str(trainingblocks+4)+'Th-Train/RandomForest_EER.csv',
            dtype=float, delimiter=',')
        arr_eer = arange(372, dtype=float32).reshape(372, 1)


        aucdata = genfromtxt(
            '../../Output Files/Experiment101/combined/12Splits Ordered/5 slide/Random '
            'Forest-'+str(trainingblocks)+'to'+str(trainingblocks+4)+'Th-Train/RandomForest_AUC.csv',
            dtype=float, delimiter=',')
        arr_auc = arange(372, dtype=float32).reshape(372, 1)

        recalldata = genfromtxt(
            "../../Output Files/Experiment101/combined/12Splits Ordered/5 slide/Random "
            "Forest-"+str(trainingblocks)+"to"+str(trainingblocks+4)+"Th-Train/RandomForest_Recall.csv",
            dtype=float, delimiter=",")
        arr_recall = arange(372, dtype=float32).reshape(372, 1)

        # find the mean of every 10 fold with 1st and 3rd block for every user so it will generate arrays of size 31*2.
        i = 0
        while i <= 371:
            arr_eer[i, 0] = np.mean(eerpreprocessdata[:, i])
            arr_auc[i, 0] = np.mean(aucdata[:, i])
            arr_recall[i, 0] = np.mean(recalldata[:, i])
            i += 1

        np.savetxt(
            "../../Output Files/Experiment101/combined/12Splits Ordered/5 slide/Random "
            "Forest-"+str(trainingblocks)+"to"+str(trainingblocks+4)+"Th-Train/RandomForestScaled_MeanRecall.csv",
            np.transpose(arr_recall.reshape(31, 12)), delimiter=",")
        np.savetxt(
            "../../Output Files/Experiment101/combined/12Splits Ordered/5 slide/Random "
            "Forest-"+str(trainingblocks)+"to"+str(trainingblocks+4)+"Th-Train/RandomForestScaled_MeanEER.csv",
            np.transpose(arr_eer.reshape(31, 12)), delimiter=",")
        np.savetxt(
            "../../Output Files/Experiment101/combined/12Splits Ordered/5 slide/Random "
            "Forest-"+str(trainingblocks)+"to"+str(trainingblocks+4)+"Th-Train/RandomForest_MeanAUC.csv",
            np.transpose(arr_auc.reshape(31, 12)), delimiter=",")

        trainingblocks += 1
        print("=========")

sixSplitTrainTest()