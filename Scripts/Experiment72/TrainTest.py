"""Create the training and testing data for the classifier model.
    Training and Testing data will contain data from both Genuine and Imposture
    with half as a genuine and half as an imposture."""

import numpy as np
from numpy import *
import os

user = 1
blocks = 1
imposture = 1
folds = 1

while user <= 31:
    currentdirectory = os.getcwd()  # get the directory.
    parentdirectory = os.path.abspath(
        currentdirectory + "/../..")  # Get the parent directory(2 levels up)
    path = parentdirectory + '/Output Files/Experiment72/GIUser/User' + str(
        user)
    if not os.path.exists(path):
        os.makedirs(path)  # Crate the directory if one does not exist.

    while blocks <= 6:
        # Get the Genuine Training and Testing file.
        arr_train = genfromtxt(
            '../../Output Files/Experiment72/Genuine User/GenuineUser' + str(
                user) + '/1-' + str(user) + '-1-G-Train-' + str(
                blocks) + '.csv', dtype=float, delimiter=',')
        arr_test = genfromtxt(
            '../../Output Files/Experiment72/Genuine User/GenuineUser' + str(
                user) + '/1-' + str(user) + '-1-G-Test-' + str(
                blocks) + '.csv', dtype=float, delimiter=',')
        while folds <= 10:
            while 1 <= imposture <= 15:
                # Get the Imposture Training and Testing files.
                if imposture != user:
                    arr_imposture_train = genfromtxt(
                        '../../Output Files/Experiment72/Impostor User/1-' + str(
                            imposture) + '-1-I-Train-' + str(folds) + '.csv',
                        dtype=float, delimiter=',')
                    arr_imposture_test = genfromtxt(
                        '../../Output Files/Experiment72/Impostor User/1-' + str(
                            imposture) + '-1-I-Test-' + str(folds) + '.csv',
                        dtype=float, delimiter=',')

                    arr_imposture_train = arr_imposture_train[1:, :]
                    arr_imposture_test = arr_imposture_test[1:, :]

                    # join genuine training and testing file with imposture training and testing files respectively.
                    arr_train = np.vstack((arr_train, arr_imposture_train))
                    arr_test = np.vstack((arr_test, arr_imposture_test))
                imposture += 1
            while 16 <= imposture <= 31:
                # Get the Imposture Training and Testing files.
                if imposture != user:
                    arr_imposture_train = genfromtxt(
                        '../../Output Files/Experiment72/Impostor User/1-' + str(
                            imposture) + '-1-I-Train-' + str(folds) + '.csv',
                        dtype=float, delimiter=',')
                    arr_imposture_test = genfromtxt(
                        '../../Output Files/Experiment72/Impostor User/1-' + str(
                            imposture) + '-1-I-Test-' + str(folds) + '.csv',
                        dtype=float, delimiter=',')

                    # join genuine training and testing file with imposture training and testing files respectively.
                    arr_train = np.vstack((arr_train, arr_imposture_train))
                    arr_test = np.vstack((arr_test, arr_imposture_test))
                imposture += 1

            # save the files.
            np.savetxt("../../Output Files/Experiment72/GIUser/User" + str(
                user) + "/1-" + str(user) + "-1-Train-" + str(
                blocks) + "-" + str(folds) + ".csv", arr_train, delimiter=",")
            np.savetxt("../../Output Files/Experiment72/GIUser/User" + str(
                user) + "/1-" + str(user) + "-1-Test-" + str(
                blocks) + "-" + str(folds) + ".csv", arr_test, delimiter=",")
            imposture = 1
            arr_train = genfromtxt(
                '../../Output Files/Experiment72/Genuine User/GenuineUser' + str(
                    user) + '/1-' + str(user) + '-1-G-Train-' + str(
                    blocks) + '.csv', dtype=float, delimiter=',')
            arr_test = genfromtxt(
                '../../Output Files/Experiment72/Genuine User/GenuineUser' + str(
                    user) + '/1-' + str(user) + '-1-G-Test-' + str(
                    blocks) + '.csv', dtype=float, delimiter=',')
            folds += 1
        folds = 1
        blocks += 1
    blocks = 1
    user += 1


