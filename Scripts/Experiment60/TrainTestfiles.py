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
    currentdirectory = os.getcwd()  # get the current directory.
    parentdirectory = os.path.abspath(
        currentdirectory + "/../..")  # Get the parent directory(2 levels up)
    path = parentdirectory + '\Output Files\Experiment60\GIUser' + str(
        user) + ''
    if not os.path.exists(path):
        os.makedirs(path)  # Create the folder if one does not present.
    while blocks <= 3:
        # Get the Genuine Training and Testing file from the experiment 4.
        arr_training = genfromtxt(
            '../../Output Files/Experiment40/GenuineUser' + str(
                user) + '/1-' + str(user) + '-1-G-Train-' + str(
                blocks) + '.csv', dtype=float, delimiter=',')
        arr_testing = genfromtxt(
            '../../Output Files/Experiment40/GenuineUser' + str(
                user) + '/1-' + str(user) + '-1-G-Test-' + str(
                blocks) + '.csv', dtype=float, delimiter=',')
        while folds <= 10:
            while imposture <= 31:
                # Get the Imposture Training and Testing files from the experiment 5.
                if imposture != user:  # As for every user we have 30 imposture
                    arr_imposture_train = genfromtxt(
                        '../../Output Files/Experiment50/1-' + str(
                            imposture) + '-1-I-Train-' + str(folds) + '.csv',
                        dtype=float, delimiter=',')
                    arr_imposture_test = genfromtxt(
                        '../../Output Files/Experiment50/1-' + str(
                            imposture) + '-1-I-Test-' + str(folds) + '.csv',
                        dtype=float, delimiter=',')

                    # join genuine training and testing file with imposture training and testing files respectively.
                    arr_training = np.vstack(
                        (arr_training, arr_imposture_train))
                    arr_testing = np.vstack((arr_testing, arr_imposture_test))
                imposture += 1

            # save the files.
            np.savetxt("../../Output Files/Experiment60/GIUser" + str(
                user) + "/1-" + str(user) + "-1-Train-" + str(
                blocks) + "-" + str(folds) + ".csv", arr_training,
                       delimiter=",")
            np.savetxt("../../Output Files/Experiment60/GIUser" + str(
                user) + "/1-" + str(user) + "-1-Test-" + str(
                blocks) + "-" + str(folds) + ".csv", arr_testing,
                       delimiter=",")
            imposture = 1
            arr_training = genfromtxt(
                '../../Output Files/Experiment40/GenuineUser' + str(
                    user) + '/1-' + str(user) + '-1-G-Train-' + str(
                    blocks) + '.csv', dtype=float, delimiter=',')
            arr_testing = genfromtxt(
                '../../Output Files/Experiment40/GenuineUser' + str(
                    user) + '/1-' + str(user) + '-1-G-Test-' + str(
                    blocks) + '.csv', dtype=float, delimiter=',')
            folds += 1
        folds = 1
        blocks += 1
    blocks = 1
    user += 1
