import numpy as np
from numpy import *
import os

def CompleteTrainTest(numberofsplits):
    user = 1
    blocks = 1
    imposture = 1
    folds = 1
    while user <= 31:
        currentdirectory = os.getcwd()  # get the directory.
        parentdirectory = os.path.abspath(
            currentdirectory + "/../..")  # Get the parent directory(2 levels up)
        path = parentdirectory + '/Output Files/Experiment100/'+str(numberofsplits)+'splits_up_ordered/GIUser/User' + \
               str(
            user)
        if not os.path.exists(path):
            os.makedirs(path)  # Crate the directory if one does not exist.

        while blocks <= numberofsplits:
            # Get the Genuine Training and Testing file.
            arr_train = genfromtxt(
                '../../Output Files/Experiment100/'+str(numberofsplits)+'splits_up_ordered/Genuine User/GenuineUser' +
                str(
                    user) + '/1-' + str(user) + '-1-G-' + str(blocks) + '.csv',
                dtype=float, delimiter=',')
            while folds <= 10:
                while imposture <= 31:
                    # Get the Imposture Training and Testing files.
                    if imposture != user:
                        arr_imposture_train = genfromtxt(
                            '../../Output Files/Experiment100/'+str(numberofsplits)+'splits_up_ordered/Impostor '
                                                                                    'User/1-' +
                            str(
                                imposture) + '-1-I-' + str(folds) + '.csv',
                            dtype=float, delimiter=',')

                        # join genuine training and testing file with imposture training and testing files respectively.
                        arr_train = np.vstack((arr_train, arr_imposture_train))
                    imposture += 1

                # save the files.
                np.savetxt("../../Output Files/Experiment100/"+str(numberofsplits)+"splits_up_ordered/GIUser/User" +
                           str(
                    user) + "/1-" + str(user) + "-1-" + str(blocks) + "-" + str(
                    folds) + ".csv", arr_train, delimiter=",")
                imposture = 1
                arr_train = genfromtxt(
                    '../../Output Files/Experiment100/'+str(numberofsplits)+'splits_up_ordered/Genuine '
                                                                            'User/GenuineUser' +
                    str(
                        user) + '/1-' + str(user) + '-1-G-' + str(blocks) + '.csv',
                    dtype=float, delimiter=',')
                folds += 1
            folds = 1
            blocks += 1
        blocks = 1
        user += 1

CompleteTrainTest(12)