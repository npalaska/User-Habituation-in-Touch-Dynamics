"""creating the Imposture training and testing data set
 to join with genuine training and testing data set respectively"""

import numpy as np
from numpy import *
import random
import os

blocks = 2
folds = 1
user = 1
currentdirectory = os.getcwd()  # get the directory.
parentdirectory = os.path.abspath(
    currentdirectory + "/../..")  # Get the parent directory(2 levels up)
path = parentdirectory + '\Output Files\Experiment71\Impostor User'
if not os.path.exists(path):
    os.makedirs(path)  # Crate the directory if one does not exist.

while folds <= 10:
    while user <= 31:
        with open("../../Output Files/Experiment30/1-" + str(
                user) + "-1-G.csv", "r") as source1:
            lines = [line for line in source1]
        # we will consider for every user, other 30 users are imposture.
        imposture = genfromtxt(
            '../../Output Files/Experiment30/1-' + str(user) + '-1-G.csv',
            dtype=float, delimiter=',')
        lengthofimpostor = 15  # length of an imposture data from every other user
        randomsample = random.sample(lines,
                                     lengthofimpostor)  # get the 15 random strokes from every genuine user

        with open("../../Output Files/Experiment71/Impostor User/1-" + str(
                user) + "-1-I-" + str(folds) + ".csv", "w") as file:
            file.write("".join(randomsample))

        imposturedata = genfromtxt(
            '../../Output Files/Experiment71/Impostor User/1-' + str(
                user) + '-1-I-' + str(folds) + '.csv', dtype=float,
            delimiter=',')
        trainingdata = imposturedata[0:7, :]  # Get the training data
        np.savetxt("../../Output Files/Experiment71/Impostor User/1-" + str(
            user) + "-1-I-Train-" + str(folds) + ".csv", trainingdata,
                   delimiter=",")
        testingdata = imposturedata[7:len(imposturedata),
                      :]  # det the testing data
        np.savetxt("../../Output Files/Experiment71/Impostor User/1-" + str(
            user) + "-1-I-Test-" + str(folds) + ".csv", testingdata,
                   delimiter=",")
        user += 1
    user = 1
    folds += 1