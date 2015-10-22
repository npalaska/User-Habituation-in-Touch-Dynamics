"""creating the Imposture training and testing data set
 to join with genuine training and testing data set respectively"""

import numpy as np
from numpy import *
import random
import os

folds = 1
user = 1
currentdirectory = os.getcwd()  # get the directory.
parentdirectory = os.path.abspath(
    currentdirectory + "/../..")  # Get the parent directory(2 levels up)
path = parentdirectory + '\Output Files\Experiment73\Impostor User'
if not os.path.exists(path):
    os.makedirs(path)  # Crate the directory if one does not exist.

# We have 76 strokes from the genuine user so we will take 76 strokes from impostor
# so we will take 2 from first 22 users and 3 from the next 9 users
# but here for simplicity of dividing the data in half we will choose random 4 strokes from the last 9 users
# But while creating the complete train and test file we will append only 3 strokes from each last 15 impostor with genuine train and test

while folds <= 10:
    while user <= 22:
        with open("../../Output Files/Experiment30/1-" + str(
                user) + "-1-G.csv", "r") as source1:
            lines = [line for line in source1]
        # we will consider for every user, other 30 users are imposture.
        imposture = genfromtxt(
            '../../Output Files/Experiment30/1-' + str(user) + '-1-G.csv',
            dtype=float, delimiter=',')
        lengthofimpostor = 2  # length of an imposture data from every other user
        # We are choosing 2 as a sample size because we can partition data easily into train and test

        randomsample = random.sample(lines,
                                     lengthofimpostor)  # get the 2 random strokes from every genuine user

        with open("../../Output Files/Experiment73/Impostor User/1-" + str(
                user) + "-1-I-" + str(folds) + ".csv", "w") as file:
            file.write("".join(randomsample))

        imposturedata = genfromtxt(
            '../../Output Files/Experiment73/Impostor User/1-' + str(
                user) + '-1-I-' + str(folds) + '.csv', dtype=float,
            delimiter=',')
        trainingdata = imposturedata[0:1, :]  # Get the training data
        np.savetxt("../../Output Files/Experiment73/Impostor User/1-" + str(
            user) + "-1-I-Train-" + str(folds) + ".csv", trainingdata,
                   delimiter=",")
        testingdata = imposturedata[1:len(imposturedata),
                      :]  # det the testing data
        np.savetxt("../../Output Files/Experiment73/Impostor User/1-" + str(
            user) + "-1-I-Test-" + str(folds) + ".csv", testingdata,
                   delimiter=",")
        user += 1
    user = 1
    folds += 1

folds = 1
user = 23

while folds <= 10:
    while 23 <= user <= 31:
        with open("../../Output Files/Experiment30/1-" + str(
                user) + "-1-G.csv", "r") as source1:
            lines = [line for line in source1]
        # we will consider for every user, other 30 users are imposture.
        imposture = genfromtxt(
            '../../Output Files/Experiment30/1-' + str(user) + '-1-G.csv',
            dtype=float, delimiter=',')
        lengthofimpostor = 4  # length of an imposture data from every other user
        # We are choosing 4 as a sample size because we can partition data easily into train and test

        randomsample = random.sample(lines,
                                     lengthofimpostor)  # get the 4 random strokes from every genuine user

        with open("../../Output Files/Experiment73/Impostor User/1-" + str(
                user) + "-1-I-" + str(folds) + ".csv", "w") as file:
            file.write("".join(randomsample))

        imposturedata = genfromtxt(
            '../../Output Files/Experiment73/Impostor User/1-' + str(
                user) + '-1-I-' + str(folds) + '.csv', dtype=float,
            delimiter=',')
        trainingdata = imposturedata[0:2, :]  # Get the training data
        np.savetxt("../../Output Files/Experiment73/Impostor User/1-" + str(
            user) + "-1-I-Train-" + str(folds) + ".csv", trainingdata,
                   delimiter=",")
        testingdata = imposturedata[2:len(imposturedata),
                      :]  # det the testing data
        np.savetxt("../../Output Files/Experiment73/Impostor User/1-" + str(
            user) + "-1-I-Test-" + str(folds) + ".csv", testingdata,
                   delimiter=",")
        user += 1
    user = 23
    folds += 1
