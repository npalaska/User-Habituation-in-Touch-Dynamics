"""creating the Imposture training and testing data set
 to join with genuine training and testing data set respectively"""

import numpy as np
from numpy import *
import random

blocks = 3
folds = 1
user = 1
while folds <= 10:
    while user <= 31:
        with open("../../Output Files/Experiment30/1-" + str(
                user) + "-1-G.csv", "r") as source1:
            lines = [line for line in source1]
        # we will consider for every user, other 30 users are imposture.
        imposture = genfromtxt(
            '../../Output Files/Experiment30/1-' + str(user) + '-1-G.csv',
            dtype=float, delimiter=',')
        lengthofimposter = (len(imposture) - (
            len(imposture) % (30 * blocks))) / (
                               30 * blocks)  # length of an imposture data from each other user
        lengthofimposter = int(lengthofimposter)
        randomsample = random.sample(lines,
                                     lengthofimposter)  # get the 10(length of imposture) random strokes from each genuine user

        # for every fold of each user we will create the new file and write the randomly generated samples to it.
        with open("../../Output Files/Experiment50/1-" + str(
                user) + "-1-I-" + str(folds) + ".csv", "w") as file:
            file.write("".join(randomsample))

        # np.savetxt("../../Output Files/Experiment5/1-"+str(user)+"-1-I-"+str(folds)+".csv", randomsample, delimiter=",")

        # save the imposture data into a local variable 'imposturedata'
        imposturedata = genfromtxt(
            '../../Output Files/Experiment50/1-' + str(user) + '-1-I-' + str(
                folds) + '.csv', dtype=float, delimiter=',')

        # Get the imposture training data by dividing the imposturedata into half
        trainingdata = imposturedata[0:len(imposturedata) / 2, :]
        np.savetxt("../../Output Files/Experiment50/1-" + str(
            user) + "-1-I-Train-" + str(folds) + ".csv", trainingdata,
                   delimiter=",")

        # get the imposture testing data
        testingdata = imposturedata[len(imposturedata) / 2:len(imposturedata),
                      :]
        np.savetxt("../../Output Files/Experiment50/1-" + str(
            user) + "-1-I-Test-" + str(folds) + ".csv", testingdata,
                   delimiter=",")
        user += 1
    user = 1
    folds += 1