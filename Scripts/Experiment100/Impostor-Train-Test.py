import numpy as np
from numpy import *
import random
import os

def Split_6_ImpostorSamples():
    #blocks = 6
    folds = 1
    user = 1
    currentdirectory = os.getcwd()  # get the directory.
    parentdirectory = os.path.abspath(
        currentdirectory + "/../..")  # Get the parent directory(2 levels up)
    path = parentdirectory + '/Output Files/Experiment30/6splits/Impostor User'
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
            lengthofimpostor = 5  # length of an imposture data from every other user
            randomsample = random.sample(lines,
                                         lengthofimpostor)  # get the 5 random strokes from every genuine user

            with open("../../Output Files/Experiment100/6splits/Impostor User/1-" + str(
                    user) + "-1-I-" + str(folds) + ".csv", "w") as file:
                file.write("".join(randomsample))

            """
            imposturedata = genfromtxt('../../Output Files/Experiment72/Impostor User/1-'+str(user)+'-1-I-'+str(folds)+'.csv',dtype = float, delimiter=',')
            trainingdata = imposturedata[0:3, :]   # Get the training data
            np.savetxt("../../Output Files/Experiment72/Impostor User/1-"+str(user)+"-1-I-Train-"+str(folds)+".csv", trainingdata, delimiter=",")
            testingdata = imposturedata[3:len(imposturedata), :]   # det the testing data
            np.savetxt("../../Output Files/Experiment72/Impostor User/1-"+str(user)+"-1-I-Test-"+str(folds)+".csv", testingdata, delimiter=",")
            """
            user += 1
        user = 1
        folds += 1

def Split_12_Impostor():
    folds = 1
    user = 1
    currentdirectory = os.getcwd()  # get the directory.
    parentdirectory = os.path.abspath(
        currentdirectory + "/../..")  # Get the parent directory(2 levels up)
    path = parentdirectory + '/Output Files/Experiment100/12splits/20 Folds Impostor User'
    if not os.path.exists(path):
        os.makedirs(path)  # Crate the directory if one does not exist.

    # We have 76 strokes from the genuine user so we will take 76 strokes from impostor
    # so we will take 2 from first 22 users and 3 from the next 9 users
    # but here for simplicity of dividing the data in half we will choose random 4 strokes from the last 9 users
    # But while creating the complete train and test file we will append only 3 strokes from each last 15 impostor with genuine train and test

    while user <= 19:
        while folds <= 20:
            with open("../../Output Files/Experiment30/1-" + str(
                    user) + "-1-G.csv", "r") as source1:
                lines = [line for line in source1]
            # we will consider for every user, other 30 users are imposture.
            imposture = genfromtxt(
                '../../Output Files/Experiment30/1-' + str(user) + '-1-G.csv',
                dtype=float, delimiter=',')
            lengthofimpostor = 1  # length of an imposture data from every other user

            randomsample = random.sample(lines,
                                         lengthofimpostor)  # get the 2 random strokes from every genuine user

            with open("../../Output Files/Experiment100/12splits/20 Folds Impostor User/1-" + str(
                    user) + "-1-I-" + str(folds) + ".csv", "w") as file:
                file.write("".join(randomsample))

            folds += 1
        folds = 1
        user += 1

    folds = 1
    user = 20

    while 20 <= user <= 31:
        while folds <= 20:
            with open("../../Output Files/Experiment30/1-" + str(
                    user) + "-1-G.csv", "r") as source1:
                lines = [line for line in source1]
            # we will consider for every user, other 30 users are imposture.
            imposture = genfromtxt(
                '../../Output Files/Experiment30/1-' + str(user) + '-1-G.csv',
                dtype=float, delimiter=',')
            lengthofimpostor = 3  # length of an imposture data from every other user

            randomsample = random.sample(lines,
                                         lengthofimpostor)  # get the 4 random strokes from every genuine user

            with open("../../Output Files/Experiment100/12splits/20 Folds Impostor User/1-" + str(
                    user) + "-1-I-" + str(folds) + ".csv", "w") as file:
                file.write("".join(randomsample))

            folds += 1
        folds = 1
        user += 1

#Split_12_Impostor()

def Split_6_ImpostorSamples_ordered():
    blocks = 1
    folds = 1
    user = 1
    currentdirectory = os.getcwd()  # get the directory.
    parentdirectory = os.path.abspath(
        currentdirectory + "/../..")  # Get the parent directory(2 levels up)
    path = parentdirectory + '/Output Files/Experiment100/6splits_ordered/Impostor User'
    if not os.path.exists(path):
        os.makedirs(path)  # Crate the directory if one does not exist.

    while folds <= 10:
        while blocks <= 5:
            while user <= 31:
                with open("../../Output Files/Experiment30/1-" + str(
                        user) + "-1-G"+str(blocks+1)+".csv", "r") as source1:
                    lines = [line for line in source1]
                # we will consider for every user, other 30 users are imposture.

                lengthofimpostor = 5  # length of an imposture data from every other user
                randomsample = random.sample(lines,
                                             lengthofimpostor)  # get the 5 random strokes from every genuine user

                with open("../../Output Files/Experiment100/6splits_ordered/Impostor User/1-" + str(
                        user) + "-1-I-" + str(folds) + ".csv", "a") as file:
                    file.write("".join(randomsample))

                """
                imposturedata = genfromtxt('../../Output Files/Experiment72/Impostor User/1-'+str(user)+'-1-I-'+str(folds)+'.csv',dtype = float, delimiter=',')
                trainingdata = imposturedata[0:3, :]   # Get the training data
                np.savetxt("../../Output Files/Experiment72/Impostor User/1-"+str(user)+"-1-I-Train-"+str(folds)+".csv", trainingdata, delimiter=",")
                testingdata = imposturedata[3:len(imposturedata), :]   # det the testing data
                np.savetxt("../../Output Files/Experiment72/Impostor User/1-"+str(user)+"-1-I-Test-"+str(folds)+".csv", testingdata, delimiter=",")
                """
                user += 1
            blocks += 1
            user = 1
        blocks = 1
        folds += 1

#Split_6_ImpostorSamples_ordered()

def doubleImpostor():
    folds = 1
    user = 1
    currentdirectory = os.getcwd()  # get the directory.
    parentdirectory = os.path.abspath(
        currentdirectory + "/../..")  # Get the parent directory(2 levels up)
    path = parentdirectory + '/Output Files/Experiment100/12splits/double Impostor User'
    if not os.path.exists(path):
        os.makedirs(path)  # Crate the directory if one does not exist.

    # We have 76 strokes from the genuine user so we will take 76 strokes from impostor
    # so we will take 2 from first 22 users and 3 from the next 9 users
    # but here for simplicity of dividing the data in half we will choose random 4 strokes from the last 9 users
    # But while creating the complete train and test file we will append only 3 strokes from each last 15 impostor with genuine train and test

    while user <= 21:
        while folds <= 10:
            with open("../../Output Files/Experiment30/1-" + str(
                    user) + "-1-G.csv", "r") as source1:
                lines = [line for line in source1]
            # we will consider for every user, other 30 users are imposture.
            imposture = genfromtxt(
                '../../Output Files/Experiment30/1-' + str(user) + '-1-G.csv',
                dtype=float, delimiter=',')
            lengthofimpostor = 4  # length of an imposture data from every other user

            randomsample = random.sample(lines,
                                         lengthofimpostor)  # get the 2 random strokes from every genuine user

            with open("../../Output Files/Experiment100/12splits/double Impostor User/1-" + str(
                    user) + "-1-I-" + str(folds) + ".csv", "w") as file:
                file.write("".join(randomsample))

            folds += 1
        folds = 1
        user += 1

    folds = 1
    user = 22

    while 22 <= user <= 31:
        while folds <= 10:
            with open("../../Output Files/Experiment30/1-" + str(
                    user) + "-1-G.csv", "r") as source1:
                lines = [line for line in source1]
            # we will consider for every user, other 30 users are imposture.
            imposture = genfromtxt(
                '../../Output Files/Experiment30/1-' + str(user) + '-1-G.csv',
                dtype=float, delimiter=',')
            lengthofimpostor = 6  # length of an imposture data from every other user

            randomsample = random.sample(lines,
                                         lengthofimpostor)  # get the 4 random strokes from every genuine user

            with open("../../Output Files/Experiment100/12splits/double Impostor User/1-" + str(
                    user) + "-1-I-" + str(folds) + ".csv", "w") as file:
                file.write("".join(randomsample))

            folds += 1
        folds = 1
        user += 1

doubleImpostor()