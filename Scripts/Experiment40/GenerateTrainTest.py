import numpy as np
from numpy import *
import os

"""Break the genuine file into 3 sub-files"""
"""Generate Train and Test Files with alternate strokes for every file"""

user = 1
while user <= 31:
    currentdirectory = os.getcwd()  # get the current directory.
    parentdirectory = os.path.abspath(
        currentdirectory + "/../..")  # Get the parent directory(2 levels up)
    path = parentdirectory + '/Output Files/Experiment40/GenuineUser' + str(
        user) + ''
    if not os.path.exists(path):
        os.makedirs(path)  # Create the folder if one does not present.
    arr_genuine = genfromtxt(
        '../../Output Files/Experiment30/1-' + str(user) + '-1-G.csv',
        dtype=float, delimiter=',')
    fileno = 1
    lengthoffile = len(arr_genuine) / 3  # set the length of sub-file
    while fileno <= 3:
        x = arr_genuine[(
                            fileno - 1) * lengthoffile: fileno * lengthoffile]  # get the sub-files from complete file
        np.savetxt("../../Output Files/Experiment40/GenuineUser" + str(
            user) + "/1-" + str(user) + "-1-G-" + str(fileno) + ".csv", x,
                   delimiter=",")
        fileno += 1

    fileno = 1
    while fileno <= 3:
        with open("../../Output Files/Experiment40/GenuineUser" + str(
                user) + "/1-" + str(user) + "-1-G-" + str(
                fileno) + ".csv") as file:
            for row, line in enumerate(
                    file):  # enumerate function will look through each line in file
                if row % 2 == 0:
                    savefile = open(
                        '../../Output Files/Experiment40/GenuineUser' + str(
                            user) + '/1-' + str(user) + '-1-G-Train-' + str(
                            fileno) + '.csv', 'a')
                    # This chooses the rows divisible by 2 and save it as a train file else save it as a test file.
                    savefile.write(line)
                    savefile.close()
                else:
                    savefile = open(
                        '../../Output Files/Experiment40/GenuineUser' + str(
                            user) + '/1-' + str(user) + '-1-G-Test-' + str(
                            fileno) + '.csv', 'a')
                    savefile.write(line)
                    savefile.close()
        fileno += 1
    user += 1

