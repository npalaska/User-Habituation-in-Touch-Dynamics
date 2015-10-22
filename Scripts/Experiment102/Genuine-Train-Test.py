import numpy as np
from numpy import *
import os

"""Break the genuine file into 3 sub-files"""

user = 1
while user <= 31:
    currentdirectory = os.getcwd()  # get the directory.
    parentdirectory = os.path.abspath(
        currentdirectory + "/../..")  # Get the parent directory(2 levels up)
    path = parentdirectory + '\Output Files\Experiment102\Genuine User\GenuineUser' + str(
        user)
    if not os.path.exists(path):
        os.makedirs(path)  # Crate the directory if one does not exist.

    arr_genuine = genfromtxt(
        '../../Output Files/Experiment30/1-' + str(user) + '-1-G.csv',
        dtype=float, delimiter=',')
    fileno = 1
    lengthoffile = (len(arr_genuine) - len(
        arr_genuine) % 3) / 3  # set the length of sub-file
    while fileno <= 3:
        x = arr_genuine[(
                            fileno - 1) * lengthoffile: fileno * lengthoffile]  # get the sub-files from complete file
        np.savetxt(
            "../../Output Files/Experiment102/Genuine User/GenuineUser" + str(
                user) + "/1-" + str(user) + "-1-G-" + str(fileno) + ".csv", x,
            delimiter=",")
        fileno += 1
    user += 1
