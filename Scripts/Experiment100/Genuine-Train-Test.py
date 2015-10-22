import numpy as np
from numpy import *
import os

"""Break the genuine file into 6 sub-files"""
def DataSplit(numberofblocks):
    user = 1
    while user <= 31:
        currentdirectory = os.getcwd()  # get the directory.
        parentdirectory = os.path.abspath(
            currentdirectory + "/../..")  # Get the parent directory(2 levels up)
        path = parentdirectory + '/Output Files/Experiment100/'+str(numberofblocks)+'splits_up_ordered/Genuine ' \
                                                                                    'User/GenuineUser' + str(
            user)
        if not os.path.exists(path):
            os.makedirs(path)  # Crate the directory if one does not exist.

        arr_genuine = genfromtxt(
            '../../Output Files/Experiment30/Up Truncation/1-' + str(user) + '-1-G.csv',
            dtype=float, delimiter=',')

        lengthoffile = (len(arr_genuine) - len(
            arr_genuine) % numberofblocks) / numberofblocks  # set the length of sub-file
        arr_baseindices = np.arange(start=0, stop=(numberofblocks+1)*lengthoffile, step=lengthoffile)

        fileno = 1
        while fileno <= numberofblocks:
            x = arr_genuine[arr_baseindices[fileno-1]: arr_baseindices[fileno]]
            #x = arr_genuine[(fileno - 1) * lengthoffile: fileno * lengthoffile]  # get the sub-files from
                                # complete file
            np.savetxt(
                "../../Output Files/Experiment100/"+str(numberofblocks)+"splits_up_ordered/Genuine User/GenuineUser" +
                str(
                    user) + "/1-" + str(user) + "-1-G-" + str(fileno) + ".csv", x,
                delimiter=",")
            fileno += 1
        user += 1

DataSplit(12)

#x = arr_genuine[arr_baseindices[i]: arr_baseindices[i+1]]

#01 12 23 34 45 56 67 78 89 99 910 1011