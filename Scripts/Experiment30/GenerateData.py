import numpy as np
from numpy import *

"""The following algorithm will generate Genuine Data for each User.
or in other words Individual User Data Set"""

def bottom():
    arr_genuine = genfromtxt('../../Raw Data/Truncated-Horizontal.csv',
                             dtype=float, delimiter=',')
    rowno = 0
    for user in range(31):
        while np.all(arr_genuine[rowno, 0:3] == [1, user, 1]):
            with open("../Raw Data/Truncated-Horizontal.csv", "r") as source1:
                lines = [line for line in source1]
            with open('../../Output Files/Experiment30/1-' + str(
                    user) + '1-G' + '.csv', 'a') as DstFile1:
                DstFile1.write("".join(str(lines[rowno])))
            rowno += 1

    """After the first run if you want to run  it again then delete all the files
    otherwise data will get added not overwritten."""


def up():
    arr_genuine = genfromtxt('../../Raw Data/Truncated-up-Horizontal.csv',
                             dtype=float, delimiter=',')
    rowno = 0
    user = 1
    while rowno in range(31):
        array = arr_genuine[rowno*915:(rowno+1)*915]
        np.savetxt("../../Output Files/Experiment30/Up Truncation/1-"+str(user)+"-1-G.csv", array, delimiter=",")
        user += 1
        rowno += 1
    a = 1


up()