import numpy as np
from numpy import *
#import Truncation
import os
import MahalanobisDistance
from sklearn import preprocessing

def MDistMean():
    #Truncation.TrucateMatrix(5, '../../Raw Data/H.csv')

    """rawdata has n rows
    Create arr_groupsmean to store mahalanobis distance between each consecutive groups of 5 rows (has n/5 rows)"""

    # DataFile = genfromtxt('../../Raw Data/Truncated-Horizontal.csv', dtype=float, delimiter=',')
    Data = genfromtxt('../../Raw Data/Truncated-Horizontal.csv', dtype=float,
                          delimiter=',')
    DataFile = preprocessing.scale(Data)
    arr_groupsmean = arange((len(DataFile)) / 5, dtype=float32).reshape(
        (len(DataFile)) / 5, 1)

    """Compute MDistance between Mean of group 1 and the Mean of another group of 5 rows.
     We use 4 most important features only
     Results in arr_groupsmean containing mdist of means for two groups for all users [x, k]"""
    i_row = -5
    k_mdist = 0
    while (i_row < len(DataFile)):
        while True:
            try:
                arr_group1 = array(DataFile[i_row + 5:i_row + 10, 3:7])
                arr_group2 = array(DataFile[i_row + 10:i_row + 15, 3:7])

                mdist = MahalanobisDistance.ComputeMDistMetrix(arr_group1,
                                                               arr_group2)
                arr_groupsmean[k_mdist, 0] = mdist
                print(arr_groupsmean[k_mdist, 0])
                break
            except:
                print(0)
                break

        i_row += 5
        k_mdist += 1

    print(os.listdir('../..'))
    """Reshape arr_groupsmean to become [31, k/31]"""
    mdistofmeans = np.transpose(arr_groupsmean.reshape(31, 183))
    np.savetxt("../../Output Files/Experiment10/ArrayMeanScaled.csv", mdistofmeans,
               delimiter=",")

#################################################################################################################

# Finding the correlation between Mahalanobis distance in output file.

def correlation():
    data = genfromtxt('../../Raw Data/Truncated-Horizontal.csv', dtype=float, delimiter=',')
    x = data[500:1500, 3:]
    y = data[2, 3:]
    c = np.corrcoef(x)
    print(c[:, 0])
    X = [0,0,1,1,0]
    Y = [1,1,0,1,1]
    print(np.corrcoef(X,Y))

MDistMean()