import numpy as np
from numpy import *
from numpy.linalg import inv
import MahalanobisDistance
from os import listdir

"""rawdata has n rows
## Create arr_groupvariance to store variance of each group of 5 rows (has n/5 rows)"""

arr_rawdata = genfromtxt('../../Raw Data/Truncated-Horizontal.csv',
                         dtype=float, delimiter=',')
arr_groupvariance = arange((len(arr_rawdata)) / 5, dtype=float32).reshape(
    (len(arr_rawdata)) / 5, 1)

"""Compute MDistance between Origin and the Variance of group of 5 rows.
 We use 4 most important features only
 Results in arr_groupvariance containing variances for all groups for all users [x, k]"""

row = -5
i_groups = 0
while row < len(arr_rawdata):
    arr_groups = array(arr_rawdata[row + 5:row + 10, [4, 8, 14, 19]])
    arr_reference = zeros_like(var(arr_groups, axis=0))
    while True:
        try:
            MDistance = MahalanobisDistance.ComputeMdistVector(arr_groups,
                                                               arr_reference)
            arr_groupvariance[i_groups, 0] = MDistance
            break
        except:
            print(0)
            break
    i_groups += 1
    row += 5

arr_allvariance = arr_groupvariance.reshape(31, ((len(arr_rawdata)) / 5) / 31)

np.savetxt("../../Output Files/Experiment20/MdistVariance1.csv",
           np.transpose(arr_allvariance), delimiter=",")
