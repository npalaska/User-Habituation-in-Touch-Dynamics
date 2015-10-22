import os
import sys
import numpy as np
from numpy import *

"""
print(os.path.abspath('Scripts'))
print(os.getcwd())
print(sys.argv[0])
x, y = os.path.split('..')
#current_dir =  os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.getcwd() + "/../..")
print (parent_dir+'\output files')
print((10-(10%3))/3)
"""

arr_genuine = genfromtxt('../../Raw Data/Truncated-Horizontal.csv',
                         dtype=float, delimiter=',')
rowno = 0
user = 1
i = 0
while user <= 31:
    data = arr_genuine[i:i + 915, :]
    np.savetxt('1-' + str(user) + '-1-G.csv', data, delimiter=',')
    user += 1
    i += 915
print(data)

