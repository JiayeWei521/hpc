"""
Created on 17.10.2023

All the matrices used for testing the performance of our algorithms

@author: Jiaye Wei <jiaye.wei@epfl.ch>
"""

from math import sin, cos
import numpy as np

def func_1(x, mu):
    numerator = sin(10 * (mu + x))
    denominator = cos(100 * (mu - x)) + 1.1
    return numerator/denominator


def matrix_1(m ,n):
    C = np.zeros((m ,n), dtype = 'd')
    for i in range(m):
        for j in range(n):
            C[i][j] = func_1(i/(m-1), j/(n-1))
    return C

