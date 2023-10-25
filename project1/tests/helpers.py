"""
Created on 17.10.2023

All the matrices used for testing the performance of our algorithms

@author: Jiaye Wei <jiaye.wei@epfl.ch>
"""

from math import sin, cos
import numpy as np

def create_communicator():
    return 0

def func_1(x, y):
    numerator = sin(10*(x+y))
    denominator = cos(100*(y-x)) + 1.1
    ratio = float(numerator/denominator)
    return ratio

def matrix_1(m ,n):
    C = np.zeros((m ,n), dtype = 'd')
    for i in range(m):
        for j in range(n):
            C[i][j] = func_1(i/(m-1), j/(n-1))
    return C

def test_matrix_1(m, n):
    return np.fromfunction(function = lambda i,j: func_1(i/(m-1), j/(n-1)), 
                           shape = (m, n), 
                           dtype=float)

def loss_of_orthogonality(Q):
    n = Q.shape[1]
    difference = np.eye(n, dtype='d') - np.transpose(Q) @ Q
    return np.linalg.norm(difference, ord=2)

def compute_condition_number():
    return 0

def is_positive_definite(M):
    is_symmetric = np.allclose(M, M.T)

    if is_symmetric:
        eigenvalues = np.linalg.eigvals(M)
        if all(eigenvalues > 0):
            True
        else:
            False
    else:
        False