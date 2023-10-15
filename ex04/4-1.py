"""
Last update on 

exercise_template.py <--- The name of the source file
<--- Description of the program

@author: Jiaye Wei <jiaye.wei@epfl.ch>

To execute the code, do (4 can be replaced by any number of processors):
mpiexec -n 4 python script.py
"""

import numpy as np
from numpy.linalg import norm
import time

# Sequential implementation of QR algorithm (just get Q)

def matrix_vector_multiplication(A, x):
    m = A.shape[0]



