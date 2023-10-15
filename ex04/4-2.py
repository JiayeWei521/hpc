"""
Last update on 10-10-2023

4-1.py 
<--- Description of the program

@author: Jiaye Wei <jiaye.wei@epfl.ch>

To execute the code, do (4 can be replaced by any number of processors):
mpiexec -n 4 python script.py
"""

from mpi4py import MPI 
import numpy as np

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


m = 3*size
n = 2*size
local_size = int(m/size)

# Define
W = None
Q = None
QT = None
P = None

if rank == 0:
    W = np.arange(1, )