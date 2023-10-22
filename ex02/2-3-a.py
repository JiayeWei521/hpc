"""
Last update on 08.10.2023

2-3-a.py 
Collective communication - scattering and broadcasting
Part a

@author: Jiaye Wei <jiaye.wei@epfl.ch>

To execute the code, do (4 can be replaced by any number of processors):
mpiexec -n 4 python 2-3.py
"""

from mpi4py import MPI 
import numpy as np

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Define the vector
if rank == 0:
    vector = np.array([16, 62, 90, 21, 7, 53])
    P = np.eye(8, dtype='d')
else:
    vector = None
    P = None

local_size = int(8/size)
P_local = np.zeros((local_size, 8), dtype='d')

data1 = comm.bcast(vector, root=0)
comm.Scatterv(P, P_local, root=0)

print("rank: ", rank, "data2: ", P_local)