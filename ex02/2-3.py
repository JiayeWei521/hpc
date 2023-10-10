"""
Last update on 08-10-2023

2-3.py 
Collective communication - scattering and broadcasting

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
    vector = np.array([16, 62, 97, 25])
else:
    vector = None

data1 = comm.bcast(vector, root=0)
data2 = comm.scatter(vector, root=0)  # type: ignore

print("rank: ", rank, "data1: ", data1, "data2: ", data2)