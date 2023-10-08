"""
Last update on 08-10-2023

2-1.py 
Reminder of a simple MPI code in Python

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

b = np.array([1, 2, 3, 4]) 
c = np.array([5, 6, 7, 8]) 
a = np.zeros_like(b) 
d = np.zeros_like(b)

if rank == 0:
    for i in range(4):
        a[i] = b[i] + c[i]
    comm.Send(a, dest = 1, tag = 77) 
else:
    comm.Recv(a, source = 0, tag = 77) 
    for i in range(4):
        d[i] = a[i] + b[i]

print("I am rank = ", rank)
print("d: ", d)
