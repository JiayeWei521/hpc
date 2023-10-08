"""
Last update on 08-10-2023

2-3-b.py 
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

# Function to perform matrix−vector multiplication 
def matrix_vector_multiplication(matrix, vector):
    result = np.dot(matrix, vector) 
    return result

# Define the matrix and vector 
A = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
v = np.array([7, 8, 9, 10])

# Check if the matrix and vector dimensions are compatible
if A.shape[1] != len(v):
    if rank == 0:
        print("Matrix and vector dimensions are not compatible for multiplication.")
elif size != A.shape[0]:
    print("Number of processors used doesn't match with the number of rows in matrix.")
else:
    # Split the work among processes
    print("rank: ", rank)
    local_row = comm.scatter(A, root=0)

    # Compute local multiplication
    local_result = matrix_vector_multiplication(local_row, v)

    # Gather results on the root process
    global_result = comm.gather(local_result, root=0)