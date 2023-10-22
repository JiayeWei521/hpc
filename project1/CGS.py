"""
Last update: 15.10.2023

CGS.py
Classical Gram-Schmidt (CGS) algorithm

@author: Jiaye Wei <jiaye.wei@epfl.ch>
"""

from mpi4py import MPI 
import numpy as np
from numpy.linalg import norm

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Function to perform classical Gram-Schmidt QR factorization
def classical_gram_schmidt(A):
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        v = A[:, j]
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v -= R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]

    return Q, R

def main_CGS(W):
    # Distribute the matrix among processors by using a block row distribution
    m, n = W.shape
    local_m = m // size
    local_W = np.empty((local_m, n), dtype=W.dtype)

    # Scatter data to each process
    comm.Scatter(W, local_W, root=0)

    # Perform QR factorization in parallel
    local_Q, local_R = classical_gram_schmidt(local_W)

    # Gather the results to construct the global Q and R matrices
    Q = np.zeros_like(W)
    R = np.zeros((n, n))

    comm.Gather(local_Q, Q, root=0)
    comm.Gather(local_R, R, root=0)

    # Print the global Q and R matrices on the root process
    if rank == 0:
        print("Matrix Q:")
        print(Q)
        print("Matrix R:")
        print(R)

