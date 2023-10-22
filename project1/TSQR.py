"""
Last update on 14.10.2023

TSQR.py
 (TSQR) algorithm

@author: Jiaye Wei <jiaye.wei@epfl.ch>
"""

from mpi4py import MPI
import numpy as np

# Function to perform Householder transformation for a given vector
def householder(x):
    v = x.copy()
    v[0] += np.sign(x[0]) * np.linalg.norm(x)
    v /= np.linalg.norm(v)
    return v

# Function to perform parallel Householder-based QR factorization
def parallel_householder_qr(A):
    m, n = A.shape
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    Q = np.eye(m)
    R = A.copy()

    for k in range(n):
        # Each process operates on its own part of the matrix
        local_R = R[k:, k:]
        local_v = local_R[:, 0].copy()

        # Perform Householder transformation on the local part
        local_v = householder(local_v)

        # Apply Householder transformation to the entire matrix
        global_v = np.zeros(m)
        comm.Allreduce(local_v, global_v, op=MPI.SUM)

        for i in range(m):
            for j in range(k, n):
                R[i, j] -= 2 * global_v[i] * global_v[j - k] * R[k, j - k]

        # Update the Q matrix
        local_Q = np.eye(m - k) - 2 * np.outer(global_v, global_v)
        Q[k:, k:] = np.dot(Q[k:, k:], local_Q)

    return Q, R

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Define the matrix A (you can replace this with your own data)
    A = np.random.rand(6, 4)

    # Split the matrix among processes
    local_m = A.shape[0] // comm.Get_size()
    local_A = np.empty((local_m, A.shape[1]), dtype=A.dtype)

    # Scatter data to each process
    comm.Scatter(A, local_A, root=0)

    # Perform Householder-based QR factorization in parallel
    local_Q, local_R = parallel_householder_qr(local_A)

    # Gather the results to construct the global Q and R matrices
    Q = None
    R = None

    if rank == 0:
        Q = np.zeros_like(A)
        R = np.zeros((A.shape[1], A.shape[1]))

    comm.Gather(local_Q, Q, root=0)
    comm.Gather(local_R, R, root=0)

    # Print the global Q and R matrices on rank 0
    if rank == 0:
        print("Matrix Q:")
        print(Q)
        print("Matrix R:")
        print(R)
