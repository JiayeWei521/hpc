from mpi4py import MPI
import numpy as np

# Function to perform Modified Gram-Schmidt QR factorization
def parallel_modified_gram_schmidt_qr(A):
    m, n = A.shape
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        # Step 1: Orthogonalization
        v = A[:, j].copy()
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], v)
            comm.Allreduce(MPI.IN_PLACE, R[i, j], op=MPI.SUM)
            v -= R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]
        comm.Allreduce(MPI.IN_PLACE, Q[:, j], op=MPI.SUM)

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

    # Perform Modified Gram-Schmidt QR factorization in parallel
    local_Q, local_R = parallel_modified_gram_schmidt_qr(local_A)

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
