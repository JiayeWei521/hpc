from mpi4py import MPI 
import numpy as np
from numpy.linalg import cholesky
from numpy.linalg import inv
from helpers import matrix_1, loss_of_orthogonality, is_positive_definite

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

wt = MPI.Wtime()

m = 5000
n = 60
local_size = int(m/size)

epsilon = 1e-4  # Small positive constant to enforce positive definiteness

# Define
W = None
Q = None
R = None
R_inv = None
WtW = None # the product W^t@W
loss = None

if rank == 0:
    W =  matrix_1(m, n)
    Q = np.zeros((m, n), dtype = 'd')
    R = np.zeros((n, n), dtype = 'd')
    WtW = np.zeros((n, n), dtype='d')
    losses = np.zeros((n, 1), dtype='d') # array storing losses of orthogonality
    
# Compute the product W^t @ W using SUMMA
W_local = np.zeros((local_size, n), dtype = 'd')

comm.Scatterv(W, W_local, root=0) # scattering W into blocks of rows
local_product = np.transpose(W_local) @ W_local # locally multiply
comm.Reduce(local_product, WtW, op=MPI.SUM, root=0) # sum-up the local product
comm.Barrier()

# Cholesky factorization to Wt*W to get R
if rank == 0:
    # if is_positive_definite(WtW) is False:
    #     WtW = epsilon * np.eye(n) + WtW
    R = np.transpose(cholesky(WtW)) # take transpose here to be consistent with python standard of Cholesky
    R_inv = inv(R) # compute the inverse of R
    Q = W @ R_inv # compute Q = W @ R^-1

    loss = loss_of_orthogonality(Q)
    
    wt = MPI.Wtime() - wt
    print("Q: \n", Q)
    print("R: \n", R)
    print("Time taken = ", wt)
    print("Loss of orthogonality = ", loss)
