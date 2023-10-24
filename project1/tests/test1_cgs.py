"""
Created on 21.10.2023

CGS

@author: Jiaye Wei <jiaye.wei@epfl.ch>
"""

from mpi4py import MPI 
import numpy as np
from numpy.linalg import norm, cond
from helpers import matrix_1
from helpers import loss_of_orthogonality

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


m = 50000
n = 600
local_m = int(m/size)
local_n = int(n/size)

# Define
W = None
Q = None
R = None
Qkreceived = None # the result of k-th iteration, without normalization
QT = None
P = None
losses = np.zeros((n, 1), dtype='d')

if rank == 0:
    W = matrix_1(m, n)
    
    Q = np.zeros((m, n), dtype = 'd')
    QT = np.zeros((n, m), dtype = 'd')
    R = np.zeros((n, n), dtype = 'd')
    
    Qkreceived = np.zeros((m, 1), dtype='d')
    P = np.eye(m, dtype = 'd')

wt = MPI.Wtime()

q_local = np.zeros((local_m, 1), dtype = 'd')
QT_local = np.zeros((local_m, m), dtype = 'd')
P_local = np.zeros((local_m, m), dtype = 'd')

W_local = comm.bcast(W, root=0)
comm.Scatterv(P, P_local, root=0)

# For the first column
q_local = P_local @ W_local[:, 0]
# Normalize
comm.Gather(q_local, Qkreceived, root = 0)
if rank == 0:
    col = Qkreceived[:,0]/norm(Qkreceived[:,0])
    Q[:, 0] = col
    QT[0, :] = np.transpose(col)
comm.Barrier()

comm.Scatterv(QT, QT_local, root=0) # We have columns of Q (or rows of Qt)
# Start iterations in columns
for k in range(1, n):
    # We have already built column 0, so we move to column 1
    # First: we must build the projector P, using SUMMA
    # localResult = 1/size * np.eye(m, dtype='d') - np.transpose(QT_local) @ QT_local
    # P = comm.reduce(localResult, op=MPI.SUM, root=0)
    if rank == 0:
        P = np.eye(m, dtype='d') - np.transpose(QT) @ QT 
    comm.Scatterv(P, P_local, root = 0)
    q_local = P_local @ W_local[:, k]
    
    # Normalize
    comm.Gather(q_local, Qkreceived, root=0)
    if rank == 0:
        col = Qkreceived[:,0]/norm(Qkreceived[:,0])
        Q[:, k] = col
        QT[k, :] = np.transpose(col)
    comm.Barrier()
    
    comm.Scatterv(QT, QT_local, root=0)
    
# Compute R as R = Q^t*W
W_rows = np.zeros((local_m, n), dtype='d')
Q_local = np.zeros((local_m, n), dtype='d')
comm.Scatterv(W, W_rows, root=0)
comm.Scatterv(Q, Q_local, root=0)
localMult_R = np.transpose(Q_local) @ W_rows
R = comm.reduce(localMult_R, op=MPI.SUM, root=0)

# Print in rank==0
if (rank == 0):
    wt = MPI.Wtime() - wt
    # print("Q: \n", Q)
    # print("R: \n", R)
    print("Time taken: ", wt)
    # print(losses)
    print("Loss of orthogonality: ", loss_of_orthogonality(Q))
    print("Condition number: ", cond(Q))