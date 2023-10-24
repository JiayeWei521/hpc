"""
Created on 21.10.2023

CGS

@author: Jiaye Wei <jiaye.wei@epfl.ch>
"""

from mpi4py import MPI 
import numpy as np
from numpy.linalg import norm

from helpers import matrix_1
from helpers import loss_of_orthogonality

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

wt = MPI.Wtime() # We are going to time this

m = 5000
n = 60
local_size = int(m/size)

# Define
W = None
Q_orthogonality = None
Q = None
R = None
Qkreceived = None
QT = None
P = None
losses = np.zeros((n,1), dtype='d')

if rank == 0:
    W = matrix_1(m, n)
    Q_orthogonality = W
    Q = np.zeros((m,n), dtype = 'd')
    QT = np.zeros((n,m), dtype = 'd')
    Qkreceived = np.zeros((m, 1), dtype = 'd')
    R = np.zeros((n,n), dtype = 'd')
    P = np.eye( m, m, dtype = 'd')

# In here: we first build Q and then we build R
W_local = np.zeros((local_size, n), dtype = 'd')
q_local = np.zeros((local_size, 1), dtype = 'd')
QT_local = np.zeros((local_size, m), dtype = 'd')
P_local = np.zeros((local_size, m), dtype = 'd')
W_local = comm.bcast(W, root=0)
comm.Scatterv(P, P_local, root=0)

# For the first column
q_local = P_local @ W_local[:, 0]
# Normalize
comm.Gather(q_local, Qkreceived, root = 0)
if rank == 0:
    col = Qkreceived[:, 0] /norm(Qkreceived)
    Q[:, 0] = col
    QT[0, :] = col
    Q_orthogonality[:, 0] = col
    losses[0] = loss_of_orthogonality(Q_orthogonality)
comm.Barrier()

comm.Scatterv(QT, QT_local, root=0) # We have columns of Q (or rows of Qt)
# Start iterations in columns
for k in range(1, n):
    # We have already built column 0, so we move to column 1
    # First: we must build the projector P, using SUMMA
    localMult = 1/size * np.eye(m) - np.transpose(QT_local) @ QT_local
    comm.Reduce(localMult, P, op=MPI.SUM, root=0) # projector
    comm.Scatterv(P, P_local, root=0) # scatter rows of projector
    q_local = P_local @ W_local[:, k]
    # Normalize
    comm.Gather(q_local, Qkreceived, root=0)
    if rank == 0:
        col = Qkreceived[:, 0] /norm(Qkreceived)
        Q[:, k] = col
        QT[k, :] = col
        Q_orthogonality[:, k] = col
        losses[k] = loss_of_orthogonality(Q_orthogonality)
    comm.Barrier()
    comm.Scatterv(QT, QT_local, root=0)
    
# Compute R as R = Q^t*W
W_rows = np.zeros((local_size, n), dtype='d')
Q_local = np.zeros((local_size, n), dtype='d')
comm.Scatterv(W, W_rows, root=0)
comm.Scatterv(Q, Q_local, root=0)
localMult_R = np.transpose(Q_local) @ W_rows
R = comm.reduce(localMult_R, op=MPI.SUM, root=0)

# Print in rank==0
if (rank == 0):
    wt = MPI.Wtime() - wt
    print("Q: \n", Q)
    print("R: \n", R)
    print("Time taken: ", wt)
    print(losses)
    print(loss_of_orthogonality(Q))