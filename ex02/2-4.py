from mpi4py import MPI
import numpy as np

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

senddata = rank * np.ones(size, dtype=int)

recvdata = comm.alltoall(senddata) # type: ignore

print("Process: ", rank, "sending: ", senddata, "receiving: ", recvdata)