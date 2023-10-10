"""
Last update on 08-10-2023

2-4.py 
Collective communication - all-to-all and reduce

@author: Jiaye Wei <jiaye.wei@epfl.ch>

To execute the code, do (4 can be replaced by any number of processors):
mpiexec -n 4 python 2-3.py
"""

""" def alltoall():
    from mpi4py import MPI
    import numpy as np

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    senddata = rank * np.ones(size, dtype=int)

    recvdata = comm.alltoall(senddata) # type: ignore

    print("Process: ", rank, "sending: ", senddata, "receiving: ", recvdata)
 """

""" def reduction():
    from mpi4py import MPI
    import numpy as np

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    senddata = rank * np.ones(size, dtype=int)
    
    global_result1 = comm.reduce(senddata, op=MPI.SUM, root=0)
    global_result2 = comm.reduce(rank, op=MPI.MAX, root=0)

    # Print 
    print("process ", rank, "sending ", senddata)

    # Print the result on the root process
    if rank == 0:
        print("Reduction operation 1: ", global_result1, 
              "\n Reduction operation 2: ", global_result2)

reduction() """


def all_reduction():
    from mpi4py import MPI
    import numpy as np

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    senddata = rank * np.ones(size, dtype=int)
    
    global_result1 = comm.allreduce(senddata, op=MPI.SUM)
    global_result2 = comm.allreduce(rank, op=MPI.MAX)

    # Print 
    print("process ", rank, "sending ", senddata)

    # Print the result on the root process
    if rank == 0:
        print("All reduction operation 1: ", global_result1, 
              "\n",
              "All reduction operation 2: ", global_result2)
        
all_reduction()