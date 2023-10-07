"""
Created on 07-10-2023

3-2.py:
Testing what comm.Split() does

@author: Jiaye Wei <jiaye.wei@epfl.ch>
"""

from mpi4py import MPI 
import numpy as np

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Defining the subset assignment
if rank%2 == 0:
    color1 = 0
else:
    color1 = 1

if int(rank/2) == 0:
    color2 = 0
else:
    color2 = 1

new_comm1 = comm.Split(color=color1, key=rank)
new_rank1 = new_comm1.Get_rank()
new_size1 = new_comm1.Get_size()

new_comm2 = comm.Split(color=color2, key=rank)
new_rank2 = new_comm2.Get_rank()
new_size2 = new_comm2.Get_size()

print("Original rank: ", rank, "\n"
      "color1: ", color1, "\n"
      "new rank1: ", new_rank1, "\n"
      "color2: ", color2, "\n"
      "new rank 2: ", new_rank2)