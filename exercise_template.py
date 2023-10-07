"""
Created on 07-10-2023

@author: Jiaye Wei <jiaye.wei@epfl.ch>
"""

from mpi4py import MPI 
import numpy as np

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
