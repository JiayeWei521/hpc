"""
Created on 07-10-2023

exercise_template.py <--- The name of the source file goes here
<--- Description of the program goes here.

@author: Jiaye Wei <jiaye.wei@epfl.ch>
"""

from mpi4py import MPI 
import numpy as np

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
