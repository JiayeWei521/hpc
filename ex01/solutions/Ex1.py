#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 17:15:04 2023

@author: tommaso
"""

import numpy as np

row1=np.linspace(1,4,4,endpoint=True) #create first row
row2=np.linspace(5,8,4,endpoint=True) #create second row

M = np.vstack((row1, row2))


M[0,2] # extract the element in the first row, third column of A
M[1,]# extract the entire second row of A;
M[:,[0,1]] #extract the first two columns of A;
np.delete(M[1,:], 2, axis=0)  #extract the vector containing all the elements of the second row of A except for the third
            #element.