#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 18:02:44 2023

@author: tommaso
"""

import numpy as np
import time
from matplotlib import pyplot as plt

m=10
n=10

M = np.random.randint(10, size=(m, n)) #generate random matrix of integers between 0 and 9
v= np.random.randint(10, size=(n, 1))

result=np.zeros((m,1))

def multiplication_matrix_vector(A,b):
    [m,n]=A.shape
    output=np.zeros((m,1))
    for i in range(0,m): #matrix vector multiplication with for loop
        sum=0
        for j in range(0,n):
            sum=sum+A[i,j]*b[j]
        output[i]=sum
    return output

result=multiplication_matrix_vector(M,v)

product=M.dot(v) #implemented library

#print(product) #compare outputs
#print(result)


#compare performances

#size=np.logspace(1,3,3,endpoint=True)
size=np.array([10,100,500,1000,3000])
timeloop=np.zeros((size.size,1))
timenp=np.zeros((size.size,1))

for l in range(size.size):
    n=int(size[l])
    m=int(size[l])
    M = np.random.randint(10, size=(m, n)) #generate random matrix of integers between 0 and 9
    v= np.random.randint(10, size=(n, 1))
    result=np.zeros((m,1))
    start = time.time()
    result=multiplication_matrix_vector(M,v)
    end = time.time()
    timeloop[l]=end-start
    start = time.time()
    product=M.dot(v)
    end=time.time()
    timenp[l]=end-start

plt.plot(size,timeloop, color='red')
plt.plot(size,timenp, color='blue')
plt.show()
