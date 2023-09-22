#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 17:33:46 2023

@author: tommaso
"""

import numpy as np
import math 
from matplotlib import pyplot as plt

f1 = lambda x: (math.sqrt(1+x)-1)/x
f2= lambda x: 1/(math.sqrt(1+x)+1)
f3= lambda x: 0.5-x/8 +x**2/16-5*x**3/128

xvec=np.logspace(-16., -10., num=4)

f1vec=np.empty((0))
f2vec=np.empty((0))
f3vec=np.empty((0))
## Part A: for loop
for i in range(xvec.size):
    f1vec=np.append(f1vec,f1(xvec[i]))
    f2vec=np.append(f2vec,f2(xvec[i]))
    f3vec=np.append(f3vec,f3(xvec[i]))

## Part B: use vectorization
    
f1vecbis=((1+xvec)**(0.5)-1)/xvec
f2vecbis=1/((1+xvec)**(0.5)+1)
f3vecbis=0.5-xvec/8 +xvec**2/16-5*xvec**3/128


plt.plot(xvec, f1vecbis, color='red')
plt.plot(xvec, f2vecbis, color='blue')
plt.plot(xvec, f3vecbis, color='black')
plt.xscale('log')
plt.show()