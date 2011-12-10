# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 16:32:24 2010

@author: musselle
"""

import pickle 
from matplotlib.pyplot import plot, figure, title
from numpy import mat, bmat
import numpy.matlib as npm 

def viewResult(N, variable = 0):  
    if variable == 0:
        filepath = './Run' + str(N) + '.dat'
        with open(filepath, 'r') as myfile:
            result = pickle.load(myfile)  
    else:
        filepath = './Run' + str(N) + '.dat'
        with open(filepath, 'r') as myfile:
            temp = pickle.load(myfile)
        result = temp[variable] 
    return result


if __name__ == '__main__':
    numRuns = 10
    
    e_qq_mat = npm.empty((10000,10))
    f_qq_mat = npm.empty((10001,10))
    
    for i in range(numRuns):    
        result = viewResult(i)
        figure(1)
        plot(result['e_qq'])
        e_qq_mat[:,i] = result['e_qq']
        
        figure(2)
        plot(result['f_qq'])
        f_qq_mat[:,i] = result['f_qq']
        
    figure(1)    
    title('Deviation of true tracked subspace') 
    figure(2)
    title('Deviation from orthonormality')    

