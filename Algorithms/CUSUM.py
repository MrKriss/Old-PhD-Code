# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 13:31:19 2010

======================
All CUSUM RElated Code 
======================

@author: musselle
"""

# pylint: disable-msg=E0611
from numpy import arange, exp, log, zeros, sqrt, inf, pi 
import numpy as np 
from ControlCharts import Tseries
import matplotlib.pyplot as plt


def gauss_kernel(inputs):
    "Gaussian Kernal Function"
    constant = 1 / sqrt(2 * pi)
    exponent =  - (abs(inputs) **2) / 2 
    return constant * exp(exponent)

def cusum(inputs):
    "My version of CUSUM"
    
    # Set parameters
    N = len(inputs)

    # initialise 
    s = zeros((N, N + 1))   # Makes S(i,j+1) all 0
    sMax = -inf  
    u = zeros(N)
    v = zeros((N, N))
    
    for j in range(N-1, 0, -1):        #  n:2 in matlab Note: 0 index
        
        # Calculate Uj
        for k in range(j-1, N):     #  k = j:n in matlab
            u[j] = u[j] + gauss_kernel(inputs[j] - inputs[k])
        
        # Think about this a bit more......
        v[j, j] = gauss_kernel(0.0) # This is effectiveky w(x(N)-x(N))
        # Used as first v_i+1,j in recursion
        
        for i in range(j-2,0,-1): # for i = j-1 : 1
            
            # Calculate V_ij
            v[i,j] = v[i+1,j] + gauss_kernel(inputs[j] - inputs[i])
            
            s[i,j] = s[i,j+1] + log(u[j]) + log((j+1)-(i+1)) -log(v[i,j]) -log(N-j+2)
            
            if s[i, j] > sMax :     
                sMax = s[i, j]
                row = i                 # Start point of first sub window
                col = j                 # Start point of second sub window        
    
    return [sMax, row, col]

#def cusum_neive(inputs): 
    
    ## Set parameters
    #N = len(inputs)

    ## initialise 
    #s = zeros((N, N + 1))   # Makes S(i,j+1) all 0
    #sMax = -inf  
    #u = zeros(N)
    #v = zeros((N, N))
    
    #for j in range(N-1,1,-1): # j = xN:x2
        #for i in range(j-1, 0): # i = xj-1:x1

            #for l in range(j,N+1): # l = j:N
                
                #for k in range(l,N+1): # k = j:N
                    ## Calculate uj
                    #u[l] = u[l] + gauss_kernel(inputs[l] - inputs[k]) 
            
                #for k in range(i,l): # k = i:l
                    ## calculate vij
                    #v[i,l] = v[i,l] + gauss_kernel(inputs[l] - inputs[k])
                
                #upper_scale = 1./ (N-l+1)
                #lower_scale = 1./ (l-i) 
                #s[i,j] = s[i,j] + log( upper_scale * u[l]) - log(lower_scale * v[i,l])
    
    #return s, u, v
    
def cusum_alg(stream, win_sizes):
    """ Main algorithm Function 
    Employs the sliding window method and performs MB-CUSUM at each window instance
    """
    # Pre allocate memory 
    score = []
    tag = np.zeros((len(stream)))
    
    sMaxScore = []  # Alternative for matlab Cell, a list 
    tagMetric = []   
    
    #=======================#
    # Sliding window Method #
    #=======================#
    
    # for each window size
    for w in range(len(win_sizes)):  # (0,...,N-1) --> 1:size(win_sizes,2)
        
        # slide along input stream 
        for i in range(len(stream)-win_sizes[w]+1): #(1:size(stream,2)-win_sizes(w)+1)
            
            #Update inputs 
            myInputs = stream[i:i + win_sizes[w]]
            # Run Cusum 
            sMax , win_i_start, win_j_start = cusum(myInputs)
            # Note change point : start of win_j
            change_point_index = i + win_j_start 
            # Sum up with previous sMax change points 
            tag[change_point_index] = tag[change_point_index] + sMax
            
            score.append(sMax)
    
        sMaxScore.append(score) # list of Smax for each win_sizes
        score = []                          # reset score
        tagMetric.append(tag)   # list of cumulative sums of Smax
        tag = np.zeros((len(stream)))          # reset tag for next win size
    
    # Plot results 
    
    x = range(1, len(stream))
    
    plt.figure()
    plt.subplot(311)
    plt.plot(stream)
    plt.title('Input Stream')
    
    plt.subplot(312)
    for i in range(len(win_sizes)):                    #   i = 1:size(win_sizes,2)   
        plt.plot(sMaxScore[i])                           #'Color', ColOrd(i,:))
    plt.title('Smax Score')
    
    plt.subplot(313)
    for i in range(len(win_sizes)):
        plt.plot(tagMetric[i])
    plt.title('Tag')
    
    
if __name__ == '__main__':
    
    series = Tseries(0)
    series.makeSeries([1,2,3,4] , [100,100,100,100], noise = 4)
    
    cusum_alg(series, [30, 60, 90])
