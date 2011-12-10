# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 10:50:07 2011

@author: -
"""

from ControlCharts import Tseries
from CUSUM import cusum 
import numpy as np
import matplotlib.pyplot as plt
from Frahst_v3_1 import FRAHST_V3_1
from SPIRIT import SPIRIT
from utils import analysis, QRsolveA, pltSummary, pltSummary2, GetInHMS
from plot_utils import plot_4x1
import scipy
import time
from PedrosFrahst import frahst_pedro


#===============================================================================
# Batch Parameters
#===============================================================================

initial_conditions = 20

num_streams = [3]           #a
SNR = [-3]                  #b
anomaly_length = [10]       #c
anomaly_magnitude = [10]    #d

baseLine = 15

#===============================================================================
# Initialise Data sets
#===============================================================================

count = 1
total = len(num_streams) * len(SNR) * len(anomaly_length) * len(anomaly_magnitude)

for a in num_streams:
    for b in SNR:
        for c in anomaly_length:
            for d in anomaly_magnitude:
                
                A = 0
                for e in range(initial_conditions):    
                    
                    # Seed random number generator 
                    np.random.seed(e)                    
                    
                    # Two ts that have anomalous shift 
                    s0 = Tseries(0)
                    s1 = Tseries(0)
                    s0.makeSeries([1,3,1],[100, c, 200 - c],[baseLine, baseLine, baseLine + d], 
                                  gradient = d/c, noise_type ='none')
                    s1.makeSeries([1,4,1],[200, c, 100 - c],[baseLine, baseLine, baseLine - d], 
                                  gradient = d/c, noise_type ='none')
                    # The rest of the ts
                    for i in range(2, a) :
                        name = 's'+ str(i)
                        vars()[name] = Tseries(0)
                        vars()[name].makeSeries([1],[300],[5], noise_type ='none')
                    
                    # concat into one matrix 
                    streams = scipy.c_[s0]
                    for i in range(1, a) :
                        streams = scipy.c_[streams, vars()[name] ]

                    # Concatonate to 3d array, timesteps x streams x initial condition 
                    if type(A) == int:
                        A = streams 
                    else:
                        A = np.dstack((A, streams))                        
                        
                # Calculate the noise                         
                Ps = np.sum(A[:,:,0] ** 2)
                Pn = Ps / (10. ** (b/10.))
                scale = Pn / (a * 300.)
                noise = np.random.randn(A.shape[0], A.shape[1], A.shape[2]) * np.sqrt(scale)
                A = A + noise
                    
                #Save
                filename = 'Data_N' + str(a) + '_SNR' + str(b) + '_L' + str(c) + '_M' + str(d) + '.npy'
                with open(filename, 'w') as savefile:
                    np.save(savefile, A)
                
                print 'Generated ' + str(count) + ' datasets out of ' + str(total)

