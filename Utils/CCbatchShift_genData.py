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

num_streams = [3]           #n
SNR = [10]                  #snr
anomaly_length = [10]       #l
anomaly_magnitude = [1]    #m

baseLine = 0.0

anomaly_type = 'shift'

#===============================================================================
# Initialise Data sets
#===============================================================================

count = 1
total = len(num_streams) * len(SNR) * len(anomaly_length) * len(anomaly_magnitude)

for n in num_streams:
    for snr in SNR:
        for l in anomaly_length:
            for m in anomaly_magnitude:
                
                A = 0
                for e in range(initial_conditions):    
                    
                    # Seed random number generator 
                    np.random.seed(e)                    
                    
                    # Two ts that have anomalous shift 
                    s0 = Tseries(0)
                    s1 = Tseries(0)
                    if anomaly_type == 'peak':
                        s0.makeSeries([1,3,4,1], [100, l/2, l/2, 200 - l], [baseLine, baseLine, baseLine + m, baseLine], 
                                  gradient = float(m)/float(l/2), noise_type ='none')
                        s1.makeSeries([1,4,3,1],[200, l/2, l/2, 100 - l],[baseLine, baseLine, baseLine - m, baseLine], 
                                  gradient = float(m)/float(l/2), noise_type ='none')
                    elif anomaly_type == 'shift':
                        s0.makeSeries([1,3,1],[100, l, 200 - l],[baseLine, baseLine, baseLine + m], 
                                  gradient = float(m)/float(l), noise_type ='none')
                        s1.makeSeries([1,4,1],[200, l, 100 - l],[baseLine, baseLine, baseLine - m], 
                                  gradient = float(m)/float(l), noise_type ='none')
                    # The rest of the ts
                    for i in range(2, n) :
                        name = 's'+ str(i)
                        vars()[name] = Tseries(0)
                        vars()[name].makeSeries([1], [300], [baseLine], noise_type ='none')
                    
                    # concat into one matrix 
                    streams = scipy.c_[s0]
                    for i in range(1, n) :
                        streams = scipy.c_[ streams, vars()['s'+ str(i)] ]

                    # Concatonate to 3d array, timesteps x streams x initial condition 
                    if type(A) == int:
                        A = streams 
                    else:
                        A = np.dstack((A, streams))                        
                    
#                B = A[:,:,0].copy()  
#                # Calculate the noise      
#                if anomaly_type == 'peak':                     
#                    B = B - baseLine
#                elif anomaly_type == 'shift' :                 
#                    # Correct for baseLine
#                    B = B - baseLine 
#
#                    # Correct for first anomalous stream                    
#                    B[100+l:, 0] = B[100+l:, 0] - m 
#                    
#                    # Correct for final third - second anomalous stream
#                    B[200+l:300, 1] = B[200+l:300, 1] + m  

                B = A                    
                Ps = np.sum((B ** 2))
                Pn = Ps / (10. ** (snr/10.))
                scale = Pn / (n * 300.)
                noise = np.random.randn(A.shape[0], A.shape[1], A.shape[2]) * np.sqrt(scale)
                A = A + noise
                    
                #Save
                filename = 'Data_N' + str(n) + '_SNR' + str(snr) + '_L' + str(l) + '_M' + str(m) + '.npy'
                with open(filename, 'w') as savefile:
                    np.save(savefile, A)
                
                print 'Generated ' + str(count) + ' datasets out of ' + str(total)

plt.figure()
plt.plot(A[:,:,0])
plt.title('SNR = ' + str(SNR))
plt.show()