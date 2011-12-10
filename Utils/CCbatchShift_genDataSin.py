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
from utils import QRsolveA, pltSummary, pltSummary2, GetInHMS
from plot_utils import plot_4x1
import scipy
import time
from PedrosFrahst import frahst_pedro

#===============================================================================
# Batch Parameters
#===============================================================================
# Data Sets 
#Fixed
baseLine = 5.0
baseLine_MA_window = 15
period = 5
amp = 0.1 
initial_conditions = 1      # i

# Varied
num_streams = [3]            # n
SNR = [0]                  # snr
anomaly_length = [10]       # l
anomaly_magnitude = [1]    # m
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
                    s0lin = Tseries(0) # linear component 
                    s0sin = Tseries(0) # Sine component 
                    s1lin = Tseries(0)
                    s1sin = Tseries(0)
                    
                    if anomaly_type == 'peak':
                        
                        s0lin.makeSeries([1,3,4,1], [100, l/2, l/2, 200 - l], [baseLine, baseLine, baseLine + m, baseLine], 
                                  gradient = float(m)/float(l/2), noise_type ='none')
                        s0sin.makeSeries([2], [300], [0.0], 
                                  amp = amp , period = period, noise_type ='none')
                        
                        # sum sin and linear components to get data stream                         
                        s0 = np.array(s0lin) + np.array(s0sin)                                    
                                  
                        s1lin.makeSeries([1,4,3,1],[200, l/2, l/2, 100 - l],[baseLine, baseLine, baseLine - m, baseLine], 
                                  gradient = float(m)/float(l/2), noise_type ='none')
                        s1sin.makeSeries([2], [300], [0.0], 
                                  amp = amp , period = period, noise_type ='none')
                                  
                        # sum sin and linear components to get data stream                                   
                        s1 = np.array(s1lin) + np.array(s1sin)                          
                                  
                    elif anomaly_type == 'shift':
                                  
                        s0lin.makeSeries([1,3,1], [100, l, 200 - l], [baseLine, baseLine, baseLine + m], 
                                  gradient = float(m)/float(l), noise_type ='none')
                        s0sin.makeSeries([2], [300], [0.0], 
                                  amp = amp, period = period, noise_type ='none')
                        
                        # sum sin and linear components to get data stream                         
                        s0 = np.array(s0lin) + np.array(s0sin)                                    
                                  
                        s1lin.makeSeries([1,4,1],[200, l, 100 - l],[baseLine, baseLine, baseLine - m], 
                                  gradient = float(m)/float(l), noise_type ='none')
                        s1sin.makeSeries([2], [300], [0.0], 
                                  amp = amp , period = period, noise_type ='none')
                                  
                        # sum sin and linear components to get data stream                                                           
                        s1 = np.array(s1lin) + np.array(s1sin)                                      
                                  
                                 
                    # The rest of the ts
                    for k in range(2, n) :
                        name = 's'+ str(k)
                        vars()[name] = Tseries(0)
                        vars()[name].makeSeries([2], [300], [0.0], 
                                  amp = amp , period = period, noise_type ='none')
                    
                    # Concat into one matrix 
                    S = scipy.c_[s0]
                    for k in range(1, n) :
                        S = scipy.c_[ S, vars()['s'+ str(k)] ]

                    # Concatonate to 3d array, timesteps x streams x initial condition 
                    if type(A) == int:
                        A = S 
                    else:
                        A = np.dstack((A, S))  
                
                if initial_conditions == 1:
                    B = A[:,:].copy()  
                else:                        
                    B = A[:,:,0].copy()  
                
                # Calculate the noise      
                if anomaly_type == 'peak':                     
                    B = B - baseLine
                elif anomaly_type == 'shift' : 
#===============================================================================
# Calculate Moving Baseline if shift                
#===============================================================================
                    baseLineMatrix = np.zeros(B.shape)
                    for k in range(n):   # for each stream
                        cnt = 0
                        window = np.ones((baseLine_MA_window)) * baseLine
                        for data in B[:,k]: 
                            # Advance window 
                            window[:-1] = window[1:]
                            # Add new value 
                            window[-1] = data
                            # Calculate average 
                            ave = window.sum() / float(len(window))
                            baseLineMatrix[cnt,k] = ave
                            cnt += 1
                            
                    # Correct for baseLine
                    B = B - baseLineMatrix
                    
                Ps = np.sum(B ** 2)
                Pn = Ps / (10. ** (snr/10.))
                scale = Pn / (n * 300.)        
                # Calculate the noise  
                if initial_conditions == 1:
                    noise = np.random.randn(A.shape[0], A.shape[1]) * np.sqrt(scale)
                else:
                    noise = np.random.randn(A.shape[0], A.shape[1], A.shape[2]) * np.sqrt(scale)
                A = A + noise
                    
                #Save
                filename = 'Data_N' + str(n) + '_SNR' + str(snr) + '_L' + str(l) + '_M' + str(m) + '.npy'
                with open(filename, 'w') as savefile:
                    np.save(savefile, A)
                
                print 'Generated ' + str(count) + ' datasets out of ' + str(total)

plt.figure()
if initial_conditions == 1:
    plt.plot(A[:,:])
else:
    plt.plot(A[:,:,0])
plt.title('SNR = ' + str(SNR))
plt.show()