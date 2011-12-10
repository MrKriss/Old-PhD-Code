# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 14:49:59 2011

Packing some of these scrips inot functios

@author: - 
"""

from ControlCharts import Tseries
from CUSUM import cusum 
import numpy as np
import matplotlib.pyplot as plt
from Frahst_v3_1 import FRAHST_V3_1
from SPIRIT import SPIRIT
from utils import QRsolveA, pltSummary, pltSummary2, GetInHMS, writeRes
from AnomalyMetrics import analysis, fmeasure, aveMetrics
from plot_utils import plot_4x1
import scipy
import time
from PedrosFrahst import frahst_pedro
import pickle as pk
import os


baseLine = 50.0
baseLine_MA_window = 15
period = 5
amp = 0.1 
initial_conditions = 1      # i

# Varied
num_streams = [3]            # n
SNRs = [0]                  # snr
anomaly_lengths = [10]       # l
anomaly_magnitudes = [75]    # m

anomaly_type = 'shift'

def genDataMatrix(anomaly_type, n, snr, l, m, initial_conditions, period, amp, baseLine = 0.0, baseLine_MA_window = 15):
    '''Generate dataset matrix with given parameters
    
    Matrix is timesteps x num_strems x initial conditions 
    '''
    A = 0
    for i in range(initial_conditions):    
                    
        # Seed random number generator 
        np.random.seed(i)                    
        
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
            vars()[name].makeSeries([2], [300], [baseLine], 
                      amp = amp , period = period, noise_type ='none')
        
        # Concat into one matrix 
        S = scipy.c_[s0]
        for k in range(1, n) :
            S = scipy.c_[ S, vars()['s'+ str(k)] ]

        # Concatonate to 3d array, timesteps x streams x initial condition 
        if type(A) == int:
            A = S # first loop only
        else:
            A = np.dstack((A, S))  
    
def addNoise(A, snr, anomaly_type = 'peak', baseLine_MA_window = None):
    '''Add noise to data matrix according to snr '''
    if initial_conditions == 1:
        B = A[:,:].copy()  # A is only 2d
    else:                        
        B = A[:,:,0].copy()  # A is 3d
    
    # Calculate the noise      
    # B is part of A that is signal, used to calculate noise based on SNR
    if anomaly_type == 'peak':                     
        B = B - baseLine
    elif anomaly_type == 'shift' : 
#===============================================================================
# Calculate Moving Baseline if shift anomaly               
#===============================================================================
        baseLineMatrix = np.zeros(B.shape)
        for k in range(B.shape[1]):   # for each stream
            cnt = 0
            window = np.ones((baseLine_MA_window)) * baseLine
            for data in B[:,k]: 
                # Advance window 
                window[:-1] = window[1:]
                # Add new value 
                window[-1] = data
                # Calculate average 
                ave = window.sum() / float(baseLine_MA_window)
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
    
    if not os.path.isdir(path):
        os.makedirs(path)
        os.chdir(path)                
    else:
        os.chdir(path)
                        
    dataFilename = 'Data_N' + str(n) + '_SNR' + str(snr) + '_L' + str(l) + '_M' + str(m)
    with open(dataFilename, 'w') as savefile:
        np.save(savefile, A)
    os.chdir(cwd)                