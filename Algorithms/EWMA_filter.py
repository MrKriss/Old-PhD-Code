#!/usr/bin/env python
#coding:utf-8
# Author:  Musselle --<>
# Purpose: EWMA filter for anomaly detection
# Created: 08/24/11

import numpy as np
import matplotlib.pyplot as plt
import sys
import os 
from load_data import load_data

def EWMA_filter(data, alpha):
    """ EWMA filter for anomaly detection:
    returns difference between predicted and actual values
    
    data Vector must be passed as input 
    """

    residual = np.zeros_like(data)
    smooth_data = np.zeros_like(data)    
    smooth_data[0] = alpha * data[0]        
    
    for t in range(data.shape[0]-1):
        # Note changed so current time step for smothed data prediction (sdp) is calculated
        # using previous values for sdp (not future sdp(t+1) based on current sdp(t) )
        smooth_data[t] = alpha * data[t] + (1-alpha) * smooth_data[t-1]    
        residual[t] = np.abs(data[t] - smooth_data[t])
    
    return residual, smooth_data

def EWMA_filter_incr(data, alpha):
    """ Incremental EWMA filter for anomaly detection:
    returns difference between predicted and actual values
    
    Actually now iterable as data generator can be passed as input 
    
    data is now an itterable 
    
    Not much use for this version. Deprecate.
    """
    last_smooth_data = 0.
    for t, dat in enumerate(data):
        pred_smooth_data = alpha * dat + (1-alpha) * last_smooth_data    
        residual = np.abs(dat - last_smooth_data)
        
        yield pred_smooth_data, residual
        last_smooth_data = pred_smooth_data


def EWMA(data, N):
    """ Exponetially Weighted Moving Average over all data"""    

    if N > 1.0:
        alpha = 2.0 / (N + 1) 
    else:
        alpha = N

    aved_data = np.zeros_like(data)
    u = np.zeros(data.shape[1])
    for i in range(data.shape[0]):
        new_data_vec = data[i]
        u = u * (1-alpha) + new_data_vec * alpha
        aved_data[i] = u
    return aved_data




if __name__=='__main__':
  
    Packets, Flows, Bytes = load_data('abilene')
    
    data = Packets
    alpha = 0.2
    
    for i in range(data.shape[1]): # per time series 
        res_name = 'residual' + str(i)
        aved_name = 'aved' + str(i)
        vars()[res_name], vars()[aved_name] = EWMA_filter(data[:,i], alpha)
    
    all_residuals, all_aved_data = EWMA_filter(data, alpha)
    EWMA_aved_data = EWMA(data, alpha)