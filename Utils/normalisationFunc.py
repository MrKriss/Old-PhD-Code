#!/usr/bin/env python
#coding:utf-8
# Author:  C Musselle --<>
# Created: 11/10/11

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import sys
import os 
from artSigs import genCosSignals_no_rand , genCosSignals
import scipy.io as sio
from utils import analysis, QRsolveA, pltSummary2
from load_syn_ping_data import load_n_store
from create_Abilene_links_data import create_abilene_links_data
from MAfunctions import MA_over_window
from ControlCharts import Tseries
from EWMA_filter import EWMA_filter

"""
Code Description:
  Normlisation functions: Min-Max, Z score and the decimal methods.
  
  Compare methods with there incremental versions based on exponentially weighted mean
  and variance estimates. 
  
  Could also simply use a sliding window method. 
  
  Comapre and contrast. 
"""

def min_max(X, low=0, high=1, minX=None, maxX=None):
    """Returns min-max normalisation for each column of array
    
    X is the full stream Matrix - timesteps x Numstreams
    """
    timeSteps = X.shape[0]
    numStreams = X.shape[1]
    norm_data = np.zeros_like(data)
    
    for i in range(numStreams):
        if minX is None:
            minX = X[:,i].min()
        if maxX is None:
            maxX = X[:,i].max()
        # Normalize to [0...1].
        norm_data[:,i] = (X[:,i] - minX) / (maxX - minX)
        # Scale to [low...high].
        norm_data[:,i] = (norm_data[:,i] * (high-low)) + low
    return norm_data

def min_max_win(data, W, low=0, high=1, minX=None, maxX=None):
    
    if data.ndim == 1:
        window = np.zeros((1, window_length))
        norm_data = np.zeros_like(data)
    else:
        window = np.zeros((data.shape[1], window_length))
        norm_data = np.zeros_like(data)
    
    for i in range(data.shape[0]):
        # Shift Window
        window[:,:-1] = window[:,1:] 
        window[:,-1] = data[i]
        # Run Function
        # Broke XXX need to fix 
        norm_data[:i:window_length] = min_max(window[i], low=low, high=high, minX=minX, maxX=maxX)
        
    
    return norm_data
 
def zscore(data):
    """Returns z-score normalisation for each column of array"""

    zscore_data = np.zeros_like(data)
    timeSteps = data.shape[0]
    if data.ndim == 1:
        numStreams = 1
    else:
        numStreams = data.shape[1]

    # Clense of NaNs
    data = np.ma.MaskedArray(data, mask = np.isnan(data))

    for i in range(numStreams):
        zscore_data[:,i] = (data[:,i] - data[:,i].mean()) / data[:,i].std() 

    return zscore_data


def zscore_win(data, win_length):
    """Returns sliding window z-score normalisation for each column of array
    Note: nieve version. Could improve easily 
    """

    if data.ndim == 1:
        numStreams = 1
    else:
        numStreams = data.shape[1]


    if data.ndim == 1:
        window = np.zeros((1, win_length))
        zscore_data = np.zeros_like(data)
    else:
        window = np.zeros((data.shape[1], win_length))
        zscore_data = np.zeros_like(data)
        
    for i in range(data.shape[0]):
        # Shift Window
        window[:,:-1] = window[:,1:] 
        window[:,-1] = data[i]
        
        # Run function 
        est_mean_vec = window.mean(axis = 1)
        est_std_vec = window.std(axis = 1)
        zscore_data[i,:] =  (data[i,:] - est_mean_vec) / est_std_vec 
        
    return zscore_data

def zscore_exp(data, alpha):
    """Returns exponential z-score normalisation for each column of array"""

    if alpha > 1 :
        alpha = 2.0 / (alpha + 1)

    exp_var_vec = 0 
    exp_mean_vec = 0
    zscore_data = np.zeros_like(data)
        
    for i in range(data.shape[0]):
        # Run function 
        diff = data[i,:] - exp_mean_vec
        incr = alpha * diff
        exp_mean_vec = exp_mean_vec + incr
        exp_var_vec = (1 - alpha) * (exp_var_vec + diff * incr)

        zscore_data[i,:] =  (data[i,:] - exp_mean_vec) / (exp_var_vec**2) 
        
    return zscore_data

def EW_mean_var(x, alpha, var, mean):
    """ Work out the exponentially weighted mean and variance of the data """
    if alpha > 1 :
        alpha = 2.0 / (alpha + 1)
    
    diff = x - mean 
    incr = alpha * diff
    mean = mean + incr
    var = (1 - alpha) * (var + diff * incr)

    return var, mean 


def decimal(X):
    """Returns decimal normalisation for each column of array"""
    decimal_data = np.zeros_like(data)
    numStreams = X.shape[1]
  
    for i in range(numStreams):        
        xmax = np.abs(X[:,i].max())
        c = 0
        while xmax / (10 ** c) > 1.0:
            c += 1 
        decimal_data[:,i] = X[:,i] / 10**c
    
    return decimal_data



if __name__=='__main__':
  
    #s1 = Tseries(0)
    s2 = Tseries(0)
    #s1.makeSeries([2,1,2], [300, 300, 300], noise = 0.5, period = 50, amp = 5)
    #s2.makeSeries([2], [900], noise = 0.5, period = 50, amp = 5)
    #data = sp.r_['1,2,0', s1, s2]
    
    s0lin = Tseries(0)
    s0sin = Tseries(0)
    s2lin = Tseries(0)
    s2sin = Tseries(0)
    
    interval_length = 300
    l = 10
    m = 10
    baseLine = 0
    amp = 5
    period = 50 
    s0lin.makeSeries([1,3,4,1], [interval_length, l/2, l/2, 2 * interval_length - l], 
                    [baseLine, baseLine, baseLine + m, baseLine], 
                    gradient = float(m)/float(l/2), noise = 0.5)
    s0sin.makeSeries([2], [3 * interval_length], [0.0], 
                    amp = amp, period = period, noise = 0.5)

    # sum sin and linear components to get data stream                         
    s1 = np.array(s0lin) + np.array(s0sin)   

    s2lin.makeSeries([1,4,3,1], [interval_length * 2, l/2, l/2, interval_length - l], 
                    [baseLine, baseLine, baseLine - m, baseLine], 
                    gradient = float(m)/float(l/2), noise = 0.5)
    s2sin.makeSeries([2], [3 * interval_length], [0.0], 
                    amp = amp, period = period, noise = 0.5)
    s2 = np.array(s2lin) + np.array(s2sin)   

    data = sp.r_['1,2,0', s1, s2]
    
    #s1lin.makeSeries([1,4,3,1],[2 * interval_length, l/2, l/2, interval_length - l],
                     #[baseLine, baseLine, baseLine - m, baseLine], 
                     #gradient = float(m)/float(l/2), noise_type ='none')
    #s1sin.makeSeries([2], [3 * interval_length], [0.0], 
                    #amp = amp, period = period, noise_type ='none')
    
    #data = genCosSignals(0, -3.0)
    
    #data, G = create_abilene_links_data()
    
    #execfile('/Users/chris/Dropbox/Work/MacSpyder/Utils/gen_Anomalous_peakORshift_data.py')
    #data = A
    
    #data = simple_sins(10,10,10,25, 0.1)
    
    #data = simple_sins_3z(10,10,13,13, 10, 27, 0.0)
    
    #data = genCosSignals_no_rand(timesteps = 10000, N = 32)  
    
    #data = array([[0,0,0], [1,2,2], [1,3,4], [3,6,6], [5,6,10], [6,8,11]])   
    
    #sig_PN, ant_PN, time_PN = load_n_store('SYN', 'PN')
    #data = sig_PN
    
    AbileneMat = sio.loadmat('/Users/chris/DataSets/Abilene/Abilene.mat')
    data = AbileneMat['P']
  
    norm_data = min_max(data)
    zscore_data = zscore(data)
    decimal_data = decimal(data)
    
    zscore_win_data_50 = zscore_win(data,50)
    zscore_win_data_100 = zscore_win(data,100)
    zscore_win_data_150 = zscore_win(data,150)
    zscore_win_data_250 = zscore_win(data,250)
    #zscore_exp_data = zscore_exp(data, 0.2) 
    
