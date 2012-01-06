# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 17:09:00 2011

@author: musselle
"""
import numpy as np
import numpy.random as npr
from numpy import  sqrt, zeros, cos, pi, float32
from numpy.random import  randn , seed, rand

def genCosSignals(randSeed, SNR, timesteps = 10000, N = 32):
    """
    Method used by Strobarch to Generate test data.
    
        N = no. of streams
    """
    seed(randSeed)    
    
    # Cheack Input     
    SNR = float(SNR)    
    
    # Create Data ##################################################
    # units  = radians
    a_11 = 1.4   
    a_12 = 1.6 
    a_21 = 2.0 
    a_22 = 1.0 
    
    w_t_11 = 2.2 
    w_t_12 = 2.8 
    w_t_21 = 2.7 
    w_t_22 = 2.3 
    
    w_s_11 = 0.5 
    w_s_12 = 0.9 
    w_s_21 = 1.1 
    w_s_22 = 0.8 
        
    s_t = zeros((timesteps,N))
    
    for k in range(1,N+1):     
        # starting phases for signals are random 
        w_0_11 = rand() * 2 * pi
        w_0_12 = rand() * 2 * pi
        w_0_21 = rand() * 2 * pi
        w_0_22 = rand() * 2 * pi 
        for t in range(1,timesteps + 1):
            if t < timesteps / 2.0:
                A = a_11 * cos(w_t_11 * t + w_s_11 * k + w_0_11)
                B = a_12 * cos(w_t_12 * t + w_s_12 * k + w_0_12)
                s_t[t-1,k-1] = A + B
            else:
                A = a_21 * cos(w_t_21 * t + w_s_21 * k + w_0_21)
                B = a_22 * cos(w_t_22 * t + w_s_22 * k + w_0_22)
                s_t[t-1,k-1] = A + B
                
    Ps = sum(s_t ** 2)
    # SNR = -3.0
    Pn = Ps / (10 ** (SNR/10))    
    
    scale = Pn / (N * timesteps)
    
    noise = randn(s_t.shape[0], s_t.shape[1]) * sqrt(scale)
    
    Sim_streams = s_t + noise       
    
    return Sim_streams

def genCosSignals_no_rand(timesteps = 10000, N = 32):
    """
        N = no. of streams
    """       
    
    # Create Data ##################################################
    # units  = radians
    a_11 = 1.4   
    a_12 = 1.6 
    a_21 = 2.0 
    a_22 = 1.0 
    
    w_t_11 = 2.2 
    w_t_12 = 2.8 
    w_t_21 = 2.7 
    w_t_22 = 2.3 
    
    w_s_11 = 0.5 
    w_s_12 = 0.9 
    w_s_21 = 1.1 
    w_s_22 = 0.8 
        
    s_t = zeros((timesteps,N))
    
    for k in range(1,N+1):     
        # starting phases for signals are random 
        for t in range(1,timesteps + 1):
            if t < timesteps / 2.0:
                A = a_11 * cos(w_t_11 * t + w_s_11 * k )
                B = a_12 * cos(w_t_12 * t + w_s_12 * k )
                s_t[t-1,k-1] = A + B
            else:
                A = a_21 * cos(w_t_21 * t + w_s_21 * k )
                B = a_22 * cos(w_t_22 * t + w_s_22 * k )
                s_t[t-1,k-1] = A + B
                     
    Sim_streams = s_t         
                     
    return Sim_streams

def genSimpleSignals(randSeed, SNR, version, N = 32):

    seed(randSeed)    
    
    # Cheack Input     
    SNR = float(SNR)
    
    # Create Data #################################################      
    s_t = zeros((10000,N))
    
    for k in range(1,N+1):     
        for t in range(1,10001):
            
            if version == '2':
                s_t[t-1,k-1] = (1. / k) + (((-1.)**t) / (k + 1.))
            else: 
                s_t[t-1,k-1] = 1. / k
                
    Ps = sum(s_t ** 2)
    # SNR = -3.0
    Pn = Ps / (10 ** (SNR/10))    
    
    scale = Pn / (N * 10000)
    
    noise = randn(s_t.shape[0], s_t.shape[1]) * sqrt(scale)
    
    Sim_streams = s_t + noise       
    
    return Sim_streams, s_t, noise

def sin_rand_combo(N, T, periods, seed = None, noise_scale = 0.1 ):
    """ Generates N time series of length T from random combinations of sin waves
    
    len(periods) sin waves of the periods given are used 
    
    sin_bank - the sine waves used
    A - the specified number of random combinations of sin_bank
    """
    
    sin_bank = np.zeros((T,len(periods))) #Â as columns are plotted as default
    
    for i,p in enumerate(periods):
        sin_bank[:,i] = np.sin( 2.0 * np.pi * np.arange(T) / float(p))

    if seed != None:
        npr.seed(seed)
  
    # A = T x N matrix
    A = np.mat(sin_bank) * npr.rand(len(periods),N)
    # Add Gaussian noise 
    A += (npr.randn(T, N) * noise_scale)
        
    return np.array(A), sin_bank

def simple_sins(p1,p11, p2,p22, noise_scale, N = 500):
    """ 2 Sine signals which change in period half way 
    
    noise_scale - scales the gaussian noise
    N - Number of time steps
    px -  first period of sine wave
    pxx - second period of sine wave
    """
    
    t = arange(N)
                
    z1 = np.sin(2 * np.pi * t / p1) + npr.randn(t.shape[0]) * noise_scale
    z2 = np.sin(2 * np.pi * t / p2) + npr.randn(t.shape[0]) * noise_scale
        
    z11 = np.sin(2 * np.pi * t / p11) + npr.randn(t.shape[0]) * noise_scale
    z22 = np.sin(2 * np.pi * t / p22) + npr.randn(t.shape[0]) * noise_scale
        
    data = sp.r_['1,2,0', sp.r_[z1, z11], sp.r_[z2, z22]]

    return data 

def simple_sins_3z(p1,p11, p2,p22, p3, p33, noise_scale, N = 500):
    """ 3 Sine signals which change in period half way 
    
    noise_scale - scales the gaussian noise
    N - Number of time steps
    px -  first period of sine wave
    pxx - second period of sine wave
    """
    t = arange(N)
                
    z1 = np.sin(2 * np.pi * t / p1) + npr.randn(t.shape[0]) * noise_scale
    z2 = np.sin(2 * np.pi * t / p2) + npr.randn(t.shape[0]) * noise_scale
    z3 = np.sin(2 * np.pi * t / p3) + npr.randn(t.shape[0]) * noise_scale
        
    z11 = np.sin(2 * np.pi * t / p11) + npr.randn(t.shape[0]) * noise_scale
    z22 = np.sin(2 * np.pi * t / p22) + npr.randn(t.shape[0]) * noise_scale
    z33 = np.sin(2 * np.pi * t / p33) + npr.randn(t.shape[0]) * noise_scale
        
    data = sp.r_['1,2,0', sp.r_[z1, z11], sp.r_[z2, z22], sp.r_[z3, z33]]

    return data 