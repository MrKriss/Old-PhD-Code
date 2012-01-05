# -*- coding: utf-8 -*-
"""
Altered on Fri Jun 24 10:50:07 2011

This Code is streamlined for experimenting with parameters.
Each itteration outputs a plot and the results structures are saved to a list.


@author: -
"""

from ControlCharts import Tseries
from CUSUM import cusum 
import numpy as np
import matplotlib.pyplot as plt
import scipy
import time
import os

#===============================================================================
# Batch Parameters
#===============================================================================

# Path Setup 
exp_name = 'singleRuns'
results_path = '/Users/chris/Dropbox/Work/MacSpyder/Results/'
cwd = os.getcwd()
path = os.path.join(results_path, exp_name)

# Data Sets 
interval_length = 300
baseLine = 10
baseLine_MA_window = 15
period = 25
amp = 5
initial_conditions = 1      # i
anomaly_type = 'shift'

seed_gen = 0

# Varied
num_streams = [3]            # n
Noises = [1.0]               # snr
anomaly_lengths = [10]       # l
anomaly_magnitudes = [20]    # m

#===============================================================================
# Initialise Data sets
#===============================================================================

# For Profiling 
start = time.time() 

for n in num_streams:
  for noise in Noises:
    for l in anomaly_lengths:
      for m in anomaly_magnitudes:
        A = 0
        for i in range(initial_conditions):    

          # Seed random number generator 
          if seed_gen == True:
            np.random.seed(i)                    

          # Two ts that have anomalous shift 
          s0lin = Tseries(0) # linear component 
          s0sin = Tseries(0) # Sine component 
          s1lin = Tseries(0)
          s1sin = Tseries(0)

          if anomaly_type == 'peak':

            s0lin.makeSeries([1,3,4,1], [interval_length, l/2, l/2, 2 * interval_length - l], 
                             [baseLine, baseLine, baseLine + m, baseLine], 
                             gradient = float(m)/float(l/2), noise_type ='none')
            s0sin.makeSeries([2], [3 * interval_length], [0.0], 
                             amp = amp, period = period, noise_type ='none')

            # sum sin and linear components to get data stream                         
            s0 = np.array(s0lin) + np.array(s0sin)                                    

            s1lin.makeSeries([1,4,3,1],[2 * interval_length, l/2, l/2, interval_length - l],
                             [baseLine, baseLine, baseLine - m, baseLine], 
                             gradient = float(m)/float(l/2), noise_type ='none')
            s1sin.makeSeries([2], [3 * interval_length], [0.0], 
                             amp = amp, period = period, noise_type ='none')

            # sum sin and linear components to get data stream                                   
            s1 = np.array(s1lin) + np.array(s1sin)                          

          elif anomaly_type == 'shift':            
            s0lin.makeSeries([1,3,1], [interval_length, l, 2 * interval_length - l], 
                             [baseLine, baseLine, baseLine + m], 
                             gradient = float(m)/float(l), noise_type ='none')
            s0sin.makeSeries([2], [3 * interval_length], [0.0], 
                             amp = amp, period = period, noise_type ='none')

            # sum sin and linear components to get data stream                         
            s0 = np.array(s0lin) + np.array(s0sin)                                    

            s1lin.makeSeries([1,4,1],[2 * interval_length, l, interval_length - l],[baseLine, baseLine, baseLine - m], 
                             gradient = float(m)/float(l), noise_type ='none')
            s1sin.makeSeries([2], [3 * interval_length], [0.0], 
                             amp = amp , period = period, noise_type ='none')

            # sum sin and linear components to get data stream                                                           
            s1 = np.array(s1lin) + np.array(s1sin)                                      

          # The rest of the ts
          for k in range(2, n):
            name = 's'+ str(k)
            vars()[name] = Tseries(0)
            vars()[name].makeSeries([2], [3 * interval_length], [baseLine], 
                                    amp = amp , period = period, noise_type ='none')

          # Concat into one matrix 
          S = scipy.c_[s0]
          for k in range(1, n) :
            S = scipy.c_[ S, vars()['s'+ str(k)] ]

          # Concatonate to 3d array, timesteps x streams x initial condition 
          if type(A) == int: # check for first iteration 
            A = S 
          else:
            A = np.dstack((A, S))  

          B = A.copy()

          # Calculate the noise  
          if initial_conditions == 1:
            added_noise = np.random.randn(A.shape[0], A.shape[1]) * noise
          else:
            added_noise = np.random.randn(A.shape[0], A.shape[1], A.shape[2]) * noise
          A = A + added_noise

          snr = 20
          Ps = np.sum(A ** 2)
          Pn = Ps / (10. ** (snr/10.))
          scale = Pn / (n * 3. * interval_length)    
          # Calculate the noise  
          if initial_conditions == 1:
            added_noise2 = np.random.randn(A.shape[0], A.shape[1]) * np.sqrt(scale)
          else:
            added_noise2 = np.random.randn(A.shape[0], A.shape[1], A.shape[2]) * np.sqrt(scale)
          B = B + added_noise2          
   
          #Save
          dataInfo = 'N = {N}, Noise = {Noise}, L = {L}, M = {M}'.format(N=n, Noise=noise, L=l, M = m)