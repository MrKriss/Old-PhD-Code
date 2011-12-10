# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 10:50:07 2011

This Code is streamlined for experimenting with parameters.
Each itteration outputs a plot and the results structures are saved to a list.


@author: -
"""

from ControlCharts import Tseries
from CUSUM import cusum 
import numpy as np
import matplotlib.pyplot as plt
from Frahst_v3_1 import FRAHST_V3_1
from Frahst_v3_2 import FRAHST_V3_2
from Frahst_v3_3 import FRAHST_V3_3
from Frahst_v3_4 import FRAHST_V3_4
from Frahst_v4_0 import FRAHST_V4_0
from Frahst_v5_0 import FRAHST_V5_0
from fastRowHouseHolder_float64 import FRHH64
from SPIRIT import SPIRIT
from utils import QRsolveA, pltSummary, pltSummary2, GetInHMS, writeRes
from AnomalyMetrics import analysis, fmeasure, aveMetrics
from plot_utils import plot_4x1
import scipy
import time
from PedrosFrahst import frahst_pedro_original
import pickle as pk
import os
import MAfunctions as MA 

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
baseLine = 0.0
baseLine_MA_window = 15
period = 5
amp = 1
initial_conditions = 1      # i
anomaly_type = 'peak'

# Algorithm Options 
# Starting r
r = 1
# Upper bound for r
upper_bound = None

# Length for z_tl : auto correlated input vector if L > 1
L = 1
# Length for z_l : multiple auto correlated frahst inputs (only for v5_0)
win_L = 1
# Ignore first x timesteps for rank changes  
ignoreUp2 = 50

# Moving Average
use_MA = 0
N = 25

# Varied
num_streams = [3]            # n
SNRs = [3]                   # snr
anomaly_lengths = [10]       # l
anomaly_magnitudes = [20]    # m

# Algorithm Parameters
e_highs = [0.90]             # eh
e_lows = [0.65]              # el 
alphas = [0.96]              # a
holdOffs = [0]               # h

# Algorithm flags 
run_frahst_v5_0 = 1
run_spirit = 0
run_frahst_eig = 1
run_frahst_newest = 0
run_frahst_previous = 0
run_fixed_frahst = 0
run_frahst_pedro = 0

#===============================================================================
# Initialise Data sets
#===============================================================================

# Results Storage Structure
results = []

# For Profiling 
start = time.time() 

for n in num_streams:
  for snr in SNRs:
    for l in anomaly_lengths:
      for m in anomaly_magnitudes:
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
          for k in range(2, n) :
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
          scale = Pn / (n * 3. * interval_length)        
          # Calculate the noise  
          if initial_conditions == 1:
            noise = np.random.randn(A.shape[0], A.shape[1]) * np.sqrt(scale)
          else:
            noise = np.random.randn(A.shape[0], A.shape[1], A.shape[2]) * np.sqrt(scale)
          A = A + noise

          #Save
          dataInfo = 'N = {N}, SNR = {SNR}, L = {L}, M = {M}'.format(N=n, SNR=snr, L=l, M = m)
#===============================================================================
# Ground Truths 
#===============================================================================
#                            # time step | length 
          ground_truths = np.array([[100, l],
                                    [200, l]])
#==============================================================================
#  Run Algorithm 
#==============================================================================   
          alg_count = 1    
          for eh in e_highs :             
            for el in e_lows :              
              for a in alphas :              
                for h in holdOffs :            
                  print 'Running Algorithm(s) with:\nE_Thresh = (' + str(el) + ',' + str(eh) + ')\n' + \
                        'alpha = ' + str(a) + '\nHoldOff = ' + str(h)  
                  for i in range(initial_conditions):

                    # Load Data 
                    if initial_conditions == 1:
                      streams = A[:,:]
                    else:
                      streams = A[:,:,i]

                    # Mean adjust data 
                    if use_MA == 1:
                      Maved_data_MAwindow = MA.MA_over_window(streams, N)
                      Maved_data_EWMA = MA.EWMA(streams, N)
                      Maved_data_CMA = MA.CMA(streams)
                      Maved_data_frac_decay = MA.fractional_decay_MA(streams, 0.9)

                      original = streams 
                      streams = streams - Maved_data_MAwindow

                    if run_spirit == 1:
                      # SPIRIT
                      res_sp = SPIRIT(streams, a, [el, eh], evalMetrics = 'F', 
                                      reorthog = False, holdOffTime = h) 
                      res_sp['Alg'] = 'SPIRIT: alpha = ' + str(a) + \
                            ' ,E_Thresh = (' + str(el) + ',' + str(eh) + ')'
                      
                      # Store
                      algInfo = 'SPIRIT: E_Thresh = (' + str(el) + ',' + str(eh) + ')\n' + \
                        'alpha = ' + str(a) + '\nHoldOff = ' + str(h)                      
                      results.append({'info': {'data': dataInfo, 'alg' : algInfo},
                                        'data' : res_sp})
                      # Plot
                      pltSummary2(res_sp, streams, (el, eh))                            

                    if run_frahst_v5_0 == 1:
                      # Multiple Auto Correlation Frahsts 
                      for n in range(streams.shape[1]):
                        res_name = 'res_ACF' + str(n)
                        vars()[res_name] = FRAHST_V5_0(streams, data_column = n, L = win_L, alpha=a, e_low=el, e_high=eh, 
                                holdOffTime=h, fix_init_Q = 1, r = 1, evalMetrics = 'T', 
                                ignoreUp2 = ignoreUp2, static_r = 0, r_upper_bound = upper_bound)
                        vars()[res_name]['Alg'] = res_name + 'alpha = ' + str(a) + ' ,E_Thresh = (' + str(el) + ',' + str(eh) + ')'
                        pltSummary2(vars()[res_name], streams[:,n] , (eh, el))
                      
                        # Store
                        algInfo = res_name + ': E_Thresh = (' + str(el) + ',' + str(eh) + ')\n' + \
                                'alpha = ' + str(a) + '\nHoldOff = ' + str(h)
                        results.append({'info': {'data': dataInfo, 'alg' : algInfo},
                                        'data' : vars()[res_name]})
                      
                    if run_frahst_eig == 1:
                      # My version of Frahst 
                      res_fr_eig = FRAHST_V4_0(streams, L = L, alpha=a, e_low=el, e_high=eh, 
                                           holdOffTime=h, fix_init_Q = 1, r = 1, evalMetrics = 'T',
                                           r_upper_bound = upper_bound, ignoreUp2 = ignoreUp2) 
                      res_fr_eig['Alg'] = 'MyFrahst-Eig: alpha = ' + str(a) + ' ,E_Thresh = (' + str(el) + ',' + str(eh) + ')'

                      # Store
                      algInfo = 'FRAHST-Eig: E_Thresh = (' + str(el) + ',' + str(eh) + ')\n' + \
                        'alpha = ' + str(a) + '\nHoldOff = ' + str(h)
                      results.append({'info': {'data': dataInfo, 'alg' : algInfo},
                                        'data' : res_fr_eig})
                      # Plot
                      pltSummary2(res_fr_eig, streams, (el, eh))
                      
                    if run_frahst_newest == 1:
                      # My version of Frahst 
                      res_fr_n = FRAHST_V3_4(streams, L = L, alpha=a, e_low=el, e_high=eh, 
                                           holdOffTime=h, fix_init_Q = 1, r = 1, evalMetrics = 'F',
                                           r_upper_bound = upper_bound, ignoreUp2 = ignoreUp2) 
                      res_fr_n['Alg'] = 'MyFrahst-NEW: alpha = ' + str(a) + ' ,E_Thresh = (' + str(el) + ',' + str(eh) + ')'

                      # Store
                      algInfo = 'FRAHST-NEW: E_Thresh = (' + str(el) + ',' + str(eh) + ')\n' + \
                        'alpha = ' + str(a) + '\nHoldOff = ' + str(h)
                      results.append({'info': {'data': dataInfo, 'alg' : algInfo},
                                        'data' : res_fr_n})
                      # Plot
                      pltSummary2(res_fr_n, streams, (el, eh))

                    if run_frahst_previous == 1:
                      # My version of Frahst 
                      res_fr_p = FRAHST_V3_3(streams, alpha=a, e_low=el, e_high=eh, 
                                           holdOffTime=h, fix_init_Q = 1, r = 1, evalMetrics = 'F',
                                           r_upper_bound = upper_bound, ignoreUp2 = ignoreUp2) 
                      res_fr_p['Alg'] = 'MyFrahst-OLD: alpha = ' + str(a) + ' ,E_Thresh = (' + str(el) + ',' + str(eh) + ')'

                      # Store
                      algInfo = 'FRAHST-OLD: E_Thresh = (' + str(el) + ',' + str(eh) + ')\n' + \
                        'alpha = ' + str(a) + '\nHoldOff = ' + str(h)
                      results.append({'info': {'data': dataInfo, 'alg' : algInfo},
                                        'data' : res_fr_p})
                      # Plot
                      pltSummary2(res_fr_p, streams, (el, eh))

                    if run_fixed_frahst == 1:

                      # My version of Frahst 
                      res_fixfr = FRAHST_V3_2(streams, alpha=a, static_r = 1, fix_init_Q = 1, r = r, 
                                           evalMetrics = 'F', r_upper_bound = upper_bound, ignoreUp2 = ignoreUp2) 
                      res_fixfr['Alg'] = 'Frahst-FIXED: alpha = ' + str(a) + ' ,E_Thresh = (' + str(el) + ',' + str(eh) + ')'
                      
                      #Store
                      algInfo = 'Fixed FRAHST: E_Thresh = (' + str(el) + ',' + str(eh) + ')\n' + \
                        'alpha = ' + str(a) + '\nHoldOff = ' + str(h)
                      results.append({'info': {'data': dataInfo, 'alg' : algInfo},
                                        'data' : res_fixfr})
                      # Plot
                      pltSummary2(res_fixfr, streams, (el, eh))                            

                    if run_frahst_pedro == 1:

                      # Pedros version of Frahst 
                      res_frped = frahst_pedro_original(streams, alpha=a, e_low=el, e_high=eh, 
                                               holdOffTime=h, r = 1, evalMetrics = 'F', ignoreUp2 = ignoreUp2) 
                      res_frped['Alg'] = 'Pedros Frahst: alpha = ' + str(a) +  \
                               ' ,E_Thresh = (' + str(el) + ',' + str(eh) + ')'
                      # Store
                      algInfo = 'Pedro FRAHST: E_Thresh = (' + str(el) + ',' + str(eh) + ')\n' + \
                      'alpha = ' + str(a) + '\nHoldOff = ' + str(h)
                      results.append({'info': {'data': dataInfo, 'alg' : algInfo},
                                        'data' : res_frped})

                      # Plot
                      pltSummary2(res_frped, streams, (el, eh))

finish  = time.time() - start
print 'Runtime = ' + str(finish) + 'seconds\n'
print 'In H:M:S = ' + GetInHMS(finish)