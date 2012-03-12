#!/usr/bin/env python
#coding:utf-8
# Author:   --<>
# Purpose: 
# Created: 12/12/11

import numpy as np
import scipy as sp
import numpy.lib.recfunctions as nprec
import sys
import os 
import time 
import pickle

from gen_anom_data import gen_a_peak_dip, gen_a_grad_persist, gen_a_step, gen_a_step_n_back, gen_a_periodic_shift
from normalisationFunc import zscore, zscore_win
from Frahst_class import FRAHST
from utils import GetInHMS

"""
Code Description: Runs Batch of experiments on synthetic anomalous data to test scalability  
"""

if __name__=='__main__':
  
  '''Batch Parameters'''
  #-----Fixed-----#
  # Path Setup 
  exp_name = 'Scalability'
  
  #results_path = '/home/pgrad/musselle/linux/FRAHST_Project/MacSpyder/Results/'
  results_path = '/Users/chris/Dropbox/Work/MacSpyder/Results/'
  
  cwd = os.getcwd()
  path = os.path.join(results_path, exp_name)
  if not os.path.exists(path):
    os.mkdir(path)
  
  # Default Shared Algorithm Parameters 
  p = {'alpha': 0.98, 'init_r' : 1, 
       # Pedro Anomal Detection
       'holdOffTime' : 0,
       # EWMA Anomaly detection
       'EWMA_filter_alpha' : 0.2, 'residual_thresh' : 0.02,
       # AR Anomaly detection 
       'ht_AR_win' : 20, 'AR_order' : 1, 'x_thresh' : 1.5, 
       # Statistical 
       'sample_N' : 20, 'dependency_lag' : 2, 't_thresh' : None, 'FP_rate' : 10**-5,
       # Eigen-Adaptive
       'F_min' : 0.95, 'epsilon' : 0.02,
       # Pedro Adaptive
       'e_low' : 0.95, 'e_high' : 0.99,
       # Other Shared
       'r_upper_bound' : 0,
       'fix_init_Q' : 0,
       'small_value' : 0.0001,
       'ignoreUp2' : 50,
       'z_win' : 100 }
  
  p['t_thresh'] = sp.stats.t.isf(1.0 * p['FP_rate'], p['sample_N'])
  
  # Default Shared Data Set Parameters
  a = { 'N' : 50, 
        'T' : 1000, 
        'periods' : [15, 50, 70, 90], 
        'L' : 10, 
        'L2' : 200, 
        'M' : 5, 
        'pA' : 0.1, 
        'noise_sig' : 0.0 }
  
  #----Varied----#
  '''Algorithms'''    
  alg_versions = ['F-7.A-recS.R-static']  
  
  # Data set changes 
  #dat_changes = {'N' : [10, 50, 100, 500, 1000]} #Â Need min one entry for loop
  dat_changes = {'N' : [5000, 10000]} #Â Need min one entry for loop
  
  # Algorithm Changes
  alg_changes = {'FP_rate' : [10**-6]} # need min one entry for loop 
  
  # Conting iterations 
  alg_ver_count = len(alg_versions)
  alg_change_count = 0
  for v in alg_changes.values(): alg_change_count += len(v)
  dat_change_count = 0
  for v in dat_changes.values(): dat_change_count += len(v)
  
  loop_count = 0
  total_loops = alg_ver_count * alg_change_count * dat_change_count
  
  # Anomaly parameters 
  initial_conditions = 10  # i - No. of generated data sets to test
    
  anomaly_type = 'peak_dip'
  gen_funcs = dict(peak_dip = gen_a_peak_dip,
                   grad_persist = gen_a_grad_persist,
                   step = gen_a_step,
                   step_n_back = gen_a_step_n_back,
                   trend = gen_a_periodic_shift)  


  '''Setup Data Structure'''  
  time_results = np.array([0.0]* dat_change_count)

  for k,n in enumerate(dat_changes['N']):
    # set time sample buffer
    time_sample_list = np.array([0.0]*initial_conditions)
    for i in range(initial_conditions):      
      '''Generate Data Set'''
      a['N'] = n
      D = gen_funcs[anomaly_type](**a)  
      data = D['data']    
      ''' Mean Centering '''
      data = zscore_win(data, 100)
      #data = zscore(data)
      data = np.nan_to_num(data)
      z_iter = iter(data)
      numStreams = data.shape[1]      
    
      # Initialise Algorithm
      F = FRAHST('F-7.A-recS.R-static.S-none', p, numStreams)

      # Start time Profiling 
      start = time.time() 
  
      '''Begin Frahst'''
      # Main iterative loop. 
      for zt in z_iter:
        
        zt = zt.reshape(zt.shape[0],1)   # Convert to a column Vector 
  
        if np.any(F.st['anomaly']):
          F.st['anomaly'][:] = False # reset anomaly var
  
        '''Frahst Version '''
        F.run(zt)
  
        '''Anomaly Detection method''' 
        F.detect_anom(zt)

        '''Rank adaptation method''' 
        F.rank_adjust(zt)
  
        '''Store Values'''
        F.track_var()
  
      # End of a single Frahst run   
      time_sample_list[i] = time.time() - start 
  
    # End of all initial conditions for N streams
    time_results[k] = time_sample_list.mean()
  
  