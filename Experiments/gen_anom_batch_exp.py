#!/usr/bin/env python
#coding:utf-8
# Author:   --<>
# Purpose: 
# Created: 12/12/11

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sys
import os 
import time 

from gen_anom_data import gen_a_peak_dip, gen_a_grad_persist, gen_a_step
from normalisationFunc import zscore, zscore_win
from Frahst_class import FRAHST


"""
Code Description: Runs Batch of experiments on synthetic anomalous data  
  .
"""
'''Batch Parameters'''
#-----Fixed-----#
# Path Setup 
exp_name = 'Test_a_signals'
results_path = '/Users/chris/Dropbox/Work/MacSpyder/Results/'
cwd = os.getcwd()
path = os.path.join(results_path, exp_name)

initial_conditions = 5      # i - No. of generated data sets to test

anomaly_type = 'peak_dip'
gen_funcs = dict(peak_dip = gen_a_peak_dip,
                 grad_persist = gen_a_grad_persist,
                 step = gen_a_step)

# Default Shared Algorithm Parameters 
p = {'alpha': 0.98, 'init_r' : 1, 
      # Pedro Anomal Detection
      'holdOffTime' : 0,
      # EWMA Anomaly detection
      'EWMA_filter_alpha' : 0.2, 'residual_thresh' : 0.02,
      # AR Anomaly detection 
      'ht_AR_win' : 30, 'AR_order' : 1, 'err_thresh' : 1.5, 
      # Statistical 
      'sample_N' : 20, 'dependency_lag' : 20, 'x_thresh' : 10, 'FP_rate' : 10**-5,
      # Eigen-Adaptive
      'F_min' : 0.9, 'epsilon' : 0.05,
      # Pedro Adaptive
      'e_low' : 0.95, 'e_high' : 0.98,
      # Other Shared
      'r_upper_bound' : None,
      'fix_init_Q' : 0,
      'small_value' : 0.0001,
      'ignoreUp2' : 50,
      'z_win' : 100 }
p['x_thresh'] = sp.stats.t.isf(0.5 * p['FP_rate'], p['sample_N'])

# Default Shared Data Set Parameters
a = { 'N' : 50, 
      'T' : 1000, 
      'periods' : [15, 40, 70, 90, 120], 
      'L' : 10, 
      'L2' : 200, 
      'M' : 3, 
      'pA' : 0.1, 
      'noise_sig' : 0.2 }


#----Varied----#
'''Algorithms'''    
alg_versions = ['F-7.A-recS.R-eig', 'F-7.A-recS.R-eng', 
                'F-7.A-forS.R-eig', 'F-7.A-forS.R-eng']

# Data set changes 
dat_changes = {'N' : [50]} # Need min one entry for loop
#dat_changes = dict(noise_sig = [0.0, 0.1, 0.2, 0.3])

# Algorithm Changes
alg_changes = {'alpha' : [0.98]} # need min one entry for loop 
#alg_changes = dict(F_min = [0.95, 0.9, 0.85, 0.8],
                   #alpha = [0.99, 0.98, 0.97, 0.96])

# Conting iterations 
alg_ver_count = len(alg_versions)
alg_change_count = 0
for v in alg_changes.values(): alg_change_count += len(v)
dat_change_count = 0
for v in dat_changes.values(): dat_change_count += len(v)

loop_count = 0
total_loops = alg_ver_count, alg_change_count, dat_change_count

# For Profiling 
start = time.time() 

'''Generate Data Set'''
# For each Change to Dataset Parameters
for var, values in dat_changes.iteritems():
  for v in values:
    # update data set parameters a
    a[var] = v
    for i in xrange(initial_conditions):
      # Generate the data
      a['seed'] = i
      D = gen_funcs[anomaly_type](**a) # so tidy!
      data = D['data']
      
      ''' Run Algorithm '''
      # For each Change to Algorith Parameters 
      for var, values in alg_changes.iteritems():
        for v in values:
          # update algorithm parameters p
          p[var] = v      
          for alg_v in alg_versions:
            '''Initialise'''
            data = zscore_win(data, p['z_win'])
            z_iter = iter(data)
            numStreams = data.shape[1]
            F = FRAHST(alg_v, p, numStreams)   

            '''Begin Frahst'''
            # Main iterative loop. 
            for zt in z_iter:
              zt = zt.reshape(zt.shape[0],1)   # Convert to a column Vector 
            
              if F.st['anomaly'] == True:
                F.st['anomaly'] = False # reset anomaly var
            
              '''Frahst Version '''
              F.run(zt)
              # Calculate reconstructed data if needed
              st = F.st
              F.st['recon'] = np.dot(st['Q'][:,:st['r']],st['ht'][:st['r']])
            
              '''Anomaly Detection method''' 
              F.detect_anom(zt)
            
              '''Rank adaptation method''' 
              F.rank_adjust(zt)
            
              '''Store data''' 
              #tracked_values = ['ht','e_ratio','r','recon', 'pred_err', 'pred_err_norm', 'pred_err_ave', 't_stat', 'pred_dsn', 'pred_zt']   
              #tracked_values = ['ht','e_ratio','r','recon','recon_err', 'recon_err_norm', 't_stat', 'rec_dsn', 'x_sample']
              tracked_values = ['ht','e_ratio','r','recon', 't_stat']
            
              F.track_var(tracked_values)
            
            ''' Plot Results '''
            #F.plot_res([data, 'ht', 't_stat'])
            F.plot_res([data, 'ht', 't_stat'])
            
                          
