#!/usr/bin/env python
#coding:utf-8
# Author:  C Musselle --<>
# Purpose: Experiment Script
# Created: 02/19/12

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sys
import os 

from SAX import SAX



"""
Code Description:
  Combine FRAHST with SAX.  
  
  Goal: to determin which of the streams are anomalous at the flagged time step. 
"""

''' Experimental Run Parameters '''
p = {'alpha': 0.98,
      'init_r' : 1, 
      # Pedro Anomal Detection
      'holdOffTime' : 0,
      # EWMA Anomaly detection
      'EWMA_filter_alpha' : 0.2,
      'residual_thresh' : 0.02,
      # AR Anomaly detection 
      'ht_AR_win' : 30,
      'AR_order' : 1,
      'x_thresh' : 1.5, 
      # Statistical 
      'sample_N' : 20,
      'dependency_lag' : 2,
      't_thresh' : None,
      'FP_rate' : 10**-2,
      # Eigen-Adaptive
      'F_min' : 0.95,
      'epsilon' : 0.02,
      # Pedro Adaptive
      'e_low' : 0.95,
      'e_high' : 0.99,
      'r_upper_bound' : 0,
      'fix_init_Q' : 0,
      'small_value' : 0.0001,
      'ignoreUp2' : 100 }
    

p['t_thresh'] = sp.stats.t.isf(1.0 * p['FP_rate'], p['sample_N'])

''' Anomalous Data Parameters '''

a = { 'N' : 50, 
      'T' : 1000, 
      'periods' : [15, 50, 70, 90], #[15, 40, 70, 90,120], 
      'L' : 10, 
      'L2' : 200, 
      'M' : 5, 
      'pA' : 0.1, 
      'noise_sig' : 0.1,
      'seed' : None}

anomaly_type = 'peak_dip'

gen_funcs = dict(peak_dip = gen_a_peak_dip,
                 grad_persist = gen_a_grad_persist,
                 step = gen_a_step,
                 trend = gen_a_periodic_shift)


''' Create/Load Dataset '''
D = gen_funcs[anomaly_type](**a)  
data = D['data']

#data = load_ts_data('isp_routers', 'full')
#execfile('/Users/chris/Dropbox/Work/MacSpyder/Utils/gen_simple_peakORshift_data.py')
#data = B

''' Mean Centering/Pre-Processing '''
data = zscore_win(data, 100)
#cdata = zscore(data)
z_iter = iter(data)
numStreams = data.shape[1]

'''Initialise'''
Frahst_alg = FRAHST('F-7.A-recS.R-static', p, numStreams)

'''Begin Frahst'''
# Main iterative loop. 
for zt in z_iter:
  zt = zt.reshape(zt.shape[0],1)   # Convert to a column Vector 

  if Frahst_alg.st['anomaly'] == True:
    Frahst_alg.st['anomaly'] = False # reset anomaly var if anomaly at t-1

  ''' Frahst Method '''
  Frahst_alg.run(zt)
  # Calculate reconstructed data if needed
  st = Frahst_alg.st
  Frahst_alg.st['recon'] = np.dot(st['Q'][:,:st['r']],st['ht'][:st['r']])

  '''Anomaly Detection Method''' 
  Frahst_alg.detect_anom(zt)

  '''Rank adaptation Method''' 
  Frahst_alg.rank_adjust(zt)

  ''' Store data ''' 
  #tracked_values = ['ht','e_ratio','r','recon', 'pred_err', 'pred_err_norm', 'pred_err_ave', 't_stat', 'pred_dsn', 'pred_zt']   
  #tracked_values = ['ht','e_ratio','r','recon','recon_err', 'recon_err_norm', 't_stat', 'rec_dsn']
  #tracked_values = ['ht','e_ratio','r', 't_stat', 'rec_dsn', 'eig_val', 'recon', 'exp_ht']
  tracked_values = ['ht','e_ratio','r', 't_stat', 'rec_dsn', 'eig_val', 'recon']

  Frahst_alg.track_var(tracked_values)
  #Frahst_alg.track_var()

''' Plot Results '''
#Frahst_alg.plot_res([data, 'ht', 't_stat'])
Frahst_alg.plot_res([data, 'ht', 'r', 'e_ratio'], hline = 0)
Frahst_alg.plot_res([data, 'ht', 'rec_dsn', 't_stat'])

#Frahst_alg.analysis(D['gt'])  