#!/usr/bin/env python
#coding:utf-8
# Author:  C Musselle --<>
# Purpose: Experiment Script file 
# Created: 12/12/11

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sys
import os 

from gen_anom_data import gen_a_peak_dip, gen_a_grad_persist, gen_a_step
from normalisationFunc import zscore, zscore_win
from Frahst_class import FRAHST

"""
Code Description: Main Experimental Script File 
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
      'err_thresh' : 1.5, 
      # Statistical 
      'sample_N' : 20,
      'dependency_lag' : 20,
      'x_thresh' : 10,
      'FP_rate' : 10**-5,
      # Q statistical 
      'Q_lag' : 5,
      'Q_alpha' : 0.05,
      # Eigen-Adaptive
      'F_min' : 0.9,
      'epsilon' : 0.05,
      # Pedro Adaptive
      'e_low' : 0.95,
      'e_high' : 0.98,
      'static_r' : 0,
      'r_upper_bound' : None,
      'fix_init_Q' : 0,
      'small_value' : 0.0001,
      'ignoreUp2' : 0 }
    
p['x_thresh'] = sp.stats.t.isf(0.5 * p['FP_rate'], p['sample_N'])

''' Anomalous Data Parameters '''

a = { 'N' : 50, 
      'T' : 1000, 
      'periods' : [15, 40, 70, 90,120], 
      'L' : 10, 
      'L2' : 200, 
      'M' : 3, 
      'pA' : 0.1, 
      'noise_sig' : 0.3 }

D = gen_a_grad_persist(**a)

#data = load_ts_data('isp_routers', 'full')
data = D['data']
data = zscore_win(data, 100)
z_iter = iter(data)
numStreams = data.shape[1]

'''Initialise'''
Frahst_alg = FRAHST('F-7.R-eig.A-recS', p, numStreams)

'''Begin Frahst'''
# Main iterative loop. 
for zt in z_iter:

  zt = zt.reshape(zt.shape[0],1)   # Convert to a column Vector 

  if Frahst_alg.st['anomaly'] == True:
    Frahst_alg.st['anomaly'] = False # reset anomaly var

  '''Frahst Version '''
  Frahst_alg.run(zt)
  # Calculate reconstructed data if needed
  st = Frahst_alg.st
  Frahst_alg.st['recon'] = np.dot(st['Q'][:,:st['r']],st['ht'][:st['r']])

  '''Anomaly Detection method''' 
  Frahst_alg.detect_anom(zt)

  '''Rank adaptation method''' 
  Frahst_alg.rank_adjust(zt)

  '''Store data''' 
  #tracked_values = ['ht','e_ratio','r','recon', 'pred_err', 'pred_err_norm', 'pred_err_ave', 't_stat', 'pred_dsn', 'pred_zt']   
  tracked_values = ['ht','e_ratio','r','recon','recon_err', 'recon_err_norm', 't_stat', 'rec_dsn', 'x_sample']
  #tracked_values = ['ht','e_ratio','r','recon', 'h_res', 'h_res_aa', 'h_res_norm']

  Frahst_alg.track_var(tracked_values)

''' Plot Results '''
#Frahst_alg.plot_res([data, 'ht', 't_stat'])
Frahst_alg.plot_res([data, 'ht', 'rec_dsn', 't_stat'])

  