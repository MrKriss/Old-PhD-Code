#!/usr/bin/env python
#coding:utf-8
# Author:   --<>
# Purpose: 
# Created: 12/12/11

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import numpy.lib.recfunctions as nprec
import sys
import os 
import time 

from gen_anom_data import gen_a_peak_dip, gen_a_grad_persist, gen_a_step
from normalisationFunc import zscore, zscore_win
from Frahst_class import FRAHST


"""
Code Description: Runs Batch of experiments on synthetic anomalous data  


TODO : - Getting too many seg faults. Rewrite so as to store datasets in files? 
     : - Store names/pointers to files/variables in an array  

  .
"""

def gen_data_str(a, dat_changes, init_c, seed = 0):
  """ Function to genrte data array structure"""

  # May wish to change pA
  if 'pA' in dat_changes:
    pA_ = np.max(dat_changes['pA'])
  else:
    pA_ = a['pA']

  # Get (max) num of Anomalies
  if pA_ < 1:
    numAnom = np.floor(pA_ * a['N'])
  else:
    numAnom = pA_ 

  dt = ([('data', np.float, (a['T'], a['N'])), 
         ('start', np.int_, (numAnom,)), 
         ('loc', np.int_, (numAnom,)),
         ('len', np.int_, (numAnom,)),
         ('mag', np.int_, (numAnom,)),
         ('type', 'a10', (numAnom,) )])

  # Big Data Structure Array
  D = np.zeros((init_c,len(dat_changes.values()[0])), dtype = dt)

  # For each Change to Dataset Parameters
  var = dat_changes.keys()[0]
  values = dat_changes.values()[0]
  for j, v in enumerate(values):
    # update data set parameters a
    a[var] = v
    for i in xrange(init_c):
      if seed == 1:
        a['seed'] = i    
      # Generate the data
      temp = gen_funcs[anomaly_type](**a) # so tidy!
      nprec.recursive_fill_fields(temp['gt'], D[i,j])
    
  return D

def gen_alg_str(p, alg_changes, alg_versions):
  """ function to generate algorithm array structure """

  dt = ([('alg', np.object), 
           ('ver', np.str)])
  
  # Big algorithm storage structure
  A = np.zeros( (len(alg_versions) , len(alg_changes.values()) ), dtype = dt)

  '''Generate algorithm instances'''
  # For each Change to Algorith Parameters 
  var = alg_changes.keys()[0]
  values = alg_changes.values()[0]
  for j, val in enumerate(values):
    p[var] = val
    for i, ver in enumerate(alg_versions): 
      A[i,j]['alg'] = FRAHST(ver, p)
      A[i,j]['ver'] = ver

  return A

def gen_res_str(D,A):
  """ generate structured array to hold results 
  
  i x j x k  =  Algorithm version x alg_params x data_parmas 
  
  """

  dt = ([('alg', np.object), 
        ('dat_idx', np.int)])

  R = np.zeros((A.shape[0], A.shape[1], D.shape[1]), dtype = dt)
  for i in xrange(A.shape[0]):
    for j in xrange(A.shape[1]):
      for k in xrange(D.shape[1]):
        R[i,j,k]['alg'] = A[i,j]['alg']
        R[i,j,k]['dat_idx'] = k
  
  return R


'''Batch Parameters'''
#-----Fixed-----#
# Path Setup 
exp_name = 'Test_a_signals'
results_path = '/Users/chris/Dropbox/Work/MacSpyder/Results/'
cwd = os.getcwd()
path = os.path.join(results_path, exp_name)

initial_conditions = 5   # i - No. of generated data sets to test

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
     'ht_AR_win' : 30, 'AR_order' : 1, 'x_thresh' : 1.5, 
     # Statistical 
     'sample_N' : 20, 'dependency_lag' : 20, 't_thresh' : None, 'FP_rate' : 10**-5,
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

p['t_thresh'] = sp.stats.t.isf(0.5 * p['FP_rate'], p['sample_N'])

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
alg_versions = ['F-7.A-recS.R-eig', 'F-7.A-recS.R-eng']
#alg_versions = ['F-7.A-recS.R-eig', 'F-7.A-recS.R-eng', 
                                #'F-7.A-forS.R-eig', 'F-7.A-forS.R-eng', 'F-7.A-forS.R-static' ]

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
anomalies_table = []
#anomaly_table = np.zeros(len(D['gt']), dtype = [('run','i4'),('start','i4'),('loc','i4'),('len','i4'),('mag','i4'),('type','a10')])

'''Generate Data Set'''
D = gen_data_str(a, dat_changes, initial_conditions)
print 'Generated Dataset array'

'''Generate algorithm instances'''
A = gen_alg_str(p,alg_changes, alg_versions)
print 'Generated Algorithm array'

''' Generate Results Structure '''
R = gen_res_str(D,A)

for alg in xrange(A.shape[0]): # for each alg version 
  for param_set in xrange(A.shape[1]): # for each parameter sey for the alg version 
    for j in xrange(D.shape[1]): # for each data set parameter set 

      anomalies_table = []
      for i in xrange(D.shape[0]):  # for each initial condition

        # Fetch alg
        F = A[alg,param_set]['alg']
        # Fetch data 
        data = D[i,j]['data']
        data = zscore_win(data, F.p['z_win'])

        '''Initialise'''      
        z_iter = iter(data)
        numStreams = data.shape[1]
        F.re_init(numStreams)

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

          '''Store Values'''
          F.track_var()

        # Record Anomalies over whole batch
        anomalies_table.append(F.res['anomalies'])
      
      # Anamlyise  FP FN TP etc. 
      F.batch_analysis(D[0,j], anomalies_table, keep_sets = 0)
      
      R[alg,param_set, j]['alg'] = F
      #R[alg,param_set, j]['dat_idx'] = 

