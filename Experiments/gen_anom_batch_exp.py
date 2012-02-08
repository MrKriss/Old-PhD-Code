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
import pickle

from gen_anom_data import gen_a_peak_dip, gen_a_grad_persist, gen_a_step, gen_a_step_n_back, gen_a_periodic_shift
from normalisationFunc import zscore, zscore_win
from Frahst_class import FRAHST
from utils import GetInHMS

"""
Code Description: Runs Batch of experiments on synthetic anomalous data  
"""

class E_str():
  """" A Class to hold all experimental data on parameters and algorithm results """

  def __init__(self, a, dat_changes, p, alg_changes, alg_versions, path, init_c):
  #def gen_res_str(self, a, dat_changes, p, alg_changes, alg_versions):
    """ generate structured array to hold results 
    
    i x j x k  =  Algorithm version x alg_params x data_parmas 
    
    Also generates FRAHST dictionary  
    
    """
    self.k = dat_changes
    self.j = alg_changes
    self.i = alg_versions
  
    # Data type for Results structure 
    dt = ([ ('key', 'a5'),  # Key to FRAHST instance 
            ('met', [('TP', 'i4'), # Metrics
                      ('FP', 'i4'),
                      ('TN', 'i4'),
                      ('FN', 'i4'),
                      ('precision', 'f4'),
                      ('recall', 'f4'),
                      ('F05', 'f4'),
                      ('F1', 'f4'),
                      ('F2', 'f4'),
                      ('ACC', 'f4'),
                      ('FDR', 'f4'),
                      ('FPR', 'f4') ]),
             ('params', [('alg', [ ('alpha', 'f4') ,   # Alg Parameters
                                   ('init_r', 'i2') ,
                                   ('holdOffTime', 'i2') ,
                                   ('EWMA_filter_alpha', 'f4') ,
                                   ('residual_thresh', 'f4') ,
                                   ('ht_AR_win', 'i2') ,
                                   ('AR_order', 'i1') ,
                                   ('x_thresh', 'f4') ,
                                   ('sample_N', 'i4') ,
                                   ('dependency_lag', 'i4') ,
                                   ('t_thresh', 'f4') ,
                                   ('FP_rate', 'f4') ,
                                   ('F_min', 'f4') ,
                                   ('epsilon', 'f4') ,
                                   ('e_low', 'f4') ,
                                   ('e_high', 'f4') ,
                                   ('r_upper_bound', 'i4') ,
                                   ('fix_init_Q', np.bool) ,
                                   ('small_value', 'f4') ,
                                   ('ignoreUp2', 'i2') ,
                                   ('z_win', 'i2') ]), 
                         ('dat', [ ('N', 'i4'),             # Data Parameters
                                  ('T', 'i4'),
                                  ('periods', np.object),
                                  ('L', 'i2'),
                                  ('L2', 'i2'),
                                  ('M', 'i2'),
                                  ('pA', 'f4'),
                                  ('noise_sig', 'f4')])
                         ])
             ])
  
    # Big results Structure 
    self.R = np.zeros((len(alg_versions),                       # i 
                   len(alg_changes.values()[0]),              # j 
                   len(dat_changes.values()[0])), dtype = dt) # k
  
    # FRahst Dictionary 
    self.F_dict = {}
  
    # For each Change to Algorith Parameters - j 
    alg_var = alg_changes.keys()[0]
    alg_values = alg_changes.values()[0]
    
    # For each Change to Dataset Parameters - k
    dat_var = dat_changes.keys()[0]
    dat_values = dat_changes.values()[0]
  
    for i in xrange(self.R.shape[0]):     # alg version
      for j in xrange(self.R.shape[1]):   # alg parameters
        # update p
        p[alg_var] = alg_values[j]
        
        if alg_var == 'FP_rate':
          # update threshold 
          p['t_thresh'] = sp.stats.t.isf(1.0 * p['FP_rate'], p['sample_N'])
        
        for k in xrange(self.R.shape[2]): # data parameters
          # update a
          a[dat_var] = dat_values[k]
          
          key = 'F' + str(i) + str(j) + str(k)
          self.F_dict[key] = FRAHST(alg_versions[i], p.copy(), a['N'])
          
          # Fill in rest of values
          self.R[i,j,k]['key'] = key
          self.R[i,j,k]['params']['alg']['alpha'] = p['alpha'] 
          self.R[i,j,k]['params']['alg']['init_r'] = p['init_r'] 
          self.R[i,j,k]['params']['alg']['holdOffTime'] = p['holdOffTime'] 
          self.R[i,j,k]['params']['alg']['EWMA_filter_alpha'] = p['EWMA_filter_alpha'] 
          self.R[i,j,k]['params']['alg']['residual_thresh'] = p['residual_thresh'] 
          self.R[i,j,k]['params']['alg']['ht_AR_win'] = p['ht_AR_win'] 
          self.R[i,j,k]['params']['alg']['AR_order'] = p['AR_order'] 
          self.R[i,j,k]['params']['alg']['x_thresh'] = p['x_thresh'] 
          self.R[i,j,k]['params']['alg']['sample_N'] = p['sample_N'] 
          self.R[i,j,k]['params']['alg']['dependency_lag'] = p['dependency_lag'] 
          self.R[i,j,k]['params']['alg']['t_thresh'] = p['t_thresh'] 
          self.R[i,j,k]['params']['alg']['FP_rate'] = p['FP_rate'] 
          self.R[i,j,k]['params']['alg']['F_min'] = p['F_min'] 
          self.R[i,j,k]['params']['alg']['epsilon'] = p['epsilon'] 
          self.R[i,j,k]['params']['alg']['e_low'] = p['e_low'] 
          self.R[i,j,k]['params']['alg']['e_high'] = p['e_high'] 
          self.R[i,j,k]['params']['alg']['r_upper_bound'] = p['r_upper_bound'] 
          self.R[i,j,k]['params']['alg']['fix_init_Q'] = p['fix_init_Q'] 
          self.R[i,j,k]['params']['alg']['small_value'] = p['small_value'] 
          self.R[i,j,k]['params']['alg']['ignoreUp2'] = p['ignoreUp2'] 
          self.R[i,j,k]['params']['alg']['z_win'] = p['z_win'] 
          self.R[i,j,k]['params']['dat']['N'] = a['N']
          self.R[i,j,k]['params']['dat']['T'] = a['T']
          self.R[i,j,k]['params']['dat']['periods'] = a['periods']
          self.R[i,j,k]['params']['dat']['L'] = a['L']
          self.R[i,j,k]['params']['dat']['L2'] = a['L2']
          self.R[i,j,k]['params']['dat']['M'] = a['M']
          self.R[i,j,k]['params']['dat']['pA'] = a['pA']
          self.R[i,j,k]['params']['dat']['noise_sig'] = a['noise_sig']
 
    # Aditional Experimental variables worth storing 
    self.alg_ver = alg_versions
    self.alg_par = alg_changes
    self.dat_par = dat_changes
    self.path = path
    self.init_c = init_c

  def search_met(self,var, val):
    """ Searches metrics and returns values """
    
    X = self.R['met'][var]
  
  def show(self,i,j,k, vars2plot, number2show = 1):
    """ Plots the Frahst instance vars for record requested """
    
    for a in xrange(number2show):
    
      # Load data file 
      data_var = E.dat_par.keys()[0]
      val = E.dat_par.values()[0][k]    
      filename = 'D_' + data_var + '=' + str(val) + '_' + str(a) + '.npy'
      data = np.load(self.path + '/' + filename)
      
      # Get Frahst instance 
      key = 'F' + str(i) + str(j) + str(k)
      F = E.F_dict[key]
      
      # plot Results for the entry  
      var = []
      var.append(data)
      var.extend(vars2plot)
      F.plot_res(var, hline = 0)
  
  def write2csv(self, metric, filename, header = 0):    
      
      data = self.R['met'][metric]
      
      # Write the array to disk
      with file(filename, 'w') as outfile:
          # I'm writing a header here just for the sake of readability
          # Any line starting with "#" will be ignored by numpy.loadtxt
          if header:
            outfile.write('# Array shape: {0}\n'.format(data.shape))
            outfile.write('# E.i: {0}\n'.format(self.i))
            outfile.write('# E.j: {0}\n'.format(self.j))
            outfile.write('# E.k: {0}\n'.format(self.k))
            outfile.write('\n')
          
          outfile.write('Values for {0} metric\n'.format(metric))
          outfile.write('\n')
      
          # Iterating through a ndimensional array produces slices along
          # the last axis. This is equivalent to data[i,:,:] in this case
          
          for i, data_slice in enumerate(data):
              
              outfile.write('Algorithm: {0}\n'.format(self.alg_ver[i]))
      
              # The formatting string indicates that I'm writing out
              # the values in left-justified columns 7 characters in width
              # with 2 decimal places.
              
              np.savetxt(outfile, data_slice, fmt='%-6d', delimiter = ', ')
      
              # Writing out a break to indicate different slices...
              outfile.write('\n')  


'''Other functions '''
def gen_data_str(a, dat_changes, init_c, path, anom_type, seed = 0):
  """ Function to genrte data array structure to hold ground truths and pointers to data files """

  # May wish to change pA
  if 'pA' in dat_changes:
    pA_ = np.max(dat_changes['pA'])
  else:
    pA_ = a['pA']
  # Changing N will also affect numAnom
  if 'N' in dat_changes:
    N_ = np.max(dat_changes['N'])
  else:
    N_ = a['N']

  # Get (max) num of Anomalies
  if pA_ < 1:
    numAnom = np.floor(pA_ * N_)
  else:
    numAnom = pA_ 

  # Double anomalous entries if 2 anomalies per run 
  if anom_type == 'grad_persist' or anom_type == 'step_n_back':
    numAnom *= 2

  # Data info Data type 
  dt = ([('file', 'a24'),
         ('gt',[('start', np.int_, (numAnom,)), 
                ('loc', np.int_, (numAnom,)),
                ('len', np.int_, (numAnom,)),
                ('mag', np.int_, (numAnom,)),
                ('type', 'a10', (numAnom,) )] ),
         ])

  # Big Results Structure Array
  D = np.zeros((init_c,
                len(dat_changes.values()[0]) ), dtype = dt)

  # For each Change to Dataset Parameters 
  dat_var = dat_changes.keys()[0]
  dat_values = dat_changes.values()[0]
  
  for j, v in enumerate(dat_values):
    # update data set parameters a
    a[dat_var] = v
    for i in xrange(init_c):
      if seed == 1:
        a['seed'] = i    

      filename = 'D_' + dat_var + '=' + str(v) + '_' + str(i) + '.npy'
      # Generate the data
      temp = gen_funcs[anomaly_type](**a.copy()) # so tidy!
      # Write to file 
      np.save(path + '/' + filename, temp['data'])
      # fill in D str
      nprec.recursive_fill_fields(temp['gt'], D[i,j]['gt'])
      D[i,j]['file'] = filename
  
  return D



if __name__=='__main__':
  
  '''Batch Parameters'''
  #-----Fixed-----#
  # Path Setup 
  exp_name = 'Exp-shift_up_down'
  
  results_path = '/Users/chris/Dropbox/Work/MacSpyder/Results/'
  cwd = os.getcwd()
  path = os.path.join(results_path, exp_name)
  if not os.path.exists(path):
    os.mkdir(path)
  
  initial_conditions = 100  # i - No. of generated data sets to test
  
  anomaly_type = 'grad_persist'
  gen_funcs = dict(peak_dip = gen_a_peak_dip,
                   grad_persist = gen_a_grad_persist,
                   step = gen_a_step,
                   step_n_back = gen_a_step_n_back,
                   trend = gen_a_periodic_shift)
  
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
  #alg_versions = ['F-7.A-recS.R-eig', 'F-7.A-recS.R-eng'] 
  #alg_versions = ['F-3.A-eng.R-eng' ]  
  alg_versions = ['F-3.A-eng.R-eng', 'F-7.A-recS.R-eng', 'F-7.A-recS.R-eig', 'F-7.A-recS.R-expht' , 'F-7.A-recS.R-static']  
  #alg_versions = ['F-7.A-recS.R-eig', 'F-7.A-recS.R-eng', 
                                  #'F-7.A-forS.R-eig', 'F-7.A-forS.R-eng', 'F-7.A-forS.R-static' ]
  
  # Data set changes 
  dat_changes = {'noise_sig' : [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]} # Need min one entry for loop
  #dat_changes = {'noise_sig' : [0.1]} # Need min one entry for loop
  #dat_changes = dict(noise_sig = [0.0, 0.1, 0.2])
  
  # Algorithm Changes
  alg_changes = {'FP_rate' : [10**-2, 10**-3, 10**-4, 10**-5, 10**-6]} # need min one entry for loop 
  #alg_changes = {'FP_rate' : [10**-2, 10**-4, 10**-5]} # need min one entry for loop 
  #alg_changes = {'FP_rate' : [10**-6]} # need min one entry for loop 
  #alg_changes = dict(F_min = [0.95, 0.9, 0.85, 0.8],
                                        #alpha = [0.99, 0.98, 0.97, 0.96])
  
  # Conting iterations 
  alg_ver_count = len(alg_versions)
  alg_change_count = 0
  for v in alg_changes.values(): alg_change_count += len(v)
  dat_change_count = 0
  for v in dat_changes.values(): dat_change_count += len(v)
  
  loop_count = 0
  total_loops = alg_ver_count * alg_change_count * dat_change_count
  
  # For Profiling 
  start = time.time() 
  anomalies_list = []
  gt_list = []
  
  '''Generate Data Set'''
  D = gen_data_str(a, dat_changes, initial_conditions, path, anomaly_type)
  print 'Generated Dataset array'
  
  ''' Generate Results Structure '''
  import gen_anom_batch_exp
  E = gen_anom_batch_exp.E_str(a, dat_changes, p, alg_changes, alg_versions, path, initial_conditions)
  
  print 'Generated Results Array'
  
  for i in xrange(E.R.shape[0]): # for each alg version 
    for j in xrange(E.R.shape[1]): # for each parameter set for the alg version 
      for k in xrange(E.R.shape[2]): # for each data set parameter set 
  
        anomalies_list = []
        gt_list = []
        for ic in xrange(D.shape[0]):  # for each initial condition
  
          # Fetch alg
          F = E.F_dict[E.R[i,j,k]['key']]
          # Fetch data 
          data = np.load(path + '/' + D[ic,k]['file'])
          #data = data.reshape(E.R[i,j,k]['params']['dat']['T'], E.R[i,j,k]['params']['dat']['N'])
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
          anomalies_list.append(F.res['anomalies'][:])
          gt_list.append(D[ic,k])
        
        # Anamlyise  FP FN TP etc. 
        F.batch_analysis(gt_list, anomalies_list, keep_sets = 0)
        
        # Fill in R
        E.R[i,j,k]['met']['TP'] = F.metric['TP']
        E.R[i,j,k]['met']['FP'] = F.metric['FP']
        E.R[i,j,k]['met']['TN'] = F.metric['TN']
        E.R[i,j,k]['met']['FN'] = F.metric['FN']
        E.R[i,j,k]['met']['precision'] = F.metric['precision']
        E.R[i,j,k]['met']['recall'] = F.metric['recall']
        E.R[i,j,k]['met']['F05'] = F.metric['F05']
        E.R[i,j,k]['met']['F1'] = F.metric['F1']
        E.R[i,j,k]['met']['F2'] = F.metric['F2']
        E.R[i,j,k]['met']['ACC'] = F.metric['ACC']
        E.R[i,j,k]['met']['FDR'] = F.metric['FDR']
        E.R[i,j,k]['met']['FPR'] = F.metric['FPR']
      
    ti = time.time() - start 
    tim = GetInHMS(ti)
    print 'Finished Algorithm ver %i of %i at %s' % (i+1, alg_ver_count, tim)
  
  fin = time.time() - start
  hms = GetInHMS(fin)
  print 'Finished a total of %i iterations in %s' % (total_loops, hms)
  print 'Average of %f seconds per iteration' % (fin / total_loops)
  
  with open( path + '/' + "E.p", "wb" ) as f:
    pickle.dump(E, f)