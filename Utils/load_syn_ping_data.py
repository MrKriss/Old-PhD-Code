#!/usr/bin/env python
#coding:utf-8
# Author:  C Musselle 
# Purpose: Utility fuctions 
# Created: 08/09/11

import numpy as np
import matplotlib.pyplot as plt
import sys
import os 
from MAfunctions import MA_over_window

"""
Code Description:
  Load in Julies Ping Scan and Syn Scan Data sets.
  
  2 possible methods 
    1) convert all data to data matrix of signals + Antigen Frequencies
    2) Read in data straight from file - may be simpler...
"""

#----------------------------------------------------------------------

data_directory = '/Users/chris/DataSets/' 

def data_loader(data2load, varient):
  """Load in SYN scan data straight from file """
  
  if data2load == 'SYN' :
    data_path = data_directory + 'Syn Scan/jqg_' + varient + '.tcr.log'
  elif data2load == 'Ping' :
    data_path = data_directory + 'Ping Scan/s' + varient + '_norm.log'

  with open(data_path, 'Ur') as dfile:
    '''Main File iterator'''

    ant_dict = {}    
    for line in dfile:
      l = line.split()
      time_stamp = l[0]
      data_type = l[1]
      if data_type == 'signal':
        # Yield all you have so far, plus signal values
        signals = np.array([float(x) for x in l[2:]])
        yield signals, ant_dict, time_stamp
        ant_dict = {}
        
      else:
        # Rack up Antigen Count 
        pid = l[2]
        if not ant_dict.has_key(pid):
          ant_dict[pid] = 1
        else:
          ant_dict[pid] += 1
  
          
#----------------------------------------------------------------------
def load_n_store(data2load, variant):
  """Creates and runs the generator and stores all values returned """
  
  data_gen = data_loader(data2load, variant)

  signal_matrix = 0
  time_stamps = []
  antigen_list = []
  for signal, antigen, time in data_gen: 
    
    # Store signals
    if type(signal_matrix) == int : # first loop
      signal_matrix = np.atleast_2d(signal)
      time_stamps.append(time)
      start_time = float(time)
      antigen_list.append(antigen)
    else:
      signal_matrix = np.vstack((signal_matrix, signal))
      time_stamps.append(time)
      # Store antigen
      antigen_list.append(antigen)        

  time_secs = np.array(time_stamps, dtype = float).round(decimals = 1)
  time_secs = time_secs - np.round(start_time, decimals = 1)
  
  return signal_matrix, antigen_list, time_secs

if __name__=='__main__':
  
  """Extract data, preprocess and plot figures in acceptable range"""
  sig_AN, ant_AN, time_AN = load_n_store('SYN', 'AN')
  sig_PN, ant_PN, time_PN = load_n_store('SYN', 'PN')

  L = 20  
  
  AN_sig_av = MA_over_window(sig_AN, L)
  PN_sig_av = MA_over_window(sig_PN, L)
  
  labs = ['s0','s1','s2','s3','s4','s5','s6','s7','s8']
  
  fig1 = plt.figure()
  plt.plot(time_AN, AN_sig_av)
  plt.xlim(0, 7000)
  plt.title('Active Normal Processes alongside Anomaly (MA of %s)' % (L))
  plt.legend(fig1.axes[0].lines, labs)
  
  fig2 = plt.figure()
  plt.plot(time_PN, PN_sig_av)
  plt.xlim(0, 7000)
  plt.title('Passive Normal Processes alongside Anomaly (MA of %s)' % (L))
  plt.legend(fig2.axes[0].lines, labs)