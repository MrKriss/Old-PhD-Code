#!/usr/bin/env python
#coding:utf-8
# Author:  C Mjusselle --<>
# Created: 09/13/11

import numpy as np
import matplotlib.pyplot as plt
import sys
import os 
from EWMA_filter import EWMA_filter
from load_data import load_data

"""
Code Description: 
    Find anomalies in TS data using EWMA and fourier analisis.
"""

def EWMA_anomalies(data, alpha, threshold, EWMA_mode = 'min', thresh_mode = 'abs'):
  """ Takes array of data and analises each stream for EWMA anomalies. 
  Records location and magnitude of each anomaly + Gives states 
  
  data is a t (time) x n (num_streams) array
  
  EWMA_mode - dictates EWMA basline from which residules are calculated 
   -forward: EWMA on data in forward direction 
   -backward: EWMA on data in backward direction
   -min: takes minimum 
   
  thres_mode - decides whether threshold is absolute or relative to moving average of data 
             - Could also just normalise each TS data to its max value, but mag of anomalies is vast 
               so would end up scaling down the remainder of the data. 
               
               leaning towards using EWMA_mode = min, and rel_res2 as default
               
               just noticed that due to initial period EWMA takes to approch correct value, the end of aved data 
               in mode = min will drop and give  false anomally
               will 
               
               Now set top use Residual2 as default 
                    Residual 2 = abs(min(EWMA_for, EWMA_back) - Time-series)
  """

  T = data.shape[0]
  N = data.shape[1]
  
  # Initalise Data Structures
  store = {'residual' : np.zeros((T,N)),
           'residual2' : np.zeros((T,N)),
           'aved_data' : np.zeros((T,N)),
           'rel_res' :  np.zeros((T,N)),
           'rel_res2' :  np.zeros((T,N)),
           'anomaly_tab' : [0] * N, 
           'anomaly_mag' : [0] * N, 
           'anomaly_mag_res2' : [0] * N, 
           'anomaly_mag_norm' : [0] * N,
           'anomaly_mag_rel' : [0] * N}
  
  for i in range(N):
    time_series = data[:,i]
    
    if EWMA_mode == 'forward':
      store['residual'][:,i], store['aved_data'][:,i] = EWMA_filter(time_series, alpha)
    elif EWMA_mode == 'backwards':
      store['residual'][:,i], store['aved_data'][:,i] = EWMA_filter(time_series[::-1], alpha)
    else:
      res_for, aved_data_for = EWMA_filter(time_series, alpha)
      res_back, aved_data_back = EWMA_filter(time_series[::-1], alpha)
      if EWMA_mode == 'min':
        store['residual'][:,i] = np.minimum(res_for, res_back[::-1])
        store['aved_data'][:,i] = np.minimum(aved_data_for, aved_data_back[::-1])
        store['residual2'][:,i] = np.abs(time_series - store['aved_data'][:,i])
      elif EWMA_mode == 'max':
        store['residual'][:,i] = np.maximum(res_for,res_back[::-1])
        store['aved_data'][:,i] = np.maximum(aved_data_for, aved_data_back[::-1])
        store['residual2'][:,i] = np.abs(time_series - store['aved_data'][:,i])
      elif EWMA_mode == 'middle':
        store['residual'][:,i] = (np.minimum(res_for,res_back[::-1]) + np.maximum(res_for,res_back[::-1])) / 2.0 
        store['aved_data'][:,i] = (np.minimum(aved_data_for, aved_data_back[::-1]) + 
                                   np.maximum(aved_data_for, aved_data_back[::-1])) / 2.0
        store['residual2'][:,i] = np.abs(time_series - store['aved_data'][:,i])
    
    # Relative Residule (Residule / Moving_aved_data)
    nonzero_idx = store['aved_data'][:,i] > 0.0
    store['rel_res'][:,i] = store['residual'][:,i] 
    store['rel_res'][:,i] = store['residual'][:,i][nonzero_idx] / store['aved_data'][:,i][nonzero_idx]
    store['rel_res2'][:,i] = store['residual2'][:,i]
    store['rel_res2'][:,i] = store['residual2'][:,i][nonzero_idx] / store['aved_data'][:,i][nonzero_idx]
    
    # find anomalies 
    # Now uses Residual 2 as default 
    if thresh_mode == 'abs':
      anomalies = store['residual2'][:,i] > threshold
      # Ignore first and last 10 time steps due to EWMA adjusting 
      anomalies[:10] = 0
      anomalies[-10:] = 0
    elif thresh_mode == 'rel':
      anomalies = store['rel_res2'][:,i] > threshold
      # Ignore first and last 10 time steps due to EWMA adjusting 
      anomalies[:10] = 0
      anomalies[-10:] = 0
    
    # Anomalies table. col1 = time step. col2 = magnitude
    anomalies_table = np.zeros((anomalies.sum(),2))   
    anomalies_table[:,0] = anomalies.nonzero()[0]
    anomalies_table[:,1] = store['residual'][:,i][anomalies] # not + 1- grrrr arg XXX
    store['anomaly_tab'][i] = anomalies_table
    
    if anomalies.any(): # if any anomalies
      # Anomaly magnitudes - sorted in decending order
      store['anomaly_mag'][i] = np.sort(anomalies_table[:,1])[::-1]
      # Normalised to max anomaly mag
      store['anomaly_mag_norm'][i] = store['anomaly_mag'][i] / store['anomaly_mag'][i].max()
      
      # Relative anomaly magnitude (Residual Mag / Moving_aved_data)
      rel_anom_mag =  store['residual'][:,i][anomalies]
      rel_anom_mag =  store['residual'][:,i][anomalies][nonzero_idx]  / store['aved_data'][:,i][anomalies][nonzero_idx]
      
      store['anomaly_mag_rel'][i] = np.sort(rel_anom_mag)[::-1]
      store['anomaly_mag_rel2'][i] = np.sort(store['residual'])[::-1]
      
  
  return store
  
def plt_anom(data, store, number):
  
  data = data[:,number]
  aved_data = store['aved_data'][:,number]
  anomalies = store['anomaly_tab'][number][:,0]
  
  fig = plt.figure()
  ax1 = fig.add_subplot(2,1,1)
  ax1.plot(data)
  for x in anomalies:
      ax1.axvline(x, ymin = 0.75, color='r')
  ax1.plot(aved_data)

  ax2 = fig.add_subplot(2,1,2, sharex = ax1)
  ax2.plot(store['residual'][:,number])
  for x in anomalies:
      ax1.axvline(x, ymin = 0.25, color='r')
  
  ax1.set_xlim(0,data.shape[0])


def plt_res(data, store, number):
  
  data = data[:,number]
  aved_data = store['aved_data'][:,number]
  anomalies = store['anomaly_tab'][number][:,0]
  
  fig = plt.figure()
  ax1 = fig.add_subplot(5,1,1)
  ax1.plot(data)
  for x in anomalies:
      ax1.axvline(x, ymin = 0.75, color='r')
  ax1.plot(aved_data)
  ax1.set_ylabel('Data')
  
  ax2 = fig.add_subplot(5,1,2, sharex = ax1)
  ax2.plot(store['residual'][:,number])
  ax2.set_ylabel('Residual')
  
  ax3 = fig.add_subplot(5,1,3, sharex = ax1)
  ax3.plot(store['residual2'][:,number])
  ax3.set_ylabel('Residual 2')
  
  ax4 = fig.add_subplot(5,1,4, sharex = ax1)
  ax4.plot(store['rel_res'][:,number])
  ax4.set_ylabel('Rel\nResidual')
  
  ax5 = fig.add_subplot(5,1,5, sharex = ax1)
  ax5.plot(store['rel_res2'][:,number])
  ax5.set_ylabel('Rel\nResidual 2')
  
  ax1.set_xlim(0,data.shape[0])
  
if __name__=='__main__':
  
  packets, flows, byte = load_data('abilene')
  data = packets

  t0 = EWMA_anomalies(data, alpha = 0.3, threshold = 0.0, EWMA_mode = 'min', thresh_mode = 'rel')
  out1 = EWMA_anomalies(data, alpha = 0.3, threshold = 0.5, EWMA_mode = 'min', thresh_mode = 'rel')
  out2 = EWMA_anomalies(data, alpha = 0.3, threshold = 0.5, EWMA_mode = 'max', thresh_mode = 'rel')
  out3 = EWMA_anomalies(data, alpha = 0.3, threshold = 0.5, EWMA_mode = 'middle', thresh_mode = 'rel')
  