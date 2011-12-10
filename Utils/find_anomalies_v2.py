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
from Frahst_v6_3 import simple_sins, simple_sins_3z
from load_syn_ping_data import load_n_store
from MAfunctions import MA_over_window

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
               
               Now set top use Residual2 as default ONLY
                    Residual 2 = abs(min(EWMA_for, EWMA_back) - Time-series)
  """

  T = data.shape[0]
  N = data.shape[1]
  
  # Initalise Data Structures
  store = {'residual' : np.zeros((T,N)),
           'aved_data' : np.zeros((T,N)),
           'rel_res' :  np.zeros((T,N)),
           'rel_res2' :  np.zeros((T,N)),
           'anomaly_tab' : [0] * N, 
           'anomaly_mag' : [0] * N, 
           'anomaly_mag_rel2' : [0] * N, 
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
        # store['residual'][:,i] = np.minimum(res_for, res_back[::-1])
        store['aved_data'][:,i] = np.minimum(aved_data_for, aved_data_back[::-1])
        store['residual'][:,i] = np.abs(time_series - store['aved_data'][:,i])
      elif EWMA_mode == 'max':
        # store['residual'][:,i] = np.maximum(res_for,res_back[::-1])
        store['aved_data'][:,i] = np.maximum(aved_data_for, aved_data_back[::-1])
        store['residual'][:,i] = np.abs(time_series - store['aved_data'][:,i])
      elif EWMA_mode == 'middle':
        # store['residual'][:,i] = (np.minimum(res_for,res_back[::-1]) + np.maximum(res_for,res_back[::-1])) / 2.0 
        store['aved_data'][:,i] = (np.minimum(aved_data_for, aved_data_back[::-1]) + 
                                   np.maximum(aved_data_for, aved_data_back[::-1])) / 2.0
        store['residual'][:,i] = np.abs(time_series - store['aved_data'][:,i])
    
    # Relative Residule (Residule / Moving_aved_data)
    # disregard small values, count data should be on order of 100s to 1000s
    nonzero_idx = store['aved_data'][:,i] > 0.1 
    store['rel_res'][:,i] = store['residual'][:,i].copy() 
    store['rel_res'][:,i][nonzero_idx] = store['residual'][:,i][nonzero_idx] / store['aved_data'][:,i][nonzero_idx]
    #store['rel_res2'][:,i] = store['residual2'][:,i]
    #store['rel_res2'][:,i] = store['residual2'][:,i][nonzero_idx] / store['aved_data'][:,i][nonzero_idx]
    
    # find anomalies 
    # Now uses Residual 2 as default 
    if thresh_mode == 'abs':
      anomalies = store['residual'][:,i] > threshold
      # Ignore first and last 10 time steps due to EWMA adjusting 
      anomalies[:10] = 0
      anomalies[-10:] = 0
    elif thresh_mode == 'rel':
      anomalies = store['rel_res'][:,i] > threshold
      # Ignore first and last 10 time steps due to EWMA adjusting 
      anomalies[:10] = 0
      anomalies[-10:] = 0
    
    # Anomalies table. col1 = time step. col2 = magnitude
    anomalies_table = np.zeros((anomalies.sum(),2))   
    anomalies_table[:,0] = anomalies.nonzero()[0]
    anomalies_table[:,1] = store['residual'][:,i][anomalies].copy() # not + 1- grrrr arg XXX
    store['anomaly_tab'][i] = anomalies_table
    
    if anomalies.any(): # if any anomalies
      # Anomaly magnitudes - sorted in decending order
      store['anomaly_mag'][i] = np.sort(anomalies_table[:,1])[::-1]
      # Normalised to max anomaly mag
      store['anomaly_mag_norm'][i] = store['anomaly_mag'][i] / store['anomaly_mag'][i].max()
      
      # Relative anomaly magnitude (Residual Mag / Moving_aved_data)
      nonzero_idx = store['aved_data'][:,i][anomalies] > 0.1
      res_anom =  store['residual'][:,i][anomalies].copy()
      aved_data_anom = store['aved_data'][:,i][anomalies].copy()
      
      rel_anom_mag =  res_anom.copy()
      rel_anom_mag[nonzero_idx] =  res_anom[nonzero_idx]  / aved_data_anom[nonzero_idx]
      
      store['anomaly_mag_rel'][i] = np.sort(rel_anom_mag)[::-1]
      store['anomaly_mag_rel'][i] = np.sort(store['rel_res'][:,i][anomalies])[::-1]
      
  
  return store
  
def plt_anom(data, store, number):
  
  data = data[:,number]
  aved_data = store['aved_data'][:,number]
  anomalies = store['anomaly_tab'][number][:,0]
  
  fig = plt.figure()
  ax1 = fig.add_subplot(2,1,1)
  ax1.plot(data)
  for x in anomalies:
      ax1.axvline(x, ymin = 0.85, color='r')
  ax1.plot(aved_data)

  ax2 = fig.add_subplot(2,1,2, sharex = ax1)
  ax2.plot(store['residual'][:,number])
  for x in anomalies:
      ax2.axvline(x, ymin = 0.85, color='r')
  
  ax1.set_xlim(0,data.shape[0])


def plt_res(data, store, number):
  
  data = data[:,number]
  aved_data = store['aved_data'][:,number]
  anomalies = store['anomaly_tab'][number][:,0]
  
  fig = plt.figure()
  ax1 = fig.add_subplot(3,1,1)
  ax1.plot(data)
  for x in anomalies:
      ax1.axvline(x, ymin = 0.75, color='r')
  ax1.plot(aved_data)
  ax1.set_ylabel('Data')
  
  ax2 = fig.add_subplot(3,1,2, sharex = ax1)
  ax2.plot(store['residual'][:,number])
  ax2.set_ylabel('Residual')
  
  ax3 = fig.add_subplot(3,1,3, sharex = ax1)
  ax3.plot(store['rel_res'][:,number])
  ax3.set_ylabel('Relative\nResidual')
  
  ax1.set_xlim(0,data.shape[0])
  
if __name__=='__main__':
  
  
  first = 1 
  if first:
      # data = genCosSignals(0, -3.0)
      
      # data, G = create_abilene_links_data()
      
      # execfile('/Users/chris/Dropbox/Work/MacSpyder/Utils/gen_Anomalous_peakORshift_data.py')
      # data = A
      
      #data = simple_sins(10,10,10,25, 0.1)
      
      #data = simple_sins_3z(10,10,13,13, 10, 27, 0.0)
      
      #data = genCosSignals_no_rand(timesteps = 10000, N = 32)  
      
      #data = array([[0,0,0], [1,2,2], [1,3,4], [3,6,6], [5,6,10], [6,8,11]])   
      
      #sig_PN, ant_PN, time_PN = load_n_store('SYN', 'PN')
      #data = sig_PN
    
      #motes = load_data('motes')
      #chlorine = load_data('chlorine')
      packet_data, IPflow_data, byte_flow_data = load_data('abilene')
      #routers = load_data('isp_routers')
      #servers = load_data('isp_servers')
      
      #data_raw = packet_data
      # Mean adjust data
      #data_mean = MA_over_window(data_raw,50)
      #data = data_raw - data_mean 
      ## Fix Nans 
      #whereAreNaNs = np.isnan(data)
      #data[whereAreNaNs] = 0
      
      data = packet_data

  # All anomalies however small the residual. NO difference between rel and abs here. 
  t0 = EWMA_anomalies(data, alpha = 0.3, threshold = 0, EWMA_mode = 'min', thresh_mode = 'abs') 
  
  #out1 = EWMA_anomalies(data, alpha = 0.3, threshold = 0.5, EWMA_mode = 'min', thresh_mode = 'rel')
  #out2 = EWMA_anomalies(data, alpha = 0.3, threshold = 0.5, EWMA_mode = 'max', thresh_mode = 'rel')
  #out3 = EWMA_anomalies(data, alpha = 0.3, threshold = 0.5, EWMA_mode = 'middle', thresh_mode = 'rel')
  
  #out4 = EWMA_anomalies(data, alpha = 0.3, threshold = 0.5, EWMA_mode = 'min', thresh_mode = 'rel')
  #out5 = EWMA_anomalies(data, alpha = 0.3, threshold = 1.0, EWMA_mode = 'min', thresh_mode = 'rel')
  #out6 = EWMA_anomalies(data, alpha = 0.3, threshold = 1.5, EWMA_mode = 'min', thresh_mode = 'rel')
  #out7 = EWMA_anomalies(data, alpha = 0.3, threshold = 2.0, EWMA_mode = 'min', thresh_mode = 'rel')
  
  #out8 = EWMA_anomalies(data, alpha = 0.3, threshold = 1000, EWMA_mode = 'min', thresh_mode = 'abs')
  #out9 = EWMA_anomalies(data, alpha = 0.3, threshold = 2000, EWMA_mode = 'min', thresh_mode = 'abs')
  #out10 = EWMA_anomalies(data, alpha = 0.3, threshold = 5000, EWMA_mode = 'min', thresh_mode = 'abs')
  #out11 = EWMA_anomalies(data, alpha = 0.3, threshold = 10000, EWMA_mode = 'min', thresh_mode = 'abs')
  
  out12 = EWMA_anomalies(data, alpha = 0.3, threshold = 1000, EWMA_mode = 'forward', thresh_mode = 'abs')
  
