#!/usr/bin/env python
#coding:utf-8
# Author:  Musselle 
# Purpose: Experimental Run on DCA Data sets
# Created: 08/10/11

import numpy as np
import matplotlib.pyplot as plt
import sys
import os 
from utils import pltSummary2
from PedrosFrahst import frahst_pedro_original
from Frahst_v3_1 import FRAHST_V3_1
from Frahst_v3_3 import FRAHST_V3_3
from load_syn_ping_data import load_n_store


def prime_data(dataset, variant, **kwargs):
  # Load Data
  sig, ant, time = load_n_store(dataset, variant)
  
  if kwargs.has_key('useCol'):
    # Use only the signals specified
    # trim rubish vectors (0, 1, 8) 
    data = sig[:, kwargs['useCol']]

  if kwargs.has_key('use2time'):
    idx = time < kwargs['use2time']
    time = time[idx]
    numRows = len(time)
    # Use only up to time step specified 
    data = sig[:numRows, :]
    ant = ant[:numRows]
  
  # Cap at max 100 (SYN_AN overshoots at two points to 1000)[likely corrupted data?]
  idx = data > 100
  if idx.any():
    data[idx] = 100
  
  return data, ant, time

if __name__ == '__main__' :
  
  first = 1
  
  if first: 
    '''Load Data'''
    #AbileneMat = sio.loadmat('/Users/chris/DataSets/Abilene/Abilene.mat')
    #data = AbileneMat['P']
    
    # Use Active Normal 
    data, ant, time = prime_data('SYN', 'AN', useCol = range(2,8), use2time = 7000)
    
    # Use Passive Normal
    #data, ant, time = prime_data('SYN', 'PN', useCol = range(2,8), use2time = 7000)
  
  """PARAMETERS"""
  e_high = 0.98
  e_low = 0.85
  alpha = 0.9
  
  holdOFF = 0
  
  '''My latest version''' 
  res_new = FRAHST_V3_3(data, alpha=alpha, e_low=e_low, e_high=e_high, 
                        holdOffTime=holdOFF, fix_init_Q = 1, r = 1, evalMetrics = 'F') 
  
  res_new['Alg'] = 'My Latest Implimentation of FRAUST '
  pltSummary2(res_new, data, (e_high, e_low))
  
  '''My older version''' 
  res_old = FRAHST_V3_1(data, alpha=alpha, e_low=e_low, e_high=e_high, 
                        holdOffTime=holdOFF, fix_init_Q = 1, r = 1, evalMetrics = 'F') 
  
  res_old['Alg'] = 'My Older Implimentation of FRAUST '
  pltSummary2(res_old, data, (e_high, e_low))
  
  '''Pedros Version'''
  res_ped = frahst_pedro_original(data, r=1, alpha=alpha, e_low=e_low, e_high=e_high,
                        holdOffTime=holdOFF, evalMetrics='F')
  
  res_ped['Alg'] = 'Pedros Original Implimentation of FRAUST'
  pltSummary2(res_ped, data, (e_high, e_low))  
  
  first = 0