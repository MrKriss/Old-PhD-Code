#!/usr/bin/env python
#coding:utf-8
# Author:   --<>
# Purpose: 
# Created: 12/07/11

import numpy as np
import numpy.random as npr
import scipy as sp
import matplotlib.pyplot as plt
import sys
import os 
from artSigs import sin_rand_combo
from ControlCharts import Tseries

""" Code Description: The functions to generate synthetic data with various anomalies. 

The baseline of the data itself is made up of a random combination of sine waves specified
by the periods p1,...,pk. Thus each time series will have at most k trends, but some may only
contribute slightly. 

The anomalies are of two main kinds, Transient and persistent, reflecting their duration.
Transient Anomalies  : Peaks/dips - sharp short gradients up/down then back to baseline
                     : Step up/step down - sudden step changes
Persistent Anomalies : Peaks/dips
                     : Step Changes
                     : Phase Changes - trends of a k 

Anomalies will be in a miminum of 1 stream in N 

Parameters 

N - total number of streams 
k - total number of sine trends making up all streams to various extents

L - Anomaly length
M - Magnitude of Anomaly
percent_Anom - percent of N streams that are anomalous  

The start time is randomised each time 


"""

def gen_a_peak_dip(N, T, periods, L, M, pA, seed = None, noise_sig = 0.1):
  """ Adds short peak or dip to data """

  if seed is not None:
    # Code to make random sins combination 
    A, sins = sin_rand_combo(N, T, periods, seed = seed, noise_scale = noise_sig)
    npr.seed(seed)
  else:
    # Code to make random sins combination 
    A, sins = sin_rand_combo(N, T, periods, noise_scale = noise_sig)

  # Anomaly will occur (and finish) Between time points 50 and T - 10 
  start_point = npr.randint(50, T - L - 10)

  # Code to make linear Anomalous trend 
  baseLine = 0

  # Randomly choose peak or dip 
  if npr.rand() < 0.5:
    anom_type = 'Peak'
    s0lin = Tseries(0)
    s0lin.makeSeries([1,3,4,1], [start_point, L/2, L/2, T - start_point - L], 
                   [baseLine, baseLine, baseLine + M, baseLine], 
                   gradient = float(M)/float(L/2), noise_type ='none')      
  else:
    anom_type = 'Dip'
    s0lin = Tseries(0)
    s0lin.makeSeries([1,4,3,1], [start_point, L/2, L/2, T - start_point - L], 
                     [baseLine, baseLine, baseLine - M, baseLine], 
                    gradient = float(M)/float(L/2), noise_type ='none')      

  # Select stream(s) to be anomalous
  if type(pA) == int:
    num_anom = pA
  elif pA < 1.0:
    num_anom = np.floor(pA * N)

  if num_anom > 1:
    anoms = []
    num_left = N
    num_needed = num_anom
    for i in xrange(N):
      # probability of selection = (number needed)/(number left)
      p = num_needed / num_left
      if npr.rand() <= p:
        anoms.append(i) 
        num_needed -= 1
      else:
        num_left -=1   
    A[:,anoms] = A[:,anoms] + np.atleast_2d(s0lin).T
  else:
    anoms = npr.randint(N+1)
    A[:,anoms] = A[:,anoms] + np.array(s0lin)

  output = dict(data = A, a_stream = anoms, a_start = start_point, 
                a_type = anom_type, a_L = L, a_M = M, trends = sins)

  return output

def gen_a_step(N, T, periods, L, M, pA, seed = None, noise_sig = 0.1):
  """ Adds short sharp step change """

  if seed is not None:
    # Code to make random sins combination 
    A, sins = sin_rand_combo(N, T, periods, seed = seed, noise_scale = noise_sig)
    npr.seed(seed)
  else:
    # Code to make random sins combination 
    A, sins = sin_rand_combo(N, T, periods, noise_scale = noise_sig)

  # Anomaly will occur (and finish) Between time points 50 and T - 10 
  start_point = npr.randint(50, T - L - 10)

  # Code to make linear Anomalous trend 
  baseLine = 0

  # Randomly choose peak or dip 
  if npr.rand() < 0.5:
    anom_type = 'Step Up'
    s0lin = Tseries(0)
    s0lin.makeSeries([1,1,1], [start_point, L, T - start_point - L], 
                   [baseLine, baseLine + M, baseLine], noise_type ='none')      
  else:
    anom_type = 'Step Down'
    s0lin = Tseries(0)
    s0lin.makeSeries([1,1,1], [start_point, L, T - start_point - L], 
                   [baseLine, baseLine - M, baseLine], noise_type ='none')      
    
  # Select stream(s) to be anomalous
  if type(pA) == int:
    num_anom = pA
  elif pA < 1.0:
    num_anom = np.floor(pA * N)

  if num_anom > 1:
    anoms = []
    num_left = N
    num_needed = num_anom
    for i in xrange(N):
      # probability of selection = (number needed)/(number left)
      p = num_needed / num_left
      if npr.rand() <= p:
        anoms.append(i) 
        num_needed -= 1
      else:
        num_left -=1   
    A[:,anoms] = A[:,anoms] + np.atleast_2d(s0lin).T
  else:
    anoms = npr.randint(N+1)
    A[:,anoms] = A[:,anoms] + np.array(s0lin)

  output = dict(data = A, a_stream = anoms, a_start = start_point, 
                a_type = anom_type, a_L = L, a_M = M, trends = sins)

  return output


def gen_a_grad_persist(N, T, periods, L, L2, M, pA, seed = None, noise_sig = 0.1):
  """ Adds longer persisted anomaly. gradient up/down to it 
  and vice versa after L steps """

  if seed is not None:
    # Code to make random sins combination 
    A, sins = sin_rand_combo(N, T, periods, seed = seed, noise_scale = noise_sig)
    npr.seed(seed)
  else:
    # Code to make random sins combination 
    A, sins = sin_rand_combo(N, T, periods, noise_scale = noise_sig)

  # Anomaly will occur (and finish) Between time points 50 and T - 10 
  start_point = npr.randint(50, T - L - L2 - 10)

  # Code to make linear Anomalous trend 
  baseLine = 0

  # Randomly choose peak or dip 
  if npr.rand() < 0.5:
    anom_type = 'Grad Up -- Down'
    s0lin = Tseries(0)
    s0lin.makeSeries([1,3,1,4,1], [start_point, L/2, L2, L/2, T - start_point - L - L2], 
                     [baseLine, baseLine, baseLine + M, baseLine + M, baseLine], 
                    gradient = float(M)/float(L/2), noise_type ='none')
  else:
    anom_type = 'Grad Down -- Up'
    s0lin = Tseries(0)
    s0lin.makeSeries([1,4,1,3,1], [start_point, L/2, L2, L/2, T - start_point - L - L2], 
                     [baseLine, baseLine, baseLine - M, baseLine - M, baseLine], 
                    gradient = float(M)/float(L/2), noise_type ='none')      
    
  # Select stream(s) to be anomalous
  if type(pA) == int:
    num_anom = pA
  elif pA < 1.0:
    num_anom = np.floor(pA * N)

  if num_anom > 1:
    anoms = []
    num_left = N
    num_needed = num_anom
    for i in xrange(N):
      # probability of selection = (number needed)/(number left)
      p = num_needed / num_left
      if npr.rand() <= p:
        anoms.append(i) 
        num_needed -= 1
      else:
        num_left -=1   
    A[:,anoms] = A[:,anoms] + np.atleast_2d(s0lin).T
  else:
    anoms = npr.randint(N+1)
    A[:,anoms] = A[:,anoms] + np.array(s0lin)

  output = dict(data = A, a_stream = anoms, a_start = start_point, 
                a_type = anom_type, a_L = L, a_M = M, trends = sins)

  return output


if __name__=='__main__':
  
  D = gen_a_peak_dip(50, 1000, [24,57,19,88,73], 10, 5,1)
  plt.plot(D['data'])
  plt.figure()
  D2 = gen_a_peak_dip(50, 1000, [24,57,19,88,73], 10, 5, 0.1)
  plt.plot(D2['data'])
  
  plt.figure()
  D3 = gen_a_step(50, 1000, [24,57,19,88,73], 20, 5, 0.1)
  plt.plot(D3['data'])
  
  plt.figure()
  D4 = gen_a_grad_persist(50, 1000, [24,57,19,88,73], 10, 100, 2, 0.1)
  plt.plot(D4['data'])