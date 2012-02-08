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
T - number of timesteps
k - total number of sine trends making up all streams to various extents
interval - interval over which they are randomly drawn from. 

L - Anomaly length
L2 - Mid anomaly length (used only for Persistent anomalies)
M - Magnitude of Anomaly
pA - if pN < 1  --> percent of N streams that are anomalous  
     else is number of anomalous streams
  

The following are randomised each time the fucntion is ran, unless seed is given, 
in which case it is set once at the start of the function. 
  -- The periods to use in specified interval 
  -- The random contributions of each to the signal 
  -- The start time of the anomaly 
  -- The streams that are anomalous

"""

def gen_a_peak_dip(N, T, L, M, pA, k = 5, interval = [10,121], seed = None, noise_sig = 0.1, L2 = None, periods = None, periods2 = None):
  """ Adds short peak or dip to data 
  
  interval used is (low (Inclusive), High (exclusive)] 
  
  """
  if seed:
    print 'Setting seed to %i' % (seed) 
    npr.seed(seed)

  if periods is None:
    periods = []
    num_left = float(interval[1] - interval[0])
    num_needed = float(k)
    for i in xrange(interval[0], interval[1]+1):
      # probability of selection = (number needed)/(number left)
      p = num_needed / num_left
      if npr.rand() <= p:
        periods.append(i) 
        num_needed -= 1
      else:
        num_left -=1     

  #if seed is not None:
    ## Code to make random sins combination 
    #A, sins = sin_rand_combo(N, T, periods, seed = seed, noise_scale = noise_sig)
  #else:
    ## Code to make random sins combination 

  # Seed already set at start of function 
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
    num_left = float(N)
    num_needed = float(num_anom)
    for i in xrange(N):
      # probability of selection = (number needed)/(number left)
      p = num_needed / num_left
      if npr.rand() <= p:
        anoms.append(i) 
        num_needed -= 1
        num_left -= 1
      else:
        num_left -= 1   
    A[:,anoms] = A[:,anoms] + np.atleast_2d(s0lin).T
  else:
    anoms = npr.randint(N)
    A[:,anoms] = A[:,anoms] + np.array(s0lin)

  ''' Grount truth Table '''
  if anoms.__class__ == int:
    anoms = [anoms]
  
  gt_table = np.zeros(num_anom, dtype = [('start','i4'),('loc','i4'),('len','i4'),('mag','i4'),('type','a10')])
  for i, a in enumerate(anoms):
    gt_table[i] = (start_point, a, L, M, anom_type)

  output = dict(data = A, gt = gt_table, trends = sins, periods = periods)

  return output

def gen_a_step_n_back(N, T, L, M, pA, k = 5, interval = [10,101], seed = None, noise_sig = 0.1, L2 = None, periods = None, periods2 = None):
  """ Adds a sharp step change which persists for a while then steps back """
  
  if seed:
    print 'Setting seed to %i' % (seed) 
    npr.seed(seed)

  if periods is None:
    periods = []
    num_left = float(interval[1] - interval[0])
    num_needed = float(k)
    for i in xrange(interval[0], interval[1]+1):
      # probability of selection = (number needed)/(number left)
      p = num_needed / num_left
      if npr.rand() <= p:
        periods.append(i) 
        num_needed -= 1
      else:
        num_left -=1     

  # Seed already set at start of function 
  A, sins = sin_rand_combo(N, T, periods, noise_scale = noise_sig)

  # Anomaly will occur (and finish) Between time points 50 and T - 10 
  start_point = npr.randint(50, T - L2 - 10)

  # Code to make linear Anomalous trend 
  baseLine = 0

  # Randomly choose peak or dip 
  if npr.rand() < 0.5:
    anom_type = 'Step Up--Down'
    s0lin = Tseries(0)
    s0lin.makeSeries([1,1,1], [start_point, L2, T - start_point - L2], 
                   [baseLine, baseLine + M, baseLine], noise_type ='none')      
  else:
    anom_type = 'Step Down--Up'
    s0lin = Tseries(0)
    s0lin.makeSeries([1,1,1], [start_point, L2, T - start_point - L2], 
                   [baseLine, baseLine - M, baseLine], noise_type ='none')      
    
  # Select stream(s) to be anomalous
  if type(pA) == int:
    num_anom = pA
  elif pA < 1.0:
    num_anom = np.floor(pA * N)

  if num_anom > 1:
    anoms = []
    num_left = float(N)
    num_needed = float(num_anom)
    for i in xrange(N):
      # probability of selection = (number needed)/(number left)
      p = num_needed / num_left
      if npr.rand() <= p:
        anoms.append(i) 
        num_needed -= 1
        num_left -=1
      else:
        num_left -=1   
    A[:,anoms] = A[:,anoms] + np.atleast_2d(s0lin).T
  else:
    anoms = npr.randint(N)
    A[:,anoms] = A[:,anoms] + np.array(s0lin)

  ''' Ground Truth Table '''
  if anoms.__class__ == int:
    anoms = [anoms]
  
  gt_table = np.zeros(num_anom*2, dtype = [('start','i4'),('loc','i4'),('len','i4'),('mag','i4'),('type','a10')])
  count = 0
  for i, a in enumerate(anoms):
    a1_type = anom_type[5:].split('--')[0]
    a2_type = anom_type[5:].split('--')[1]    
    gt_table[i+count] = (start_point, a, L, M, 'Grad ' + a1_type)
    gt_table[i+count+1] = (start_point+L2+(L/2.0), a, 10, M, 'Grad ' + a2_type)
    count += 1
  
  output = dict(data = A, gt = gt_table, trends = sins, periods = periods)  
  
  return output

def gen_a_step(N, T, L, M, pA, k = 5, interval = [10,101], seed = None, noise_sig = 0.1, L2 = None, periods = None, periods2 = None):
  """ Adds a single sharp step change """
  
  if seed:
    print 'Setting seed to %i' % (seed) 
    npr.seed(seed)

  if periods is None:
    periods = []
    num_left = float(interval[1] - interval[0])
    num_needed = float(k)
    for i in xrange(interval[0], interval[1]+1):
      # probability of selection = (number needed)/(number left)
      p = num_needed / num_left
      if npr.rand() <= p:
        periods.append(i) 
        num_needed -= 1
      else:
        num_left -=1     

  # Seed already set at start of function 
  A, sins = sin_rand_combo(N, T, periods, noise_scale = noise_sig)

  # Anomaly will occur (and finish) Between time points 50 and T - 10 
  start_point = npr.randint(50, T - L - 10)

  # Code to make linear Anomalous trend 
  baseLine = 0

  # Randomly choose peak or dip 
  if npr.rand() < 0.5:
    anom_type = 'Step Up'
    s0lin = Tseries(0)
    s0lin.makeSeries([1,1], [start_point, T - start_point], 
                   [baseLine, baseLine + M], noise_type ='none')      
  else:
    anom_type = 'Step Down'
    s0lin = Tseries(0)
    s0lin.makeSeries([1,1], [start_point, T - start_point], 
                   [baseLine, baseLine - M, baseLine], noise_type ='none')      
    
  # Select stream(s) to be anomalous
  if type(pA) == int:
    num_anom = pA
  elif pA < 1.0:
    num_anom = np.floor(pA * N)

  if num_anom > 1:
    anoms = []
    num_left = float(N)
    num_needed = float(num_anom)
    for i in xrange(N):
      # probability of selection = (number needed)/(number left)
      p = num_needed / num_left
      if npr.rand() <= p:
        anoms.append(i) 
        num_needed -= 1
        num_left -=1
      else:
        num_left -=1   
    A[:,anoms] = A[:,anoms] + np.atleast_2d(s0lin).T
  else:
    anoms = npr.randint(N)
    A[:,anoms] = A[:,anoms] + np.array(s0lin)

  ''' Ground Truth Table '''
  if anoms.__class__ == int:
    anoms = [anoms]
  
  gt_table = np.zeros(num_anom, dtype = [('start','i4'),('loc','i4'),('len','i4'),('mag','i4'),('type','a10')])
  for i, a in enumerate(anoms):
    gt_table[i] = (start_point, a, 10, M, anom_type)

  output = dict(data = A, gt = gt_table, trends = sins, periods = periods)
  
  return output



def gen_a_grad_persist(N, T, L, M, pA, k = 5, interval = [10,101], seed = None, noise_sig = 0.1, L2 = None, periods = None, periods2 = None):
  """ Adds longer persisted anomaly. gradient up/down to it 
  and vice versa after L steps """

  if seed:
      print 'Setting seed to %i' % (seed) 
      npr.seed(seed)
  
  if periods is None:
    periods = []
    num_left = float(interval[1] - interval[0])
    num_needed = float(k)
    for i in xrange(interval[0], interval[1]+1):
      # probability of selection = (number needed)/(number left)
      p = num_needed / num_left
      if npr.rand() <= p:
        periods.append(i) 
        num_needed -= 1
      else:
        num_left -=1       

  # Seed already set at start of function 
  A, sins = sin_rand_combo(N, T, periods, noise_scale = noise_sig)

  # Anomaly will occur (and finish) Between time points 50 and T - 10 
  start_point = npr.randint(50, T - L - L2 - 10)

  # Code to make linear Anomalous trend 
  baseLine = 0

  # Randomly choose peak or dip 
  if npr.rand() < 0.5:
    anom_type = 'Grad Up--Down'
    s0lin = Tseries(0)
    s0lin.makeSeries([1,3,1,4,1], [start_point, L/2, L2, L/2, T - start_point - L - L2], 
                     [baseLine, baseLine, baseLine + M, baseLine + M, baseLine], 
                    gradient = float(M)/float(L/2), noise_type ='none')
  else:
    anom_type = 'Grad Down--Up'
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
    num_left = float(N)
    num_needed = float(num_anom)
    for i in xrange(N):
      # probability of selection = (number needed)/(number left)
      p = num_needed / num_left
      if npr.rand() <= p:
        anoms.append(i) 
        num_needed -= 1
        num_left -=1
      else:
        num_left -=1   
    A[:,anoms] = A[:,anoms] + np.atleast_2d(s0lin).T
  else:
    anoms = npr.randint(N)
    A[:,anoms] = A[:,anoms] + np.array(s0lin)
  
  ''' Ground Truth Table '''
  if anoms.__class__ == int:
    anoms = [anoms]
  
  gt_table = np.zeros(num_anom*2, dtype = [('start','i4'),('loc','i4'),('len','i4'),('mag','i4'),('type','a10')])
  count = 0
  for i, a in enumerate(anoms):
    a1_type = anom_type[5:].split('--')[0]
    a2_type = anom_type[5:].split('--')[1]    
    gt_table[i+count] = (start_point, a, L, M, 'Grad ' + a1_type)
    gt_table[i+count+1] = (start_point+L2+(L/2.0), a, L, M, 'Grad ' + a2_type)
    count += 1

  output = dict(data = A, gt = gt_table, trends = sins, periods = periods)
  return output


def gen_a_periodic_shift(N, T, L, M, pA, k = 5, interval = [10,101], seed = None, noise_sig = 0.1, L2 = None, periods = None, periods2 = None):
  """ Switches signal periods from one set of trends to another and back again
   after L2 steps 
   
   Tests for a group of signals momentarily breaking correlations.
   """

  if seed:
      print 'Setting seed to %i' % (seed) 
      npr.seed(seed)
  
  if periods is None:
    periods = []
    num_left = float(interval[1] - interval[0])
    num_needed = float(k)
    for i in xrange(interval[0], interval[1]+1):
      # probability of selection = (number needed)/(number left)
      p = num_needed / num_left
      if npr.rand() <= p:
        periods.append(i) 
        num_needed -= 1
      else:
        num_left -=1   
    
  if periods2 is None:
      periods2 = []
      num_left = float(interval[1] - interval[0])
      num_needed = float(k)
      for i in xrange(interval[0], interval[1]+1):
        # probability of selection = (number needed)/(number left)
        p = num_needed / num_left
        if npr.rand() <= p:
          periods2.append(i) 
          num_needed -= 1
        else:
          num_left -=1     

  # Seed already set at start of function 
  A, sins1 = sin_rand_combo(N, T, periods, noise_scale = noise_sig)
  B, sins2 = sin_rand_combo(N, T, periods2, noise_scale = noise_sig)

  # Anomaly will occur (and finish) Between time points 50 and T - 10 
  start_point = npr.randint(50, T - L - L2 - 10)

  # Select stream(s) to be anomalous
  if type(pA) == int:
    num_anom = pA
  elif pA < 1.0:
    num_anom = np.floor(pA * N)

  if num_anom > 1:
    anoms = []
    num_left = float(N)
    num_needed = float(num_anom)
    for i in xrange(N):
      # probability of selection = (number needed)/(number left)
      p = num_needed / num_left
      if npr.rand() <= p:
        anoms.append(i) 
        num_needed -= 1
        num_left -=1
      else:
        num_left -=1   
    A[start_point:start_point+L2,anoms] = B[start_point:start_point+L2,anoms]
  else:
    anoms = npr.randint(N)
    A[start_point:start_point+L2,anoms] = B[start_point:start_point+L2,anoms]


  ''' Ground Truth Table '''
  if anoms.__class__ == int:
    anoms = [anoms]
  
  gt_table = np.zeros(num_anom*2, dtype = [('start','i4'),('loc','i4'),('len','i4'),('mag','i4'),('type','a10')])
  count = 0
  for i, a in enumerate(anoms):   
    gt_table[i+count] = (start_point, a, 10, M, 'Break Trends')
    gt_table[i+count+1] = (start_point+L2, a, 10, M, 'Make Trends')
    count += 1

  output = dict(data = A, gt = gt_table, trends = sins1, trends2 = sins2, periods = periods, periods2 = periods2)
  return output

if __name__=='__main__':
  
  D = gen_a_peak_dip(50, 1000, 10, 5, 1, periods = [10, 38 , 88])
  plt.plot(D['data'])
  plt.figure()
  
  D2 = gen_a_step(50, 1000, 10, 5, 0.1, periods = [10, 38 , 88])
  plt.plot(D2['data'])
  
  plt.figure()
  D3 = gen_a_step_n_back(50, 1000, 10, 5, 0.1, periods = [10, 38 , 88], L2 = 100)
  plt.plot(D3['data'])
  
  plt.figure()
  D4 = gen_a_grad_persist(50, 1000, 10, 5, 0.1, periods = [10, 38 , 88], L2 = 200)
  plt.plot(D4['data'])
  
  plt.figure()
  D5 = gen_a_periodic_shift(50, 1000, 0, 0, 0.1, L2 = 200)
  plt.plot(D5['data'])