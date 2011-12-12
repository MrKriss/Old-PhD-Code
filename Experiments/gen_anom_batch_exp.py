#!/usr/bin/env python
#coding:utf-8
# Author:   --<>
# Purpose: 
# Created: 12/12/11

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sys
import os 

"""
Code Description:
  .
"""

#===============================================================================
# Batch Parameters
#===============================================================================

#-----Fixed-----#
# Path Setup 
exp_name = 'Test_a_signals'
results_path = '/Users/chris/Dropbox/Work/MacSpyder/Results/'
cwd = os.getcwd()
path = os.path.join(results_path, exp_name)

amp = 0.1 
initial_conditions = 5      # i

anomaly_type = 'shift'

# Detection Parameters 
epsilon = 5
ignore = 25 

# Algorithm flags    
run_spirit = 0
run_frahst = 1
run_frahst_pedro = 0

#----Varied----#

# Data Sets 
num_streams = [3,5]            # n
SNRs = [-3, 0]                  # snr
anomaly_lengths = [10,20]       # l
anomaly_magnitudes = [1,2]    # m

# Algorithm Parameters
e_highs = [0.995, 0.96]               # eh
e_lows = [0.95, 0.9 ]                 # el 
alphas = [1.0, 0.98]            # a
holdOffs = [0, 3]                 # h

#===============================================================================
# Initialise Data sets
#===============================================================================

dataset_count = 0
loop_count = 0
total_data = len(num_streams) * len(SNRs) * len(anomaly_lengths) * len(anomaly_magnitudes)
total_alg = len(e_highs) * len(e_lows) * len(alphas) * len(holdOffs)
total_loops = total_data * total_alg 

# For Profiling 
start = time.time() 

for n in num_streams:
  for snr in SNRs:
    for l in anomaly_lengths:
      for m in anomaly_magnitudes:
        A = 0
        for i in range(initial_conditions):    



if __name__=='__main__':
