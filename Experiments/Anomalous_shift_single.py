# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 10:50:07 2011

@author: -
"""

from ControlCharts import Tseries
from CUSUM import cusum 
import numpy as np
import matplotlib.pyplot as plt
from Frahst_v3_1 import FRAHST_V3_1
from SPIRIT import SPIRIT
from utils import QRsolveA, pltSummary, pltSummary2, GetInHMS, writeRes
from AnomalyMetrics import analysis, fmeasure, aveMetrics
from plot_utils import plot_4x1
import scipy
import time
from PedrosFrahst import frahst_pedro
import pickle as pk
import os

#===============================================================================
# Batch Parameters
#===============================================================================

# Path Setup 
exp_name = 'singleRuns'
results_path = '/Users/chris/Dropbox/Work/MacSpyder/Results/'
cwd = os.getcwd()
path = os.path.join(results_path, exp_name)

# N = 3, SNR = 10, L = 10, M = 20, E = (0.95,0.99), A = 0.96, H = 5
# N = 3, SNR = -3, L = 20, M = 1, E = (0.93,0.99), A = 0.98, H = 10

# Data Sets 
baseLine = 1.0

num_streams = [3]            # n
SNRs = [50]                  # snr
anomaly_lengths = [10]       # l
anomaly_magnitudes = [1]    # m

initial_conditions = 1      # i

# Algorithm Parameters
e_highs = [0.99]             # eh
e_lows = [0.95]              # el 
alphas = [0.96]              # a
holdOffs = [0]             # h
    
# Algorithm flags    
run_spirit = 0
run_frahst = 1
run_frahst_pedro = 0

#===============================================================================
# Initialise Data sets
#========-=======================================================================

# For Profiling 
start = time.time() 

for n in num_streams:
    for snr in SNRs:
        for l in anomaly_lengths:
            for m in anomaly_magnitudes:
                A = 0
                for i in range(initial_conditions):    
                    
                    # Seed random number generator 
                    np.random.seed(i)                    
                    
                    # Two ts that have anomalous shift 
                    s0 = Tseries(0)
                    s1 = Tseries(0)
                    s0.makeSeries([1,3,1],[100, l, 200 - l],[baseLine, baseLine, baseLine + m], 
                                  gradient = float(m)/float(l), noise_type ='none')
                    s1.makeSeries([1,4,1],[200, l, 100 - l],[baseLine, baseLine, baseLine - m], 
                                  gradient = float(m)/float(l), noise_type ='none')
                    # The rest of the ts
                    for k in range(2, n) :
                        name = 's'+ str(k)
                        vars()[name] = Tseries(0)
                        vars()[name].makeSeries([1], [300], [baseLine], noise_type ='none')
                    
                    # Concat into one matrix 
                    S = scipy.c_[s0]
                    for k in range(1, n) :
                        S = scipy.c_[ S, vars()['s'+ str(k)] ]

                    # Concatonate to 3d array, timesteps x streams x initial condition 
                    if type(A) == int:
                        A = S 
                    else:
                        A = np.dstack((A, S))                        
                        
                # Calculate the noise  
                if initial_conditions == 1:
                    Ps = np.sum(A[:,:] ** 2)
                    Pn = Ps / (10. ** (snr/10.))
                    scale = Pn / (n * 300.)
                    noise = np.random.randn(A.shape[0], A.shape[1]) * np.sqrt(scale)
                    A = A + noise
                else:
                    Ps = np.sum(A[:,:,0] ** 2)
                    Pn = Ps / (10. ** (snr/10.))
                    scale = Pn / (n * 300.)
                    noise = np.random.randn(A.shape[0], A.shape[1], A.shape[2]) * np.sqrt(scale)
                    A = A + noise
                        
                #Save
                
                if not os.path.isdir(path):
                    os.makedirs(path)
                    os.chdir(path)                
                else:
                    os.chdir(path)
                                    
                dataFilename = 'Data_N' + str(n) + '_SNR' + str(snr) + '_L' + str(l) + '_M' + str(m)
                with open(dataFilename, 'w') as savefile:
                    np.save(savefile, A)
                os.chdir(cwd)                

#===============================================================================
# Ground Truths 
#===============================================================================
#                                 # time step | length 
                ground_truths = np.array([[100, l],
                                          [200, l]])
                                    
#==============================================================================
#  Run Algorithm 
#==============================================================================   
                alg_count = 1    
    
                for eh in e_highs :             
                    for el in e_lows :              
                        for a in alphas :              
                            for h in holdOffs :            
                                
                                print 'Running Algorithm(s) with:\nE_Thresh = (' + str(el) + ',' + str(eh) + ')\n' + \
                                'alpha = ' + str(a) + '\nHoldOff = ' + str(h)  
                                
                                SPIRIT_metricList = []                                
                                SPIRIT_RawResults = []                                
                                FRAHST_metricList = []                                
                                FRAHST_RawResults = []                                
                                PEDRO_FRAHST_metricList = []
                                PEDRO_FRAHST_RawResults = []
                                
                                for i in range(initial_conditions):
                                    
                                    # Load Data 
                                    if initial_conditions == 1:
                                        streams = A[:,:]
                                    else:
                                        streams = A[:,:,i]
                                    
                                    if run_spirit == 1:
                                        # SPIRIT
                                        res_sp = SPIRIT(streams, a, [el, eh], evalMetrics = 'F', 
                                                        reorthog = False, holdOffTime = h) 
                                        res_sp['Alg'] = 'SPIRIT: alpha = ' + str(a) + \
                                        ' ,E_Thresh = (' + str(el) + ',' + str(eh) + ')'
     
                                        pltSummary2(res_sp, streams, (el, eh))                            

                                        SPIRIT_RawResults.append(res_sp)                            
                            
                                        SPIRIT_metricList.append(analysis(res_sp, ground_truths, 300 ))
                                                                            
#                                        data = res_sp
#                                        Title = 'SPIRIT alpha = ' + str(a) + ' ,E_Thresh = (' + str(el) + ',' + str(eh) + ')'
#                            
    #                                    plot_4x1(streams, data['hidden'], data['e_ratio'], data['orthog_error'], 
    #                                             ['Input Data','Hidden\nVariables',
    #                                    'Energy Ratio', 'Orthogonality\nError (dB)'] , 'Time Steps', Title)
                                        
#                                        plot_4x1(streams, data['hidden'], data['orthog_error'], data['subspace_error'],
#                                ['Input Data','Hidden\nVariables', 'Orthogonality\nError (dB)','Subspace\nError (dB)'], 
#                                                                 'Time Steps', Title)             
#                            
                                    if run_frahst == 1:
                                    
                                        # My version of Frahst 
                                        res_fr = FRAHST_V3_1(streams, alpha=a, e_low=el, e_high=eh, 
                                                         holdOffTime=h, fix_init_Q = 1, r = 1, evalMetrics = 'F') 
                                        res_fr['Alg'] = 'MyFrahst: alpha = ' + str(a) + ' ,E_Thresh = (' + str(el) + ',' + str(eh) + ')'
                                    
                                        FRAHST_metricList.append(analysis(res_fr, ground_truths, 300 ))                                    

                                        FRAHST_RawResults.append(res_fr)
                                        
                                        pltSummary2(res_fr, streams, (el, eh))
#                                      
 
                                    if run_frahst_pedro == 1:
                                    
                                        # Pedros version of Frahst 
                                        res_frped = frahst_pedro(streams, alpha=a, e_low=el, e_high=eh, 
                                                         holdOffTime=h, r = 1, evalMetrics = 'F') 
                                        res_frped['Alg'] = 'Pedros Frahst: alpha = ' + str(a) +  \
                                        ' ,E_Thresh = (' + str(el) + ',' + str(eh) + ')'

                                        PEDRO_FRAHST_metricList.append(analysis(res_frped, ground_truths, 300 ))

                                        PEDRO_FRAHST_RawResults.append(res_frped)

                                        pltSummary2(res_frped, streams, (el, eh))
                                                                 
                                
finish  = time.time() - start
print 'Runtime = ' + str(finish) + 'seconds\n'
print 'In H:M:S = ' + GetInHMS(finish)