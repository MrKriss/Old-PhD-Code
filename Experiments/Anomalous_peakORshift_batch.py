# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 10:50:07 2011

Recesnt Changes:
    Fixed gradient not used issue (int/int problem)
    Added saving of all Raw Results data in pickle dump. Allows further 
    analysis later. Still need to write funtion to do so. 

@author: - Musselle
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

#-----Fixed-----#
# Path Setup 
exp_name = 'Test_sine_signals'
results_path = '/Users/chris/Dropbox/Work/MacSpyder/Results/'
cwd = os.getcwd()
path = os.path.join(results_path, exp_name)

baseLine = 5.0
baseLine_MA_window = 15
period = 5
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
                    
                    # Seed random number generator 
                    np.random.seed(i)                    
                    
                    # Two ts that have anomalous shift 
                    s0lin = Tseries(0) # linear component 
                    s0sin = Tseries(0) # Sine component 
                    s1lin = Tseries(0)
                    s1sin = Tseries(0)
                    
                    if anomaly_type == 'peak':
                        
                        s0lin.makeSeries([1,3,4,1], [100, l/2, l/2, 200 - l], [baseLine, baseLine, baseLine + m, baseLine], 
                                  gradient = float(m)/float(l/2), noise_type ='none')
                        s0sin.makeSeries([2], [300], [0.0], 
                                  amp = amp , period = period, noise_type ='none')
                        
                        # sum sin and linear components to get data stream                         
                        s0 = np.array(s0lin) + np.array(s0sin)                                    
                                  
                        s1lin.makeSeries([1,4,3,1],[200, l/2, l/2, 100 - l],[baseLine, baseLine, baseLine - m, baseLine], 
                                  gradient = float(m)/float(l/2), noise_type ='none')
                        s1sin.makeSeries([2], [300], [0.0], 
                                  amp = amp , period = period, noise_type ='none')
                                  
                        # sum sin and linear components to get data stream                                   
                        s1 = np.array(s1lin) + np.array(s1sin)                          
                                  
                    elif anomaly_type == 'shift':
                                  
                        s0lin.makeSeries([1,3,1], [100, l, 200 - l], [baseLine, baseLine, baseLine + m], 
                                  gradient = float(m)/float(l), noise_type ='none')
                        s0sin.makeSeries([2], [300], [0.0], 
                                  amp = amp, period = period, noise_type ='none')
                        
                        # sum sin and linear components to get data stream                         
                        s0 = np.array(s0lin) + np.array(s0sin)                                    
                                  
                        s1lin.makeSeries([1,4,1],[200, l, 100 - l],[baseLine, baseLine, baseLine - m], 
                                  gradient = float(m)/float(l), noise_type ='none')
                        s1sin.makeSeries([2], [300], [0.0], 
                                  amp = amp , period = period, noise_type ='none')
                                  
                        # sum sin and linear components to get data stream                                                           
                        s1 = np.array(s1lin) + np.array(s1sin)                                      
                                  
                                 
                    # The rest of the ts
                    for k in range(2, n) :
                        name = 's'+ str(k)
                        vars()[name] = Tseries(0)
                        vars()[name].makeSeries([2], [300], [0.0], 
                                  amp = amp , period = period, noise_type ='none')
                    
                    # Concat into one matrix 
                    S = scipy.c_[s0]
                    for k in range(1, n) :
                        S = scipy.c_[ S, vars()['s'+ str(k)] ]

                    # Concatonate to 3d array, timesteps x streams x initial condition 
                    if type(A) == int:
                        A = S 
                    else:
                        A = np.dstack((A, S))  
                
                if initial_conditions == 1:
                    B = A[:,:].copy()  
                else:                        
                    B = A[:,:,0].copy()  
                
                # Calculate the noise      
                if anomaly_type == 'peak':                     
                    B = B - baseLine
                elif anomaly_type == 'shift' : 
#===============================================================================
# Calculate Moving Baseline if shift                
#===============================================================================
                    baseLineMatrix = np.zeros(B.shape)
                    for k in range(n):   # for each stream
                        cnt = 0
                        window = np.ones((baseLine_MA_window)) * baseLine
                        for data in B[:,k]: 
                            # Advance window 
                            window[:-1] = window[1:]
                            # Add new value 
                            window[-1] = data
                            # Calculate average 
                            ave = window.sum() / float(len(window))
                            baseLineMatrix[cnt,k] = ave
                            cnt += 1
                            
                    # Correct for baseLine
                    B = B - baseLineMatrix
                    
                Ps = np.sum(B ** 2)
                Pn = Ps / (10. ** (snr/10.))
                scale = Pn / (n * 300.)        
                # Calculate the noise  
                if initial_conditions == 1:
                    noise = np.random.randn(A.shape[0], A.shape[1]) * np.sqrt(scale)
                else:
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

                dataset_count += 1                
                print 'Generated ' + str(dataset_count) + ' datasets out of ' + str(total_data)


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
                                    streams = A[:,:,i]
                                    
                                    if run_spirit == 1:
                                        # SPIRIT
                                        res_sp = SPIRIT(streams, a, [el, eh], evalMetrics = 'F', 
                                                        reorthog = False, holdOffTime = h) 
                                        res_sp['Alg'] = 'SPIRIT: alpha = ' + str(a) + \
                                        ' ,E_Thresh = (' + str(el) + ',' + str(eh) + ')'
     
#                                        pltSummary2(res_sp, streams, (el, eh))                            
                            
                                        SPIRIT_RawResults.append(res_sp)
                                        
                                        SPIRIT_metricList.append(analysis(res_sp, ground_truths, 300, 
                                                                          epsilon = epsilon, ignoreUpTo = ignore))
                                                                            
                                      
                                    if run_frahst == 1:
                                    
                                        # My version of Frahst 
                                        res_fr = FRAHST_V3_1(streams, alpha=a, e_low=el, e_high=eh, 
                                                         holdOffTime=h, fix_init_Q = 1, r = 1, evalMetrics = 'F') 
                                        res_fr['Alg'] = 'MyFrahst: alpha = ' + str(a) + ' ,E_Thresh = (' + str(el) + ',' + str(eh) + ')'
                                    
                                        FRAHST_RawResults.append(res_fr)                                        
                                        
                                        FRAHST_metricList.append(analysis(res_fr, ground_truths, 300, 
                                                                          epsilon = epsilon, ignoreUpTo = ignore))                                    

#                                    pltSummary2(res_fr, streams, (el, eh))

                                        
                                    if run_frahst_pedro == 1:
                                    
                                        # Pedros version of Frahst 
                                        res_frped = frahst_pedro(streams, alpha=a, e_low=el, e_high=eh, 
                                                         holdOffTime=h, r = 1, evalMetrics = 'F') 
                                        res_frped['Alg'] = 'Pedros Frahst: alpha = ' + str(a) +  \
                                        ' ,E_Thresh = (' + str(el) + ',' + str(eh) + ')'

                                        PEDRO_FRAHST_RawResults.append(res_frped)                                        
                                        
                                        PEDRO_FRAHST_metricList.append(analysis(res_frped, ground_truths, 300,
                                                                                epsilon = epsilon, ignoreUpTo = ignore))

                                        
#                                        pltSummary2(res_frped, streams, (el, eh))
#                                
#                                                                
                                print 'Finished running alg(s) with parameter set ' + str(alg_count) + ' of ' + str(total_alg) + ' for this dataset\n'
                                print  '\t' + str(total_data - dataset_count) + ' datasets remaining\n'
                                alg_count += 1                                
                                
                                
#===============================================================================
# # Calculate Average Metrics  and Store Results                                
#===============================================================================
                                
                                parameters = 'N = {N}, SNR = {SNR}, L = {L}, M = {M}, E = ({EL:.2f},{EH:.2f}), A = {A:.2f}, H = {H}' \
                                            .format(N = n, SNR = snr, L = l,M = m,EL = el, EH = eh, A =a, H =h)
                                
                                if run_spirit:
                                   SPIRIT_AveMetrics =  aveMetrics(SPIRIT_metricList)
                                   
                                   #Save
                                   resFilename = 'SPIRIT_Results.txt'
                                   writeRes(resFilename, SPIRIT_AveMetrics, parameters, dataFilename, path=path, mode = 'a')

                                   results_filename = 'SPIRITRes_' + parameters + '.pk'
                                                                      
                                   os.chdir(path)
                                   with open(results_filename, 'w') as savefile:
                                       pk.dump(SPIRIT_RawResults, savefile)                                    
                                   os.chdir(cwd)
                                   
                                if run_frahst:
                                   FRAHST_AveMetrics =  aveMetrics(FRAHST_metricList)

                                   #Save
                                   resFilename = 'FRAHST_Results.txt'
                                   writeRes(resFilename, FRAHST_AveMetrics, parameters, dataFilename, path=path, mode = 'a')

                                   results_filename = 'FRAHSTRes_' + parameters + '.pk'

                                   os.chdir(path)
                                   with open(results_filename, 'w') as savefile:                    
                                       pk.dump(FRAHST_RawResults, savefile)
                                   os.chdir(cwd)
                                
                                if run_frahst_pedro:
                                   FRAHST_PEDRO_AveMetrics =  aveMetrics(PEDRO_FRAHST_metricList)

                                   #Save
                                   resFilename = 'FRAHST_PEDRO_Results.txt'
                                   writeRes(resFilename, FRAHST_PEDRO_AveMetrics, parameters, dataFilename, path=path, mode = 'a')

                                   results_filename = 'PedroFRAHSTRes_' + parameters + '.pk'

                                   os.chdir(path)                                   
                                   with open(results_filename, 'w') as savefile:                    
                                       pk.dump(PEDRO_FRAHST_RawResults, savefile)
                                   os.chdir(cwd)
                                   
                                   
    loop_count += 1
    print 'Progress: ' + str(loop_count) + ' of ' + str(total_data) 
    
    
print 'Finished Batch: writing batch info file....'

os.chdir(path)
with open('batch_info.txt', 'w') as f :
    
    f.write('Experiment: ' + exp_name + '\n\n') 

    f.write('Dataset Parameters\n')
    f.write('--------------------\n')
    f.write('Initial Conditions:      ' + str(initial_conditions) + '\n')
    f.write('Base Line:               ' + str(baseLine) + '\n')
    f.write('Number of Streams:       ' + str(num_streams) + '\n')
    f.write('Signal to Noise Ratios:  ' + str(SNRs) + '\n')
    f.write('Anomaly Lengths:         ' + str(anomaly_lengths) + '\n')
    f.write('Anomaly Magnitudes:      ' + str(anomaly_magnitudes) + '\n.\n')

    # Formate float strings     
    elowStr = '[ '
    for s in e_lows:
        elowStr = elowStr + "%.2f" % s  + ', '

    ehighStr = '[ '
    for s in e_highs:
        ehighStr = ehighStr + "%.2f" % s + ', '
    
    alphasStr = '[ '
    for s in alphas:
        alphasStr = alphasStr + "%.2f" % s + ', '
        
        
    f.write('Algorithm Parameters\n')
    f.write('--------------------\n')
    f.write('Lower Energy Thresholds: ' + elowStr + ']\n')
    f.write('Upper Energy Thresholds: ' + ehighStr + ']\n')
    f.write('Alphas:                  ' + alphasStr + ']\n')
    f.write('Hold Current r lengths:  ' + str(holdOffs) + '\n.\n')
    
    f.write('Algorithms Run\n')
    f.write('--------------------\n')
    if run_spirit :
        f.write('SPIRIT\n')
    if run_frahst :
        f.write('FRAHST - My Most Current Version\n')
    if run_frahst_pedro :
        f.write('FRAHST - Pedros Published Version\n')    
        
os.chdir(cwd)     
    
finish  = time.time() - start
print 'Runtime = ' + str(finish) + 'seconds\n'
print 'In H:M:S = ' + GetInHMS(finish)
