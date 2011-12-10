# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 12:00:51 2011

@author: -
"""

from ControlCharts import Tseries
from CUSUM import cusum 
import numpy as np
import matplotlib.pyplot as plt
from SPIRIT_Pedro import SPIRIT_pedro
from SPIRIT import SPIRIT
from SPIRIT2 import SPIRIT2
from utils import analysis, QRsolveA, pltSummary
from load_data import load_data
from plot_utils import plot_31

# Load Data 
Alldata = load_data('chlorine')
#data[1500:3000, 2] = 0.6
#data[1500:3000, 1] = 0.7

streams = Alldata[:,:]

# Parameters

e_high = 0.99
e_low = 0.95
alpha = 0.94

holdOFF = 10
        
#var_vals = [0.98, 0.96, 0.94, 0.92, 0.9]
#var_vals = [0.88, 0.86, 0.84, 0.82, 0.8]    
#var_vals = [0.98, 0.96, 0.94, 0.92, 0.90, 0.88, 0.86, 0.84, 0.82, 0.8]

var_vals = [1.0]

# Run mini Batch     
run_spirit_me = 0
run_spirit_original = 1
    
for val in var_vals :     
    
    # Variable to alter    
    alpha = val
    if run_spirit_me == 1:
        # SPIRIT
        res_spme, all_weights = SPIRIT(streams, [e_low, e_high], alpha, evalMetrics = 'T') 
        res_spme['Alg'] = 'My SPIRIT with alpha = ' + str(val)    
        
#        data = res_spme
        pltSummary(res_spme, streams)                 
                 
    if run_spirit_original == 1:
    
        # My version of Frahst 
        res_spor = SPIRIT2(streams, alpha, [e_low, e_high], k0 = 1, holdOffTime = 10,                            
                         reorthog = False, evalMetrics = 'T')
        
        res_spor['Alg'] = 'Original FRAHST with alpha = ' + str(val)
    
        #plot_31(res_fr['RSRE'], res_fr['e_ratio'], res_fr['orthog_error'],
        #        ['RSRE','Energy Ratio','Orthogonality\nError (dB)'], 'Time Steps', 
        #         'Error Analysis of FRAHST' )    
    
        # metric_me, sets_me, anom_det_tab_me = analysis(res_me, AbileneMat['P_g_truth_tab'], n)

        pltSummary(res_spor, streams)