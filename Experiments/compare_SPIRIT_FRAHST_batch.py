# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 12:00:51 2011

@author: -
"""

from ControlCharts import Tseries
from CUSUM import cusum 
import numpy as np
import matplotlib.pyplot as plt
from Frahst_v3_1 import FRAHST_V3_1
from SPIRIT import SPIRIT
from utils import analysis, QRsolveA, pltSummary2
from load_data import load_data
from plot_utils import plot_31
import time

start = time.time()

# Load Data 
data = load_data('motes')
#data[1500:3000, 2] = 0.6
#data[1500:3000, 1] = 0.7

streams = data[:,:]

# Parameters

e_high = 0.98
e_low = 0.95
alpha = 0.96

holdOff = 0.0
        
varA_vals = [1.0, 0.99, 0.98, 0.96, 0.94, 0.92]
#varA_vals = [0.88, 0.86, 0.84, 0.82, 0.8]    
#varA_vals = [0.96]
    
varB_vals = [0.99, 0.98, 0.97, 0.96, 0.94, 0.92]    
#varB_vals = [0.95]
    
# Run mini Batch     
run_spirit = 1
run_frahst = 1
    
for valA in varA_vals :     
    for valB in varB_vals :    
        
        # Variables to alter    
        alpha = valA
        e_low = valB
        
        print 'Running Alpha =', valA, ' and e_low = ', valB        
        
        if run_spirit == 1:
            # SPIRIT
            res_sp = SPIRIT(streams, alpha, [e_low, e_high], evalMetrics = 'F', 
                            reorthog = False, holdOffTime = holdOff) 
            res_sp['Alg'] = 'SPIRIT with alpha = ' + str(valA) + ' and e_low = ' + str(valB)
            
            #plot_31(res_sp['RSRE'], res_sp['e_ratio'], res_sp['orthog_error'],
            #        ['RSRE','Energy Ratio','Orthogonality\nError (dB)'], 'Time Steps', 
            #         'Error Analysis of SPIRIT' )
            
            pltSummary2(res_sp, streams, (e_low, e_high))                 
                    
        if run_frahst == 1:
        
            # My version of Frahst 
            res_fr = FRAHST_V3_1(streams, alpha=alpha, e_low=e_low, e_high=e_high, 
                             holdOffTime=holdOff, fix_init_Q = 1, r = 1, evalMetrics = 'F') 
            res_fr['Alg'] = 'Frahst with alpha = ' + str(valA) + ' and e_low = ' + str(valB)
            
            #plot_31(res_fr['RSRE'], res_fr['e_ratio'], res_fr['orthog_error'],
            #        ['RSRE','Energy Ratio','Orthogonality\nError (dB)'], 'Time Steps', 
            #         'Error Analysis of FRAHST' )    
        
            # metric_me, sets_me, anom_det_tab_me = analysis(res_me, AbileneMat['P_g_truth_tab'], n)
    
            pltSummary2(res_fr, streams, (e_low, e_high))
            
finish  = time.time() - start
print 'Total Runtime = ', finish