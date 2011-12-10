# -*- coding: utf-8 -*-
"""
Created on Wed May 18 11:23:36 2011

Test SPIRIT and Frahst on Contol Time Series: Gradient

@author: - Musselle
"""

from ControlCharts import Tseries
from CUSUM import cusum 
import numpy as np
import matplotlib.pyplot as plt
from Frahst_v3_1 import FRAHST_V3_1
from SPIRIT import SPIRIT
from utils import analysis, QRsolveA, pltSummary, pltSummary2, GetInHMS
from plot_utils import plot_4x1
import scipy
import time
from PedrosFrahst import frahst_pedro


#===============================================================================
# Runscript 
#===============================================================================

#============
# Initialise
#============

'Create time series string'
start = time.time()
# Initialise stream 
series1 = Tseries(0)
series2 = Tseries(0)
series3 = Tseries(0)
#series4 = Tseries(0)

series1.makeSeries([1],[300],[5], noise = 1)
series2.makeSeries([1,4,1],[200, 10, 90],[6, 6, -4], gradient = 1, noise = 1)
series3.makeSeries([1,3,1],[100,10,190],[2,2,12], gradient = 1, noise = 1) 
#series4.makeSeries([2],[300],[5], period = 10, amp = 1, noise = 1)  
                   
streams = scipy.c_[series1, series2, series3]  # ,series4] 

# Run SPIRIT and Frahst

# Parameters

e_high = 0.98
e_low = 0.95
alpha = 0.96

holdOff = 5.0
    
#varA_vals = [1.0, 0.99, 0.98, 0.96, 0.94, 0.92]
#varA_vals = [0.88, 0.86, 0.84, 0.82, 0.8]    
varA_vals = [0.96]
    
#varB_vals = [0.99, 0.98, 0.97, 0.96, 0.94, 0.92]    
varB_vals = [0.95]
    
# Run mini Batch     
run_spirit = 1
run_frahst = 1
run_frahst_pedro = 0
    
for valA in varA_vals :     
    for valB in varB_vals :    
        
        # Variables to alter    
        alpha = valA
        e_low = valB
        
        print valA, valB        
        
        if run_spirit == 1:
            # SPIRIT
            res_sp = SPIRIT(streams, alpha, [e_low, e_high], evalMetrics = 'T', 
                            reorthog = False, holdOffTime = holdOff) 
            res_sp['Alg'] = 'SPIRIT with alpha = ' + str(valA) + ' and e_low = ' + str(valB)
            
            #plot_31(res_sp['RSRE'], res_sp['e_ratio'], res_sp['orthog_error'],
            #        ['RSRE','Energy Ratio','Orthogonality\nError (dB)'], 'Time Steps', 
            #         'Error Analysis of SPIRIT' )
            
            pltSummary2(res_sp, streams, (e_low, e_high))

            data = res_sp

            Title = 'SPIRIT with alpha = ' + str(valA) + ' and e_low = ' + str(valB)

            plot_4x1(streams, data['hidden'], data['e_ratio'], data['orthog_error'], 
                     ['Input Data','Hidden\nVariables',
            'Energy Ratio', 'Orthogonality\nError (dB)'] , 'Time Steps', Title)
            
            #TODO
#             Why does changing the above for this             
            plot_4x1(streams, data['hidden'], data['orthog_error'], data['subspace_error'],
    ['Input Data','Hidden\nVariables', 'Orthogonality\nError (dB)','Subspace\nError (dB)'], 
                                     'Time Steps', Title)             
            
#             Have such a big effect?????            
            

#            plot_4x1(streams, data['hidden'], data['orthog_error'], 
#                     data['subspace_error'], 
#    ['Input Data','Hidden\nVariables',
#            'Orthogonality\nError (dB)','Subspace\nError (dB)'] , 'Time Steps', Title)

                 
        if run_frahst == 1:
        
            # My version of Frahst 
            res_fr = FRAHST_V3_1(streams, alpha=alpha, e_low=e_low, e_high=e_high, 
                             holdOffTime=holdOff, fix_init_Q = 1, r = 1, evalMetrics = 'T') 
            res_fr['Alg'] = 'My Frahst with alpha = ' + str(valA) + ' and e_low = ' + str(valB)
        
            pltSummary2(res_fr, streams, (e_low, e_high))
    
            data = res_fr

            Title = 'My Frahst with alpha = ' + str(valA) + ' and e_low = ' + str(valB)

            plot_4x1(streams, data['hidden'], data['orthog_error'], data['subspace_error'],
    ['Input Data','Hidden\nVariables', 'Orthogonality\nError (dB)','Subspace\nError (dB)'], 
                                     'Time Steps', Title)            
            
        if run_frahst_pedro == 1:
        
            # My version of Frahst 
            res_frped = frahst_pedro(streams, alpha=alpha, e_low=e_low, e_high=e_high, 
                             holdOffTime=holdOff, r = 1, evalMetrics = 'T') 
            res_frped['Alg'] = 'Pedros Frahst with alpha = ' + str(valA) + ' and e_low = ' + str(valB)
        
            pltSummary2(res_frped, streams, (e_low, e_high))
    
            data = res_frped

            Title = 'Pedros Frahst with alpha = ' + str(valA) + ' and e_low = ' + str(valB)

            plot_4x1(streams, data['hidden'], data['orthog_error'], data['subspace_error'],
    ['Input Data','Hidden\nVariables', 'Orthogonality\nError (dB)','Subspace\nError (dB)'], 
                                     'Time Steps', Title)            
            
finish  = time.time() - start
print 'Runtime = ', finish