# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 18:59:19 2010

@author: musselle
"""

from artSigs import genCosSignals
from fastRowHouseHolder import FRHH, plotEqqFqq
from matplotlib.pyplot import figure, plot
import numpy.matlib as npm
import pickle 
import os

numRuns = 10

BatchName = 'CosData Rads sci +1'

#===============================================================================
# Note: 
#   Use simple data gen version 2
#   All seeds the same
# 
#===============================================================================

mainPath = 'C:/DataSets/Results/Fraust'

os.chdir(mainPath)

# Parameters List  (of Dictionaries)
param = {'alpha' : 0.996,
         'rr' : 4,
         'sci' : 1}

# Results list (of dictionaris) 
# results = []
e_qq_mat = npm.empty((10000,numRuns))

for i in range(numRuns):
    # Generate artificial data streams
    streams = genCosSignals(i, -3)
    
    # Run Fast row householder subspace tracker
    Q_t, S_t, rr, E_t, E_dash_t, hid_var, z_dash, RSRE, no_inp_count, \
    no_inp_marker = FRHH(streams, param['rr'], param['alpha'], param['sci'])
    
    # Calculate deviations from orthogonality and subspace
    e_qq, f_qq  = plotEqqFqq(streams, Q_t, param['alpha'],0, 0)
    
    # Store results in Dictionary
    dic_name = 'res_' + str(i) # string of the name of the Dictionary
    vars()[dic_name] = {'param' : param,
                        'Q_t' : Q_t,
                        'S_t': S_t,
                        'rr' : rr,
                        'E_t' : E_t, 
                        'E_dash_t' : E_dash_t, 
                        'hid_var' : hid_var, 
                        'z_dash' : z_dash,
                        'RSRE' : RSRE, 
                        'no_inp_count' : no_inp_count,
                        'no_inp_marker' : no_inp_marker,
                        'e_qq' : e_qq.T,
                        'f_qq' : f_qq.T}                   
       
    myDic =  vars()[dic_name]
    
    e_qq_mat[:,i] = e_qq.T    
     
    # Save data   
    # streams.tofile('stream_inputs.dat')

    mypath = mainPath + '\\' + BatchName 
    
    if not os.path.exists(mypath):
        os.makedirs(mypath)    
    
    runfile = mypath + '/Run' + str(i) + '.dat'
    with open(runfile, 'w') as outfile:  
        pickle.dump(myDic, outfile)
    
    streamfile = mypath + '/stream_inputs'  + str(i) + '.dat'
    streams.tofile(streamfile)
    
    print 'finished ', i, 'th batch'
    
    del myDic, vars()[dic_name]
      
e_qq_path = mypath + '/e_qq_mat.dat'
e_qq_mat.tofile(e_qq_path)

# summary plot
#figure(1)

#for i in range(numRuns):   
#    plot(results[i]['e_qq'])
    
#figure(2)
#for i in range(numRuns):   
#    plot(results[i]['f_qq'])