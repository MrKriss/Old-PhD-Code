# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 18:59:19 2010

@author: musselle
"""

from artSigs import genCosSignals
from fastRowHouseHolder_float32 import FRHH32, plotEqqFqq32
from basic_Frhh import frhh_min, plotEqqFqqA, genSimSignalsA 
import numpy.matlib as npm
import pickle 
import os


def runBatch(numRuns,BatchName,param):
    """
    Function to run Batch of 'numRuns' initial conditions for FRHH using the 
    'param' parameters  
    """
    
    # Directory in which to create new data directory
    mainPath = 'C:/DataSets/Results/Fraust/'   
    os.chdir(mainPath)
             
    # Results matrix
    e_qq_mat = npm.empty((10000,numRuns))
    f_qq_mat = npm.empty((10001,numRuns))    
    g_qq_mat = npm.empty((10000,numRuns))    
     
    for i in range(numRuns):
        # Generate artificial data streams
        streams = genSimSignalsA(i, -3.0)
        
        # Run Basic Fast row householder subspace tracker
        Q_t, S_t = frhh_min(streams, param['alpha'], param['rr'])
        
        #Q_t, S_t, rr, E_t, E_dash_t, hid_var, z_dash, RSRE, no_inp_count, \
        #no_inp_marker = FRHH32(streams, param['rr'], param['alpha'], param['sci'])
         
        # Calculate deviations from orthogonality and subspace
        e_qq, f_qq, g_qq  = plotEqqFqqA(streams, Q_t, param['alpha'])
     
        # Store results in Dictionary
        dic_name = 'res_' + str(i) # string of the name of the Dictionary
        vars()[dic_name] = {'param' : param,
                            'Q_t' : Q_t,
                            'S_t': S_t,
                            'e_qq' : e_qq,
                            'f_qq' : f_qq,
                            'g_qq' : g_qq} 
                            
                           # 'rr' : rr,
                           # 'E_t' : E_t, 
                           # 'E_dash_t' : E_dash_t, 
                           # 'hid_var' : hid_var, 
                           # 'z_dash' : z_dash,
                           # 'RSRE' : RSRE, 
                           # 'no_inp_count' : no_inp_count,
                           # 'no_inp_marker' : no_inp_marker,
       
        myDic =  vars()[dic_name]
       
        e_qq_mat[:,i] = e_qq   
        f_qq_mat[:,i] = f_qq
        g_qq_mat[:,i] = g_qq
        
        # Save data files  
        mypath = mainPath + '\\' + BatchName 
    
        if not os.path.exists(mypath):
            os.makedirs(mypath)    
    
        runfile = mypath + '/Run' + str(i) + '.dat'
        with open(runfile, 'w') as outfile:  
            pickle.dump(myDic, outfile)
        
        streamfile = mypath + '/stream_inputs'  + str(i) + '.dat'
        streams.tofile(streamfile)
        
        print 'finished ', i, 'th batch'
        
        del myDic, vars()[dic_name] # saves memory
    
    # Save summary Matrixs
    e_qq_path = mypath + '/e_qq_mat.dat'
    e_qq_mat.tofile(e_qq_path)
    f_qq_path = mypath + '/f_qq_mat.dat'
    f_qq_mat.tofile(f_qq_path)
    g_qq_path = mypath + '/g_qq_mat.dat'
    g_qq_mat.tofile(g_qq_path)

    return e_qq_mat, f_qq_mat, g_qq_mat 
    
#===============================================================================
# If main method 
#===============================================================================
if __name__ == '__main__':
    
    numRuns = 10    
    
    param = {'alpha' : 0.996,
             'rr' : 4,
             'sci' : 1}
    
    BatchName = 'Trial 0_1'    
    
    e_qq_mat, f_qq_mat, g_qq_mat = runBatch(numRuns,BatchName,param)
