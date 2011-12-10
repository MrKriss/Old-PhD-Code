# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 15:12:11 2011

@author: -
"""

import mdp
from numpy import array, vstack, eye, zeros, atleast_2d, arange,  \
transpose, concatenate, empty 
from numpy.linalg import norm 
from Frahst_single_itter import FRAHST_itter
import numpy as np
import os
import pickle as pk


# Playing arround with generators
def gen_data(blocks):
    """" Data Generator, iterable
    
    Each Call yeilds a new block of data. I think. May use this as a data 
    loading generator for Abilene or other data sets. 
    INPUTs: - file_name
            - 
    """
    
    for line in xrange(blocks):
        block_x = atleast_2d(arange(2.,1001,2))
        block_y = atleast_2d(arange(1.,1001,2))
        # put variables on columns and observations on rows
        block = transpose(concatenate([block_x,block_y]))
        yield block

class FrahstNode(mdp.Node):
    """ Frahst data processing node 
    # define frahst node.
    # OPTIONS: - Fixed r, i.e. Multi FHST
    #          - dynamic Frahst
    Template: 
        # input_dim
        # output_dim 
        # data_type 
        # _train
        # _stop_training
        # _execute 
        
    # Remember to use mdp.numx, mdp.numx_linalg, mdp.numx_rand
    """
    
    def __init__(self, alpha, e_high, e_low, r = 1, holdOffTime = 0, 
                 evalMetrics = False, input_dim = None, dtype = None):
        
        mdp.Node.__init__(self, input_dim = input_dim, dtype = dtype)
        
        # Parameter Setup 
        self.param  = { 'alpha' : alpha,  
              'e_high' : e_high,
              'e_low' :  e_low,
              'r'  : r, 
              't' : 0,
              'lastChangeAt' : 1, 
              'holdOffTime' : holdOffTime,
              'evalMetrics' : evalMetrics}        
        
    def is_trainable(self):
         return False
    def is_invertible(self):
         return False
    def _get_supported_dtypes(self):
         return ['float32', 'float64']
         
    def _execute(self, data_vec):
        
        if self.param['t'] == 0 : 
            numStreams = len(data_vec)
            # Initialisations 
            Q = eye(numStreams)       
            S = eye(numStreams) * 0.0001 # Avoids Singularity    
            v = zeros((numStreams,1)) 
            res = {}
            res['E_x'] = array([0])
            res['E_y'] = array([0])

        # Main Function
        self.param['t'] += 1 # Increment t        
        # run 1 line Frahst.     
        Q, S, v, E_x, E_y, recon, hidden, anomaly = FRAHST_itter(data_vec,
                    Q, S, v, res['E_x'][-1], res['E_y'][-1], self.param)
        
#        # Store Results         
#        if not hasattr(res,'hidden') : # Create res dectionary       
#            if self.param['t'] == 1:
#                top = 0.0
#                bot = 0.0
#                
#            top = top + (norm(data_vec - recon) ** 2 )
#            bot = bot + (norm(data_vec) ** 2)
#            res['RSRE'] = array(top / bot)   
#            res['e_ratio'] = array(E_x / E_y)
#            res['hidden'] = hidden 
#            res['r_hist'] = array(self.param['r'])
#            res['E_x'] = array(E_x) 
#            res['E_y'] = array(res['E_y'], E_y) 
#            res['recon'] = recon 
#            if anomaly : 
#                res['anomalies'].append(self.param['t']) 
#        else: # Concatonate with previous
#             top = top + (norm(data_vec - recon) ** 2 )
#             bot = bot + (norm(data_vec) ** 2)
#             res['RSRE'] = vstack((res['RSRE'], top / bot))        
#             res['e_ratio'] = vstack((res['e_ratio'], E_x / E_y))    
#             res['hidden'] = vstack((res['hidden'], hidden)) 
#             res['r_hist'] = vstack((res['r_hist'], self.param['r']))
#             res['E_x'] = vstack((res['E_x'], E_x)) 
#             res['E_y'] = vstack((res['E_y'], E_y)) 
#             res['recon'] = vstack((res['recon'], recon)) 
#             if anomaly : 
#                 res['anomalies'].append(self.param['t'])
#        
        return  self.param['r'], hidden, recon    
        

class SequenceBufferNode(mdp.Node):
    """ Frahst data processing node 
    # define frahst node.
    # OPTIONS: - Fixed r, i.e. Multi FHST
    #          - dynamic Frahst
    Template: 
        # input_dim
        # output_dim 
        # data_type 
        # _train
        # _stop_training
        # _execute 
        
    # Remember to use mdp.numx, mdp.numx_linalg, mdp.numx_rand
    
    This node stores the last x sequences in a window object and returns it
    Newer entries are on the right, older on the left.
    """
    
    def __init__(self, win_size, input_dim = None, dtype = None):
        
        mdp.Node.__init__(self, input_dim = input_dim, dtype = dtype)
        
        # Initalise objects and variables
        self.win_size = win_size
        self.time = 0

        self.window = empty((win_size), dtype = np.object)
        
        for i in range(win_size):
            self.window[i] = []

    def is_trainable(self):
         return False
    def is_invertible(self):
         return False
         
    def _execute(self, seq_vec):
        
        # Update window, drop oldest sequence data.
        # Newest at the right, oldest at the left
        new_window = empty((self.win_size), dtype = np.object)
        new_window[:self.win_size-1] = self.window[1:self.win_size]
        new_window[self.win_size-1] = seq_vec
        
        return new_window


class  MA_Node(mdp.Node):
    """ Frahst data processing node 
    # define frahst node.
    # OPTIONS: - Fixed r, i.e. Multi FHST
    #          - dynamic Frahst
    Template: 
        # input_dim
        # output_dim 
        # data_type 
        # _train
        # _stop_training
        # _execute 
        
    # Remember to use mdp.numx, mdp.numx_linalg, mdp.numx_rand
    
    Calculated the CMA and EMA of the sequence data events. Returns a dictionary
    of results.
    
    Currently keeps all entries.
    """
    
    def __init__(self, win_size, input_dim = None, dtype = None):
        
        mdp.Node.__init__(self, input_dim = input_dim, dtype = dtype)
        
        # Initalise objects and variables
        self.time = 0
        
        self.MAs = {} # Moving Averages Dictionay
        self.EMA_alpha = 0.98

    def is_trainable(self):
         return False
    def is_invertible(self):
         return False
         
    def _execute(self, seq_vec):
        
        self.moving_ave(seq_vec)
        
        self.time += 1
        
        return self.MAs
    
    def moving_ave(self, seq_vec):
        """ Update all moving averages """
        
        for packet in seq_vec:
            
            if not self.MAs.has_key(packet): # if packet not in dictionary
                self.MAs[packet] = {'CMA' : 0.0, 'EMA' : 0.0}
            
            # Update CMA
            num_t = seq_vec.count(packet)     
            CA_t_min_1 = self.MAs[packet]['CMA']
            self.MAs[packet]['CMA'] = ((num_t + CA_t_min_1 * (self.time - 1)) 
                                                    / self.time)
            
            # Update EMA
            EA_t_min_1 = self.MAs[packet]['EMA']
            self.MAs[packet]['EMA'] = (num_t * self.EMA_alpha + 
                                        (1 - self.EMA_alpha) * EA_t_min_1)




class DetectAnomaly(mdp.Node):
    """ Node to Process whether anomaly is detected"""
    
    def __init__(self, ignore_up_to, hold_off_time, input_dim = None, dtype = None):
        
        mdp.Node.__init__(self, input_dim = input_dim, dtype = dtype)
    
        # Set internal Variables 
        self.t = 0 
        self.hold_off_time = hold_off_time
        self.ignore_up_to = ignore_up_to
        self.last_change_at = 0
        self.last_r = 0
        self.anomaly = False
    
    def is_trainable(self):
         return False
    def is_invertible(self):
         return False
    
    def _execute(self, r_t, hidden_t, recon_t):
    
        # Analyise r for recent change
            # Flag anomally if so         
        if r_t > self.last_r and self.last_change_at < (self.t - self.hold_off_time):
            self.anomaly = True 
        
        
        #TODO
        # Analyise hidden for CUSUM Change
            # Flag Anomally if so
                
    
        # Update internal Variables
    
        self.last_r = r_t    
    

if __name__ == '__main__' :

    # Load OD network dataset from folder.

    os.chdir('/Users/Main/DataSets/Python/Network_ODs')

    # The packet flows vector
    with open('P_t_data.pk', 'r') as data_file :
        P_t = pk.load(data_file)

    # The Link flows vector
    with open('Y_t_data.pk', 'r') as data_file :
        Y_t = pk.load(data_file)    
 
    # construcnt appropriate flow/nodes 

    

    # Test 





     
    
