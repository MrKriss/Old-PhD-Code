# -*- coding: utf-8 -*-
"""
Created on Tue May 10 20:21:20 2011

@author: -
"""

from numpy import array, vstack, eye, zeros, atleast_2d, arange,  \
transpose, concatenate, empty, nan, dot, trace, log10, arccos, sqrt, unique
from numpy.linalg import norm, eig 
from Frahst_single_itter import FRAHST_itter
import numpy as np
import os
import pickle as pk
import scipy.io as sio
from utils import analysis, QRsolveA, pltSummary, GetInHMS

class FrahstNode(object) :
    ''' Node that performs Single iteration of Frahst for input data vec
    
    Outputs r, hidden vars and recon 
    
    All other variables are stored internaly and overwriiten each itteration
    
    '''
    def __init__(self, alpha, e_high, e_low, r = 1, holdOffTime = 0, 
                 evalMetrics = False):
        
        # Parameter Setup 
        self.param  = { 'alpha' : alpha,  
              'e_high' : e_high,
              'e_low' :  e_low,
              'r'  : r, 
              't' : 0,
              'lastChangeAt' : 1, 
              'holdOffTime' : holdOffTime,
              'evalMetrics' : evalMetrics}        
         
    def _execute(self, data_vec):
        
        # Initialisations
        if self.param['t'] == 0 : 
             numStreams = len(data_vec)              
             self.Q = eye(numStreams)       
             self.S = eye(numStreams) * 0.0001 # Avoids Singularity    
             self.v = zeros((numStreams,1)) 
             self.res = {}
             self.E_x = array([0])
             self.E_y = array([0])       
         
        # run single iterable Frahst
        self.Q, self.S, self.v, \
        self.E_x, self.E_y, self.recon, \
        self.hidden, anom = FRAHST_itter(data_vec, 
                                    self.Q, self.S, self.v, 
                                    self.E_x, self.E_y, 
                                    self.param)
        
        ### FOR EVALUATION ###
        #deviation from truth
        if self.param['evalMetrics'] == True :
            
            #alias to matrices for current r
            Qt  = self.Q[:, :self.param['r']]          
                
            if self.param['t'] == 0 :
                self.res['subspace_error'] = zeros((1,1))
                self.res['orthog_error'] = zeros((1,1))                
                self.res['angle_error'] = zeros((1,1))
                self.Cov_mat = zeros([len(data_vec), len(data_vec)])
                
            # Calculate Covarentce Matrix of data up to time t   
            self.Cov_mat = self.param['alpha'] * self.Cov_mat +  dot(data_vec,  data_vec.T)
            # Get eigenvalues and eigenvectors             
            W , V = eig(self.Cov_mat)
            # Use this to sort eigenVectors in according to deccending eigenvalue
            eig_idx = W.argsort() # Get sort index
            eig_idx = eig_idx[::-1] # Reverse order (default is accending)
            # v_r = highest r eigen vectors (accoring to thier eigenvalue if sorted).
            V_r = V[:, eig_idx[:self.param['r']]]          
            # Calculate subspace error        
            C = dot(V_r , V_r.T) - dot(Qt , Qt.T) 
            
            sub_error = 10 * log10(trace(dot(C.T , C)))
            self.res['subspace_error'] = vstack((self.res['subspace_error'], 
                                sub_error)) #frobenius norm in dB
            
            # Calculate angle between projection matrixes
            D = dot(dot(dot(V_r.T, Qt), Qt.T), V_r) 
            eigVal, eigVec = eig(D)
            angle = arccos(sqrt(max(eigVal)))        
            self.res['angle_error'] = vstack((self.res['angle_error'], angle))        
    
            # Calculate deviation from orthonormality
            F = dot(Qt.T , Qt) - eye(self.param['r'])
            orth_error = 10 * log10(trace(dot(F.T , F)))
            self.res['orthog_error'] = vstack((self.res['orthog_error'],
                      orth_error )) #frobenius norm in dB
              
        # Main Function
        self.param['t'] += 1 # Increment t 
                
        return  self.param['r'], self.hidden, self.recon, \
        self.E_x, self.E_y, anom
         
         
class SequenceBufferNode(object):
    """ Sequence Buffer / Sliding window Node
    
    This node stores the last x sequences in a window object and returns it
    Newer entries are on the right, older on the left.
    """
    
    def __init__(self, win_size):
        
        # Initalise objects and variables
        self.win_size = win_size
        self.time = 0

        self.window = empty((win_size), dtype = np.object)
        
        for i in range(win_size):
            self.window[i] = []
         
    def _execute(self, seq_vec):
        
        # Update window, drop oldest sequence data.
        # Newest at the right, oldest at the left
        new_window = empty((self.win_size), dtype = np.object)
        new_window[:self.win_size-1] = self.window[1:self.win_size]
        new_window[self.win_size-1] = seq_vec
        
        return new_window

class  MA_Node(object):
    """  Moving Average Node
    
    Calculates the Cumulative Moving Average (CMA) and Exponential Moving Average
    (EMA) of the sequence data events. Returns a dictionary of results.
    
    Currently keeps all entries.
    """
    
    def __init__(self, EMA_alpha = None):
        
        # Initalise objects and variables
        self.time = 1
        
        self.MAs = {} # Moving Averages Dictionay
        if EMA_alpha == None:
            self.EMA_alpha = 0.98
        else:
            self.EMA_alpha = EMA_alpha
    
    def moving_ave(self, seq_vec):
        """ Update all moving averages """       
        
        unique_packets = unique(array(seq_vec))      
        
        for packet in unique_packets:
            
            if not self.MAs.has_key(packet): # if packet not in dictionary
                self.MAs[packet] = {'CMA' : 0.0, 'EMA' : 0.0}
            
            # Update CMA
            # TODO: note integer devision, could be why getting strange results?
            num_t = seq_vec.count(packet)     
            CA_t_min_1 = self.MAs[packet]['CMA']
            self.MAs[packet]['CMA'] = ((num_t + CA_t_min_1 * (self.time)) 
                                                    / self.time + 1)
            
            # Update EMA
            EA_t_min_1 = self.MAs[packet]['EMA']
            self.MAs[packet]['EMA'] = (num_t * self.EMA_alpha + 
                                        (1 - self.EMA_alpha) * EA_t_min_1)

    def _execute(self, seq_vec):
        
        self.moving_ave(seq_vec)
        self.time += 1
        
        return self.MAs

#===============================================================================
# IF Main Method          
#===============================================================================
if __name__ == '__main__' :    
    
    # Test for FrahstNode -- Correct 
    test_FrahstNode_Abilene = 0 
    test_FrahstNode_OD = 1
    
        
    #############################        
    if test_FrahstNode_Abilene:
    
        # Load data    
        AbileneMat = sio.loadmat('/Users/Main/DataSets/Abilene/Abilene.mat')
        # Number of Packets 
        packet_data = AbileneMat['P']
        packet_data_iter = iter(packet_data)
        
        # Create Frahst node object  (alpha, e_high, e_low)
        F1 = FrahstNode(0.96, 0.98, 0.96, evalMetrics = True)
        
        # Test iteration
        numStreams = AbileneMat['P'].shape[1]          
        # Data Stores
        res = {'hidden' :  zeros((1, numStreams)) * nan,  # Array for hidden Variables
               'E_x' : array([0]),                     # total energy of data 
               'E_y' : array([0]),                # hidden var energy
               'e_ratio' : zeros([1, 1]),              # Energy ratio 
               'RSRE' : zeros([1, 1]),           # Relative squared Reconstruction error 
               'recon' : zeros([1, numStreams]),  # reconstructed data
               'r_hist' : zeros([1, 1]), # history of r values 
               'anomalies' : []}  
        
        t = 0    
        
        for data_vec in packet_data_iter: 
            
            r, hidden, recon, E_x, E_y, anomaly = F1._execute(data_vec)
    
            # Post Processing and Data Storeage
            if t == 0:
                top = 0.0
                bot = 0.0
                    
            top = top + (norm(data_vec - recon) ** 2 )
            bot = bot + (norm(data_vec) ** 2)
            res['RSRE'] = vstack((res['RSRE'], top / bot))        
            res['e_ratio'] = vstack((res['e_ratio'], E_x / E_y))    
            res['hidden'] = vstack((res['hidden'], hidden)) 
            res['r_hist'] = vstack((res['r_hist'], r))
            res['E_x'] = vstack((res['E_x'], E_x)) 
            res['E_y'] = vstack((res['E_y'], E_y)) 
            res['recon'] = vstack((res['recon'], recon.T))
            if anomaly : 
                res['anomalies'].append(t)
            
            t += 1        
            
        res['Alg'] = 'My Incremetal FrahstNode Implimentation of FRAUST'
    
        pltSummary(res, packet_data)    
        
    #######################    
    if test_FrahstNode_OD:
    
        # Load OD network dataset from folder.
        os.chdir('/Users/Main/DataSets/Python/Network_ODs')
    
        # The Link flows vector
        with open('Y_t_data.pk', 'r') as data_file :
            Y_t = pk.load(data_file)     

        packet_data_iter = iter(Y_t)
        
        # Create Frahst node object  (alpha, e_high, e_low)
        F1 = FrahstNode(0.96, 0.98, 0.96, evalMetrics = True)
        
        # Test iteration
        numStreams = Y_t.shape[1]          
        # Data Stores
        res = {'hidden' :  zeros((1, numStreams)) * nan,  # Array for hidden Variables
               'E_x' : array([0]),                     # total energy of data 
               'E_y' : array([0]),                # hidden var energy
               'e_ratio' : zeros([1, 1]),              # Energy ratio 
               'RSRE' : zeros([1, 1]),           # Relative squared Reconstruction error 
               'recon' : zeros([1, numStreams]),  # reconstructed data
               'r_hist' : zeros([1, 1]), # history of r values 
               'anomalies' : []}  
        
        t = 0    
        
        for data_vec in packet_data_iter: 
            
            r, hidden, recon, E_x, E_y, anomaly = F1._execute(data_vec)
    
            # Post Processing and Data Storeage
            if t == 0:
                top = 0.0
                bot = 0.0
                    
            top = top + (norm(data_vec - recon) ** 2 )
            bot = bot + (norm(data_vec) ** 2)
            res['RSRE'] = vstack((res['RSRE'], top / bot))        
            res['e_ratio'] = vstack((res['e_ratio'], E_x / E_y))    
            res['hidden'] = vstack((res['hidden'], hidden)) 
            res['r_hist'] = vstack((res['r_hist'], r))
            res['E_x'] = vstack((res['E_x'], E_x)) 
            res['E_y'] = vstack((res['E_y'], E_y)) 
            res['recon'] = vstack((res['recon'], recon.T))
            if anomaly : 
                res['anomalies'].append(t)
            
            t += 1        
            
        res['Alg'] = 'My Incremetal FrahstNode Implimentation of FRAUST'
    
        pltSummary(res, Y_t)    
        