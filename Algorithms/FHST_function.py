# -*- coding: utf-8 -*-
"""
Created on Sat Mar 05 23:19:11 2011

THIS IS THE VERSION TO RUN ON MAC, NOT WINDOWS

@author: musselle
"""
from numpy import eye, zeros, dot, sqrt, log10, trace, arccos, nan, arange, ones
import numpy as np
import scipy as sp
import numpy.random as npr
from numpy.linalg import qr, eig, norm, solve
from matplotlib.pyplot import plot, figure, title, step, ylim
from artSigs import genCosSignals_no_rand , genCosSignals
import scipy.io as sio
from utils import analysis, QRsolveA, pltSummary2
from PedrosFrahst import frahst_pedro_original
from Frahst_v3_1 import FRAHST_V3_1
from Frahst_v3_3 import FRAHST_V3_3
from Frahst_v3_4 import FRAHST_V3_4
from Frahst_v4_0 import FRAHST_V4_0
from load_syn_ping_data import load_n_store
from QR_eig_solve import QRsolve_eigV
from create_Abilene_links_data import create_abilene_links_data
from MAfunctions import MA_over_window

def FHST(data, init_Q, init_S, init_U, init_v, r=4, alpha=0.96, evalMetrics = 'F', 
                ignoreUp2 = 0):
    """
    Fast Rank Adaptive Householder Subspace Tracking Algorithm (FRAHST)  
    
    Version 6.3 of FRAHST, but only simple FHST component, no adaptaion. static r.
    
    Iterative 
    
    returns Subspace tracked - Q
    
    """   

    
    # Initialise variables and data structures 
    #########################################
    # Derived Variables 
    numStreams = data.shape[1] 
    timeSteps = data.shape[0]
    
    # Data Stores
    res = {'hidden' :  zeros((timeSteps, numStreams)) * nan,  # Array for hidden Variables
           'RSRE' : zeros([timeSteps, 1]),           # Relative squared Reconstruction error 
           'recon' : zeros([timeSteps, numStreams]),  # reconstructed data
           'r_hist' : zeros([timeSteps, 1]),         # history of r values 
           'eig_val': zeros((timeSteps, numStreams)) * nan,  # Estimated Eigenvalues 
           'zt_mean' : zeros((timeSteps, numStreams)), # history of data mean 
           'skips'   : zeros((timeSteps, 1)),          # tracks time steps where Z < 0 
           'EWMA_res' : zeros((timeSteps, 1)),         # residual of energy ratio not acounted for by EWMA
           'S' : [],
           'Q' : []}      
        
    
    Q = init_Q
    S = init_S
    v = init_v
    U = init_U
    
    iter_data = iter(data)
    
    # NOTE algorithm's state (constant memory), S, Q and v and U are kept at max size
    # Main Loop #
    #############
    for t in range(1, timeSteps + 1):
    
        #alias to matrices for current r
        Qt  = Q[:, :r]
        vt  = v[:r, :]
        St  = S[:r, :r]
        Ut  = U[:r, :r]
    
        zt = iter_data.next()  
        
        '''Data Preprocessing'''
        # Convert to a column Vector 
        zt = zt.reshape(zt.shape[0],1) 
    
        small_value = 0.0001
        # Check S remains non-singular
        for idx in range(r):
            if S[idx, idx] < small_value:
                S[idx,idx] = small_value
        
        '''Begin main algorithm'''        
        ht = dot(Qt.T , zt) 
        
        Z = dot(zt.T,  zt) - dot(ht.T , ht)

        if Z > 0 :
            
            # Refined version, use of extra terms
            u_vec = dot(St , vt)
            X = (alpha * St) + (2 * alpha * dot(u_vec, vt.T)) + dot(ht , ht.T)
    
            # Estimate eigenValues + Solve Ax = b using QR decomposition 
            b_vec, e_values, Ut = QRsolve_eigV(X.T, Z, ht, Ut)
            
            beta  = 4 * (dot(b_vec.T , b_vec) + 1)
        
            phi_sq = 0.5 + (1.0 / sqrt(beta))
        
            phi = sqrt(phi_sq)
    
            gamma = (1.0 - 2 * phi_sq) / (2 * phi)
            
            delta = phi / sqrt(Z)
            
            vt = gamma * b_vec 
            
            St = X - ((1 /delta) * dot(vt , ht.T))
            
            w = (delta * ht) - (vt) 
            
            ee = delta * zt - dot(Qt , w) 
            
            Qt = Qt - 2 * dot(ee , vt.T) 
            
        else: # if Z is not > 0

            if norm(zt) > 0 and norm(ht) > 0 : # May be due to zt <= ht 
                res['skips'][t-1] = 2 # record Skips
            else: # or may be due to zt and ht = 0
                St = alpha * St # Continue decay of St 
                res['skips'][t-1] = 1 # record Skips
        
        #restore data structures
        Q[:,:r] = Qt
        v[:r,:] = vt
        S[:r, :r] = St
        U[:r,:r] = Ut
        
        ''' EVALUATION '''
        # Deviations from true dominant subspace 
        if evalMetrics == 'T' :
            if t == 1 :
                res['subspace_error'] = zeros((timeSteps,1))
                res['orthog_error'] = zeros((timeSteps,1))
                res['angle_error'] = zeros((timeSteps,1))
                res['true_eig_val'] = ones((timeSteps, numStreams)) * np.NAN
                Cov_mat = zeros([numStreams,numStreams])
                
            # Calculate Covarentce Matrix of data up to time t   
            Cov_mat = alpha * Cov_mat +  dot(zt,  zt.T)
            
            # Get eigenvalues and eigenvectors             
            W , V = eig(Cov_mat)
            # Use this to sort eigenVectors in according to deccending eigenvalue
            eig_idx = W.argsort() # Get sort index
            eig_idx = eig_idx[::-1] # Reverse order (default is accending)
            # v_r = highest r eigen vectors (accoring to thier eigenvalue if sorted).
            V_r = V[:, eig_idx[:r]]          
            # Calculate subspace error        
            C = dot(V_r , V_r.T) - dot(Qt , Qt.T)  
            res['subspace_error'][t-1,0] = 10 * log10(trace(dot(C.T , C))) #frobenius norm in dB
        
            # Store True r Dominant Eigenvalues
            res['true_eig_val'][t-1,:r] = W[eig_idx[:r]]
    
            # Calculate deviation from orthonormality
            F = dot(Qt.T , Qt) - eye(r)
            res['orthog_error'][t-1,0] = 10 * log10(trace(dot(F.T , F))) #frobenius norm in dB
        
        '''Store Values''' 
        # Record S 
        res['S'].append(St)
        res['Q'].append(Qt)
        
        # Store eigen values
        if 'e_values' not in locals():
            res['eig_val'][t-1,:r] = 0.0
        else:
            res['eig_val'][t-1,:r] = e_values[:r]
        
        # Record reconstrunted z
        z_hat = dot(Qt , ht)
        res['recon'][t-1,:] = z_hat.T[0,:]

        # Record hidden variables
        res['hidden'][t-1, :r] = ht.T[0,:]
        
        # Record RSRE
        if t == 1:
            top = 0.0
            bot = 0.0
            
        top = top + (norm(zt - z_hat) ** 2 )
        bot = bot + (norm(zt) ** 2)
        res['RSRE'][t-1, 0] = top / bot
        
        # Record r
        res['r_hist'][t-1, 0] = r
            
    return res 
