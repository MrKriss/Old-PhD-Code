#!/usr/bin/env python
#coding:utf-8
# Author:  C Musselle --<>
# Purpose: Main Experiment Script 
# Created: 11/23/11

import numpy as np
from numpy import dot
from math import sqrt
import numpy.linalg as npl
import scipy as sp
import matplotlib.pyplot as plt
import sys
import os 
from load_data import load_data, load_ts_data
from utils import QRsolveA, QRsolve_eigV, pltSummary2
from normalisationFunc import zscore
from burg_AR import burg_AR
from artSigs import sin_rand_combo, simple_sins, simple_sins_3z
from plot_utils import plot_3x1, plot_4x1

"""
Code Description:
  .
"""

def initialise(p, numStreams):
    """ Initialise all Frahst variables """
    
    r = p['init_r']
    
    # Q_0
    if p['fix_init_Q'] != 0:  # fix inital Q as identity 
        q_0 = np.eye(numStreams);
        Q = q_0
    else: # generate random orthonormal matrix N x r 
        Q = np.eye(numStreams) # Max size of Q
        Q_0, R_0 = npl.qr(np.random.rand(numStreams,r))   
        Q[:,:r] = Q_0          
    # S_0
    small_value = p['small_value']
    S = np.eye(numStreams) * small_value # Avoids Singularity    
    # v-1
    v = np.zeros((numStreams,1)) 
    # U(t-1) for eigenvalue estimation
    U = np.eye(numStreams)
    
    # Define st dictionary 
    store  = {'Q' : Q,         # Orthogonal dominant subspace vectors
                  'S' : S,     # Energy
                  'v' : v,     # used for S update
                  'U' : U,     # Used for eigen value calculations 
                  'r' : r,     # Previous rank of Q and number of hidden variables h
                  't' : 0,     # Timestep, used for ignoreup2  
                  'sumEz' : 0.0,        # Exponetial sum of zt Energy 
                  'sumEh': 0.0,     # Exponential sum of ht energy  
                  'last_Z_pos' : bool(1), # Tracks last Z value, used for eigenvalue calculation  
                  'U2t' : 0,   # st for U2t, used for alternative eigen value tracking
                  'anomaly': bool(0),
                  'eig_val': np.zeros(1)}
    
    if p.has_key('e_low'):
        store['lastChangeAt'] = 0.0
    if p.has_key('AR_order'):
        store['pred_zt'] = 0.0
        store['pred_err'] = np.zeros(numStreams)
        store['pred_err_norm'] = 0.0
        store['pred_err_ave'] = 0.0
    
    return store


def FRAHST_V3_1_iter(zt, st, p):
    ''' 
    zt = data at next time step 
    st = {'Q' : Q,     - Orthogonal dominant subspace vectors
              'S' : S,     - Energy 
              'U' : U,     - Orthonormal component of Orthogonal iteration around X.T
              'v' : v,     - Used for speed up of calculating X update 
              'r' : r,     - Previous rank of Q and number of hidden variables h
              't' : t,     - Timestep, used for ignoreup2  
              'sumEz' : Et,  - Exponetial sum of zt Energy 
              'sumEh': E_dash_t }- Exponential sum of ht energy 
              
    p = {'alpha': 0.96,
          'init_r' : 1, 
          'holdOffTime' : 0,
          'evalMetrics' : 'F',
          'EW_mean_alpha' : 0.1,
          'EWMA_filter_alpha' : 0.3,
          'residual_thresh' : 0.1,
          'e_high' : 0.98,
          'e_low' : 0.95,
          'static_r' : 0,
          'r_upper_bound' : None,
          'fix_init_Q' : 0,
          'ignoreUp2' : 0 }
    '''
    """
    Fast Rank Adaptive Householder Subspace Tracking Algorithm (FRAHST)  
    
    Version 7.0 - An attempt to cull unused aprts of the algorithm, and make it iterative 
    
    Version 6.4 - Problem with skips if Z < 0. happens when zt< ht. Only a problem when r --> N. Eigen values not updataed.
                - Fixed by using Strobarchs alternative eigenvalue approx method in this case. Still relies on alpha ~ 1.
                - Introduced Z normalisation as preprocessing method. MA/EWMA removes correlations. 
    
    Version 6.3 - Now uses only a single threshold F_min and the tollerance parameter epsilon.
                - Fixed error in rank adaptation (keeper deleted row and col of Q, instead of just col) 
    
    Version 6.2 - In light of 6.1.5, EWMA incrementally incorperated, and cleaned up a bit.
                - Now uses an extra parameter epsilon to buffer thresholding condition. 
    
    Version 6.1.5 - Tried useing CUSUM on Energy ratio to detect anomalous points. 
                - Also Have the option to fix r or allow to adapt. Though right parameters 
                for adaptation require some experimentation.  
                - NOt yet incorperated, tried to run just as a batch on res['e_ratio'], but was
                a lot slower than previously thought < 300 seconds. W^2 time with window length W. A quick test with
                EWMA_filter was MUCH MUCH quicker < 1 second.
                Will likely use EWMA instead of CUSUM. 
                To_do: add EWMA filter to algorithm output....
        
    Version 6.1 - basicly 6.0 but without the junk func + the actual new eigen(enegy)tracking 
                - Turns out E_dash_t ~ S_trace or sum(eig_val)
                            E_t ~ EW_var2(zt) discounted by alpha a la covarience matrix    
                - no need to calculate incremental mean and var anymore 
                - Thresholding mechanism now uses two thresholds.
                     - if below the lowest -- > increment r
                     - if abouve the higher --> test if (E_dast_t - eig_i ) / E_t is above e_high, 
                       if so remove dimentions. 
                     - difference between e_low and e_high acts as a 'safety' buffer, as removing an eig can 
                      result in too much variance being subtracted because eigs are only smoothed estimates 
                      of the true values. Takes time for est_eit to reach true eigs.    

                - NEXT (maybe) Normalisation of data optional as a preprocessing of data.
                
    Version 6.0 - Aim: Different rank adjusting mechanism
                      compares sum of r eigenvalues to variance of entire data.
                - Performes incremental calculation of data mean and variance. (no longer in later version )

    Version 5.0 - No changes of 5.0 incorperated in this version 
        
    Version 4.0 - Now also approximates eigenvalues for the approximated tracked basis for the eignevectors          
                - Approach uses an orthogonal iteration arround X.T 
                - Note, only a good approximation if alpha ~< 1. Used as its the fastest method 
                as X.T b --> b must be solved anyway. 
                - New entries in res
                    ['eig_val'] - estimated eigenvalues
                    ['true_eig_val'] - explicitly calculated eigenvalues (only if evalMetrics = T) 
        
    VErsion 3.4 - input data z is time lagged series up to length l. 
                - Algorithm is essentially same as 3.3, just adds pre processing to data vector
                - input Vector z_t is now of length (N times L) where L is window length
                - Use L = 1 for same results as 3.3 
                - Q is increased accordingly 
                
    Version 3.3 - Add decay of S and in the event of vanishing S
                - Make sure rank of S does not drop (and work out what that means!) - stops S going singular
        
    Version 3.2 -  Added ability to fix r to a static value., and also give it an upper bound.
                   If undefined, defaults to num of data streams. 
        
    Version 3.1 - Combines good bits of Pedros version, with my correction of the bugs
    
    Changed how the algorithm deals with sci. only difference, but somehow has a bigish 
    effect on the output.
    
    """   
    # Derived Variables 
    # Length of z or numStreams = N x L
    numStreams = data.shape[1] 
    r = st['r']    
    
    # NOTE algorithm'st st Q, S, v and U are kept at max size (constant memory)
    # alias to st for current value of r
    Qt  = st['Q'][:, :r]
    vt  = st['v'][:r, :]
    St  = st['S'][:r, :r]
    #Ut  = st['U'][:r, :r]

    # Main Loop # (but only go through once)
    #############
    
    '''Data Preprocessing'''        
    # Convert to a column Vector 
    #zt = zt.reshape(zt.shape[0],1) 

    # Define upper bound for r if not predefined  
    if p['r_upper_bound'] == None :
        p['r_upper_bound'] = len(zt)
    
    # Check S remains non-singular
    #for idx in range(r):
        #if St[idx, idx] < small_value:
            #St[idx,idx] = small_value
    
    '''Begin main algorithm'''        
    ht = dot(Qt.T, zt) 
    Z = dot(zt.T, zt) - dot(ht.T , ht)

    if Z > 0 :
        
        # Flag for whether Z(t-1) > 0
        # Used for alternative eigenvalue calculation if Z < 0
        st['last_Z_pos'] = bool(1)
            
        # Refined version, use of extra terms
        u_vec = dot(St , vt)
        X = (p['alpha'] * St) + (2 * p['alpha'] * dot(u_vec, vt.T)) + dot(ht, ht.T)
    
        # Estimate eigenValues + Solve Ax = b using QR decomposition 
        #b_vec, e_values, Ut = QRsolve_eigV(X.T, Z, ht, Ut)
        # Solve Ax = b using QR updates - not strictly needed 
        A = X.T
        B = sqrt(Z) * ht
        b_vec = QRsolveA(A,B)
        
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
        ## Implies norm of ht is > zt or zt --> 0
        
        #St = p['alpha'] * St # Continue decay of S matrix 
        
        ## Recalculate Eigenvalues using other method 
        ## (less fast, but does not need Z to be positive)
        #if st['last_Z_pos'] == 1:
            ## New U                 
            #U2t_min1 = np.eye(r)
            ## PHI = np.dot(Qt_min1.T, Qt) - Unnecessary as S* Phi ~ S if phi ~ I
            ## Also the above causes complications when rank r changes
            #Wt = np.dot(St, U2t_min1)
            #U2t, R2 = qr(Wt) # Decomposition
            #PHI_U = np.dot(U2t_min1.T,U2t)
            #e_values = np.diag(np.dot(R2,PHI_U))
            #st['U2t'] = U2t
        #elif st['last_Z_pos'] == 0:
            
            #U2t_min1 = st['U2t']                
            ##PHI = np.dot(Qt_min1.T, Qt) - Unnecessary as S* Phi ~ S if phi ~ I
            ## Also the above causes complications when rank r changes 
            #Wt = np.dot(St, U2t_min1)
            #U2t, R2 = qr(Wt) # Decomposition
            #PHI_U = np.dot(U2t_min1.T,U2t)
            #e_values = np.diag(np.dot(R2,PHI_U))
            #st['U2t'] = U2t
        pass

    # update input data 
    st['Q'][:,:r] = Qt
    st['v'][:r,:] = vt
    st['S'][:r, :r] = St
    #st['U'][:r,:r] = Ut
        
    '''Store Values''' 
    # store eigen values
    #st['eig_val'] = e_values[:r]
    # Record hidden variables
    ht_vec = np.hstack((ht.T[0,:], np.array([np.nan]*(numStreams-r))))
    st['ht'] = ht_vec
        
    # Energy Ratio Calculations    
    st['sumEz'] = p['alpha']*st['sumEz'] + np.sum(zt ** 2) # Energy of Data
    st['sumEh'] = p['alpha']*st['sumEh'] + np.sum(ht ** 2) # Energy of Hidden Variables
    
    if st['sumEz'] == 0 : # Catch NaNs 
        st['e_ratio']  = 0.0
    else:
        st['e_ratio']  = st['sumEh'] / st['sumEz']
        
    return st
    
def FRAHST_V7_0_iter(zt, st, p):
    
    ''' 
    zt = data at next time step 
    st = {'Q' : Q,     - Orthogonal dominant subspace vectors
              'S' : S,     - Energy 
              'U' : U,     - Orthonormal component of Orthogonal iteration around X.T
              'v' : v,     - Used for speed up of calculating X update 
              'r' : r,     - Previous rank of Q and number of hidden variables h
              't' : t,     - Timestep, used for ignoreup2  
              'sumEz' : Et,  - Exponetial sum of zt Energy 
              'sumEh': E_dash_t }- Exponential sum of ht energy 
              
    p = {'alpha': 0.96,
          'lag' :  0,
          'init_r' : 1, 
          'holdOffTime' : 0,
          'evalMetrics' : 'F',
          'EW_mean_alpha' : 0.1,
          'EWMA_filter_alpha' : 0.3,
          'residual_thresh' : 0.1,
          'F_min' : 0.9,
          'epsilon' : 0.05,
          'static_r' : 0,
          'r_upper_bound' : None,
          'fix_init_Q' : 0,
          'ignoreUp2' : 0 }
    '''
    """
    Fast Rank Adaptive Householder Subspace Tracking Algorithm (FRAHST)  
    
    Version 7.0 - An attempt to cull unused aprts of the algorithm, and make it iterative 
    
    Version 6.4 - Problem with skips if Z < 0. happens when zt< ht. Only a problem when r --> N. Eigen values not updataed.
                - Fixed by using Strobarchs alternative eigenvalue approx method in this case. Still relies on alpha ~ 1.
                - Introduced Z normalisation as preprocessing method. MA/EWMA removes correlations. 
    
    Version 6.3 - Now uses only a single threshold F_min and the tollerance parameter epsilon.
                - Fixed error in rank adaptation (keeper deleted row and col of Q, instead of just col) 
    
    Version 6.2 - In light of 6.1.5, EWMA incrementally incorperated, and cleaned up a bit.
                - Now uses an extra parameter epsilon to buffer thresholding condition. 
    
    Version 6.1.5 - Tried useing CUSUM on Energy ratio to detect anomalous points. 
                - Also Have the option to fix r or allow to adapt. Though right parameters 
                for adaptation require some experimentation.  
                - NOt yet incorperated, tried to run just as a batch on res['e_ratio'], but was
                a lot slower than previously thought < 300 seconds. W^2 time with window length W. A quick test with
                EWMA_filter was MUCH MUCH quicker < 1 second.
                Will likely use EWMA instead of CUSUM. 
                To_do: add EWMA filter to algorithm output....
        
    Version 6.1 - basicly 6.0 but without the junk func + the actual new eigen(enegy)tracking 
                - Turns out E_dash_t ~ S_trace or sum(eig_val)
                            E_t ~ EW_var2(zt) discounted by alpha a la covarience matrix    
                - no need to calculate incremental mean and var anymore 
                - Thresholding mechanism now uses two thresholds.
                     - if below the lowest -- > increment r
                     - if abouve the higher --> test if (E_dast_t - eig_i ) / E_t is above e_high, 
                       if so remove dimentions. 
                     - difference between e_low and e_high acts as a 'safety' buffer, as removing an eig can 
                      result in too much variance being subtracted because eigs are only smoothed estimates 
                      of the true values. Takes time for est_eit to reach true eigs.    

                - NEXT (maybe) Normalisation of data optional as a preprocessing of data.
                
    Version 6.0 - Aim: Different rank adjusting mechanism
                      compares sum of r eigenvalues to variance of entire data.
                - Performes incremental calculation of data mean and variance. (no longer in later version )

    Version 5.0 - No changes of 5.0 incorperated in this version 
        
    Version 4.0 - Now also approximates eigenvalues for the approximated tracked basis for the eignevectors          
                - Approach uses an orthogonal iteration arround X.T 
                - Note, only a good approximation if alpha ~< 1. Used as its the fastest method 
                as X.T b --> b must be solved anyway. 
                - New entries in res
                    ['eig_val'] - estimated eigenvalues
                    ['true_eig_val'] - explicitly calculated eigenvalues (only if evalMetrics = T) 
        
    VErsion 3.4 - input data z is time lagged series up to length l. 
                - Algorithm is essentially same as 3.3, just adds pre processing to data vector
                - input Vector z_t is now of length (N times L) where L is window length
                - Use L = 1 for same results as 3.3 
                - Q is increased accordingly 
                
    Version 3.3 - Add decay of S and in the event of vanishing S
                - Make sure rank of S does not drop (and work out what that means!) - stops S going singular
        
    Version 3.2 -  Added ability to fix r to a static value., and also give it an upper bound.
                   If undefined, defaults to num of data streams. 
        
    Version 3.1 - Combines good bits of Pedros version, with my correction of the bugs
    
    Changed how the algorithm deals with sci. only difference, but somehow has a bigish 
    effect on the output.
    
    """   
    # Derived Variables 
    # Length of z or numStreams = N x L
    numStreams = data.shape[1] 
    r = st['r']
    
    # NOTE algorithm's s Q, S, v and U are kept at max size (constant memory)
    # alias to s for current value of r
    Qt  = st['Q'][:, :r]
    vt  = st['v'][:r, :]
    St  = st['S'][:r, :r]
    Ut  = st['U'][:r, :r]

    # Main Loop # (but only go through once)
    #############
    
    '''Data Preprocessing'''       
    # Define upper bound for r if not predefined  
    if p['r_upper_bound'] == None :
        p['r_upper_bound'] = len(zt)
    
    # Check S remains non-singular
    for idx in range(r):
        if St[idx, idx] < p['small_value']:
            St[idx,idx] = p['small_value']
    
    '''Begin main algorithm'''        
    ht = dot(Qt.T, zt) 
    Z = dot(zt.T, zt) - dot(ht.T , ht)

    if Z > 0 :
        
        # Flag for whether Z(t-1) > 0
        # Used for alternative eigenvalue calculation if Z < 0
        st['last_Z_pos'] = bool(1)
            
        # Refined version, use of extra terms
        u_vec = dot(St , vt)
        X = (p['alpha'] * St) + (2 * p['alpha'] * dot(u_vec, vt.T)) + dot(ht, ht.T)
    
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
        # Implies norm of ht is > zt or zt --> 0
        
        St = p['alpha'] * St # Continue decay of S matrix 
        
        # Recalculate Eigenvalues using other method 
        # (less fast, but does not need Z to be positive)
        if st['last_Z_pos'] == 1:
            # New U                 
            U2t_min1 = np.eye(r)
            # PHI = np.dot(Qt_min1.T, Qt) - Unnecessary as S* Phi ~ S if phi ~ I
            # Also the above causes complications when rank r changes
            Wt = np.dot(St, U2t_min1)
            U2t, R2 = npl.qr(Wt) # Decomposition
            PHI_U = np.dot(U2t_min1.T,U2t)
            e_values = np.diag(np.dot(R2,PHI_U))
            st['U2t'] = U2t
        elif st['last_Z_pos'] == 0:
            
            U2t_min1 = st['U2t']                
            #PHI = np.dot(Qt_min1.T, Qt) - Unnecessary as S* Phi ~ S if phi ~ I
            # Also the above causes complications when rank r changes 
            Wt = np.dot(St, U2t_min1)
            U2t, R2 = npl.qr(Wt) # Decomposition
            PHI_U = np.dot(U2t_min1.T,U2t)
            e_values = np.diag(np.dot(R2,PHI_U))
            st['U2t'] = U2t

    # update input data 
    st['Q'][:,:r] = Qt
    st['v'][:r,:] = vt
    st['S'][:r, :r] = St
    st['U'][:r,:r] = Ut
        
    '''Store Values''' 
    # Store eigen values
    st['eig_val'] = e_values[:r]
    # Record hidden variables
    ht_vec = np.hstack((ht.T[0,:], np.array([np.nan]*(numStreams-r))))
    st['ht'] = ht_vec
        
    # Energy Ratio Calculations    
    st['sumEz'] = p['alpha']*st['sumEz'] + np.sum(zt ** 2) # Energy of Data
    st['sumEh'] = p['alpha']*st['sumEh'] + np.sum(ht ** 2) # Energy of Hidden Variables
    
    if st['sumEz'] == 0 : # Catch NaNs 
        st['e_ratio']  = 0.0
    else:
        st['e_ratio']  = st['sumEh'] / st['sumEz']
        
    return st



def rank_adjust_energy(st, zt):
    """ Adjust rank r of subspace accoring to Pedros Energy-adaptive method """
   
    Q = st['Q']
    S = st['S']
    U = st['U']
    r = st['r']
    t = st['t']

    ''' Rank Adjust - needs ht, Q, zt, '''
    # Adjust Q_t, St and Ut for change in r
    if st['e_ratio'] < p['e_low'] and st['r'] < p['r_upper_bound'] and t > p['ignoreUp2'] \
    and st['lastChangeAt'] < (t - p['holdOffTime']) :
                
        # Extend Q by z_bar
        # new h with updated Q 
        h_dash = dot(Q[:, :r].T,  zt)
        # Error 
        z_bar = zt - dot(Q[:, :r] , h_dash)
        z_bar_norm = npl.norm(z_bar)
        z_bar = z_bar / z_bar_norm
        # Extend Q
        Q[:, r] = z_bar.T[0,:]
                
        # Set next row and column to zero
        S[r, :] = 0.0
        S[:, r] = 0.0
        s_end  = z_bar_norm ** 2
        S[r, r] = s_end # change last element
                
        # Update Ut_1
        # Set next row and column to zero
        U[r, :] = 0.0
        U[:, r] = 0.0
        U[r, r] = 1.0 # change last element
                
        # Flag as anomaly 
        st['anomaly'] = bool(1)
        
        st['lastChangeAt'] = t
        
        # new r, increment and st
        st['r'] = r + 1
        st['Q'] = Q 
        st['S'] = S 
        st['U'] = U 
                
    elif st['e_ratio'] > p['e_high'] and r > 1 and t > p['ignoreUp2'] \
    and st['lastChangeAt'] < (t - p['holdOffTime']):
        
        st['lastChangeAt'] = t
        
        st['r'] = r  - 1
    
    return st
    
    
def rank_adjust_eigen(st, zt):
    """ Adjust rank r of subspace accoring to my eigvalue-adaptive method """
    
    Q = st['Q']
    S = st['S']
    U = st['U']
    r = st['r']
    t = st['t']

    ''' Rank Adjust - needs ht, Q, zt, '''
    # Adjust Q_t, St and Ut for change in r
    if st['e_ratio'] < p['F_min'] and st['r'] < p['r_upper_bound'] and t > p['ignoreUp2']:
                
        # Extend Q by z_bar
        # new h with updated Q 
        h_dash = dot(Q[:, :r].T,  zt)
        # Error 
        z_bar = zt - dot(Q[:, :r] , h_dash)
        z_bar_norm = npl.norm(z_bar)
        z_bar = z_bar / z_bar_norm
        # Extend Q
        Q[:, r] = z_bar.T[0,:]
                
        # Set next row and column to zero
        S[r, :] = 0.0
        S[:, r] = 0.0
        s_end  = z_bar_norm ** 2
        S[r, r] = s_end # change last element
                
        # Update Ut_1
        # Set next row and column to zero
        U[r, :] = 0.0
        U[:, r] = 0.0
        U[r, r] = 1.0 # change last element
                
        # Update eigenvalue 
        st['eig_val'] = np.hstack((st['eig_val'], z_bar_norm ** 2)) 
        # This is the bit where the estimate is off? dont really have anything better 
                
        # new r, increment and store
        st['r'] = r + 1
        st['Q'] = Q 
        st['S'] = S 
        st['U'] = U 
                
    elif st['e_ratio'] > p['F_min'] and r > 1 and t > p['ignoreUp2']:
                
        keeper = np.ones(r, dtype = bool)
        # Sorted in accending order
        # Causing problems, skip sorting, (quicker/simpler), and just cylce from with last 
        # added eignevalue through to newest.  
        # sorted_eigs = e_values[e_values.argsort()]
        
        acounted_var = st['sumEh']
        for idx in range(r)[::-1]:
            if ((acounted_var - st['eig_val'][idx]) / st['sumEz']) > p['F_min'] + p['epsilon']:
                keeper[idx] = 0
                acounted_var = acounted_var - st['eig_val'][idx]
        
        # use keeper as a logical selector for S and Q and U 
        if not keeper.all():
            
            # Delete rows/cols in Q, S, and U. 
            newQ = Q[:,:r].copy()
            newQ = newQ[:,keeper] # cols eliminated                        
            st['Q'][:newQ.shape[0], :newQ.shape[1]] = newQ
            
            newS = S[:r,:r].copy()
            newS = newS[keeper,:][:,keeper] # rows/cols eliminated                        
            st['S'][:newS.shape[0], :newS.shape[1]] = newS
            
            newU = U[:r,:r].copy()
            newU = newU[keeper,:][:,keeper] # rows/cols eliminated                        
            st['U'][:newU.shape[0], :newU.shape[1]] = newU
            
            r = keeper.sum()
            if r == 0 :
                r = 1
                       
            st['r'] = r  
    
    return st





def anomaly_EWMA(st, p):
    """ Track e_ratio for sharp dips and spikes using EWMA """
    # Run EWMA on e_ratio 
    if st.has_key('pred_data') :  
        pred_data = st['pred_data']
    else:
        pred_data = 0.0 # initialise value

    # Calculate residual usung last time steps prediction 
    residual = np.abs(st['e_ratio'] - pred_data)

    # Update prediction for next time step
    st['pred_data'] = p['EWMA_filter_alpha'] * st['e_ratio'] + \
        (1-p['EWMA_filter_alpha']) * pred_data    
                
    # Threshold residual for anomaly
    if residual > p['residual_thresh'] and st['t'] > p['ignoreUp2']:
        # Record time step of anomaly            
        st['anomaly'] = bool(1)

    st['EWMA_res'] = residual

    return st




def anomaly_AR_forcasting(st, p):
    """ Use Auto Regressive prediction to calculate anomalies 
    
    h_window is ht_AR_win x numStreams 
    """

    # Build/Slide h_window
    if  st.has_key('h_window'):
        #dropped_data_vec = st['h_window'][:,0].copy()
        #new_data_vec = st['ht']
        st['h_window'][:-1,:] = st['h_window'][1:,:] # Shift Window
        st['h_window'][-1,:] = np.nan_to_num(st['ht'])
    else:
        st['h_window'] = np.zeros((p['ht_AR_win'], st['ht'].size))
        st['h_window'][-1,:] = np.nan_to_num(st['ht'])
    
    ''' Forcasting '''
    if st['t'] > p['ht_AR_win']:
        # Get Coefficents for ht+1
        # Get h-buffer window (can speed this up latter)
        #h_buffer = np.nan_to_num(res['hidden'][t-h_AR_buff:t, :])
        pred_h = np.zeros((st['r'],1))
        for i in range(st['r']):
            coeffs = burg_AR(p['AR_order'], st['h_window'][:,i])
            for j in range(p['AR_order']):
                pred_h[i,0] -= coeffs[j] * st['h_window'][-1-j, i]
        

        # Calculate Prediction error based on last time step prediction  
        st['pred_err'] = np.abs(st['pred_zt'] - zt.T)
        st['pred_err_ave'] = np.abs(st['pred_zt'] - zt.T).sum() / numStreams
        st['pred_err_norm'] = npl.norm(st['pred_zt'] - zt.T)
    
        # Update prediction for next time step 
        st['pred_zt'] = dot(st['Q'][:,:st['r']], pred_h).T
        
        '''Anomaly Test'''
        if st['pred_err_norm'] > p['err_thresh']:
            st['anomaly'] = True
            
    return st

def anomaly_AR_Qstat(st, p):
    """ Use Auto Regressive prediction and Q statistic to calculate anomalies 
    
    Step 1: Calculate prediction of next data point z_t+1 by fitting 
    an AR model to each hidden Variable, predicting each hi_t+1 and then 
    projecting h_t+1 using the basis Q at current time t.
    
    Step 2: Track a sample of the residual 
    res_t,l = [(z_t-l - predicted_z_t-l)+ ... + (z_t - predicted_z_t)]
    
    Step 3: Calculate Q statistic of residual sample and test 
    H0 = No correlation in residuals, all iid, just noise. 
    Ha = Not all iid. 
    
    h_window is ht_AR_win x numStreams 
    """

    if not st.has_key('Q_stat'):
        st['Q_stat'] = np.zeros(st['ht'].size) * np.nan
        st['pred_h'] = np.zeros(st['ht'].size) * np.nan
        st['coeffs'] = np.zeros((st['ht'].size,p['AR_order'])) * np.nan
        st['h_res'] =  np.zeros(st['ht'].size)
        st['h_res_aa'] =  np.zeros(1)
        st['h_res_norm'] =  np.zeros(1)
        

    # Build/Slide h_window
    if  st.has_key('h_window'):
        st['h_window'][:-1,:] = st['h_window'][1:,:] # Shift Window
        st['h_window'][-1,:] = np.nan_to_num(st['ht'])
    else:
        st['h_window'] = np.zeros((p['ht_AR_win'], st['ht'].size))
        st['h_window'][-1,:] = np.nan_to_num(st['ht'])
    
    ''' Forcasting '''
    if st['t'] > p['ht_AR_win']:
        
        # Calculate error in last h prediction 
        if st.has_key('pred_h'):
            st['h_res'] = np.nan_to_num(st['pred_h']) - np.nan_to_num(st['ht'])
            st['h_res_aa'] = np.abs(st['h_res']).sum() / st['ht'].size
            st['h_res_norm'] = npl.norm(st['h_res'])
            
            # Build/Slide h_residual_window
            if  st.has_key('h_res_win'):
                st['h_res_win'][:-1,:] = st['h_res_win'][1:,:] # Shift Window
                st['h_res_win'][-1,:] = np.nan_to_num(st['h_res'])
            else:
                st['h_res_win'] = np.zeros((p['ht_AR_win'], st['h_res'].size))
                st['h_res_win'][-1,:] = np.nan_to_num(st['h_res'])
                
            # Calculate Q statistic: per h: arhhhg
            for i in xrange(st['coeffs'].shape[0]): # coeffs.shape[0] correspond to r at last time step  
                st['Q_stat'][i] = Q_stat(st['h_res_win'][:,i], st['coeffs'][i,:], p['Q_lag'])
    
        # Get Coefficents for ht+1        
        # st['pred_h'] = np.zeros((st['r'],1))
        st['coeffs'] = np.zeros((st['ht'].size,p['AR_order'])) * np.nan
        for i in range(st['r']):
            st['coeffs'][i, :] = burg_AR(p['AR_order'], st['h_window'][:,i])
            for j in range(p['AR_order']):
                st['pred_h'][i] -= st['coeffs'][i,j] * st['h_window'][-1-j, i]
        
        # Calculate Prediction error based on last time step prediction  
        st['zt_res'] = np.abs(st['pred_zt'] - zt.T)
        st['zt_res_aa'] = np.abs(st['pred_zt'] - zt.T).sum() / numStreams
        st['zt_res_norm'] = npl.norm(st['pred_zt'] - zt.T)
    
        # Update prediction for next time step 
        st['pred_zt'] = dot(st['Q'][:,:st['r']], st['pred_h'][:st['r']]).T
        
        '''Anomaly Test'''
        #if st['pred_err_norm'] > p['err_thresh']:
            #st['anomaly'] = True
            
    return st


def Q_stat(sample, phi, lag):
    """ Calculate the Q statistic over the sample of residuals
    with the specified AR coefficients and lag 
    
    sample - h_residual_window of past L residuals 
    phi - AR coefficients
    lag - Number of auto correlations to look at 
    """

    N = sample.size
    
    if len(phi) == 1:
        # AR(1) Process
        Q = 0.0
        for k in xrange(1,lag+1):
            Q += (phi**k)**2/(N-k)
        Q *= N*(N+2)
    elif len(phi) == 2:
        # AR(2) Process
        Q = 0.0
        for k in xrange(1,lag+1):
            y1 = (phi[0]/2.) + (sqrt((phi[0]**2) - 4*phi[1])/2.)
            y2 = (phi[0]/2.) + (sqrt((phi[0]**2) - 4*phi[1])/2.)
            roe = phi[0]*(y1**-k) + phi[1]*(y2**-k)
            Q += (roe)**2/(N-k)
        Q *= N*(N+2)
        
    return Q        
    
def anomaly_recon_stats(st, p, zt):
    """ Working on a test statistic for ressidul of zt_reconstructed """

    if not st.has_key('t_stat'):
        st['t_stat'] = 0
        st['rec_dsn'] = 0
        st['x_sample'] = 0
    
    st['recon'] = dot(st['Q'][:,:st['r']],st['ht'][:st['r']])
    
    st['recon_err'] = zt.T - st['recon']
    st['recon_err_norm'] = npl.norm(st['recon_err'])
    
    # Build/Slide recon_err_window
    if  st.has_key('recon_err_win'):
        st['recon_err_win'][:-1] = st['recon_err_win'][1:] # Shift Window
        st['recon_err_win'][-1] = st['recon_err_norm']**2
        #st['recon_err_win'][-1] = st['recon_err_norm']
    else:
        st['recon_err_win'] = np.zeros(((p['sample_N'] + p['dependency_lag']) *2, st['recon_err_norm'].size))
        st['recon_err_win'][-1] = st['recon_err_norm']**2
        #st['recon_err_win'][-1] = st['recon_err_norm']
    
    if st['t'] >=  (p['sample_N'] + p['dependency_lag']) : 
        # Differenced squared norms of the residules. 
        #st['rec_diff_sq_norm'] = st['recon_err_win'][::2] - st['recon_err_win'][1::2] 
        #st['rec_diff_sq_norm'] = np.diff(st['recon_err_win'], axis = 0)[::2]
        st['rec_diff_sq_norm'] = np.diff(st['recon_err_win'], axis = 0)
        st['rec_dsn'] = st['rec_diff_sq_norm'][-1]
        if True : #not st['t'] % p['sample_N']: # if no remainder calc T_stat
            
            st['x_sample'] = (st['rec_diff_sq_norm'][-(p['sample_N'] + p['dependency_lag']):-p['dependency_lag']]**2).sum()
            st['t_stat'] = st['rec_diff_sq_norm'][-1] / np.sqrt(st['x_sample']/ p['sample_N']) 
    
        if np.abs(st['t_stat']) > p['x_thresh']:
            st['anomaly'] = True
    
    return st


if __name__=='__main__':

    '''Experimental Run Parameters '''
    p = {'alpha': 0.98,
              'init_r' : 1, 
              # Pedro Anomal Detection
              'holdOffTime' : 0,
              # EWMA Anomaly detection
              'EWMA_filter_alpha' : 0.2,
              'residual_thresh' : 0.02,
              # AR Anomaly detection 
              'ht_AR_win' : 30,
              'AR_order' : 1,
              'err_thresh' : 1.5, 
              # Statistical 
              'sample_N' : 50,
              'dependency_lag' : 1,
              'x_thresh' : 10,
              'FP_rate' : 10**-6,
              # Q statistical 
              'Q_lag' : 5,
              'Q_alpha' : 0.05,
              # Eigen-Adaptive
              'F_min' : 0.9,
              'epsilon' : 0.05,
              # Pedro Adaptive
              'e_low' : 0.95,
              'e_high' : 0.98,
              'static_r' : 0,
              'r_upper_bound' : None,
              'fix_init_Q' : 0,
              'small_value' : 0.0001,
              'ignoreUp2' : 0 }

    p['x_thresh'] = sp.stats.t.isf(0.5* p['FP_rate'], p['sample_N'])

    ''' Load Data '''
    data = load_ts_data('isp_routers', 'full')
    #data, sins = sin_rand_combo(5, 1000, [10, 35, 60], noise_scale = 0.2)
    data = zscore(data)
    z_iter = iter(data)
    numStreams = data.shape[1]

    '''Initialise'''
    st = initialise(p, numStreams)
    
    '''Begin Frahst'''
    # Main iterative loop. 
    for zt in z_iter:

        zt = zt.reshape(zt.shape[0],1)   # Convert to a column Vector 
        st['anomaly'] = False
        '''Frahst Version '''
        st = FRAHST_V7_0_iter(zt, st, p)
        # Calculate reconstructed data if needed

        '''Anomaly Detection method''' 
        #st = anomaly_EWMA(st, p)
        #st = anomaly_AR_forcasting(st, p)
        st['recon'] = dot(st['Q'][:,:st['r']],st['ht'][:st['r']])
        st = anomaly_recon_stats(st, p, zt)
        #st = anomaly_AR_Qstat(st,p)
        
        '''Rank adaptation method''' 
        if p['static_r'] != 1:
            #st = rank_adjust_energy(st, zt)
            st = rank_adjust_eigen(st, zt)
  
        '''Store data''' 
        #tracked_values = ['ht','e_ratio','r','recon', 'pred_err', 'pred_err_norm', 'pred_err_ave']   
        tracked_values = ['ht','e_ratio','r','recon','recon_err', 'recon_err_norm', 't_stat', 'rec_dsn', 'x_sample']
        #tracked_values = ['ht','e_ratio','r','recon','Q_stat', 'coeffs', 'h_res', 'h_res_aa', 'h_res_norm']
        
        if 'res' not in locals(): 
            # initalise res
            res = {}
            for k in tracked_values:
                res[k] = st[k]
            res['anomalies'] = []

        else:
            # stack values for all keys
            for k in tracked_values:
                res[k] = np.vstack((res[k], st[k]))
            if st['anomaly'] == True:
                print 'Found Anomaly at t = {0}'.format(st['t'])
                res['anomalies'].append(st['t'])
        # increment time        
        st['t'] += 1
        
        
    res['Alg'] = 'My FRAHST'
    res['hidden'] = res['ht']
    res['r_hist'] = res['r']
    pltSummary2(res, data, (p['F_min'] + p['epsilon'], p['F_min']))
    
    plot_4x1(data, res['ht'], res['rec_dsn'], res['t_stat'], ['']*4, ['']*4)
    plt.hlines(p['x_thresh'], 0, 1000)
    plt.hlines(-p['x_thresh'], 0, 1000)
    plt.ylim(-p['x_thresh']-5, p['x_thresh']+5)
    
    N = 20 
    
    a = res['recon_err_norm']**2
    b = np.zeros_like(a)

    for i in range(1,len(a)):
        b[i] = a[i] - a[i-1]

    c = np.atleast_1d(b[::2])

    means = [0.0]*(len(c)/N)
    stds = [0.0]*(len(c)/N)

    for i in range((len(c)/N)):
        means[i] = c[N*i:N*(i+1)].mean()
        stds[i] = c[N*i:N*(i+1)].std()

    T_stat = [0]*(len(c)/N)
    for i in range((len(c)/N)):
        T_stat[i] = means[i] / (stds[i] / sqrt(N)) 
    
    
    # ok so b == diff(x)
    # c = diff(x)[::2]
    
    # What didi I do differently here?
# hmm, I think igot the x_sample wrong     
    
    D = 20 
    T_stat2 = [0]*(len(c)-(2*D))
    for i in range(len(c)-(2*D)):
        T_stat2[i] = c[i] / np.sqrt((c[i+D : i+(2*D)]**2).sum()/ D)
    
    T_stat3 = [0]*len(c)
    for i in range(len(c)-(2*D)):
        T_stat3[i+(2*D)] = c[i+(2*D)] / np.sqrt((c[i:i+D]**2).sum()/ D)
    # actuall goes 
    