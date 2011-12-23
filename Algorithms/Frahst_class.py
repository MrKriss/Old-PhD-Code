#!/usr/bin/env python
#coding:utf-8
# Author:  C Musselle --<>
# Purpose: Main Experiment Script 
# Created: 11/23/11

import numpy as np
from numpy import dot
import scipy as sp
from math import sqrt
import numpy.linalg as npl
import matplotlib.pyplot as plt

from plot_utils import plot_2x1, plot_3x1, plot_4x1
from utils import QRsolveA, QRsolve_eigV, fmeasure, analysis
from normalisationFunc import zscore, zscore_win
from burg_AR import burg_AR

from load_data import load_data, load_ts_data
from artSigs import sin_rand_combo, simple_sins, simple_sins_3z
from gen_anom_data import gen_a_grad_persist, gen_a_peak_dip, gen_a_step

"""
Code Description: The class that hlds all Frahst related Functions 
"""

class FRAHST():
  """ Class that holds all variants of FRAHST modules 

  version is a string specifying the combination of mudules to use 

  F = Frahst version 
  R = Rank adjustment Version
  A = Anomaly Version

  Format = 'F-xxxx.A-xxxx.R-xxxx'
  """

  def __init__(self, version, p, numStreams = 1):
    self.p = p
    self.p['version'] = version

    self.F_version = version.split('.')[0]
    self.A_version = version.split('.')[1]
    self.R_version = version.split('.')[2]
    self.numStreams = numStreams

    """ Initialise all Frahst variables """

    r = self.p['init_r']
    
    # Q_0
    if self.p['fix_init_Q'] != 0:  # fix inital Q as identity 
      q_0 = np.eye(numStreams);
      Q = q_0
    else: # generate random orthonormal matrix N x r 
      Q = np.eye(numStreams) # Max size of Q
      Q_0, R_0 = npl.qr(np.random.rand(numStreams,r))   
      Q[:,:r] = Q_0          
    # S_0
    small_value = self.p['small_value']
    S = np.eye(numStreams) * small_value # Avoids Singularity    
    # v-1
    v = np.zeros((numStreams,1)) 
    # U(t-1) for eigenvalue estimation
    U = np.eye(numStreams)

    # Define st dictionary 
    self.st  = {'Q' : Q,         # Orthogonal dominant subspace vectors
                'S' : S,     # Energy
                'v' : v,     # used for S update
                'U' : U,     # Used for eigen value calculations 
                'r' : r,     # Previous rank of Q and number of hidden variables h
                't' : 0,     # Timestep, used for ignoreup2  
                'sumEz' : 0.0,        # Exponetial sum of zt Energy 
                'sumEh': 0.0,     # Exponential sum of ht energy  
                'anomaly': bool(0)}

    if 'F-7' in version.split('.')[0]:
      # Extra variales used for alternative eigen value tracking
      self.st['last_Z_pos'] = True
      self.st['U2t'] = 0.0
      self.st['eig_val'] = np.zeros(1)
    elif 'F-3' in version.split('.')[0]:        
      self.st['lastChangeAt'] = 0.0
      
    if 'eng' in self.R_version:
      self.st['lastChangeAt'] = 0.0
      
    # Preliminary setup for forcasting methods.
    # May put all initialisation checks here eventually
    if 'for' in self.A_version: 
      self.st['pred_zt'] = np.zeros(numStreams)
      self.st['pred_err'] = np.zeros(numStreams)
      self.st['pred_err_norm'] = 0
      self.st['pred_err_ave'] = 0  
      self.st['pred_dsn'] = 0
    if 'rec' in self.A_version:
      self.st['recon_err'] = np.zeros(numStreams)
      self.st['rec_err_norm'] = 0
      self.st['rec_dsn'] = 0
    if 'S' in self.A_version:
      self.st['t_stat'] = 0
    if 'eng' in self.A_version:
      self.st['increased_r'] = bool(0)

  def re_init(self, numStreams):
  
    self.numStreams = numStreams
    
    # This deletes all tracked values 
    if hasattr(self, 'res'):
      del self.res
    
    """ Initialise all Frahst variables """
   
    r = self.p['init_r']
    # Q_0
    if self.p['fix_init_Q'] != 0:  # fix inital Q as identity 
      q_0 = np.eye(numStreams);
      Q = q_0
    else: # generate random orthonormal matrix N x r 
      Q = np.eye(numStreams) # Max size of Q
      Q_0, R_0 = npl.qr(np.random.rand(numStreams,r))   
      Q[:,:r] = Q_0          
    # S_0
    small_value = self.p['small_value']
    S = np.eye(numStreams) * small_value # Avoids Singularity    
    # v-1
    v = np.zeros((numStreams,1)) 
    # U(t-1) for eigenvalue estimation
    U = np.eye(numStreams)
  
    # Define st dictionary 
    self.st  = {'Q' : Q,          # Orthogonal dominant subspace vectors
                'S' : S,          # Energy
                'v' : v,          # used for S update
                'U' : U,          # Used for eigen value calculations 
                'r' : r,          # Previous rank of Q and number of hidden variables h
                't' : 0,          # Timestep, used for ignoreup2  
                'sumEz' : 0.0,    # Exponetial sum of zt Energy 
                'sumEh': 0.0,     # Exponential sum of ht energy  
                'anomaly': bool(0)}
  
    if 'F-7' in self.F_version:
      # Extra variales used for alternative eigen value tracking
      self.st['last_Z_pos'] = True
      self.st['U2t'] = 0.0
      self.st['eig_val'] = np.zeros(1)
    elif 'F-3' in self.F_version:        
      self.st['lastChangeAt'] = 0.0
      
    if 'eng' in self.R_version:
      self.st['lastChangeAt'] = 0.0
      
    # Preliminary setup for forcasting methods.
    if 'for' in self.A_version: 
      self.st['pred_zt'] = np.zeros(numStreams)
      self.st['pred_err'] = np.zeros(numStreams)
      self.st['pred_err_norm'] = 0
      self.st['pred_err_ave'] = 0  
      self.st['pred_dsn'] = 0
    if 'rec' in self.A_version:
      self.st['recon_err'] = np.zeros(numStreams)
      self.st['rec_err_norm'] = 0
      self.st['rec_dsn'] = 0
    if 'S' in self.A_version:
      self.st['t_stat'] = 0
    if 'eng' in self.A_version:
      self.st['increased_r'] = bool(0)  


  def run(self, zt):
    if 'F-7' in self.F_version:
      self.FRAHST_V7_0_iter(zt)
    elif 'F-3' in self.F_version:
      self.FRAHST_V3_1_iter(zt)
    else:
      print 'Did not run: %s not recognised' % (self.F_version)

  def rank_adjust(self, zt):
    # Check whether r is static 
    if 'static' not in self.R_version:
      if 'eig' in self.R_version:
        self.rank_adjust_eigen(zt)
      elif 'eng' in self.R_version:
        self.rank_adjust_energy(zt)
      else:
        print 'Did not run rank adjust: %s not recognised' % (self.R_version)

  def detect_anom(self, zt):
    if 'forS' in self.A_version:
      self.anomaly_AR_forcast_stat(zt)
    elif 'forT' in self.A_version:
      self.anomaly_AR_forcast_thresh(zt)
    elif 'recS' in self.A_version:
      self.anomaly_recon_stat(zt)
    elif 'ewma' in self.A_version:
      self.anomaly_EWMA()
    elif 'eng' in self.A_version:
      self.anomaly_eng()
    else:
      print 'Did not run detect anomalies:  %s not recognised' % (self.A_version)

  def FRAHST_V3_1_iter(self, zt):
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
          'r_upper_bound' : 0,
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

    st = self.st
    p = self.p

    # Derived Variables 
    # Length of z or numStreams = N x L
    numStreams = self.numStreams
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
    if p['r_upper_bound'] == 0 :
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
    ht_vec = np.hstack((ht.T[0,:], np.array([np.nan]*(self.numStreams-r))))
    st['ht'] = ht_vec

    # Energy Ratio Calculations    
    st['sumEz'] = p['alpha']*st['sumEz'] + np.sum(zt ** 2) # Energy of Data
    st['sumEh'] = p['alpha']*st['sumEh'] + np.sum(ht ** 2) # Energy of Hidden Variables

    if st['sumEz'] == 0 : # Catch NaNs 
      st['e_ratio']  = 0.0
    else:
      st['e_ratio']  = st['sumEh'] / st['sumEz']

    self.st = st 

  
  def FRAHST_V7_0_iter(self, zt):
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
          'r_upper_bound' : 0,
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

    st = self.st 
    p = self.p

    # Derived Variables 
    # Length of z or numStreams = N x L
    numStreams = self.numStreams
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
    if p['r_upper_bound'] == 0 :
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
    ht_vec = np.hstack((ht.T[0,:], np.array([np.nan]*(self.numStreams-r))))
    st['ht'] = ht_vec

    # Energy Ratio Calculations    
    st['sumEz'] = p['alpha']*st['sumEz'] + np.sum(zt ** 2) # Energy of Data
    st['sumEh'] = p['alpha']*st['sumEh'] + np.sum(ht ** 2) # Energy of Hidden Variables

    if st['sumEz'] == 0 : # Catch NaNs 
      st['e_ratio']  = 0.0
    else:
      st['e_ratio']  = st['sumEh'] / st['sumEz']

    self.st = st

  def rank_adjust_energy(self, zt):
    """ Adjust rank r of subspace accoring to Pedros Energy-adaptive method """

    st = self.st
    p = self.p

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

      # Flag as increased r 
      st['increased_r'] = bool(1)

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

    self.st = st


  def rank_adjust_eigen(self, zt):
    """ Adjust rank r of subspace accoring to my eigvalue-adaptive method """

    st = self.st
    p = self.p

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

      # Flag as increased r
      if 'eng' in self.A_version:
        st['increased_r'] = bool(1)

      # new r, increment and store
      st['r'] = r + 1
      st['Q'] = Q 
      st['S'] = S 
      st['U'] = U 

    elif st['e_ratio'] > p['F_min'] and r > 1 and t > p['ignoreUp2']:

      keeper = np.ones(r, dtype = bool)
      # Sorted in accending order
      #Â Causing problems, skip sorting, (quicker/simpler), and just cylce from with last 
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

    self.st = st


  def anomaly_eng(self):
    
    if self.st['increased_r'] == True:
      self.st['anomaly'] = True
      self.st['increased_r'] = False
  

  def anomaly_EWMA(self):
    """ Track e_ratio for sharp dips and spikes using EWMA """

    p = self.p
    st = self.st

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

    self.st = st


  def anomaly_AR_forcast_stat(self, zt):
    """ Use Auto Regressive prediction and T statistic to calculate anomalies 

        Step 1: Calculate prediction of next data point z_t+1 by fitting 
        an AR model to each hidden Variable, predicting each hi_t+1 and then 
        projecting h_t+1 using the basis Q at current time t.

        Step 2: Track a sample of the residual norm squared
        res_t,l = [norm(z_t-l - predicted_z_t-l)**2 , ... , norm(z_t - predicted_z_t)**2]

        Step 3: Difference the sample to get zero mean.

        Step 3: Calculate T statistic of the differenced norm sqared of the residual (res_dns)

        T = red_dns[i] / sqrt( sample / sample_N)
        sample =  red_dns[i-(lag + sample_N)]**2 + ... + red_dns[i-(lag)**2]

        Step 4: Compare with calclated threshold based on N_sample and  

        h_window is ht_AR_win x numStreams 
        """    

    st = self.st
    p = self.p

    # Build/Slide h_window
    if  st.has_key('h_window'):
      st['h_window'][:-1,:] = st['h_window'][1:,:] # Shift Window
      st['h_window'][-1,:] = np.nan_to_num(st['ht'])
    else:
      st['h_window'] = np.zeros((p['ht_AR_win'], st['ht'].size))
      st['h_window'][-1,:] = np.nan_to_num(st['ht'])

    ''' Forcasting '''
    #if st['t'] > p['ht_AR_win']: # Sod it, do every time step 
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
    st['pred_err_ave'] = np.abs(st['pred_zt'] - zt.T).sum() / self.numStreams
    st['pred_err_norm'] = npl.norm(st['pred_zt'] - zt.T)

    # Update prediction for next time step 
    st['pred_zt'] = dot(st['Q'][:,:st['r']], pred_h).T

    '''Anomaly Test'''
    # Build/Slide pred_err_window
    if  st.has_key('pred_err_win'):
      st['pred_err_win'][:-1] = st['pred_err_win'][1:] # Shift Window
      st['pred_err_win'][-1] = st['pred_err_norm']**2
      #st['pred_err_win'][-1] = st['pred_err_norm']
    else:
      st['pred_err_win'] = np.zeros(p['sample_N'] + p['dependency_lag'])
      st['pred_err_win'][-1] = st['pred_err_norm']**2
      #st['pred_err_win'][-1] = st['pred_err_norm']

    #if st['t'] >=  (p['sample_N'] + p['dependency_lag']) : 
    # Differenced squared norms of the residules. 
    # If i wish to sample every other point instead, I will need to adjust size of pred_err_win above         
    #st['pred_dsn_sample'] = st['pred_err_win'][::2] - st['pred_err_win'][1::2] 
    #st['pred_dsn_sample'] = np.diff(st['pred_err_win'], axis = 0)[::2]

    st['pred_dsn_sample'] = np.diff(st['pred_err_win'], axis = 0)
    st['pred_dsn'] = st['pred_dsn_sample'][-1]

    x_sample = (st['pred_dsn_sample'][-(p['sample_N'] + p['dependency_lag']):-p['dependency_lag']]**2).sum()
    st['t_stat'] = st['pred_dsn'] / np.sqrt( x_sample / p['sample_N'])

    if st['t'] > p['ignoreUp2'] and np.abs(st['t_stat']) > p['t_thresh']:
      st['anomaly'] = True

    self.st = st

  def anomaly_AR_forcast_thresh(self, zt):
    """ Use Auto Regressive prediction and T statistic to calculate anomalies 

        Step 1: Calculate prediction of next data point z_t+1 by fitting 
        an AR model to each hidden Variable, predicting each hi_t+1 and then 
        projecting h_t+1 using the basis Q at current time t.

        Step 2: Compare with Arbritrary threshold 

        h_window is ht_AR_win x numStreams 
        """    

    st = self.st
    p = self.p

    # Build/Slide h_window
    if  st.has_key('h_window'):
      st['h_window'][:-1,:] = st['h_window'][1:,:] # Shift Window
      st['h_window'][-1,:] = np.nan_to_num(st['ht'])
    else:
      st['h_window'] = np.zeros((p['ht_AR_win'], st['ht'].size))
      st['h_window'][-1,:] = np.nan_to_num(st['ht'])

    ''' Forcasting '''
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
    st['pred_err_ave'] = np.abs(st['pred_zt'] - zt.T).sum() / self.numStreams
    st['pred_err_norm'] = npl.norm(st['pred_zt'] - zt.T)

    # Update prediction for next time step 
    st['pred_zt'] = dot(st['Q'][:,:st['r']], pred_h).T

    '''Anomaly Test'''
    if st['t'] > p['ignoreUp2'] and st['pred_err_norm']**2 > p['x_thresh']:
      st['anomaly'] = True

    self.st = st

  def anomaly_recon_stat(self, zt):
    """ Working on a test statistic for ressidul of zt_reconstructed """

    st = self.st
    p = self.p  

    st['recon'] = dot(st['Q'][:,:st['r']],st['ht'][:st['r']])

    st['recon_err'] = zt.T - st['recon']
    st['recon_err_norm'] = npl.norm(st['recon_err'])

    # Build/Slide recon_err_window
    if  st.has_key('recon_err_win'):
      st['recon_err_win'][:-1] = st['recon_err_win'][1:] # Shift Window
      st['recon_err_win'][-1] = st['recon_err_norm']**2
      #st['recon_err_win'][-1] = st['recon_err_norm']
    else:
      st['recon_err_win'] = np.zeros(((p['sample_N'] + p['dependency_lag']), st['recon_err_norm'].size))
      st['recon_err_win'][-1] = st['recon_err_norm']**2
      #st['recon_err_win'][-1] = st['recon_err_norm']

    # Differenced squared norms of the residules. 
    #st['rec_dsn_sample'] = st['recon_err_win'][::2] - st['recon_err_win'][1::2] 
    #st['rec_dsn_sample'] = np.diff(st['recon_err_win'], axis = 0)[::2]
    st['rec_dsn_sample'] = np.diff(st['recon_err_win'], axis = 0)
    st['rec_dsn'] = st['rec_dsn_sample'][-1]

    x_sample = (st['rec_dsn_sample'][-(p['sample_N'] + p['dependency_lag']):-p['dependency_lag']]**2).sum()
    st['t_stat'] = st['rec_dsn'] / np.sqrt( x_sample / p['sample_N']) 

    if st['t'] > p['ignoreUp2'] and np.abs(st['t_stat']) > p['t_thresh']:
      st['anomaly'] = True

    self.st = st 

  def track_var(self, values = ()):
    
    if not hasattr(self, 'res'):
      # initalise res
      self.res = {}
      for k in values:
        self.res[k] = self.st[k]
      self.res['anomalies'] = []
    
    else:
      # stack values for all keys
      for k in values:
        self.res[k] = np.vstack((self.res[k], self.st[k]))
      if self.st['anomaly'] == True:
        print 'Found Anomaly at t = {0}'.format(self.st['t'])
        self.res['anomalies'].append(self.st['t'])
        
    # Increment time        
    self.st['t'] += 1    
    
 
  def plot_res(self, var, xname = 'time steps', ynames = None, title = None, hline= 1, anom = 1):
    
    if ynames is None:
      ynames = ['']*4
      
    if title is None:
      title = (self.p['version'])
      
    if 'S' in self.A_version:
      thresh = self.p['t_thresh']
    elif 'T' in self.A_version:
      thresh = self.p['x_thresh']
    
    num_plots = len(var)
    
    for i, v in enumerate(var):
      if type(v) == str :
        var[i] = getattr(self, 'res')[v]
    
    if num_plots == 1:
      plt.figure()
      plt.plot(var[0])
      plt.title(title)
      if anom == 1:
        for x in self.res['anomalies']:
          plt.axvline(x, ymin=0.25, color='r')        
      
    elif num_plots == 2:
      plot_2x1(var[0], var[1], ynames[:2], xname, main_title = title)
      
      if hline == 1:
        plt.hlines(-thresh, 0, self.res['ht'].shape[0], linestyles = 'dashed')
        plt.hlines(+thresh, 0, self.res['ht'].shape[0], linestyles = 'dashed')
        plt.ylim(-2*thresh,2*thresh)
        
      if anom == 1:
        f = plt.gcf()
        for ax in f.axes[:-1]:
          for x in self.res['anomalies']:
            ax.axvline(x, ymin=0.25, color='r')              
        
    elif num_plots == 3:
      plot_3x1(var[0], var[1], var[2], ynames[:3] , xname, main_title = title) 

      
      if hline == 1:
        plt.hlines(-thresh, 0, self.res['ht'].shape[0], linestyles = 'dashed')
        plt.hlines(+thresh, 0, self.res['ht'].shape[0], linestyles = 'dashed') 
        plt.ylim(-2*thresh,2*thresh)
        
      if anom == 1:
        f = plt.gcf()
        for ax in f.axes[:-1]:
          for x in self.res['anomalies']:
            ax.axvline(x, ymin=0.25, color='r')         
               
    elif num_plots == 4:
      plot_4x1(var[0], var[1], var[2], var[3], ynames[:4], xname, main_title = title)
      plt.title(title)
      
      if hline == 1:
        plt.hlines(-thresh, 0, self.res['ht'].shape[0], linestyles = 'dashed')
        plt.hlines(+thresh, 0, self.res['ht'].shape[0], linestyles = 'dashed')               
        plt.ylim(-2*thresh,2*thresh)
        
      if anom == 1:
        f = plt.gcf()
        for ax in f.axes[:-1]:
          for x in self.res['anomalies']:
            ax.axvline(x, ymin=0.25, color='r')           

  
  def analysis(self, gt_table, epsilon = 0, accumulative = 1, keep_sets = 1):
      ''' Calculate all anomally detection Metrics 
      
      # epsilon: used to allow for lagged detections: if Anomaly occurs in time window
      anom_start - anom_end + eplsilon it is considered a TP
      
      # Acts accumulateive 
      
      '''
      # Detections  
      D = np.array(self.res['anomalies'])
      index =  D > self.p['ignoreUp2'] 
      D = set(list(D[index]))        
      
      # Total Neg 
      pred_total_N = self.st['t'] - self.p['ignoreUp2'] - len(D)    
      
      # initalise metrics 
      if 'metric' not in locals() or accumulative == 0:
        self.metric = { 'TP' : 0.0 ,
                   'FP' : 0.0 ,
                   'FN' : 0.0 ,
                   'TN' : 0.0,
                   'precision' : 0.0 ,
                   'recall' : 0.0 ,
                   'F1' : 0.0, 
                   'F2' : 0.0, 
                   'F05' : 0.0,
                   'FPR' : 0.0,
                   'FDR' : 0.0,
                   'ACC' : 0.0}
        self.detection_sets = []
        self.anom_detect_tab = []

      # set of point anomalies detected as true
      anom_TP = set()
      
      # Set of anomalous segments detected           
      anom_segments_detected_set  = set()  
      
      # Table to record frequency of anomalous segment detections
      anomalies_detected_tab  = np.zeros((len (gt_table), 2))
      anomalies_detected_tab[:,0] = gt_table['start']
      
      # TRUE POSITIVES
      # Run through ground truths 
      idx = 0
      for gt in gt_table:
          count = 0
          # Run through the list of detections    
          for d in D :
              if d >= gt['start']  and d <= gt['start'] + gt['len'] + epsilon:
                  # if set does not yet contain the anomaly, add it and increment TP
                  if not anom_segments_detected_set.issuperset(set([gt['start']])):
                      
                      anom_segments_detected_set.add(gt['start'])
                      anom_TP.add(d)
                      self.metric['TP'] += 1
                      count += 1
                  else: # if multiple detections in anomalous segment 
                      count += 1 
                      anom_TP.add(d)                    
                      
          anomalies_detected_tab[idx,1] = count   
          idx += 1     
      
      # FALSE Pos 
      anom_FP = D - anom_TP    
      self.metric['FP'] += len(anom_FP)
      # FALSE Neg     
      anom_FN = set(gt_table['start']) - anom_segments_detected_set
      self.metric['FN'] += len(anom_FN)
      # True Negatives
      self.metric['TN'] += pred_total_N - len(anom_FN)

      if self.metric['FP'] == 0 and self.metric['TP'] == 0:
        self.metric['precision'] += 0
        self.metric['FDR'] += 0
      else:
        self.metric['precision'] = self.metric['TP'] / (self.metric['TP'] + self.metric['FP'])          
        self.metric['FDR'] = self.metric['FP'] / (self.metric['FP'] + self.metric['TP'])    

      self.metric['recall'] = self.metric['TP'] / (self.metric['TP'] + self.metric['FN'])      
      self.metric['FPR'] = self.metric['FP'] / (self.metric['TN'] + self.metric['FP'])      
      self.metric['ACC'] = (self.metric['TP'] + self.metric['TN']) /  \
                      ( self.metric['TP'] + self.metric['FN'] + self.metric['TN'] + self.metric['FP'] )
                      
      self.metric['F1'] = self.fmeasure(1, self.metric['TP'], self.metric['FN'], self.metric['FP'])
      self.metric['F2'] = self.fmeasure(2, self.metric['TP'], self.metric['FN'], self.metric['FP'])
      self.metric['F05'] = self.fmeasure(0.5, self.metric['TP'], self.metric['FN'], self.metric['FP']) 
      
      if keep_sets == 1:
        sets = {'TP' : anom_TP,
                'anom_seg_detected' : anom_segments_detected_set,
                'FN' : anom_FN,
                'FP' : anom_FP}     
        self.detection_sets.append(sets)
        self.anom_detect_tab.append(anomalies_detected_tab)

  def batch_analysis(self, data_list, anomalies_list, epsilon = 0, accumulative = 1, keep_sets = 1):
      ''' Calculate all anomally detection Metrics 
      
      # epsilon: used to allow for lagged detections: if Anomaly occurs in time window
      anom_start - anom_end + eplsilon it is considered a TP
      
      # Acts accumulateive 
      
      data 
      
      Need to go through And check this in detail!!!!
      Something not right.
      
      tHINK I KNOW WHAT IT IS !!
      
      '''
      
      # For each initial condition 
      for k in xrange(len(anomalies_list)):
        
        gt_table = data_list[k]['gt']
        anomalies = anomalies_list[k] 
      
        # Detections  
        D = np.array(anomalies)
        index =  D > self.p['ignoreUp2'] 
        D = set(list(D[index]))        
        
        # initalise metrics 
        if not hasattr(self, 'metric') or accumulative == 0:
          self.metric = { 'TP' : 0.0 ,
                     'FP' : 0.0 ,
                     'FN' : 0.0 ,
                     'TN' : 0.0,
                     'precision' : 0.0 ,
                     'recall' : 0.0 ,
                     'F1' : 0.0, 
                     'F2' : 0.0, 
                     'F05' : 0.0,
                     'FPR' : 0.0,
                     'FDR' : 0.0,
                     'ACC' : 0.0}
          self.detection_sets = []
          self.anom_detect_tab = []
  
        # set of point anomalies detected as true
        anom_TP = set()
        
        # Set of anomalous segments detected           
        anom_segments_detected_set  = set()  
        
        # Table to record frequency of anomalous segment detections
        anomalies_detected_tab  = np.zeros((len(gt_table['start']), 2))
        anomalies_detected_tab[:,0] = gt_table['start']
        
        # TRUE POSITIVES
        
        idx = 0
        for i in xrange(len(gt_table['start'])):
            count = 0
            # Run through the list of detections    
            for d in D:
              if d >= gt_table['start'][i]  and d <= gt_table['start'][i] + gt_table['len'][i] + epsilon:
                # if set does not yet contain the anomaly, add it and increment TP
                if not anom_segments_detected_set.issuperset(set([gt_table['start'][i]])):
                  
                  anom_segments_detected_set.add(gt_table['start'][i])
                  anom_TP.add(d)
                  self.metric['TP'] += 1
                  count += 1
                else: # if multiple detections in anomalous segment 
                  count += 1 
                  anom_TP.add(d)                    
                      
            anomalies_detected_tab[idx,1] = count   
            idx += 1     
        
        # FALSE Pos 
        anom_FP = D - anom_TP    
        self.metric['FP'] += len(anom_FP)
        # FALSE Neg     
        anom_FN = set(gt_table['start']) - anom_segments_detected_set
        self.metric['FN'] += len(anom_FN)
        # True Negatives
        self.metric['TN'] += (self.st['t'] - self.p['ignoreUp2'] - len(anom_FN) - len(anom_FP) - len(anom_TP))
  
        if self.metric['FP'] == 0 and self.metric['TP'] == 0:
          self.metric['precision'] += 0
          self.metric['FDR'] += 0
        else:
          self.metric['precision'] = self.metric['TP'] / (self.metric['TP'] + self.metric['FP'])          
          self.metric['FDR'] = self.metric['FP'] / (self.metric['FP'] + self.metric['TP'])    
  
        self.metric['recall'] = self.metric['TP'] / (self.metric['TP'] + self.metric['FN'])      
        self.metric['FPR'] = self.metric['FP'] / (self.metric['TN'] + self.metric['FP'])      
        self.metric['ACC'] = (self.metric['TP'] + self.metric['TN']) /  \
                        ( self.metric['TP'] + self.metric['FN'] + self.metric['TN'] + self.metric['FP'] )
                        
        self.metric['F1'] = self.fmeasure(1, self.metric['TP'], self.metric['FN'], self.metric['FP'])
        self.metric['F2'] = self.fmeasure(2, self.metric['TP'], self.metric['FN'], self.metric['FP'])
        self.metric['F05'] = self.fmeasure(0.5, self.metric['TP'], self.metric['FN'], self.metric['FP']) 
        
        if keep_sets == 1:
          sets = {'TP' : anom_TP,
                  'anom_seg_detected' : anom_segments_detected_set,
                  'FN' : anom_FN,
                  'FP' : anom_FP}     
          self.detection_sets.append(sets)
          self.anom_detect_tab.append(anomalies_detected_tab)
    
  def fmeasure(self, B, hits, misses, falses) :
      """ General formular for F measure 
      
      Uses TP(hits), FN(misses) and FP(falses)
      """
      x = ((1 + B**2) * hits) / ((1 + B**2) * hits + B**2 * misses + falses)
      return x


if __name__=='__main__':

  ''' Experimental Run Parameters '''
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
        'x_thresh' : 1.5, 
        # Statistical 
        'sample_N' : 20,
        'dependency_lag' : 1,
        't_thresh' : None,
        'FP_rate' : 10**-4,
        # Q statistical 
        'Q_lag' : 5,
        'Q_alpha' : 0.05,
        # Eigen-Adaptive
        'F_min' : 0.9,
        'epsilon' : 0.05,
        # Pedro Adaptive
        'e_low' : 0.95,
        'e_high' : 0.98,
        'r_upper_bound' : 0,
        'fix_init_Q' : 0,
        'small_value' : 0.0001,
        'ignoreUp2' : 50 }
      
  
  p['t_thresh'] = sp.stats.t.isf(0.5 * p['FP_rate'], p['sample_N'])

  ''' Anomalous Data Parameters '''
  
  a = { 'N' : 50, 
        'T' : 1000, 
        'periods' : [15, 40, 70, 90,120], 
        'L' : 10, 
        'L2' : 200, 
        'M' : 3, 
        'pA' : 0.1, 
        'noise_sig' : 0.3 }
  
  D = gen_a_grad_persist(**a)
  
  #data = load_ts_data('isp_routers', 'full')
  data = D['data']
  data = zscore_win(data, 100)
  z_iter = iter(data)
  numStreams = data.shape[1]
  
  '''Initialise'''
  Frahst_alg = FRAHST('F-7.A-recS.R-eig', p, numStreams)
  
  '''Begin Frahst'''
  # Main iterative loop. 
  for zt in z_iter:
  
    zt = zt.reshape(zt.shape[0],1)   # Convert to a column Vector 
  
    if Frahst_alg.st['anomaly'] == True:
      Frahst_alg.st['anomaly'] = False # reset anomaly var
  
    '''Frahst Version '''
    Frahst_alg.run(zt)
    # Calculate reconstructed data if needed
    st = Frahst_alg.st
    Frahst_alg.st['recon'] = np.dot(st['Q'][:,:st['r']],st['ht'][:st['r']])
  
    '''Anomaly Detection method''' 
    Frahst_alg.detect_anom(zt)
  
    '''Rank adaptation method''' 
    Frahst_alg.rank_adjust(zt)
  
    '''Store data''' 
    #tracked_values = ['ht','e_ratio','r','recon', 'pred_err', 'pred_err_norm', 'pred_err_ave', 't_stat', 'pred_dsn', 'pred_zt']   
    tracked_values = ['ht','e_ratio','r','recon','recon_err', 'recon_err_norm', 't_stat', 'rec_dsn']
    #tracked_values = ['ht','e_ratio','r','recon']
  
    Frahst_alg.track_var(tracked_values)
  
  ''' Plot Results '''
  Frahst_alg.plot_res([data, 'ht', 't_stat'])
  #Frahst_alg.plot_res([data, 'ht', 'r', 'e_ratio'], hline = 0)
  #Frahst_alg.plot_res([data, 'ht', 'rec_dsn', 't_stat'])

  Frahst_alg.analysis(D['gt'])
