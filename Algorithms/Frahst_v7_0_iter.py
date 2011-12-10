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
from Frahst_v6_3 import FRAHST_V6_3
from load_syn_ping_data import load_n_store
from QR_eig_solve import QRsolve_eigV
from create_Abilene_links_data import create_abilene_links_data
from MAfunctions import MA_over_window
from ControlCharts import Tseries
from EWMA_filter import EWMA_filter
from normalisationFunc import zscore, zscore_win 
from load_data import load_data, load_ts_data


def FRAHST_V7_0_iter(zt, inputs, alpha=0.96, holdOffTime=0, evalMetrics = 'F',
                EW_mean_alpha = 0.1, EWMA_filter_alpha = 0.3, residual_thresh = 0.1, 
                F_min = 0.9, epsilon = 0.05, small_value = 0.0001 
                static_r = 0, r_upper_bound = None,
                ignoreUp2 = 0):
    
    ''' 
    zt = data at next time step 
    inputs = {'Q' : Q,     - Orthogonal dominant subspace vectors
              'S' : S,     - Energy 
              'U' : U,     - Orthonormal component of Orthogonal iteration around X.T
              'v' : v,     - Used for speed up of calculating X update 
              'r' : r,     - Previous rank of Q and number of hidden variables h
              't' : t,     - Timestep, used for ignoreup2  
              'sumEz' : Et,  - Exponetial sum of zt Energy 
              'sumEh': E_dash_t }- Exponential sum of ht energy   
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
                
    Version 3.3 - Add decay of S and in the event of vanishing inputs 
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
    
    if r_upper_bound == None :
        r_upper_bound = numStreams
    
    #for energy test
    #last_Z_pos = bool() # bool flag
    #lastChangeAt = 1
    #sumYSq = 0.
    #sumXSq = 0.
    
    # NOTE algorithm's inputs Q, S, v and U are kept at max size (constant memory)
    # alias to inputs for current value of r
    Qt  = inputs['Q'][:, :r]
    vt  = inputs['v'][:r, :]
    St  = inputs['S'][:r, :r]
    Ut  = inputs['U'][:r, :r]

    # Main Loop # (but only go through once)
    #############
    
    '''Data Preprocessing'''        
    # Convert to a column Vector 
    # Already taken care of in this version
    # zt = zt.reshape(zt.shape[0],1) 
    
    # Check S remains non-singular
    for idx in range(r):
        if St[idx, idx] < small_value:
            St[idx,idx] = small_value
    
    '''Begin main algorithm'''        
    ht = dot(Qt.T, zt) 
    Z = dot(zt.T, zt) - dot(ht.T , ht)

    if Z > 0 :
        
        # Flag for whether Z(t-1) > 0
        # Used for alternative eigenvalue calculation if Z < 0
        last_Z_positive = 1
            
        # Refined version, use of extra terms
        u_vec = dot(St , vt)
        X = (alpha * St) + (2 * alpha * dot(u_vec, vt.T)) + dot(ht, ht.T)
    
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
            
        Qt_min1 = Qt
        Qt = Qt - 2 * dot(ee , vt.T) 
            
    else: # if Z is not > 0
        # Implies norm of ht is > zt or zt --> 0
        
        St = alpha * St # Continue decay of S matrix 
        
        # Recalculate Eigenvalues using other method 
        # (less fast, but does not need Z to be positive)
        if last_Z_pos == 1:
            # New U                 
            U2t_min1 = np.eye(r)
            #PHI = np.dot(Qt_min1.T, Qt) - Unnecessary as S* Phi ~ S if phi ~ I
            # Also the above causes complications when rank r changes
            Wt = np.dot(St, U2t_min1)
            U2t, R2 = qr(Wt) # Decomposition
            PHI_U = np.dot(U2t_min1.T,U2t)
            e_values = np.diag(np.dot(R2,PHI_U))
        elif last_Z_pos == 0:
            U2t_min1 = U2t                
            #PHI = np.dot(Qt_min1.T, Qt) - Unnecessary as S* Phi ~ S if phi ~ I
            # Also the above causes complications when rank r changes 
            Wt = np.dot(St, U2t_min1)
            #Wt = np.dot(np.dot(St, PHI), U2t_min1)
            U2t, R2 = qr(Wt) # Decomposition
            PHI_U = np.dot(U2t_min1.T,U2t)
            e_values = np.diag(np.dot(R2,PHI_U))

    # update input data 
    inputs['Q'][:,:r] = Qt
    inputs['v'][:r,:] = vt
    inputs['S'][:r, :r] = St
    inputs['U'][:r,:r] = Ut
        
    '''Store Values''' 
    outputs = {}
        
    # Store eigen values
    outputs['eig_val'] = e_values[:r]
        
    # Record hidden variables
    outputs['hidden'] = ht.T[0,:]
        
    return inputs, outputs
       
       
def rank_adjust_e_ratio(inputs):

    # Calculate energies 
    inputs['sumEz'] = alpha * inputs['sumEz'] + np.sum(zt ** 2) # Energy of Data
    inputs['sumEh'] = alpha * inputs['sumEh'] + np.sum(ht ** 2) # Energy of hidden Variables
                
    if sumXSq == 0 : # Catch NaNs 
        e_ratio = 0.0
    else:
        e_ratio = sumYSq / sumXSq
        
    outputs['e_ratio'] = e_ratio
    
    
    
    
def EWMA_e_ratio(EWMA_filter_alpha, e_ratio, pred_data):    
        
    # Run EWMA on e_ratio 
    if inputs['t'] == 1:  
        pred_data = 0.0 # initialise value

    # Calculate residual usung last time steps prediction 
    residual = np.abs(outputs['e_ratio'] - pred_data)
    outputs['EWMA_res'] = residual
    # Update prediction for next time step
    outputs['pred_data'] = EWMA_filter_alpha * e_ratio + (1-EWMA_filter_alpha) * pred_data    
                
    # Threshold residual for anomaly
    if residual > residual_thresh and t> ignoreUp2:
        # Record time step of anomaly            
        res['anomalies'].append(t-1) 
        
    return pred_data
    
            ''' Rank Adjust - needs ht, Q, zt, '''
        if static_r == 0: # optional parameter to keep r unchanged
            # Adjust Q_t, St and Ut for change in r
            if sumYSq < (F_min * sumXSq) and lastChangeAt < (t - holdOffTime) and r < r_upper_bound and t > ignoreUp2:
                
                """Note indexing with r works like r + 1 as index is from 0 in python"""

                # Extend Q by z_bar
                h_dash = dot(Q[:, :r].T,  zt)
                z_bar = zt - dot(Q[:, :r] , h_dash)
                z_bar_norm = norm(z_bar)
                z_bar = z_bar / z_bar_norm
                Q[:numStreams, r] = z_bar.T[0,:]
                
                s_end  = z_bar_norm ** 2
                
                # Set next row and column to zero
                S[r, :] = 0.0
                S[:, r] = 0.0
                S[r, r] = s_end # change last element
                
                # Update Ut_1
                # Set next row and column to zero
                U[r, :] = 0.0
                U[:, r] = 0.0
                U[r, r] = 1.0 # change last element
                
                # Update eigenvalue 
                e_values = sp.r_[e_values, z_bar_norm ** 2] 
                # This is the bit where the estimate is off? dont really have anything better 
                
                # new r, increment
                r = r + 1
                
                # Reset lastChange             
                lastChangeAt = t
                
            elif sumYSq > (F_min * sumXSq) and lastChangeAt < t - holdOffTime and r > 1 and t > ignoreUp2:
                
                keeper = ones(r, dtype = bool)
                # Sorted in accending order
                #Â Causing problems, skip sorting, (quicker/simpler), and just cylce from with last 
                # added eignevalue through to newest.  
                #sorted_eigs = e_values[e_values.argsort()]
                
                acounted_var = sumYSq
                for idx in range(r)[::-1]:
                    
                    if ((acounted_var - e_values[idx]) / sumXSq) > F_min + epsilon:
                        keeper[idx] = 0
                        acounted_var = acounted_var - e_values[idx]
                
                # use keeper as a logical selector for S and Q and U 
                if not keeper.all():
                    
                    # Delete rows/cols in Q, S, and U. 
                    newQ = Q[:,:r].copy()
                    newQ = newQ[:,keeper] # cols eliminated                        
                    Q[:newQ.shape[0], :newQ.shape[1]] = newQ
                    
                    newS = S[:r,:r].copy()
                    newS = newS[keeper,:][:,keeper] # rows/cols eliminated                        
                    S[:newS.shape[0], :newS.shape[1]] = newS
                    
                    newU = U[:r,:r].copy()
                    newU = newU[keeper,:][:,keeper] # rows/cols eliminated                        
                    U[:newU.shape[0], :newU.shape[1]] = newU
                    
                    r = keeper.sum()
                    if r == 0 :
                        r = 1
            
                    # Reset lastChange
                    lastChangeAt = t
            
    return res 

def lag_inputs(data, L):
    """Generator function to construct an input vector ztl that is the lagged zt 
    up to time l.
    
    z_tl = [zt, zt-t, zt-2,..., zt-l]
    
    Takes input data as a matrix. 
    """
    N = data.shape[1]
    total_timesteps = data.shape[0]
    
    z_tl = np.zeros((L*N,1))
    
    for i in range(total_timesteps):
        #shift values 
        z_tl[N:] = z_tl[:-N]
        # add new one to start of vector 
        z_tl[:N] = np.atleast_2d(data[i,:]).T
        
        yield z_tl

def alpha_var(x, alpha, var):
    """ Simple exponential forgetting of Variance """
    var = alpha * var + ( np.power(x,2))
    
    return var

def EW_mean_var(x, alpha, var, mean):
    """ Work out the exponentially weighted mean and variance of the data """
    if alpha > 1 :
        alpha = 2.0 / (alpha + 1)
    
    diff = x - mean 
    incr = alpha * diff
    mean = mean + incr
    var = (1 - alpha) * (var + diff * incr)

    return var, mean 
    
def simple_sins(p1,p11, p2,p22, noise_scale, N = 500):
    
    t = arange(N)
                
    z1 = np.sin(2 * np.pi * t / p1) + npr.randn(t.shape[0]) * noise_scale
    z2 = np.sin(2 * np.pi * t / p2) + npr.randn(t.shape[0]) * noise_scale
        
    z11 = np.sin(2 * np.pi * t / p11) + npr.randn(t.shape[0]) * noise_scale
    z22 = np.sin(2 * np.pi * t / p22) + npr.randn(t.shape[0]) * noise_scale
        
    data = sp.r_['1,2,0', sp.r_[z1, z11], sp.r_[z2, z22]]

    return data 

def simple_sins_3z(p1,p11, p2,p22, p3, p33, noise_scale, N = 500):
    
    t = arange(N)
                
    z1 = np.sin(2 * np.pi * t / p1) + npr.randn(t.shape[0]) * noise_scale
    z2 = np.sin(2 * np.pi * t / p2) + npr.randn(t.shape[0]) * noise_scale
    z3 = np.sin(2 * np.pi * t / p3) + npr.randn(t.shape[0]) * noise_scale
        
    z11 = np.sin(2 * np.pi * t / p11) + npr.randn(t.shape[0]) * noise_scale
    z22 = np.sin(2 * np.pi * t / p22) + npr.randn(t.shape[0]) * noise_scale
    z33 = np.sin(2 * np.pi * t / p33) + npr.randn(t.shape[0]) * noise_scale
        
    data = sp.r_['1,2,0', sp.r_[z1, z11], sp.r_[z2, z22], sp.r_[z3, z33]]

    return data 

if __name__ == '__main__' : 
    
    first = 1
    
    if first:
            
        #s1 = Tseries(0)
        s2 = Tseries(0)
        #s1.makeSeries([2,1,2], [300, 300, 300], noise = 0.5, period = 50, amp = 5)
        #s2.makeSeries([2], [900], noise = 0.5, period = 50, amp = 5)
        #data = sp.r_['1,2,0', s1, s2]
        
        s0lin = Tseries(0)
        s0sin = Tseries(0)
        s2lin = Tseries(0)
        s2sin = Tseries(0)
        
        interval_length = 300
        l = 10
        m = 10
        baseLine = 0
        amp = 5
        period = 50 
        s0lin.makeSeries([1,3,4,1], [interval_length, l/2, l/2, 2 * interval_length - l], 
                        [baseLine, baseLine, baseLine + m, baseLine], 
                        gradient = float(m)/float(l/2), noise = 0.5)
        s0sin.makeSeries([2], [3 * interval_length], [0.0], 
                        amp = amp, period = period, noise = 0.5)

        # sum sin and linear components to get data stream                         
        s1 = np.array(s0lin) + np.array(s0sin)   

        s2lin.makeSeries([1,4,3,1], [interval_length * 2, l/2, l/2, interval_length - l], 
                        [baseLine, baseLine, baseLine - m, baseLine], 
                        gradient = float(m)/float(l/2), noise = 0.5)
        s2sin.makeSeries([2], [3 * interval_length], [0.0], 
                        amp = amp, period = period, noise = 0.5)
        s2 = np.array(s2lin) + np.array(s2sin)   

        data = sp.r_['1,2,0', s1, s2]
        
        #s1lin.makeSeries([1,4,3,1],[2 * interval_length, l/2, l/2, interval_length - l],
                         #[baseLine, baseLine, baseLine - m, baseLine], 
                         #gradient = float(m)/float(l/2), noise_type ='none')
        #s1sin.makeSeries([2], [3 * interval_length], [0.0], 
                        #amp = amp, period = period, noise_type ='none')
        
        #data = genCosSignals(0, -3.0)
        
        #data, G = create_abilene_links_data()
        
        #execfile('/Users/chris/Dropbox/Work/MacSpyder/Utils/gen_Anomalous_peakORshift_data.py')
        #data = A
        
        #execfile('/Users/chris/Dropbox/Work/MacSpyder/Utils/gen_sin_signals_Anomaly_monitoring.py')
        
        #data = simple_sins(10,10,10,25, 0.1)
        
        #data = simple_sins_3z(10,10,13,13, 10, 27, 0.0)
        
        #data = genCosSignals_no_rand(timesteps = 10000, N = 32)  
        
        #data = array([[0,0,0], [1,2,2], [1,3,4], [3,6,6], [5,6,10], [6,8,11]])   
        
        #sig_PN, ant_PN, time_PN = load_n_store('SYN', 'PN')
        #data = sig_PN
        
        #AbileneMat = sio.loadmat('/Users/chris/DataSets/Abilene/Abilene.mat')
        #data = AbileneMat['P']
        
        #D = load_data('isp_routers')
        data = load_data('chlorine')
        
        #inserted anomaly 
        data[1500:2000, 10:20] = 0.0
        #data = data[:,:20]
        
        #data = D['data']
        
        ## Missout low valued TS 
        #mask = data.mean(axis=0) > 50
        #data = data[:, mask]


        #data = load_ts_data('isp_routers', 'mid')
        
        # Z score data
        data = zscore(data)
        #data = zscore_win(data, 250)
        # Fix Nans 
        whereAreNaNs = np.isnan(data)
        data[whereAreNaNs] = 0
    
    # old rank adaptation - thresholds  
    e_high = 0.99
    e_low = 0.94
    
    alpha = 0.96
    # New rank adaptation - EWMA
    F_min = 0.9
    epsilon = 0.05
    
    EW_mean_alpha = 0.1 # for incremental mean 
    EWMA_filter_alpha = 0.3 # for EWMA detector 
    residual_thresh = 0.02 # for EWMA detector
    
    R = 3 # if r is fixed, r = R
    
    ignoreUp2 = 0 # ignore first 50 inputs 
    
    holdOFF = 0 # delay between changes in r

    L = 1 # No. of Time lagged vectors concatonated to make input vector. 1 = just original vec.
    
    # Run Flags
    v6_4 = 1
    v6_4_fix = 1
    v6_3 = 0
    v6_3_fix = 0
    v6_2 = 0
    v6_2_fix = 0
    v6_1 = 0
    v6_0 = 0
    v4_0 = 0
    v3_4 = 0
    v3_3 = 0
    v3_1 = 0
    pedro = 0
    
    if v6_4:
        '''EWMA detection with F_min-epsilon thresholding'''
        res_v6_4 = FRAHST_V6_4(data, r=1, alpha=alpha, L = L, holdOffTime=holdOFF, evalMetrics = 'F',
                               EW_mean_alpha = EW_mean_alpha, EWMA_filter_alpha = EWMA_filter_alpha, 
                               residual_thresh = residual_thresh, 
                               F_min = F_min, epsilon = epsilon,  
                               static_r = 0, r_upper_bound = None,
                               fix_init_Q = 1, ignoreUp2 = ignoreUp2)

        res_v6_4['Alg'] = 'My FRAUST V6.4 Eigen-Adaptive'
        pltSummary2(res_v6_4, data, (F_min + epsilon, F_min))
        ylim(F_min - 0.05 , 1.02)
    
    if v6_4_fix:
        '''EWMA detection with F_min-epsilon thresholding'''
        res_v6_4f = FRAHST_V6_4(data, r=R, alpha=alpha, L = L, holdOffTime=holdOFF, evalMetrics = 'F',
                               EW_mean_alpha = EW_mean_alpha, EWMA_filter_alpha = EWMA_filter_alpha, 
                               residual_thresh = residual_thresh, 
                               F_min = F_min, epsilon = epsilon,  
                               static_r = 1, r_upper_bound = None,
                               fix_init_Q = 1, ignoreUp2 = ignoreUp2)

        res_v6_4f['Alg'] = 'My FRAUST V6.4 Fixed'
        pltSummary2(res_v6_4f, data, (F_min + epsilon, F_min))
        ylim(F_min - 0.05 , 1.02)
    
    if v6_3:
        '''EWMA detection with F_min-epsilon thresholding'''
        res_v6_3 = FRAHST_V6_3(data, r=1, alpha=alpha, L = L, holdOffTime=holdOFF, evalMetrics = 'T',
                               EW_mean_alpha = EW_mean_alpha, EWMA_filter_alpha = EWMA_filter_alpha, 
                               residual_thresh = residual_thresh, 
                               F_min = F_min, epsilon = epsilon,  
                               static_r = 0, r_upper_bound = None,
                               fix_init_Q = 1, ignoreUp2 = ignoreUp2)

        res_v6_3['Alg'] = 'My FRAUST V6.3 Eigen-Adaptive'
        pltSummary2(res_v6_3, data, (F_min+epsilon, F_min))
        ylim(F_min - 0.05 , 1.02)
        
    if v6_3_fix:
        '''EWMA detection with F_min-epsilon thresholding'''
        res_v6_3f = FRAHST_V6_3(data, r=R, alpha=alpha, L = L, holdOffTime=holdOFF, evalMetrics = 'F',
                               EW_mean_alpha = EW_mean_alpha, EWMA_filter_alpha = EWMA_filter_alpha, 
                               residual_thresh = residual_thresh, 
                               F_min = F_min, epsilon = epsilon,  
                               static_r = 1, r_upper_bound = None,
                               fix_init_Q = 1, ignoreUp2 = ignoreUp2)

        res_v6_3f['Alg'] = 'My FRAUST V6.3 Fixed'
        pltSummary2(res_v6_3f, data, (F_min+epsilon, F_min))
        ylim(F_min - 0.05 , 1.02)
        
    if v6_2:
        '''My Latest version''' 
        res_v6_2 = FRAHST_V6_2(data, L = L, alpha = alpha, EW_mean_alpha = EW_mean_alpha, 
                               EWMA_filter_alpha = EWMA_filter_alpha, 
                               residual_thresh = residual_thresh, epsilon = epsilon,
                               e_low=e_low, e_high = e_high, holdOffTime=holdOFF, 
                               fix_init_Q = 1, r = 1, evalMetrics = 'T', 
                                ignoreUp2 = ignoreUp2, static_r = 0, r_upper_bound = None) 
        
        res_v6_2['Alg'] = 'My FRAUST Version 6.2 Eigen-Adaptive'
        pltSummary2(res_v6_2, data, (e_high, e_low))
        ylim(e_low - 0.05 , 1.02)
        
    if v6_2_fix:
        '''My Latest version''' 
        res_v6_2_fix = FRAHST_V6_2(data, L = L, alpha = alpha, EW_mean_alpha = EW_mean_alpha, 
                               EWMA_filter_alpha = EWMA_filter_alpha, 
                               residual_thresh = residual_thresh,  
                               e_low=e_low, e_high = e_high, holdOffTime=holdOFF, 
                               fix_init_Q = 1, r = R, evalMetrics = 'T', 
                                ignoreUp2 = ignoreUp2, static_r = 1, r_upper_bound = None) 
        
        res_v6_2_fix['Alg'] = 'My FRAUST Version 6.2 Fixed'
        pltSummary2(res_v6_2_fix, data, (e_high, e_low))
        ylim(e_low - 0.05 , 1.02)
        
    if v6_1:
        '''My Latest version''' 
        res_v6_1 = FRAHST_V6_1(data, L = L, alpha = alpha, EW_mean_alpha = EW_mean_alpha, e_low=e_low, 
                                e_high = e_high, holdOffTime=holdOFF, fix_init_Q = 1, r = 1, evalMetrics = 'T', 
                                ignoreUp2 = ignoreUp2, static_r = 0, r_upper_bound = None) 
        
        res_v6_1['Alg'] = 'My Implimentation of FRAUST Version 6.1 '
        pltSummary2(res_v6_1, data, (e_high, e_low))
        ylim(e_low - 0.05 , 1.02)
        
    if v6_0:
        '''My Latest version''' 
        res_v6_0 = FRAHST_V6_0(data, L = L, alpha = alpha, EW_mean_alpha = EW_mean_alpha, e_low=e_low, e_high=e_high, 
                                holdOffTime=holdOFF, fix_init_Q = 1, r = 1, evalMetrics = 'T', 
                                ignoreUp2 = ignoreUp2, static_r = 0, r_upper_bound = None) 
        
        res_v6_0['Alg'] = 'My Implimentation of FRAUST Version 6.0 '
        pltSummary2(res_v6_0, data, (e_high, e_low))

    if v4_0:
        '''My Latest version''' 
        res_v4_0 = FRAHST_V4_0(data, L = L, alpha=alpha, e_low=e_low, e_high=e_high, 
                                holdOffTime=holdOFF, fix_init_Q = 1, r = 1, evalMetrics = 'T', 
                                ignoreUp2 = ignoreUp2, static_r = 0, r_upper_bound = None) 
        
        res_v4_0['Alg'] = 'My Implimentation of FRAUST Version 4.0 '
        pltSummary2(res_v4_0, data, (e_high, e_low))
    
    if v3_4:
        '''My Latest version''' 
        res_new = FRAHST_V3_4(data, L = L, alpha=alpha, e_low=e_low, e_high=e_high, 
                              holdOffTime=holdOFF, fix_init_Q = 1, r = 1, evalMetrics = 'F', 
                              ignoreUp2 = ignoreUp2) 
        
        res_new['Alg'] = 'My Implimentation of FRAUST Version 3.4'
        pltSummary2(res_new, data, (e_high, e_low))
    
    if v3_3:
        '''My previous version''' 
        res_v3_3 = FRAHST_V3_3(data, alpha=alpha, e_low=e_low, e_high=e_high, 
                              holdOffTime=holdOFF, fix_init_Q = 1, r = 1, evalMetrics = 'F', 
                              ignoreUp2 = ignoreUp2) 
    
        res_v3_3['Alg'] = 'My Implimentation of FRAUST V3.3 Best simple version'
        pltSummary2(res_v3_3, data, (e_high, e_low))
    
    if v3_1:
        '''My older version''' 
        res_v3_1 = FRAHST_V3_1(data, alpha=alpha, e_low=e_low, e_high=e_high, 
                              holdOffTime=holdOFF, fix_init_Q = 1, r = 1, evalMetrics = 'F') 
    
        res_v3_1['Alg'] = 'My Implimentation of FRAUST Version 3.1'
        pltSummary2(res_v3_1, data, (e_high, e_low))
    
    if pedro:    
        '''Pedros Version'''
        res_ped = frahst_pedro_original(data, r=1, alpha=alpha, e_low=e_low, e_high=e_high,
                              holdOffTime=holdOFF, evalMetrics='F')
    
        res_ped['Alg'] = 'Pedros Original Implimentation of FRAUST'
        pltSummary2(res_ped, data, (e_high, e_low))

    first = 0