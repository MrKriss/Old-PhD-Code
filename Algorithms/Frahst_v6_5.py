#!/usr/bin/env python
#coding:utf-8
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
from artSigs import genCosSignals_no_rand , genCosSignals, rand_sin_trends, simple_sins \
     simpl
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
from burg_AR import burg_AR
from Frahst_v6_4 import FRAHST_V6_4

def FRAHST_V6_5(data, r=1, alpha=0.96, L = 1, h_AR_buff = 30, AR_order = 5,  holdOffTime=0, evalMetrics = 'F',
                EW_mean_alpha = 0.1, EWMA_filter_alpha = 0.3, residual_thresh = 0.1, 
                F_min = 0.9, epsilon = 0.05,  
                static_r = 0, r_upper_bound = None,
                fix_init_Q = 0, ignoreUp2 = 0):
    """
    Fast Rank Adaptive Householder Subspace Tracking Algorithm (FRAHST)  
    
    Version 6.5 - Investigate forcasting on hidden variables as anomaly measure.
    
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
    
    # Initialise variables and data structures 
    #########################################

    # Derived Variables 
    # Length of z or numStreams is now N x L 
    numStreams = data.shape[1] * L
    timeSteps = data.shape[0]
    
    if r_upper_bound == None :
        r_upper_bound = numStreams
    
    #for energy test
    last_Z_pos = bool(1) # bool flag
    lastChangeAt = 1
    sumYSq = 0.
    sumXSq = 0.
    
    # Data Stores
    res = {'hidden' :  zeros((timeSteps, numStreams)) * nan,  # Array for hidden Variables
           'E_t' : zeros([timeSteps, 1]),                     # total energy of data 
           'E_dash_t' : zeros([timeSteps, 1]),                # hidden var energy
           'e_ratio' : zeros([timeSteps, 1]),              # Energy ratio 
           'RSRE' : zeros([timeSteps, 1]),           # Relative squared Reconstruction error 
           'recon' : zeros([timeSteps, numStreams]),  # reconstructed data
           'r_hist' : zeros([timeSteps, 1]),         # history of r values 
           'eig_val': zeros((timeSteps, numStreams)) * nan,  # Estimated Eigenvalues 
           'zt_mean' : zeros((timeSteps, numStreams)), # history of data mean 
           'zt_var' : zeros((timeSteps, numStreams)), # history of data var  
           'zt_var2' : zeros((timeSteps, numStreams)), # history of data var  
           'S_trace' : zeros((timeSteps, 1)),          # history of S trace
           'skips'   : zeros((timeSteps, 1)),          # tracks time steps where Z < 0 
           'EWMA_res' : zeros((timeSteps, 1)),         # residual of energy ratio not acounted for by EWMA
           'Phi' : [],      
           'S' : [],
           'Q' : [],
           'w' : zeros((timeSteps, numStreams)),
           'e' : zeros((timeSteps, numStreams)),
           'anomalies' : [],
           'forecast_err' : zeros((timeSteps, numStreams)),
           'for_err_sum' : zeros((timeSteps, 1)),
           'pred_zt' : zeros((timeSteps, numStreams))} # Error from forcasting with hidden vars
        
    # Initialisations 
    # Q_0
    if fix_init_Q != 0:  # fix inital Q as identity 
        q_0 = eye(numStreams);
        Q = q_0
        Qt_min1 = q_0
    else: # generate random orthonormal matrix N x r 
        Q = eye(numStreams) # Max size of Q
        Qt_min1 = eye(numStreams) # Max size of Q
        Q_0, R_0 = qr(rand(numStreams,r))   
        Q[:,:r] = Q_0          
        Qt_min1[:,:r] = Q_0          
    
    # S_0
    small_value = 0.0001
    S = eye(numStreams) * small_value # Avoids Singularity    
    # v-1
    v = zeros((numStreams,1)) 
    # U(t-1) for eigenvalue estimation
    U = eye(numStreams)
    # zt mean and var
    zt_mean = zeros((numStreams,1))
    zt_var = zeros((numStreams,1))
    zt_var2 = zeros((numStreams,1))
    
    # NOTE algorithm's state (constant memory), S, Q and v and U are kept at max size
    
    # Use iterable for data 
    # Now a generator to calculate z_tl
    iter_data = lag_inputs(data, L)         
    
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
        # Update zt mean and var
        zt_var, zt_mean = EW_mean_var(zt, EW_mean_alpha, zt_var, zt_mean)
        zt_var2 = alpha_var(zt, alpha, zt_var2)
        
        # Convert to a column Vector 
        # Already taken care of in this version
        # zt = zt.reshape(zt.shape[0],1) 
    
        # Check S remains non-singular
        for idx in range(r):
            if S[idx, idx] < small_value:
                S[idx,idx] = small_value
        
        '''Begin main algorithm'''        
        ht = dot(Qt.T, zt) 
        
        Z = dot(zt.T, zt) - dot(ht.T , ht)

        if Z > 0 :
            
            last_Z_pos = 1
            
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
            
            # store e and w
            res['w'][t-1,:r] = w.T[0,:]
            res['e'][t-1,:] = ee.T[0,:]
            
        else: # if Z is not > 0

            if norm(zt) > 0 and norm(ht) > 0 : # May be due to zt <= ht 
                St = alpha * St # Continue decay of St
                res['skips'][t-1] = 2 # record Skips
                
            else: # or may be due to zt and ht = 0
                St = alpha * St # Continue decay of St 
                res['skips'][t-1] = 1 # record Skips
            
            # Recalculate Eigenvalues using other method 
            # (less fast, but does not need Z to be positive)
            if last_Z_pos == 1:
                # New U                 
                U2t_min1 = np.eye(r)
                #PHI = np.dot(Qt_min1.T, Qt)
                Wt = np.dot(St, U2t_min1)
                U2t, R2 = qr(Wt) # Decomposition
                PHI_U = np.dot(U2t_min1.T,U2t)
                e_values = np.diag(np.dot(R2,PHI_U))
            elif last_Z_pos == 0:
                U2t_min1 = U2t                
                #PHI = np.dot(Qt_min1.T, Qt)
                Wt = np.dot(St, U2t_min1)
                #Wt = np.dot(np.dot(St, PHI), U2t_min1)
                U2t, R2 = qr(Wt) #Decomposition
                PHI_U = np.dot(U2t_min1.T,U2t)
                e_values = np.diag(np.dot(R2,PHI_U))

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
            #
            res['Phi'].append(Cov_mat)
            #
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
            
            # Calculate angle between projection matrixes
            #D = dot(dot(dot(V_r.T, Qt), Qt.T), V_r) 
            #eigVal, eigVec = eig(D)
            #angle = arccos(sqrt(max(eigVal)))        
            #res['angle_error'][t-1,0] = angle        
    
            # Calculate deviation from orthonormality
            F = dot(Qt.T , Qt) - eye(r)
            res['orthog_error'][t-1,0] = 10 * log10(trace(dot(F.T , F))) #frobenius norm in dB
        
        '''Store Values''' 
        # Record data mean and Var
        res['zt_mean'][t-1,:] = zt_mean.T[0,:]
        res['zt_var'][t-1,:] = zt_var.T[0,:]
        res['zt_var2'][t-1,:] = zt_var2.T[0,:]
        
        # REcord S & Q
        res['S'].append(St)
        res['Q'].append(Qt)
        
        # Record S trace
        res['S_trace'][t-1] = np.trace(St)
        
        # Store eigen values
        if 'e_values' not in locals():
            e_values = zt_var2 # Why this?
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
        
        ''' Forcasting '''
        
        if t > h_AR_buff:
            # Get Coefficents for ht+1
            # Get h-buffer window (can speed this up latter)
            h_buffer = np.nan_to_num(res['hidden'][t-h_AR_buff:t, :])
            pred_h = np.zeros((r,1))
            for i in range(r):
                coeffs = burg_AR(AR_order, h_buffer[:,i])
                for j in range(AR_order):
                    pred_h[i,0] -= coeffs[j] * h_buffer[-1-j, i]
            
            res['pred_zt'][t-1,:] = dot(Qt, pred_h).T 
            res['forecast_err'][t-1,:] = np.abs(res['pred_zt'][t-2,:] - zt.T)
            res['for_err_sum'][t-1] = np.abs(res['pred_zt'][t-2,:] - zt.T).sum() / numStreams
        
        
        '''Rank Estimation''' 
        # Calculate energies 
        sumXSq = alpha * sumXSq + np.sum(zt ** 2) # Energy of Data
        sumYSq = alpha * sumYSq + np.sum(ht ** 2) # Energy of hidden Variables
                
        res['E_t'][t-1,0] = sumXSq 
        res['E_dash_t'][t-1,0] = sumYSq
        
        if sumXSq == 0 : # Catch NaNs 
            e_ratio = 0.0
        else:
            e_ratio = sumYSq / sumXSq
        
        res['e_ratio'][t-1, 0] = e_ratio
        
        # Run EWMA on e_ratio 
        if t == 1:  
            pred_data = 0.0 # initialise value

        # Calculate residual usung last time steps prediction 
        residual = np.abs(e_ratio - pred_data)
        res['EWMA_res'][t-1,0] = residual
        # Update prediction for next time step
        pred_data = EWMA_filter_alpha * e_ratio + (1-EWMA_filter_alpha) * pred_data    
        
        # Threshold residual for anomaly
        if residual > residual_thresh and t> ignoreUp2:
            # Record time step of anomaly            
            res['anomalies'].append(t-1) 
    
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
        s3 = Tseries(0)
        s4 = Tseries(0)
        
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
        
        s3.makeSeries([2], [3 * interval_length], [0.0], 
                        amp = amp, period = period, noise = 0.5)
        s4.makeSeries([2], [3 * interval_length], [0.0], 
                        amp = amp, period = period, noise = 0.5)

        data = sp.r_['1,2,0', s1, s2, s3, s4]
        
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
        
        #data = simple_sins_3z(10,10,10,10, 10, 27, 0.1)
        
        #data = genCosSignals_no_rand(timesteps = 10000, N = 32)  
        
        #data = array([[0,0,0], [1,2,2], [1,3,4], [3,6,6], [5,6,10], [6,8,11]])   
        
        #sig_PN, ant_PN, time_PN = load_n_store('SYN', 'PN')
        #data = sig_PN
        
        #AbileneMat = sio.loadmat('/Users/chris/DataSets/Abilene/Abilene.mat')
        #data = AbileneMat['P']
        
        
        #data1, trends1 = rand_sin_trends(20, 500, [10,35,50], seed = 1,  noise_scale = 0.0)
        #data2, trends2 = rand_sin_trends(20, 500, [10,20,25], seed = None,  noise_scale = 0.0)
        #data3, trends3 = rand_sin_trends(20, 500, [10,35,50], seed = 1,  noise_scale = 0.0)
        #d2 = data1.copy()
        #d2[:,2] = data2[:,2]
        #data = sp.r_['0', data1, data2, data1]
        #data[250:255, 0] = data[250:255, 0] + 1.5
        #data = data1
        dorig = data.copy()
        
        #D = load_data('isp_routers')
        
        #data = load_data('chlorine')
        
        #inserted anomaly 
        #data[1500:1505, 0] = data[1500:1505, 0] + 1.5
        #data = data[:,:20]
        
        #data = D['data']
        
        ## Missout low valued TS 
        #mask = data.mean(axis=0) > 50
        #data = data[:, mask]

        #data = load_ts_data('isp_routers', 'full')
        
        # Z score data
        data = zscore(data)
        #data = zscore_win(data, 250)
        # Fix Nans 
        whereAreNaNs = np.isnan(data)
        data[whereAreNaNs] = 0
    
    # old rank adaptation - thresholds  
    e_high = 0.98
    e_low = 0.90
    
    alpha = 0.98
    # New rank adaptation - EWMA
    F_min = 0.90
    epsilon = 0.05
    
    EW_mean_alpha = 0.1 # for incremental mean 
    EWMA_filter_alpha = 0.2 # for EWMA detector 
    residual_thresh = 0.02 # for EWMA detector
    
    R = 3 # if r is fixed, r = R
    
    ignoreUp2 = 0 # ignore first 50 inputs 
    
    holdOFF = 0 # delay between changes in r

    L = 1 # No. of Time lagged vectors concatonated to make input vector. 1 = just original vec.
    
    # Run Flags
    v6_5 = 1
    v6_5_fix = 1
    v6_4 = 0
    v6_4_fix = 0
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
    pedro = 1
    
    if v6_5:
        '''EWMA detection with F_min-epsilon thresholding'''
        res_v6_5 = FRAHST_V6_5(data, r=1, alpha=alpha, L = L, holdOffTime=holdOFF, evalMetrics = 'F',
                               EW_mean_alpha = EW_mean_alpha, EWMA_filter_alpha = EWMA_filter_alpha, 
                               residual_thresh = residual_thresh, 
                               F_min = F_min, epsilon = epsilon,  
                               static_r = 0, r_upper_bound = None,
                               fix_init_Q = 1, ignoreUp2 = ignoreUp2)

        res_v6_5['Alg'] = 'My FRAUST V6.5 Eigen-Adaptive'
        pltSummary2(res_v6_5, data, (F_min + epsilon, F_min))
        ylim(F_min - 0.05 , 1.02)
        f = figure()
        ax1 = f.add_subplot(311)
        ax1.plot(data)
        ax2 = f.add_subplot(312, sharex = ax1)
        ax2.plot(res_v6_5['pred_zt'])
        ax3 = f.add_subplot(313, sharex = ax1)
        ax3.plot(res_v6_5['for_err_sum'])
        
    
    if v6_5_fix:
        '''EWMA detection with F_min-epsilon thresholding'''
        res_v6_5f = FRAHST_V6_5(data, r=R, alpha=alpha, L = L, holdOffTime=holdOFF, evalMetrics = 'F',
                               EW_mean_alpha = EW_mean_alpha, EWMA_filter_alpha = EWMA_filter_alpha, 
                               residual_thresh = residual_thresh, 
                               F_min = F_min, epsilon = epsilon,  
                               static_r = 1, r_upper_bound = None,
                               fix_init_Q = 1, ignoreUp2 = ignoreUp2)

        res_v6_5f['Alg'] = 'My FRAUST V6.5 Fixed'
        pltSummary2(res_v6_5f, data, (F_min + epsilon, F_min))
        ylim(F_min - 0.05 , 1.02)
        f = figure()
        ax1 = f.add_subplot(311)
        ax1.plot(data)
        ax2 = f.add_subplot(312, sharex = ax1)
        ax2.plot(res_v6_5f['pred_zt'])
        ax3 = f.add_subplot(313, sharex = ax1)
        ax3.plot(res_v6_5f['for_err_sum'])
    
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