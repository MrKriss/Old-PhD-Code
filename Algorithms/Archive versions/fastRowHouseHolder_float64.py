# -*- coding: utf-8 -*-
"""
Created on Thu Dec 09 16:43:49 2010

@author: musselle
"""

import numpy.matlib as npm
from numpy import  sqrt, zeros, mat, multiply, log10, trace, cos, pi, array, \
arccos
from numpy.linalg import norm , qr
from numpy.random import rand, randn , seed
from matplotlib.pyplot import plot, figure, title
from utils import QRsolveM 


def FRHH64(streams, rr, alpha, sci = 0, fix_init_Q = 0):
    """ Fast row-Householder Subspace Traking Algorithm, Non adaptive version 
    
    """
#===============================================================================
#     #Initialise variables and data structures 
#===============================================================================     
    
    N = streams.shape[1] # No. of streams 
    
    # Data Stores
    E_t = [0] # time series of total energy 
    E_dash_t = [0] # time series of reconstructed energy
    
    z_dash = npm.zeros((1,N)) # time series of reconstructed data
    # z_dash = npm.zeros(N)
    
    RSRE = mat([0])  # time series of Root squared Reconstruction Error
    hid_var = npm.zeros((streams.shape[0], N)) # Array of hidden Variables 
    
    seed(111)
    
    # Initialisations 
    # Q_0
    if fix_init_Q != 0:  # fix inital Q as identity 
        # Identity     
        q_I = npm.eye(N, rr) 
        Q_t = [q_I]
    else: # generate random orthonormal matrix N x r 
        # Random     
        qq,RR = qr(rand(N,rr))   # generate random orthonormal matrix N x r 
        Q_t = [mat(qq)]   # Initialise Q_t - N x r
    
    S_t = [npm.eye(rr) * 0.00001]   # Initialise S_t - r x r 
    
    No_inp_count = 0 # count of number of times there was no input i.e. z_t = [0,...,0]
    No_inp_marker = zeros((1,streams.shape[0] + 1))
    
    v_vec_min_1 = npm.zeros((rr,1))
    
    iter_streams = iter(streams)
    
    for t in range(1, streams.shape[0] + 1):
        
        z_vec = mat(iter_streams.next())
        
        z_vec = z_vec.T # Now a column Vector
        
        hh = Q_t[t-1].T * z_vec                  # 13a

        Z = z_vec.T * z_vec - hh.T * hh           # 13b
        
        # Z = float(Z) # cheak that Z is really scalar
        
        if Z > 0 :       
            
            # Refined version, sci accounts better for tracked eigen values
            if sci != 0: 
                u_vec = S_t[t-1] * v_vec_min_1 
                extra_term = 2 * alpha * sci * u_vec * v_vec_min_1.T
            else:
                extra_term = 0
                
            X = alpha * S_t[t-1]  + hh * hh.T - extra_term
            
            # QR method - hopefully more stable 
            aa = X.T
            b = sqrt(Z[0,0]) * hh
            # b_vec = solve(aa,b)
            b_vec = QRsolveM(aa,b)   
            
            
            beta = 4 * (b_vec.T * b_vec + 1)
            
            phi_sq_t = 0.5 + (1 / sqrt(beta)) 
            
            phi_t = sqrt(phi_sq_t) 
            
            gamma = (1 - 2 * phi_sq_t) / (2 * phi_t)   
            
            delta = phi_t / sqrt(Z)          
            
            v_vec_t = multiply(gamma , b_vec)
            
            S_t.append(X - multiply( 1/delta , v_vec_t * hh.T))         
            
            w_vec = multiply(delta , hh) - v_vec_t        
            
            e_vec = multiply(delta, z_vec) - (Q_t[t-1] * w_vec)
            
            Q_t.append(Q_t[t-1] - 2 * e_vec * v_vec_t.T)
        
            v_vec_min_1 = v_vec_t # update for next time step
        
            # Record hidden variables
            hid_var[t-1,:hh.shape[0]] = hh.T
                
            # Record reconstrunted z 
            new_z_dash = Q_t[t-1] * hh
            z_dash = npm.vstack((z_dash, new_z_dash.T))
        
            # Record RSRE
            new_RSRE = RSRE[0,-1] + (((norm(new_z_dash - z_vec)) ** 2) / 
                                        (norm(z_vec) ** 2))                           
            RSRE = npm.vstack((RSRE, mat(new_RSRE))) 
        
        else:
            
            # Record hidden variables
            hid_var[t-1,:hh.shape[0]] = hh.T
            
            # Record reconstrunted z 
            new_z_dash = Q_t[t-1] * hh
            z_dash = npm.vstack((z_dash, new_z_dash.T))
        
            # Record RSRE
            new_RSRE = RSRE[0,-1] + (((norm(new_z_dash - z_vec)) ** 2) / 
                                    (norm(z_vec) ** 2))                           
            RSRE = npm.vstack((RSRE, mat(new_RSRE)))            
            
            # Repeat last entries
            Q_t.append(Q_t[-1])
            S_t.append(S_t[-1])            
                         
            # increment count
            No_inp_count += 1                        
            No_inp_marker[0,t-1] = 1 
            
    # convert to tuples to save memory        
    Q_t = tuple(Q_t)
    S_t = tuple(S_t)
    rr = array(rr)
    E_t = array(E_t)
    E_dash_t = array(E_dash_t)
            
    return  Q_t, S_t, rr, E_t, E_dash_t, hid_var, z_dash, RSRE, No_inp_count, No_inp_marker

def plotEqqFqq64(streams, Q_t, alpha, p = 0, flag = 0):
    """
     p = plot e_qq and f_qq (YES/NO)
     flag = record all data for cov_mat and eigenvalues (YES/NO)
    """
    N = len(Q_t) 

    if flag != 0:
        cov_mat_record = []
        V_r_record = []

    # Calculate F_qq #  (deviation fron orthogonality)
    f_qq = npm.zeros((N,1))                                                                
    index = 0
    for q_t_i in Q_t:       
        X = q_t_i.T * q_t_i 
        FQQ = X - npm.eye(X.shape[0])  
        f_qq[index, 0] = 10 * log10(trace(FQQ.T * FQQ))
        index += 1

    # Calculate E_qq (deviation from eigenvector subspace)
    e_qq = npm.zeros((N-1,1))
    g_qq = npm.zeros((N-1,1))
    cov_mat = npm.zeros((streams.shape[1],streams.shape[1]))    
    for i in range(streams.shape[0]):
        data = mat(streams[i,:])
        data = data.T # store as column vector 
        cov_mat = alpha * cov_mat + (data * data.T)
        W , V = npm.linalg.eig(cov_mat)
            
        # sort eigenVectors in according to deccending eigenvalue
        eig_idx = W.argsort() # Get sort index
        eig_idx = eig_idx[::-1] # Reverse order (default is accending)
        
        # v_r = highest r eigen vectors accoring to thier eigenvalue.
        V_r = V[:, eig_idx[:Q_t[i+1].shape[1]]]
        # Hopefuly have sorted correctly now 
        # Currently V_r is [1st 2nd, 3rd 4th] highest eigenvector 
        # according to eigenvalue     
        
        Y = V_r * V_r.T - Q_t[i+1] * Q_t[i+1].T  
        e_qq[i, 0] = 10 * log10(trace(Y.T * Y))
        
        if flag != 0:
            cov_mat_record.append(cov_mat)
            V_r_record.append(V_r)
        
        # Calculate angle between projection matrixes
        A = V_r.T * Q_t[i+1] * Q_t[i+1].T * V_r 
        eigVal , eigVec = npm.linalg.eig(A)
        angle = arccos(sqrt(max(eigVal)))        
        g_qq[i,0] = angle
        
    if p != 0:    
        figure()
        plot(f_qq)
        title('Deviation from orthonormality')    
        figure()
        plot(e_qq)
        title('Deviation of true tracked subspace') 
    
    if flag != 0:
        cov_mat_record = tuple(cov_mat_record)
        V_r_record = tuple(V_r_record) 
        return e_qq, f_qq, cov_mat_record, V_r_record
    else:
        return e_qq, f_qq, g_qq

def genSimSignals64(randSeed, SNR):

    seed(randSeed)    
    
    # Cheack Input     
    SNR = float(SNR)    
    
    # units 
    # # 1 = radians 
    # # pi / 180 = degrees    
    
    units = 1     
    
    # Create Data ##################################################
    a_11 = 1.4 * units
    a_12 = 1.6 * units
    a_21 = 2.0 * units
    a_22 = 1.0 * units
    
    w_t_11 = 2.2 * units
    w_t_12 = 2.8 * units
    w_t_21 = 2.7 * units
    w_t_22 = 2.3 * units
    
    w_s_11 = 0.5 * units
    w_s_12 = 0.9 * units
    w_s_21 = 1.1 * units
    w_s_22 = 0.8 * units
        
    N = 32  # No. Signals
    s_t = zeros((10000,N))
    
    for k in range(1,N+1):     
        # starting phases for signals are random 
        w_0_11 = rand() * 2 * pi
        w_0_12 = rand() * 2 * pi
        w_0_21 = rand() * 2 * pi
        w_0_22 = rand() * 2 * pi 
        for t in range(1,10001):
            if t < 4000:
                A = a_11 * cos(w_t_11 * t + w_s_11 * k + w_0_11)
                B = a_12 * cos(w_t_12 * t + w_s_12 * k + w_0_12)
                s_t[t-1,k-1] = A + B
            else:
                A = a_21 * cos(w_t_21 * t + w_s_21 * k + w_0_21)
                B = a_22 * cos(w_t_22 * t + w_s_22 * k + w_0_22)
                s_t[t-1,k-1] = A + B
                
    Ps = sum(s_t ** 2)
    # SNR = -3.0
    Pn = Ps / (10 ** (SNR/10))    
    
    scale = Pn / (N * 10000)
    
    noise = randn(s_t.shape[0], s_t.shape[1]) * sqrt(scale)
    
    Sim_streams = s_t + noise       
    
    return Sim_streams
    
#===============================================================================
# If main method 
#===============================================================================

if __name__ == '__main__':
            
    # Sim_streams = genSimSignals64(0, -3.0)
    
    # Sim_streams = mat('1,2,2; 1,3,4; 3,6,6; 5,6,10; 6,8,11')
    
    Sim_streams = npm.fromfile('/Users/chris/Dropbox/Work/MacSpyder/Experiments/shiftstreams.dat')
    Sim_streams = Sim_streams.reshape(900, 3)
    
    # import scipy.io
    
    # data = scipy.io.loadmat('S0-S4_streams.mat')
    
    # Sim_streams = data['S0']    
    
    # FRRHH ###############################################
    alpha = 0.996
    rr = 1
    sci = 0
    Q_t, S_t, rr, E_t, E_dash_t, hid_var, z_dash, RSRE, No_inp_count, \
    No_inp_marker = FRHH64(Sim_streams, rr, alpha, sci)  
    
    # Calculate deviations from orthogonality and subspace
    e_qq, f_qq, g_qq = plotEqqFqq64(Sim_streams, Q_t, alpha, p = 1)
    
    fRHH64_data = {'alpha' : alpha,
                   'sci' : sci,
                   'Q_t' : Q_t,
                   'S_t' : S_t,
                   'rr' : rr,
                   'E_t' : E_t,
                   'E_dash_t' : E_dash_t,
                   'hid_var' : hid_var,
                   'z_dash' : z_dash,
                   'RSRE' : RSRE,
                   'No_inp_count' : No_inp_count,
                   'No_inp_marker' : No_inp_marker,
                   'e_qq' : e_qq,
                   'f_qq' : f_qq,
                   'g_qq' : g_qq }
    
