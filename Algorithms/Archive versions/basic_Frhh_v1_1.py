# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 13:26:34 2011

@author: musselle
"""

from numpy import eye, zeros, dot, sqrt, array, pi, cos, log10, trace, arccos, \
vstack
from numpy.random import rand, randn , seed
from numpy.linalg import qr, eig, norm
from matplotlib.pyplot import plot, figure, title

from QRsolve import QRsolveA

def frhh_A(streams, alpha, rr, sci = 0, init_Q = 0):
    """
    Inputs
    #  streams - t x N array of input data streams.
    #  energyThresh - tuple of energy thresholds (low, high)
    #  alpha - forgetting factor in range [0,1], closer to 1 gives more wieght 
        to the past. Lower values adapt quicker 
    #  sci - flag parameter to use extra terms: 0 - No extra terms 
                                                [-1, +1] - Extra terms 
    """

    # Derived Variables 
    NumStreams = streams.shape[1]
    
    
    # Data Stores
    E_t = [0] # time series of total energy 
    E_dash_t = [0] # time series of reconstructed energy
    z_dash = zeros((1,NumStreams)) # time series of reconstructed data
    RSRE = array([[0]])  # time series of Root squared Reconstruction Error
    hidden_var = zeros((streams.shape[0], NumStreams)) # Array of hidden Variables     
    
    # Initialisations 
    # Q_0
    if init_Q != 0:
        q_0 = eye(NumStreams,rr);
        Q_t = [q_0]
    else:
        q_0, r_0 = qr(rand(NumStreams,rr))   # generate random orthonormal matrix N x r 
        Q_t = [q_0]         
        
    # S_0
    S_t = [eye(rr) * 0.00001]
    
    # v-1
    v_minus_1 = zeros((rr,1))    
    
    iter_streams = iter(streams) 

    for t in range(1, streams.shape[0] + 1):
    
        z_t = iter_streams.next()  
        
        # Convert to a column Vector 
        z_t = z_t.reshape(z_t.shape[0],1)
    
        h_t = dot(Q_t[t-1].T , z_t)
        
        Z = dot(z_t.T,  z_t) - dot(h_t.T , h_t)
    
        if Z > 0 :
            
            # Refined version, sci accounts better for tracked eigen values
            if sci != 0: 
                u_vec = dot(S_t[t-1] , v_minus_1) 
                extra_term = 2 * alpha * sci * dot(u_vec, v_minus_1.T)
            else:
                extra_term = 0            
            
            X = alpha * S_t[t-1] + dot(h_t , h_t.T) - extra_term
    
            A = X.T
            B = sqrt(Z) * h_t
            b_vec = QRsolveA(A,B)  
    
            beta  = 4 * (dot(b_vec.T , b_vec) + 1)
        
            phi_sq = 0.5 + (1 / sqrt(beta))
        
            phi = sqrt(phi_sq)
    
            gamma = (1 - 2 * phi_sq) / (2 * phi)
            
            delta = phi / sqrt(Z)
            
            v = gamma * b_vec 
            
            new_S_t = X - ((1 /delta) * dot(v , h_t.T))
            S_t.append(new_S_t)
            
            w = (delta * h_t) - (v) 
            
            ee = delta * z_t - dot(Q_t[t-1] , w)
            
            new_Q_t = Q_t[t-1] - 2 * dot(ee , v.T) 
            Q_t.append(new_Q_t) 
            
            v_minus_1 = v # update for next time step
        
            # Record hidden variables
            hidden_var[t-1,:h_t.shape[0]] = h_t.T
                
            # Record reconstrunted z 
            new_z_dash = dot(Q_t[t-1] , h_t)
            z_dash = vstack((z_dash, new_z_dash.T))
        
            # Record RSRE
            new_RSRE = RSRE[0,-1] + (((norm(new_z_dash - z_t)) ** 2) / 
                                        (norm(z_t) ** 2))                           
            RSRE = vstack((RSRE, new_RSRE)) 
        
        else:
            
            # Record hidden variables
            hidden_var[t-1, :h_t.shape[0]] = h_t.T
            
            # Record reconstrunted z 
            new_z_dash = dot(Q_t[t-1] , h_t)
            z_dash = vstack((z_dash, new_z_dash.T))
        
            # Record RSRE
            new_RSRE = RSRE[0,-1] + (((norm(new_z_dash - z_t)) ** 2) / 
                                    (norm(z_t) ** 2))                           
            RSRE = vstack((RSRE, new_RSRE))            
            
            # Repeat last entries
            Q_t.append(Q_t[-1])
            S_t.append(S_t[-1])            
                        
            
    # convert to tuples to save memory        
    Q_t = tuple(Q_t)
    S_t = tuple(S_t)
    rr = array(rr)
    E_t = array(E_t)
    E_dash_t = array(E_dash_t)           
    
    return Q_t, S_t, RSRE, rr, E_t, E_dash_t, z_dash, hidden_var 
    
def genSimSignalsA(randSeed, SNR, timesteps = 10000, N = 32):
    """
        N = no. of streams
    """
    seed(randSeed)    
    
    # Cheack Input     
    SNR = float(SNR)    
    
    # Create Data ##################################################
    # units  = radians
    a_11 = 1.4   
    a_12 = 1.6 
    a_21 = 2.0 
    a_22 = 1.0 
    
    w_t_11 = 2.2 
    w_t_12 = 2.8 
    w_t_21 = 2.7 
    w_t_22 = 2.3 
    
    w_s_11 = 0.5 
    w_s_12 = 0.9 
    w_s_21 = 1.1 
    w_s_22 = 0.8 
        
    s_t = zeros((timesteps,N))
    
    for k in range(1,N+1):     
        # starting phases for signals are random 
        w_0_11 = rand() * 2 * pi
        w_0_12 = rand() * 2 * pi
        w_0_21 = rand() * 2 * pi
        w_0_22 = rand() * 2 * pi 
        for t in range(1,timesteps + 1):
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
    
    scale = Pn / (N * timesteps)
    
    noise = randn(s_t.shape[0], s_t.shape[1]) * sqrt(scale)
    
    Sim_streams = s_t + noise       
    
    return Sim_streams
    
def plotEqqFqqA(streams, Q_t, alpha, p = 0):
    """
     p = plot e_qq and f_qq (YES/NO)
     flag = record all data for cov_mat and eigenvalues (YES/NO)
    """
    # N = number of timesteps + 1 for initial Q_0
    N = len(Q_t) 

    # Calculate F_qq #  (deviation fron orthogonality)
    f_qq = zeros((N,1))                                                                
    index = 0
    for q_t_i in Q_t:       
        X = dot(q_t_i.T , q_t_i) 
        FQQ = X - eye(X.shape[0])  
        f_qq[index, 0] = 10 * log10(trace(dot(FQQ.T, FQQ)))
        index += 1

    # Calculate E_qq (deviation from eigenvector subspace)
    e_qq = zeros((N-1,1))
    g_qq = zeros((N-1,1))
    cov_mat = zeros((streams.shape[1],streams.shape[1]))    
    for i in range(streams.shape[0]):
        
        data = streams[i,:]
        data = data.reshape(data.shape[0],1) # store as column vector 
        cov_mat = alpha * cov_mat + dot(data , data.T)
        W , V = eig(cov_mat)
                    
        # sort eigenVectors in according to deccending eigenvalue
        eig_idx = W.argsort() # Get sort index
        eig_idx = eig_idx[::-1] # Reverse order (default is accending)
        
        # v_r = highest r eigen vectors accoring to thier eigenvalue.
        V_r = V[:, eig_idx[:Q_t[i+1].shape[1]]]
        # Hopefuly have sorted correctly now 
        # Currently V_r is [1st 2nd, 3rd 4th] highest eigenvector 
        # according to eigenvalue     
        
        Y = dot(V_r , V_r.T) - dot(Q_t[i+1] , Q_t[i+1].T)  
        e_qq[i, 0] = 10 * log10(trace(dot(Y.T , Y)))
        
        # Calculate angle between projection matrixes
        A = dot(dot(dot(V_r.T , Q_t[i+1]) , Q_t[i+1].T) , V_r) 
        eigVal , eigVec = eig(A)
        angle = arccos(sqrt(max(eigVal)))        
        g_qq[i,0] = angle
        
    if p != 0:    
        figure()
        plot(f_qq)
        title('Deviation from orthonormality')    
        figure()
        plot(e_qq)
        title('Deviation of true tracked subspace') 
    
    return e_qq, f_qq, g_qq     
     
if __name__ == '__main__' : 
    
    streams = genSimSignalsA(0, -3.0)    
    
    # streams = array([[0,0,0], [1,2,2], [1,3,4], [3,6,6], [5,6,10], [6,8,11]])   
    
    alpha = 0.996    
    rr = 4
    sci = -1
    
    Q_t, S_t, RSRE, rr, E_t, E_dash_t, z_dash, hidden_var \
    = frhh_A(streams, alpha, rr, sci)
    
    e_qq, f_qq, g_qq = plotEqqFqqA(streams, Q_t, alpha, p = 1)