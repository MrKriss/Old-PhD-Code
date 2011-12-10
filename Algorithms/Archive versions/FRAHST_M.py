# -*- coding: utf-8 -*-
"""
Created on Wed Dec 01 00:07:06 2010

The Frahst Algorithm in using the Matrix class instead of array.

@author: musselle
"""

from ControlCharts import Tseries
import numpy.matlib as npm
from numpy import hstack, vstack, sqrt, expand_dims, dot, array, zeros, \
mat, multiply, bmat, log10, trace 
from numpy.linalg import norm , inv, qr
from numpy.random import rand 
from matplotlib.pyplot import plot, figure, title
import scipy as sp

def FRAHST_M(streams, energyThresh, alpha):
    """ Fast rank adaptive row-Householder Subspace Traking Algorithm   
    
    """
    #Initialise 
    N = streams.shape[1]
    rr = [1]    
    hiddenV = npm.zeros((streams.shape[0], N))   
    # generate random orthonormal  - N x r 
    qq,RR = qr(rand(N,1))    
    Q_t = [mat(qq)]    
    S_t = [mat([0.000001])] 
    E_t = [0]
    E_dash_t = [0]
    z_dash = npm.zeros(N)
    RSRE = mat([0])
    No_inp_count = 0
    
    iter_streams = iter(streams)
    
    for t in range(1, streams.shape[0] + 1):
        
        z_vec = mat(iter_streams.next())
        
        z_vec = z_vec.T # Now a column Vector
        
        hh = Q_t[t-1].T * z_vec                       # 13a

        Z = z_vec.T * z_vec - hh.T * hh           # 13b
        
        Z = float(Z) # cheak that Z is really scalar
        
        if Z > 0.0000001:        
            
            X = alpha * S_t[t-1] + hh * hh.T               # 13c
        
            # X.T * b = sqrt(Z) * hh                           # 13d        
        
            b = multiply(inv(X.T), sqrt(Z)) * hh  # inverse method 
        
            phi_sq_t = 0.5 + (1 / sqrt(4 *((b.T * b) + 1)))   # 13e

            phi_t = sqrt(phi_sq_t)        

            delta = phi_t / sqrt(Z)                        # 13f
        
            gamma = (1 - 2 * phi_sq_t) / (2 * phi_t)         #13 g
        
            v = multiply(gamma, b)  
        
            S_t.append(X - multiply(1/delta , v * hh.T))         # 13 h  S_t[t] = 

            
            e = multiply(delta, z_vec) - (Q_t[t-1] * (multiply(delta, hh) - v))  # 13 i
            
            Q_t.append(Q_t[t-1] - 2 * (e * v.T))                 # 13 j  Q[t] = 

            # Record hidden variables
            hiddenV[t-1,:hh.shape[0]] = hh.T
            
            # Record reconstrunted z 
            new_z_dash = Q_t[t-1] * hh
            z_dash = npm.vstack((z_dash, new_z_dash.T))
        
            # Record RSRE
            new_RSRE = RSRE[0,-1] + (((norm(new_z_dash - z_vec)) ** 2) / 
                                    (norm(z_vec) ** 2))                           
            RSRE = npm.vstack((RSRE, mat(new_RSRE))) 
        
            E_t.append(alpha * E_t[-1] + norm(z_vec) ** 2)        # 13 k
                
            E_dash_t.append( alpha * E_dash_t[-1] + norm(hh) ** 2)  # 13 l
        
            if E_dash_t[-1] < energyThresh[0] * E_t[-1] and rr[-1] < N: # 13 m 
        
                z_dag_orthog =  z_vec - Q_t[t] * Q_t[t].T * z_vec 
            
                # try Q[t], not Q[t + 1]
                
                Q_t[t] = npm.bmat([Q_t[t], z_dag_orthog/norm(z_dag_orthog)])
                                                 
                TR = npm.zeros((S_t[t].shape[0], 1))
                BL = npm.zeros((1 ,S_t[t].shape[1]))
                BR = mat(norm(z_dag_orthog) ** 2 )
                                
                S_t[t] = npm.bmat([[S_t[t],  TR],
                                   [  BL  ,  BR]])
                  
                rr.append(rr[-1] + 1)
            
            elif E_dash_t[-1] > energyThresh[1] * E_t[-1] and rr[-1] > 1 :
            
                Q_t[t] = Q_t[t][:, :-1]   # delete the last column of Q_t
        
                S_t[t] = S_t[t][:-1, :-1] # delete last row and colum of S_t 
        
                rr.append(rr[-1] - 1)
        
        else:
            
            # Record hidden variables
            hiddenV[t-1,:hh.shape[0]] = hh.T
            
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
            rr.append(rr[-1])  
            E_t.append(E_t[-1])        
            E_dash_t.append(E_dash_t[-1])              
            
            # increment count
            No_inp_count += 1                        
            
    return Q_t, S_t, rr, E_t, E_dash_t, hiddenV, z_dash, RSRE, No_inp_count
    
    
if __name__ == '__main__':

    # Initialise stream 
    series1 = Tseries(0)
    series2 = Tseries(0)

    series3 = Tseries(0)

    series1.makeSeries([3],[5],[2], gradient = 10, noise = 0.000000001)
    series2.makeSeries([3],[4],[2], gradient = 10, noise = 0.000000001)
    series3.makeSeries([3],[8],[2], gradient = 5, noise = 0.000000001)   
    
    series1.makeSeries([2,1,2],[95, 100, 100],[50, 50, 50], amp = 10,
                       noise = 0.00000001)
                       
    series2.makeSeries([2],[296],[40], amp = 10,
                   noise = 0.000000001, period = 10, phase = 5)
                   
    series3.makeSeries([2,4,2],[142, 10, 140],[40, 40, 2], gradient = 2,
                       amp = 10, noise = 0.00000001)                   
                   
                   
    streams = sp.c_[series1, series2, series3]        

    alpha = 0.96
    energyThresh = [0.96, 0.98]    
        
    Q_t, S_t, rr, E_t, E_dash_t,  \
    hiddenV, z_dash, RSRE, No_inp_count = FRAHST_M(streams, energyThresh, alpha)  
    
    # Calculate F_qq
    f_qq = npm.zeros(streams.shape[0] + 1)                                                                
    index = 0
    for q_t_i in Q_t:       
        X = q_t_i * q_t_i.T 
        FQQ = X - npm.eye(X.shape[0])  
        f_qq[0, index] = 10 * log10(trace(FQQ.T * FQQ))
        index += 1
        
    # Calculate E_qq
    e_qq = npm.zeros(streams.shape[0]) 
    
    for i in range(2,streams.shape[0]+1):
        data = streams[:i,:]
        cov_mat = npm.cov(data.T)
        W , V = npm.linalg.eig(cov_mat)
        Y = V * V.T - Q_t[i] * Q_t[i].T 
        e_qq[0, i-1] = 10 * log10(trace(Y.T * Y))
                                                
                                                                                                                  
    plot(streams)
    title('Input Data')
    figure()
    plot(z_dash)
    title('Reconstructed Data')
    figure()
    plot(hiddenV[:,0])
    title('Hidden Var 1')
    figure()
    plot(hiddenV[:,1])
    title('Hidden Var 2')    
    figure()    
    plot(hiddenV[:,2])
    title('Hidden Var 3')
    figure()
    plot(RSRE)
    title('Residual Squared Reconstruction Error')
    figure()    
    plot(E_t)
    plot(E_dash_t)
    title('Energy vs Reconstructed Energy') 
    figure()
    plot(f_qq.T)
    title('Deviation from orthonormality')    
    figure()
    plot(e_qq.T)
    title('Deviation of true tracked subspace')