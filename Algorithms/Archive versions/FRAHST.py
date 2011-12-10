# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 14:30:44 2010

@author: musselle
"""

from ControlCharts import Tseries
from numpy import hstack, vstack, sqrt, expand_dims, dot, array, zeros
from numpy.linalg import norm , inv, qr
from numpy.random import rand 
from matplotlib.pyplot import plot, figure
import scipy as sp

def FRAHST(streams, energyThresh, alpha):
    """ Fast rank adaptive row-Householder Subspace Traking Algorithm   
    
    Working on allowing to take File or array input     
    
    """
#    
#    if type(streams) == file :
#    
#    elif type(streams) == array :
#--------------------------------------
    
    #Initialise 
    N = streams.shape[1]
    rr = [1]
    
    hiddenV = zeros((streams.shape[0], N))
    count = 0
    
    # generate random orthonormal  - N x r 
    qq,RR = qr(rand(N,1))    
    Q_t = [qq]
    # Q_t[0] = expand_dims(Q_t[0], axis=1)  
    
    S_t = [0.000001] 
    E_t = [0]
    E_dash_t = [0]
    
    iter_streams = iter(streams)
    
    for t in range(1, streams.shape[0] + 1):
        
        z_vec = iter_streams.next()
        
        z_vec = expand_dims(z_vec, axis=1)        
        
        hh = dot(Q_t[t-1].T , z_vec)                       # 13a

        Z = dot(z_vec.T, z_vec) - dot(hh.T, hh)           # 13b
        
        if Z > 0.0000001:        
            
            X = alpha * S_t[t-1] + dot(hh , hh.T)               # 13c
        
            # X.T * b = sqrt(Z) * hh                           # 13d        
        
            b = dot((inv(X.T) * sqrt(Z)), hh)  # inverse method 
        
            phi_sq_t = 0.5 + 1 / sqrt(4 * (dot(b.T, b) + 1))   # 13e

            phi_t = sqrt(phi_sq_t)        

            delta = phi_t / sqrt(Z)                        # 13f
        
            v = (1 - 2 * phi_sq_t) / (2 * phi_t)         #13 g
        
            v = v * b  
        
            S_t.append( X - (1/delta) * dot(v, hh.T))              # 13 h  S_t[t] = 
            
            e = delta * z_vec - dot(Q_t[t-1], (delta * hh - v))    # 13 i
        
            Q_t.append(Q_t[t-1] - 2 * dot(e, v.T))                 # 13 j  Q[t] = 

            E_t.append(alpha * E_t[-1] + norm(z_vec) ** 2)        # 13 k
                
            E_dash_t.append( alpha * E_dash_t[-1] + norm(hh) ** 2)  # 13 l
        
            if E_dash_t[-1] < energyThresh[0] * E_t[-1] and rr[-1] < N: # 13 m 
        
                z_dag_orthog =  z_vec - dot(dot(Q_t[t], Q_t[t].T), z_vec) 
            
                # try Q[t], not Q[t + 1]
                Q_t[t] = hstack((Q_t[t], z_dag_orthog/norm(z_dag_orthog))) 

                top = hstack((S_t[t], zeros((S_t[t].shape[1], 1))))
                
                br = array([[norm(z_dag_orthog) ** 2 ]])

                bot = hstack((zeros((1 ,S_t[t].shape[0])), br))
            
                S_t[t] = vstack((top,bot))          
            
                # S_t[t] = vstack((hstack((S_t[t], [[0]]))), 
                  #          hstack(([[0]], norm(z_dag_orthog) ** 2 )))
                  
                rr.append(rr[-1] + 1)
            
            elif E_dash_t[-1] > energyThresh[1] * E_t[-1] and rr[-1] > 1 :
            
                Q_t[t] = Q_t[t][:, :-1]   # delete the last column of Q_t
        
                S_t[t] = S_t[t][:-1, :-1] # delete last row and colum of S_t 
        
                rr.append(rr[-1] - 1)
        
        else:
            # Repeat last entries
            count = count + 1
            Q_t.append(Q_t[-1])
            S_t.append(S_t[-1])            
            rr.append(rr[-1])            
        
        hiddenV[t-1,:len(hh[0])] = hh[0]      
        
    return Q_t, S_t, rr, E_t, E_dash_t, hiddenV, count
    
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
                   
    series3.makeSeries([2,4,2],[142, 10, 140],[40, 40, 2], gradient = 20,
                       amp = 10, noise = 0.00000001)                   
                   
                   
    streams = sp.c_[series1, series2, series3]        

    alpha = 0.96
    energyThresh = [0.95, 0.99]    
        
    Q_t, S_t, rr, E_t, E_dash_t, hiddenV, count = FRAHST(
                                            streams, energyThresh, alpha)  

    



     
    