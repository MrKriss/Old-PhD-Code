# -*- coding: utf-8 -*-
"""
Created on Thu Feb 03 20:56:15 2011

@author: musselle
"""

from numpy.linalg import norm

from Fraust_V2 import genSimSignalsA

def track_energy(streams, alpha):
    
    N = streams.shape[0]

    z_t_norm = []
    z_t_norm_sq = []
    
    for i in range(N):
        
        z_t_norm.append(norm(streams[i,:]))
        
        z_t_norm_sq.append(z_t_norm[-1] ** 2)


    # Calculate Energy 

    E_t = [0]         

    for i in range(1, N+1):
    
        temp_E1 = ((i - 1) * alpha * E_t[-1])/i
    
        temp_E2 = z_t_norm[N-1] / i
    
        E_t.append(temp_E1 + temp_E2)
        
    return E_t, z_t_norm, z_t_norm_sq
    
if __name__ == '__main__' : 
    
    streams = genSimSignalsA(0, -3.0)        
    
    alpha = 1.0
    
    E_t, z_t_norm, z_t_norm_sq = track_energy(streams, alpha)