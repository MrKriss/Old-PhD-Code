# -*- coding: utf-8 -*-
"""
Created on Wed May 18 13:26:17 2011

Reimplimentation of the SPIRIT algorithm by Spiros Papadimitriou, 
Jimeng Sun and Christos Faloutsos available at;
http://www.cs.cmu.edu/afs/cs/project/spirit-1/www/

@author: - Chris Musselle
"""

import numpy.matlib as npm
import numpy as np
import matplotlib.pyplot as plt
from load_data import load_data
from utils import pltSummary2


def updateW(old_x, old_w, old_d, lamb):
    ''' w = new weight 
        d = new energy 
        x = remaining x 
    '''
    y = old_w.T * old_x
    d = lamb * old_d + y ** 2
    e = old_x - old_w * y
    w = old_w + e * y / d
    x = old_x - w * y
    w = w / npm.linalg.norm(w)
    
    return w,d,x
    
    
def SPIRIT(A, lamb, energy, k0 = 1, holdOffTime = 0, reorthog = False, evalMetrics = 'F'):

    A = np.mat(A)

    n = A.shape[1]
    totalTime = A.shape[0]
    Proj = npm.ones((totalTime, n)) * np.nan
    recon = npm.zeros((totalTime, n))
    
    # initialize w_i to unit vectors
    W = npm.eye(n)
    d = 0.01 * npm.ones((n, 1))
    m = k0  # number of eigencomponents
    
    relErrors = npm.zeros((totalTime, 1))
    
    sumYSq = 0.
    E_t = []
    sumXSq = 0.
    E_dash_t = []
    
    res = {}
    k_hist = []
    W_hist = []
    anomalies = []
    
    # incremental update W
    lastChangeAt = 0
        
    for t in range(totalTime):
        
        k_hist.append(m)        
        
        # update W for each y_t
        x = A[t,:].T # new data as column vector 
      
        for j in range(m):
            W[:,j], d[j], x = updateW(x, W[:,j], d[j], lamb)
            Wj = W[:,j]
    
        # Grams smit reorthog  
        if reorthog == True:
            W[:,:m], R = npm.linalg.qr(W[:,:m])
      
        # compute low-D projection, reconstruction and relative error
        Y = W[:, :m].T * A[t, :].T  # project to m-dimensional space
        xActual = A[t, :].T        # actual vector of the current time
        xProj = W[:, :m] * Y       # reconstruction of the current time
        Proj[t, :m] = Y.T 
        recon[t,:] = xProj.T
        xOrth = xActual - xProj
        relErrors[t] = npm.sum(npm.power(xOrth, 2)) / npm.sum(npm.power(xActual, 2))
    
        # update energy
        sumYSq = lamb * sumYSq + npm.sum(npm.power(Y, 2))
        E_dash_t.append(sumYSq)
        sumXSq = lamb * sumXSq + npm.sum(npm.power(A[t,:] , 2))
        E_t.append(sumXSq)
    
        # Record RSRE
        if t == 0:
            top = 0.0
            bot = 0.0
            
        top = top + npm.power(npm.linalg.norm(xActual - xProj) , 2 )

        bot = bot + npm.power(npm.linalg.norm(xActual) , 2)
        
        new_RSRE = top / bot   
                  
        if t == 0:
            RSRE = new_RSRE
        else:                  
            RSRE = npm.vstack((RSRE, new_RSRE))

        ### Metric EVALUATION ###
        #deviation from truth
        if evalMetrics == 'T' :
            
            Qt = W[:,:m]            
            
            if t == 0 :
                res['subspace_error'] = npm.zeros((totalTime,1))
                res['orthog_error'] = npm.zeros((totalTime,1))                
                res['angle_error'] = npm.zeros((totalTime,1))
                Cov_mat = npm.zeros([n,n])
                
            # Calculate Covarentce Matrix of data up to time t   
            Cov_mat = lamb * Cov_mat +  npm.dot(xActual,  xActual.T)
            # Get eigenvalues and eigenvectors             
            WW , V = npm.linalg.eig(Cov_mat)
            # Use this to sort eigenVectors in according to deccending eigenvalue
            eig_idx = WW.argsort() # Get sort index
            eig_idx = eig_idx[::-1] # Reverse order (default is accending)
            # v_r = highest r eigen vectors (accoring to thier eigenvalue if sorted).
            V_k = V[:, eig_idx[:m]]          
            # Calculate subspace error        
            C = npm.dot(V_k , V_k.T) - npm.dot(Qt , Qt.T)  
            res['subspace_error'][t,0] = 10 * np.log10(npm.trace(npm.dot(C.T , C))) #frobenius norm in dB
            # Calculate angle between projection matrixes
            D = npm.dot(npm.dot(npm.dot(V_k.T, Qt), Qt.T), V_k) 
            eigVal, eigVec = npm.linalg.eig(D)
            angle = npm.arccos(np.sqrt(max(eigVal)))        
            res['angle_error'][t,0] = angle        
    
            # Calculate deviation from orthonormality
            F = npm.dot(Qt.T , Qt) - npm.eye(m)
            res['orthog_error'][t,0] = 10 * np.log10(npm.trace(npm.dot(F.T , F))) #frobenius norm in dB
            
        # Energy thresholding 
        ######################
        # check the lower bound of energy level
        if sumYSq < energy[0] * sumXSq and lastChangeAt < t - holdOffTime and m < n :
            lastChangeAt = t
            m = m + 1
            anomalies.append(t)
           # print 'Increasing m to %d at time %d (ratio %6.2f)\n' % (m, t, 100 * sumYSq/sumXSq)
        # check the upper bound of energy level
        elif sumYSq > energy[1] * sumXSq and lastChangeAt < t - holdOffTime and m < n and m > 1 :
            lastChangeAt = t
            m = m - 1
           # print 'Decreasing m to %d at time %d (ratio %6.2f)\n' % (m, t, 100 * sumYSq/sumXSq)  
        W_hist.append(W[:,:m])
    # set outputs
    
    # Grams smit reorthog  
    if reorthog == True:
        W[:,:m], R = npm.linalg.qr(W[:,:m])
   
    # Data Stores
    res2 = {'hidden' :  Proj,                        # Array for hidden Variables
           'E_t' : np.array(E_t),                     # total energy of data 
           'E_dash_t' : np.array(E_dash_t),                # hidden var energy
           'e_ratio' : np.array(E_dash_t) / np.array(E_t),  # Energy ratio 
           'rel_orth_err' : relErrors,          # orthoX error
           'RSRE' : RSRE,                        # Relative squared Reconstruction error 
           'recon' : recon,                     # reconstructed data
           'r_hist' : k_hist, # history of r values 
           'W_hist' : W_hist, # history of Weights
           'anomalies' : anomalies}  
           
    res.update(res2)

    return res
    
    
if __name__ == '__main__' : 
    
    #A1 = np.mat(np.sin(np.arange(1,1001)/20.0)).T 
    #A11 =  A1 * npm.rand((1,10))
    #plt.figure()
    #plt.plot(A11)

    #a1 = np.sin(npm.arange(1,1001)/10.0)
    #a2 = np.sin(npm.arange(1,1001)/50.0)
    #A2 = np.mat([a1,a2]).T * npm.rand((2,10))
    #plt.figure()    
    #plt.plot(A2)

    data = load_data('chlorine')
    data[1500:2000, 0:4] = 0.7
    data = data[:,:15]
    
    res = SPIRIT(data,1,[0.8,0.99]) 
    res['Alg'] = 'SPIRIT'
    pltSummary2(res, data, (0.8, 0.99))
    
    
    