# -*- coding: utf-8 -*-
"""
Created on Sat Mar 05 09:00:25 2011

@author: musselle
"""

from numpy import eye, zeros, dot, sqrt, nan, diag
import numpy as np
from numpy.linalg import eig, solve
from artSigs import genCosSignals_no_rand , genCosSignals
from utils import analysis, pltSummary, QRsolveA
import scipy.io as sio

# fast rank-adaptive row-Householder subspace tracker                               
def frahst_pedro_original(data, r=1, alpha=0.996, e_low=0.96, e_high=0.98, 
                          holdOffTime=0, evalMetrics='F', ignoreUp2 = 0): 
    
    """ Implimentation of Pedros version of the code.
    Only changes are:
        inclusion of iterable for data for speed/mem reasons; and
        No need to load/unload variables each loop.
        eval --> res
        Have not implimented QR.update or LU.update version. No need.    
    """    
    
    #for evaluation
    res = {'hidden' :  zeros([data.shape[0], data.shape[1]]) * nan, 
                'e_ratio' : zeros([data.shape[0],1]),
                'error' : zeros([data.shape[0],1]),
                'recon' : zeros([data.shape[0], data.shape[1]]),
                'angle_error' : zeros([data.shape[0],1]),
                'r_hist' : zeros([data.shape[0],1]),
                'anomalies' : [],
                'anomalous_streams' : [],
                'Q' : [],
                'S' : []}  
        
    #for energy ratio thresholding
    lastChangeAt = 1    
    sumYSq = 0
    sumXSq = 0
    
    # algorithm's state (constant memory)
    n = data.shape[1]    
    Q = eye(n)             # Note Q starts as Identity
    v = zeros((n, 1))            
    #initialize S matrix
    pI = eye(n) * 10 ** -4 #avoid singularity   
    S = pI            
    
    # Use iterable for data 
    iter_data = iter(data)     
    
    '''Simulation loop'''
    for t in range(1,data.shape[0] + 1):
  
        #data input vector
        z = iter_data.next()  
        
        # Convert to a column Vector 
        z = z.reshape(z.shape[0],1)   # now [ n x 1]
   
        #z = as.matrix( data[t,] )  
        
        #alias to matrices for current r
        Qt  = Q[:, :r]
        vt  = v[:r, :]
        St  = S[:r, :r]        
        
        h =  dot(Qt.T , z)               # r x 1        
        Z = dot(z.T, z)  - dot(h.T, h)   #scalar
        
        if Z > 0 :
        
            Z_sqrt = sqrt(Z)
            
            u = dot(St, vt)  # rx1 
               
            X = alpha * St + 2 * alpha * dot(u, vt.T) + dot(h , h.T)     #r x r

            #X -> b (solve  for b)
            ###################
            # Potential Error # 
            # A = X is used instead of A = X.T            
            ###################
            # Use standard numpy solver
            b = np.linalg.solve( X, Z_sqrt * h )   #rx1                
            
            B = 4 * (dot(b.T, b) + 1.0)  #scalar        
            phi_sqrt = 0.5 + 1.0/sqrt(B)        
            phi = sqrt(phi_sqrt)            
            gamma = ( 1 - 2 * phi_sqrt ) / (2 * phi )            
            delta = phi / Z_sqrt            
            vt = gamma * b  
            
            St = X - 1.0/delta * dot(vt, h.T)            
            
            w = delta * h - vt           # rx1
            e = delta * z - dot(Qt, w)   # nx1
            Qt = Qt - 2 * dot(e, vt.T)

        #restore data structures
        Q[:,:r] = Qt
        v[:r,:] = vt
        S[:r, :r] = St        
        
        '''FOR EVALUATION'''
        # Deviation from true dominant subspace
        if evalMetrics == 'T' :
            if t == 1 : 
                 Cxx = zeros([n,n])
                 res['deviation'] = zeros((data.shape[0],1))
            
            Cxx = alpha * Cxx +  dot(z,  z.T)
            W , V = eig(Cxx)
                
            ###################
            # Error RESOLVED - Not an Error after all#
            ################### 
            # In R eigen automaticly sorts the eigenvectors 
            # Use this to sort eigenVectors in according to deccending eigenvalue
            eig_idx = W.argsort() # Get sort index
            eig_idx = eig_idx[::-1] # Reverse order (default is accending)      
            # v_r = highest r eigen vectors (accoring to thier eigenvalue if sorted).
            V_r = V[:, eig_idx[:r]]            
        
            E = dot(V_r, V_r.T) - dot(Qt, Qt.T)
        
            deviation = np.sum(diag(dot(E.T, E)))            
            res['deviation'][t-1] = deviation
              
        # Store Values 
        z_hat = dot(Q[:,:r] , h)  # z reconstructed 
        res['hidden'][t-1, :r] = h.T[0,:] # t-1 for index adjustment. 
        res['recon'][t-1,:] = z_hat.T[0,:]
        
        # Record S & Q
        res['S'].append(St)
        res['Q'].append(Qt)
        
        ####################    
        # Potiential Error #  error is not the RSRE, but does not effect output
        ####################       
        res['error'][t-1] = np.sum((z_hat - z) ** 2) # Not normalised, so not RSRE
        angle_error = dot(Q[:, :r].T , Q[:, :r]) - eye(r)
        angle_error = sqrt(np.sum(diag(dot(angle_error.T,  angle_error)))) #frobenius norm
        res['angle_error'][t-1] = angle_error  # Akin to orthogonality error       
        res['r_hist'][t-1] = r
        
        ''' RANK ESTIMATION '''
        sumYSq = alpha * sumYSq + np.sum(h ** 2)
        sumXSq = alpha * sumXSq + np.sum(z ** 2)

        # check the lower bound of energy level
        if sumYSq < (e_low * sumXSq) and lastChangeAt < (t - holdOffTime) and r < n and t > ignoreUp2:
        
            #by deflation
            h_dash = dot(Q[:, :r].T,  z)
            z_bar = z - dot(Q[:, :r] , h_dash)
            z_bar = z_bar / sqrt(np.sum(z_bar ** 2))
            Q[:n, r+1-1] = z_bar.T[0,:]
            
            ####################
            # Potiential Error #   Uses the sum Squared not sum of the squares.
            ####################  
            s  = np.sum(z_bar) ** 2 
            
            ####################
            # Potiential Error #  Does not zero the last row and column of S, just
            ####################  changes last element             
            S[r, r] = s # Change the corner       
            
            r = r+1
     #       print "Increasing r to ", r," at time ", t, " (ratio energy", 100*sumYSq/sumXSq, ")\n"
            
            res['anomalies'].append(t-1)
            
            # Original sorted the orthogonal basis Q in decreasing order along 2nd axis 
            # Potential Error? # Cant find reference to it. Does nothing???? Q[:,x] are unit vectors  
            # Skip for now
            #res['anomalous_streams'][[t]] <- apply( Q[,1:r], 2, function(c) { sort(c, decreasing=TRUE, index.return=TRUE)$ix } )
         
            lastChangeAt = t
            
        # check the upper bound of energy level
        elif sumYSq > e_high*sumXSq and lastChangeAt < t - holdOffTime and r > 1 and t > ignoreUp2:

            lastChangeAt = t
            r = r-1
     #       print "Decreasing r to ", r," at time ", t, " (ratio energy", 100*sumYSq/sumXSq, ")\n"            
            
        res['e_ratio'][t-1] = sumYSq / sumXSq 
         
    return res


'''Fixed Version of Pedros FRAHST (Errors indicated)'''
def frahst_pedro_fixed(data, r=1, alpha=0.996, e_low=0.96, e_high=0.98, holdOffTime=0, evalMetrics='F', ignoreUp2 = 0): 
    """ Fixed version of Pedros version of the code.
    
    Fixed Version - changes are:
        - A = X.T instead of A = X
        - increasing rank of S now sets last col and row to zeros (except the diagonl)
        - The added diagonal is now sum of squares of z_perp, not the sum squared.
        
    """    
    
    #for evaluation
    res = {'hidden' :  zeros([data.shape[0], data.shape[1]]) * nan, 
                'e_ratio' : zeros([data.shape[0],1]),
                'error' : zeros([data.shape[0],1]),
                'recon' : zeros([data.shape[0], data.shape[1]]),
                'angle_error' : zeros([data.shape[0],1]),
                'r_hist' : zeros([data.shape[0],1]),
                'anomalies' : [],
                'anomalous_streams' : []}  
        
    #for energy test
    lastChangeAt = 1    
    sumYSq = 0
    sumXSq = 0
    
    #algorithm's state (constant memory)
    n = data.shape[1]    
    Q = eye(n)             # Note Q starts as Identity
    v = zeros((n, 1))            
    #initialize S matrix
    pI = eye(n) * 10 ** -4 #avoid singularity   
    S = pI            
    
    # Use iterable fro data 
    iter_data = iter(data)     
    
    #simulation loop
    for t in range(1,data.shape[0] + 1):
  
        #data input vector
        z = iter_data.next()  
        
        # Convert to a column Vector 
        z = z.reshape(z.shape[0],1)   #[ n x 1]
   
        #alias to matrices for current r
        Qt  = Q[:, :r]
        vt  = v[:r, :]
        St  = S[:r, :r]        
        
        h =  dot(Qt.T , z)  # r x 1        
        Z = dot(z.T, z)  - dot(h.T, h)   #scalar
        
        if Z > 0 :
        
            Z_sqrt = sqrt(Z)
            
            u = dot(St, vt)  # rx1 
               
            X = alpha * St + 2 * alpha * dot(u, vt.T) + dot(h , h.T)     #r x r

            #X -> b (solve  for b)
            ###################
            # Error # A = X is used instead of the correct A = X.T   
            # NOTE: corrected in this version         
            ###################
            b = np.linalg.solve( X.T, Z_sqrt * h )   #rx1                
            
            B = 4 * (dot(b.T, b) + 1.0)  #scalar        
            phi_sqrt = 0.5 + 1.0/sqrt(B)        
            phi = sqrt(phi_sqrt)            
            gamma = ( 1 - 2 * phi_sqrt ) / (2 * phi )            
            delta = phi / Z_sqrt            
            vt = gamma * b  
            
            St = X - 1.0/delta * dot(vt, h.T)            
            
            w = delta * h - vt    # rx1
            e = delta * z - dot(Qt, w)   # nx1
            Qt = Qt - 2 * dot(e, vt.T)
                
        #restore data structures
        Q[:,:r] = Qt
        v[:r,:] = vt
        S[:r, :r] = St        
        
        ''' FOR EVALUATION '''
        #deviation from truth
        if evalMetrics == 'T' :
            if t == 1 : 
                 Cxx = zeros([n,n])
                 res['deviation'] = zeros((data.shape[0],1))
            
            Cxx = alpha * Cxx +  dot(z,  z.T)
            W , V = eig(Cxx)
                
            # The eigen function in R automaticly sorts the eigenvectors 
            # Use the following to sort eigenVectors in according to deccending eigenvalue
            eig_idx = W.argsort() # Get sort index
            eig_idx = eig_idx[::-1] # Reverse order (default is accending)      
            # V_r = highest r eigen vectors (accoring to thier eigenvalue if sorted in decending order).
            V_r = V[:, eig_idx[:r]]             
        
            E = dot(V_r, V_r.T) - dot(Qt, Qt.T)
        
            deviation = np.sum(diag(dot(E.T, E)))            
            res['deviation'][t-1] = deviation
              
        # Store Values 
        z_hat = dot(Q[:,:r] , h)  # z reconstructed 
        res['hidden'][t-1, :r] = h.T[0,:] # t-1 for index adjustment. 
        res['recon'][t-1,:] = z_hat.T[0,:]
        
        ####################    
        # Potiential Error #   error is not the RSRE, but does not effect output
        ####################       
        res['error'][t-1] = np.sum((z_hat - z) ** 2) # the un-normalised error  
        angle_error = dot(Q[:, :r].T , Q[:, :r]) - eye(r)
        angle_error = sqrt(np.sum(diag(dot(angle_error.T,  angle_error)))) #frobenius norm
        res['angle_error'][t-1] = angle_error        
        res['r_hist'][t-1] = r
        
        ''' RANK ESTIMATION '''
        sumYSq = alpha * sumYSq + np.sum(h ** 2)
        sumXSq = alpha * sumXSq + np.sum(z ** 2)

        # check the lower bound of energy level
        if sumYSq < (e_low * sumXSq) and lastChangeAt < (t - holdOffTime) and r < n and t > ignoreUp2 :
        
            #by deflation
            h_dash = dot(Q[:, :r].T,  z)
            z_bar = z - dot(Q[:, :r] , h_dash)
            z_bar = z_bar / sqrt(np.sum(z_bar ** 2))
            Q[:n, r+1-1] = z_bar.T[0,:]
            
            ####################
            # Potiential Error #   Uses the sum Squared not sum of the squares.
            ####################    # Corrected to sum of squares in this version
            s  = np.sum(z_bar ** 2) 
            
            ####################
            # Potiential Error #  Does not zero the last row and column of S, just
            ####################  changes last diagonal element 
            # Corrected # 
            S[:r, r] = 0 # zero the  rth column, leaves corner
            S[r, :r] = 0 # zeroes the rth row, leaves corner             
            S[r, r] = s # Change the corner       
            
            r = r+1

     #       print "Increasing r to ", r," at time ", t, " (ratio energy", 100*sumYSq/sumXSq, ")\n"
            
            res['anomalies'].append(t-1)
            
            lastChangeAt = t
            
        # check the upper bound of energy level
        elif sumYSq > e_high*sumXSq and lastChangeAt < t - holdOffTime and r > 1 and t > ignoreUp2 :

            lastChangeAt = t
            r = r-1
     #       print "Decreasing r to ", r," at time ", t, " (ratio energy", 100*sumYSq/sumXSq, ")\n"            
            
        res['e_ratio'][t-1] = sumYSq / sumXSq 
         
    return res

if __name__ == '__main__' : 
    
    # streams = genSimSignalsA(0, -3.0)    
    # streams = array([[0,0,0], [1,2,2], [1,3,4], [3,6,6], [5,6,10], [6,8,11]])   
    # streams = genCosSignals(timesteps = 20, N = 10)    
    
    AbileneMat = sio.loadmat('/Users/chris/DataSets/Abilene/Abilene.mat')
    data = AbileneMat['P']
    
    n = data.shape[0]    
    
    e_high = 0.98
    e_low = 0.95
    alpha = 0.96

    holdOFF = 0    
    
    energyThresh = [0.96, 0.98]
    alpha = 0.96
    # rr = 4

    # Pedros Original Version 
    res_ped_orig = frahst_pedro_original(data, r=1, alpha=alpha, e_low=e_low, e_high=e_high,
                           holdOffTime=holdOFF, evalMetrics='F')    
    res_ped_orig['Alg'] = 'Pedros Original Implimentation of FRAUST'    
    pltSummary(res_ped_orig, data)
    
    # Pedros Version Fixed by me 
    res_ped_fix = frahst_pedro_fixed(data, r=1, alpha=alpha, e_low=e_low, e_high=e_high,
                           holdOffTime=holdOFF, evalMetrics='F')
    res_ped_fix['Alg'] = 'Pedros Version (Fixed) of FRAUST'    
    pltSummary(res_ped_fix, data)
   

