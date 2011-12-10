# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 16:02:33 2011

@author: -
"""

from numpy import dot, sqrt, log10, trace, arccos, nan, vstack, zeros, eye, \
array, allclose
from numpy.linalg import qr, eig, norm, solve
from matplotlib.pyplot import plot, figure, title, step
#from artSigs import genCosSignals_no_rand, genCosSignals
from utils import analysis, QRsolveA, pltSummary
import scipy.io as sio
from PedrosFrahst import frahst_pedro
from Frahst_v3 import FRAHST_V3
from Frahst_v3_1 import FRAHST_V3_1

def FRAHST_itter(data_vec, Q, S, v, E_x, E_y, param):
    """
        Fast Rank Adaptive Householder Subspace Tracking Algorithm (FRAHST)  
    
    Version 4.0 - Combines good bits of Pedros version, with my correction of the bugs
    
    Keeps Sci as -1 so like 3.1, but ONLY ITERATES ONCE.    
    
    # INPUTS 
        data_vec - New data at x_t+1
        S_t = S at t minus 1
        Q_t = Q at t minus 1
        V_t = V at t minus 1
        E_y = energy of hidden variables 
        E_x = energy of data_vectors  
        param - dict{ 'alpha' : alpha   ,  
                     'e_high' : e_high  ,
                     'e_low' :  e_low   ,
                     'r'    : r 
                     't' : timestep,
                     'lastChangeAt' : lastChangeAt , 
                     'holdTime' : hold_off_time,
                     'evalMetrics' : evaluate_metrics}
        
        Note: r and lastChangeAt are altered in run
        
    PROGRESS: Succesfuly carris out Frahst_v3_1 iteratively. 
                      
    NOTE: have altered Z calculation slightly.
    Z = sqrt(zT * z) - sqrt(hT * h)                      
                      
    """   
    
    # Initialise variables and data structures 
    #########################################

    # Derived Variables 
    numStreams = len(data_vec) 
    anomaly = False
    
    # Passed parameters for this itteration
    alpha = param['alpha']
    e_high = param['e_high']
    e_low = param['e_low']
    r = param['r']    
    t = param['t']    
    
    #alias to matrices for current r
    Qt  = Q[:, :r]
    vt  = v[:r, :]
    St  = S[:r, :r]
    
    ##################
    # One Itteration #
    ##################
    zt = data_vec
        
    # Convert to a column Vector if not already 
    zt = zt.reshape(zt.shape[0],1) 
    
    ht = dot(Qt.T , zt) 
    
    # EXPERIMENTAL CODE
    # Q is not orthodonal, and is causing negative Z    
    
#    if not allclose(dot(Qt.T, Qt), eye(r)):    
#        Z = dot((zt.T - dot(ht.T, Qt.T)), (zt - dot(Qt, ht)))  
#        print 'QTQ - I != 0'      
#    else:
#        Z = dot(zt.T,  zt) - dot(ht.T , ht)
    
    Z = abs(dot(zt.T,  zt) - dot(ht.T , ht))
    
    if Z > 0 :
            
        # Refined version, sci accounts better for tracked eigenvalues
        u_vec = dot(St , vt)            
        X = alpha * St + 2 * alpha * dot(u_vec, vt.T) + dot(ht , ht.T) 
                           
        # Solve Ax = b using QR updates - not strictly needed 
        A = X.T
        B = sqrt(Z) * ht
        b_vec = QRsolveA(A,B)  

        beta  = 4 * (dot(b_vec.T , b_vec) + 1)
    
        phi_sq = 0.5 + (1 / sqrt(beta))
    
        phi = sqrt(phi_sq)

        gamma = (1 - 2 * phi_sq) / (2 * phi)
        
        delta = phi / sqrt(Z)
        
        vt = gamma * b_vec 
        
        St = X - ((1 /delta) * dot(vt , ht.T))
        
        w = (delta * ht) - (vt) 
        
        ee = delta * zt - dot(Qt , w) 
        
        Qt = Qt - 2 * dot(ee , vt.T) 
    
        # Restore Values 
        Q[:,:r] = Qt
        v[:r,:] = vt
        S[:r, :r] = St            
    
        ################        
        # Store Values #
        ################
        
        # Record reconstrunted z
        recon = dot(Qt , ht)
        
        # Record hidden variables
        # Pad with nans
        hidden = zeros((1,numStreams)) * nan     
        hidden[0, :r] = ht.T[0,:]

        ###################
        # Rank Estimation #
        ###################

        # Calculate energies 
        E_x = alpha * E_x + sum(zt ** 2) # Energy of Data
        E_y = alpha * E_y + sum(ht ** 2) # Energy of hidden Variables
        
        # Adjust Q_t ans St for change in rr 
        if E_y < (e_low * E_x) and param['lastChangeAt'] < (t - param['holdOffTime']) and r < numStreams :
            
            # Note indexing with r works like r + 1 as index is from 0 in python
                        
            # Extend Q by z_bar
            h_dash = dot(Q[:, :r].T,  zt)
            z_bar = zt - dot(Q[:, :r] , h_dash)
            z_bar = z_bar / norm(z_bar)
            Q[:numStreams, r] = z_bar.T[0,:]
            
            s_end  = sum(z_bar ** 2)
            
            # Set next row and column to zero
            S[r, :] = 0.0
            S[:, r] = 0.0
            S[r, r] = s_end # change last element
            
            # new r, increment
            param['r'] = r + 1
 #           print "Increasing r to ", r," at time ", t, " (ratio energy", 100*E_y/E_x, ")\n"
            
            # Record time step of anomaly            
            anomaly = True

            # Reset lastChange             
            param['lastChangeAt'] = t
            
        elif E_y > e_high*E_x and param['lastChangeAt'] < t - param['holdOffTime'] and r > 1 :

            # Reset lastChange
            param['lastChangeAt'] = t
            # new r, decrement
            param['r'] = r - 1
  #          print "Decreasing r to ", r," at time ", t, " (ratio energy", 100*E_y/E_x, ")\n"            
            
            # No need to change S and Q as r index is decreased   
    
    else: 
        
        ###############################        
        #  When Z < 0 #
        ###############################
        
        # Record reconstrunted z
        recon = dot(Qt , ht)
        
        # Record hidden variables
        # Pad with nans
        hidden = zeros((1,numStreams)) * nan     
        hidden[0, :r] = ht.T[0,:]

    
    return Q, S, v, E_x, E_y, recon, hidden, anomaly   
    
    
#===============================================================================
# IF Main Method     
#===============================================================================
#
#Compares my original Frahst implimentation with sci = 1,-1 and 0 with this 
#incrimental one,alonside pedros, and my second implimentation  

if __name__ == '__main__' :

    ''' The following compares alll versions of Frahst so far '''
    

    # Data setup 
    AbileneMat = sio.loadmat('/Users/Main/DataSets/Abilene/Abilene.mat')
    data = AbileneMat['P']
    data_iter = iter(AbileneMat['P'])
    
    # Parameter Setup 
    param  = { 'alpha' : 0.96  ,  
              'e_high' : 0.98  ,
              'e_low' :  0.96   ,
              'r'    : 1 , 
              't' : 0 ,
              'lastChangeAt' : 1 , 
              'holdOffTime' : 0,
              'evalMetrics' : False}   
    
    numStreams = AbileneMat['P'].shape[1]          
              
    # Data Stores
    res = {'hidden' :  zeros((1, numStreams)) * nan,  # Array for hidden Variables
           'E_x' : array([0]),                     # total energy of data 
           'E_y' : array([0]),                # hidden var energy
           'e_ratio' : zeros([1, 1]),              # Energy ratio 
           'RSRE' : zeros([1, 1]),           # Relative squared Reconstruction error 
           'recon' : zeros([1, numStreams]),  # reconstructed data
           'r_hist' : zeros([1, 1]), # history of r values 
           'anomalies' : []}  
    
    # Initialisations 
    # set inital S and Q   
    # Q_0
    Q = eye(numStreams)       
    S = eye(numStreams) * 0.0001 # Avoids Singularity    
    v = zeros((numStreams,1)) 

    # Simulate continuous data
    for i in xrange(2010):
        param['t'] += 1 # Increment t
        data_vec = data_iter.next()    
    
        # run 1 line Frahst.     
        Q, S, v, E_x, E_y, recon, hidden, anomaly  = FRAHST_itter(data_vec,
                                    Q, S, v, res['E_x'][-1], res['E_y'][-1], param)
    
    # Post Processing and Data Storeage
        if param['t'] == 1:
            top = 0.0
            bot = 0.0
                
        top = top + (norm(data_vec - recon) ** 2 )
        bot = bot + (norm(data_vec) ** 2)
        res['RSRE'] = vstack((res['RSRE'], top / bot))        
        res['e_ratio'] = vstack((res['e_ratio'], E_x / E_y))    
        res['hidden'] = vstack((res['hidden'], hidden)) 
        res['r_hist'] = vstack((res['r_hist'], param['r']))
        res['E_x'] = vstack((res['E_x'], E_x)) 
        res['E_y'] = vstack((res['E_y'], E_y)) 
        res['recon'] = vstack((res['recon'], recon.T)) 
        if anomaly : 
            res['anomalies'].append(param['t'])
            
    res['Alg'] = 'My Incremetal Implimentation of FRAUST'

    pltSummary(res, data)
    
    #################### End of incremental version of Frahst #############
    
    # My Original version 
    res_me1 = FRAHST_V3(data, r = 1, 
                        alpha = param['alpha'], 
                        e_low = param['e_low'], 
                        e_high = param['e_high'],
                        holdOffTime=param['holdOffTime'],
                        fix_init_Q = 1, 
                        sci = 1,
                        evalMetrics = 'F') 

    res_me1['Alg'] = 'My First Implimentation of FRAUST sci = 1'
    
    pltSummary(res_me1, data)
    
    # My Original version 
    res_me2 = FRAHST_V3(data, r = 1, 
                        alpha = param['alpha'], 
                        e_low = param['e_low'], 
                        e_high = param['e_high'],
                        holdOffTime=param['holdOffTime'],
                        fix_init_Q = 1, 
                        sci = 0,
                        evalMetrics = 'F') 

    res_me2['Alg'] = 'My First Implimentation of FRAUST sci = 0'
    
    pltSummary(res_me2, data)
    
    # My Original version 
    res_me3 = FRAHST_V3(data, r = 1, 
                        alpha = param['alpha'], 
                        e_low = param['e_low'], 
                        e_high = param['e_high'],
                        holdOffTime=param['holdOffTime'],
                        fix_init_Q = 1, 
                        sci = -1,
                        evalMetrics = 'F') 

    res_me3['Alg'] = 'My First Implimentation of FRAUST sci = -1'
    
    pltSummary(res_me3, data)
    
    # My second version
    
    res_me2 = FRAHST_V3_1(data, r = 1,
                        alpha = param['alpha'], 
                        e_low = param['e_low'], 
                        e_high = param['e_high'],
                        holdOffTime=param['holdOffTime'],
                        fix_init_Q = 1, 
                        evalMetrics = 'F')   
    
    res_me2['Alg'] = 'My Second Implimentation of FRAUST'
    
    pltSummary(res_me2, data)
    
    # Pedros Version
    res_ped = frahst_pedro(data, r = 1,
                            alpha = param['alpha'],
                            energy_low = param['e_low'],
                            energy_high = param['e_high'],
                            holdOffTime=param['holdOffTime'],
                            evaluateTruth='FALSE')

    res_ped['Alg'] = 'Pedros Implimentation of FRAUST'

    pltSummary(res_ped, data)
    