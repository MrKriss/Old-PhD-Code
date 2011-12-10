# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 21:38:33 2010

SPIRIT Algorithm 
@author: musselle
"""

from ControlCharts import Tseries
import numpy.matlib as npm
from numpy import delete, ma, mat, multiply, nan
from numpy.linalg import norm, eig
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy
from plot_utils import adjust_spines, adjust_ticks

#===============================================================================
# Algorithm TrackW
#===============================================================================
def track_W(x_t_plus_1, k, pc_weights, d_i, num_streams, lamb):
    """
    Inputs --> x_t+1, k, pc_weights, energy
    Ouptputs --> pc_weights, 
    """  
                           
    # Calculate new hidden variables 
    y_t_i_final = npm.zeros(k)
    y_t_i_dash = npm.zeros(k)
    error = []

    # initialise x-dash-i
    x_dash_i = npm.zeros((k, num_streams))    
    x_dash_i[0] = mat(x_t_plus_1)

    # MAIN LOOP
    for i in range(k):  

        # Claculate temperary new hidden variable
        y_t_i_dash[0, i] = pc_weights[i] * x_dash_i[i].T
        
        # its energy 
        d_i[0, i] = (lamb * d_i[0, i]) + (y_t_i_dash[0, i] ** 2)
        
        # estimate error 
        error.append(x_dash_i[i] - y_t_i_dash[0, i] * pc_weights[i])
        
        # update PC estimate
        pc_weights[i] = pc_weights[i] + multiply(error[-1], 
                                        abs(y_t_i_dash[0, i])) / d_i[0, i] 
        
        # Calculate true hidden variable
        y_t_i_final[0, i] = pc_weights[i] * x_dash_i[i].T
        
        # update x_dash_i - the variability not yet accounted for
        if i != k-1: # not used on last loop 
            x_dash_i[i+1] = x_dash_i[i] - y_t_i_final[0, i] * pc_weights[i] 
    
    return pc_weights, y_t_i_final, error

#===============================================================================
# Algorithm SPIRIT
#===============================================================================
def SPIRIT(streams, energyThresh, lamb, evalMetrics):

    # Make 
    if type(streams) == np.ndarray:
        streams_iter = iter(streams)

    # Max No. Streams
    if streams.ndim == 1:
        streams = np.expand_dims(streams, axis=1)
        num_streams = streams.shape[1]
    else: 
        num_streams = streams.shape[1]

    count_over = 0
    count_under = 0

#===============================================================================
#      Initalise k, w and d, lamb
#===============================================================================

    k = 1 # Hidden Variables, initialise to one 
    
    # Weights
    pc_weights = npm.zeros(num_streams)
    pc_weights[0, 0] = 1
    
    # initialise outputs
    res = {}
    all_weights = []
    k_hist = []
    anomalies = []
    x_dash = npm.zeros((1,num_streams))
    
    Eng = mat([0.00000001, 0.00000001])    
    
    E_xt = 0  # Energy of X at time t
    E_rec_i = mat([0.000000000000001]) # Energy of reconstruction

    Y = npm.zeros(num_streams)
    
    timeSteps = streams.shape[0]
    
#===============================================================================
# Main Loop 
#===============================================================================
    for t in range(1, timeSteps + 1): # t = 1,...,200

        k_hist.append(k)

        x_t_plus_1 = mat(streams_iter.next()) # Read in next signals

        d_i = E_rec_i * t

        # Step 1 - Update Weights 
        pc_weights, y_t_i, error = track_W(x_t_plus_1, 
                                               k, pc_weights, d_i,
                                               num_streams, 
                                               lamb)
        # Record hidden variables
        padding = num_streams - k
        y_bar_t = npm.hstack((y_t_i, mat([nan] * padding)))
        Y = npm.vstack((Y,y_bar_t))
        
        # Record Weights
        all_weights.append(pc_weights)  
        # Record reconstrunted z and RSRE
        x_dash = npm.vstack((x_dash, y_t_i * pc_weights))
               
        # Record RSRE
        if t == 1:
            top = 0.0
            bot = 0.0
            
        top = top + (norm(x_t_plus_1 - x_dash) ** 2 )

        bot = bot + (norm(x_t_plus_1) ** 2)
        
        new_RSRE = top / bot   
                  
        if t == 1:
            RSRE = new_RSRE
        else:                  
            RSRE = npm.vstack((RSRE, new_RSRE))

        ### FOR EVALUATION ###
        #deviation from truth
        if evalMetrics == 'T' :
            
            Qt = pc_weights.T            
            
            if t == 1 :
                res['subspace_error'] = npm.zeros((timeSteps,1))
                res['orthog_error'] = npm.zeros((timeSteps,1))                


                res['angle_error'] = npm.zeros((timeSteps,1))
                Cov_mat = npm.zeros([num_streams,num_streams])
                
            # Calculate Covarentce Matrix of data up to time t   
            Cov_mat = lamb * Cov_mat +  npm.dot(x_t_plus_1,  x_t_plus_1.T)
            # Get eigenvalues and eigenvectors             
            W , V = eig(Cov_mat)
            # Use this to sort eigenVectors in according to deccending eigenvalue
            eig_idx = W.argsort() # Get sort index
            eig_idx = eig_idx[::-1] # Reverse order (default is accending)
            # v_r = highest r eigen vectors (accoring to thier eigenvalue if sorted).
            V_k = V[:, eig_idx[:k]]          
            # Calculate subspace error        
            C = npm.dot(V_k , V_k.T) - npm.dot(Qt , Qt.T)  
            res['subspace_error'][t-1,0] = 10 * np.log10(npm.trace(npm.dot(C.T , C))) #frobenius norm in dB
        
            # Calculate angle between projection matrixes
            D = npm.dot(npm.dot(npm.dot(V_k.T, Qt), Qt.T), V_k) 
            eigVal, eigVec = eig(D)
            angle = npm.arccos(np.sqrt(max(eigVal)))        
            res['angle_error'][t-1,0] = angle        
    
            # Calculate deviation from orthonormality
            F = npm.dot(Qt.T , Qt) - npm.eye(k)
            res['orthog_error'][t-1,0] = 10 * np.log10(npm.trace(npm.dot(F.T , F))) #frobenius norm in dB
              

        # Step 2 - Update Energy estimate
        E_xt = ((lamb * (t-1) * E_xt) + norm(x_t_plus_1) ** 2) / t
    
        for i in range(k):
            E_rec_i[0, i] = ((lamb * (t-1) * E_rec_i[0, i]) + (y_t_i[0, i] ** 2)) / t

        # Step 3 - Estimate the retained energy
        E_retained = npm.sum(E_rec_i,1)
    
        # Record Energy  
        Eng_new = npm.hstack((E_xt, E_retained[0,0]))
        Eng = npm.vstack((Eng, Eng_new))
    
        if E_retained < energyThresh[0] * E_xt:
            if k != num_streams:
                k = k + 1       
                # Initalise Ek+1 <-- 0 
                E_rec_i = npm.hstack((E_rec_i, mat([0]))) 
                # Initialise W_i+1
                new_weight_vec = npm.zeros(num_streams)  
                new_weight_vec[0, k-1] = 1
                pc_weights = npm.vstack((pc_weights, new_weight_vec))
                anomalies.append(t -1)
            else:
                count_over += 1
        elif E_retained > energyThresh[1] * E_xt:
            if k > 1 :
                k = k - 1
                # discard w_k and error
                pc_weights = delete(pc_weights, -1, 0)    
                # Discard E_rec_i[k]
                E_rec_i = delete(E_rec_i, -1)
            else:
                count_under += 1
          
          
    # Data Stores
    res2 = {'hidden' :  Y,                        # Array for hidden Variables
           'weights' : all_weights,
           'E_t' : Eng[:,0],                     # total energy of data 
           'E_dash_t' : Eng[:,1],                # hidden var energy
           'e_ratio' : np.divide(Eng[:,1], Eng[:,0]),      # Energy ratio 
           'RSRE' : RSRE,                        # Relative squared Reconstruction error 
           'recon' : x_dash,                     # reconstructed data
           'r_hist' : k_hist, # history of r values 
           'anomalies' : anomalies}  
           
    res.update(res2)
              
    return res, all_weights
        
#===============================================================================
# Initialise and test functions        
#===============================================================================
         
if __name__ == '__main__':

    plot_res = 1

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
                   
    series3.makeSeries([2,4,2],[142, 10, 140],[40, 40, 20], gradient = 2,
                       amp = 10, noise = 0.00000001)                   
                   
    # Concatonate streams                
    streams = scipy.c_[series1, series2, series3]        
    
    # run SPIRIT    
    energyThresh = [0.95, 0.99]
    lamb = 0.96    
    res_sp, all_weights = SPIRIT(streams, energyThresh, lamb, evalMetrics = 'T' )        
        
    # Plot Results
    if plot_res == True:
                
        # Plot data 
        fig = plt.figure()
        ax1 = fig.add_subplot(311)
        ax1.plot(res_sp['RSRE'])
        #plt.grid(True)
        
        ax2 = fig.add_subplot(312, sharex = ax1)
        ax2.plot(res_sp['e_ratio'])
        #plt.grid(True)
        
        ax3 = fig.add_subplot(313, sharex = ax1)
        ax3.plot(res_sp['orthog_error'])
        #plt.grid(True)
        
        # Shift spines
        adjust_spines(ax1, ['left', 'bottom'])
        adjust_spines(ax2, ['left', 'bottom'])        
        adjust_spines(ax3, ['left', 'bottom'])
        
        adjust_ticks(ax1, 'y', 5)
        adjust_ticks(ax2, 'y', 5)
        adjust_ticks(ax3, 'y', 5)
                        
        # Axis Labels
        plt.suptitle('Error Analysis of SPIRIT', size = 18, verticalalignment = 'top' )
        
        ylabx = -0.1        
        ax1.set_ylabel('RSRE', horizontalalignment = 'right', transform = ax1.transAxes )
        ax1.yaxis.set_label_coords(ylabx, 0.5)
        ax2.set_ylabel('Energy Ratio', horizontalalignment = 'right')
        ax2.yaxis.set_label_coords(ylabx, 0.5)
        ax3.set_ylabel('Orthogonality\nError (dB)', multialignment='center',
                       horizontalalignment = 'right')
        ax3.yaxis.set_label_coords(ylabx, 0.5)
        
        ax3.set_xlabel('Time Steps')        

#        ax1.text(-0.09, 0.5, 'RSRE', transform=ax1.transAxes, rotation=90,
#                                         ha='right', va='center')        
#        ax2.text(-0.09, 0.5, 'Energy Ratio', transform=ax2.transAxes, rotation=90,
#                                         ha='right', va='center')        
#        ax3.text(-0.09, 0.5, 'Orthogonality\nError (dB)', transform=ax3.transAxes, rotation=90,
#                                         ha='right', va='center', multialignment = 'center')

        # Set ylabel format  
#        formatter = mpl.ticker.Formatter                              
#        ax1.yaxis.set_major_formatter(formatter)
#        ax2.yaxis.set_major_formatter(formatter)
#        ax3.yaxis.set_major_formatter(formatter)

        # Remove superflous ink 
        plt.setp(ax1.get_xticklabels(), visible = False)
        plt.setp(ax2.get_xticklabels(), visible = False)    

        fig.show()