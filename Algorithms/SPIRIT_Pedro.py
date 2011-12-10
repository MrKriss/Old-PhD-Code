# -*- coding: utf-8 -*-
"""
Created on Wed May 18 13:26:17 2011


Pedros Version of SPIRIT

@author: -
"""

import numpy.matlib as npm
from numpy import delete, ma, mat, multiply, sqrt, nan, diag, eye, matrix
from numpy.linalg import norm 
import numpy as np
from matplotlib.pyplot import plot, figure
import scipy
from ControlCharts import Tseries


def SPIRIT_pedro(A, k0 = 1, lamb=0.96, holdOffTime=0, 
           energy_low=0.95, energy_high=0.98):

    n = A.shape[1]
    totalTime = A.shape[0]
    
    W = npm.eye(n)      #initialize the k (up to n) w_i to unit vectors
    d = 0.01 * npm.ones((n, 1)) #energy associated with given eigenvalue of covariance X_t'X_t
    m = k0              # number of eigencomponents
        
    sumYSq=0
    sumXSq=0
    
    #data structures for evaluating (~ totalTime)
    print "Running incremental simulation on ", n, " streams with total of ", totalTime, "ticks.\n"
    anomalies = []
    hidden = npm.zeros((totalTime, n)) * nan
    m_hist = npm.zeros((totalTime, 1)) * nan
    ratio_energy_hist = npm.zeros((totalTime, 1)) * nan
    Proj = npm.zeros((totalTime, n))
    recon = npm.zeros((totalTime, n))    
    relErrors = npm.zeros((totalTime, 1))
    W_hist = []    
    errors = npm.zeros((totalTime, 1)) 
    angle_error = []
    E_t = []
    E_dash_t = []
  
    
    #incremental update W
    lastChangeAt = 1
    for t in range(totalTime):      
        #actual vector (transposed) of the current time
        xActual = matrix(A[t,:]).T  
                
        #project to m-dimensional space    
        Y = W[:,m:] * xActual
        
        #reconstruction of the current time
        xProj = W[:,:m] * Y
        Proj[t,:m] = Y
        recon[t,:] = xProj
        xOrth = xActual - xProj        
        errors[t] = sum(xOrth**2)
        relErrors[t] = sum(xOrth**2) / sum(xActual**2)

        #update W for each y_t        
        x = xActual        
        for j in range(m):       
            w,d,x = updateW(x, W[:,j], d[j], lamb)
            W[:,j] = w  
            d[j] = d
            x = x
            
        #keep the weights orthogonal 
        #if(qr) {                
#            W[,1:m] <- qr.Q(qr(W[,1:m]))*-1  
  #      }

        #eval
        Y = W[:, :m].T * xActual
        hidden[t, :m] = Y
        ang_err = W[:, :m].T * W[:, :m] - eye(m)
        ang_err = sqrt(sum(diag(ang_err.T * ang_err))) #frobenius norm        
        angle_error[t] = ang_err
        
        # Record RSRE
        if t == 1:
            top = 0.0
            bot = 0.0
            
        top = top + (norm(xActual - xProj) ** 2 )

        bot = bot + (norm(xActual) ** 2)
        
        new_RSRE = top / bot   
                  
        if t == 1:
            RSRE = new_RSRE
        else:                  
            RSRE = npm.vstack((RSRE, new_RSRE))        
        

        #update energy
        sumYSq = lamb * sumYSq + sum(Y ** 2)
        sumXSq = lamb * sumXSq + sum(xActual ** 2)
        
        E_t.append(sumXSq)
        E_dash_t.append(sumYSq)
        
        #for evaluating:
        m_hist[t] = m        
        ratio_energy_hist[t] = sumYSq/sumXSq


        # check the lower bound of energy level
        if (sumYSq < energy_low * sumXSq and 
                    lastChangeAt < t - holdOffTime and m < n) :
            lastChangeAt = t
            m = m + 1
            print "Increasing m to ", m," at time ", t, " (ratio energy", \
                    100*sumYSq/sumXSq, ")\n"
            print "Max stream for each hidden variable", \
                    (W[:,:m].T).argmax(axis=0), "\n"
            anomalies.append(t)
            W_hist.append(W[:,:m])

        # check the upper bound of energy level
        elif (sumYSq >= energy_high * sumXSq and 
                    lastChangeAt < t - holdOffTime and  m > 1):
            lastChangeAt = t 
            m = m - 1
            print "Increasing m to ", m," at time ", t, " (ratio energy", \
                    100*sumYSq/sumXSq, ")\n"
            print "Max stream for each hidden variable", \
                    (W[:,:m].T).argmax(axis=0), "\n"
            
            W_hist.append(W[:,:m])
        
    # Data Stores
    res = {'hidden' :  hidden,
            'Proj' : Proj,                        # Array for hidden Variables
           'weights' : W_hist,
           'E_t' : np.array(E_t),                     # total energy of data 
           'E_dash_t' : np.array(E_dash_t),                # hidden var energy
           'e_ratio' : ratio_energy_hist,
           'relErrors' : relErrors,  
           'errors' : errors,           # Energy ratio 
           'RSRE' : RSRE,                        # Relative squared Reconstruction error 
           'recon' : recon,                     # reconstructed data
           'r_hist' : m_hist, # history of r values 
           'angle_err' : angle_error,
           'anomalies' : anomalies}  
        
    return res

#old_x and old_w are column vectors
def updateW(old_x, old_w, old_d, lamb):
    #projection onto w
    y = old_w.T * old_x
    #energy value
    d = lamb * old_d + y**2
    # error (orthogonal to w)
    e = old_x - old_w * y
    # update PC estimal
    w = old_w + e * y / d
    x = old_x - w * y
    #keep it normal
    w = w/(sqrt(sum(w**2)))
    return w,d,x

if __name__ == '__main__' :
    
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
                   

#temp_streams = scipy.c_[series1, series2]

# min-max Normalisation 

#mymin = np.min(temp_streams) 
#mymax = np.max(temp_streams) 

#series1 = (series1 - mymin) / (mymax - mymin) 
#series2 = (series2 - mymin) / (mymax - mymin) 

# min-max Normalisation 
#series1 = (series1 - np.min(series1)) / (np.max(series1) - np.min(series1)) 
#series2 = (series2 - np.min(series2)) / (np.max(series2) - np.min(series2)) 

    streams = scipy.c_[series1, series2, series3]        
    
    res = SPIRIT_pedro(streams, k0 = 1, lamb=0.96, holdOffTime=0, 
           energy_low=0.95, energy_high=0.98)
