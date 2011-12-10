# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 16:07:31 2010

@author: musselle
"""
from numpy import  zeros
from numpy.linalg import norm 
from viewResults import viewResult

def diff_search(time_interval, matrix_list):
    '''
    Function to search for differences between recorded data matrixes
    '''
    
    #var = viewResult(run, variable)
    var = matrix_list

    norm_var = zeros((1,time_interval[1] - time_interval[0] + 1))
    cum_diff_var = zeros((1,time_interval[1] - time_interval[0] + 1))
    
    cum_diff = var[time_interval[0]]  
    
    index = 0

    for i in range(time_interval[0], time_interval[1]+1):
        X = var[i]
        cum_diff = cum_diff - norm_var[0, index]        
        cum_diff_var[0, index] = cum_diff
        index += 1

    return   norm_var, cum_diff_var      

def diff_proj_M(time_interval, run):
    
    Q_t = viewResult(run, 'Q_t')
    Vr_t = viewResult(run, 'V_r_record')

    Q_proj_t = []
    Q_proj_norm = zeros((1,time_interval[1] - time_interval[0] + 1)) 
    Vr_proj_t = []
    Vr_proj_norm = zeros((1,time_interval[1] - time_interval[0] + 1))     
    
    diff_t = [] 
    diff_norm = zeros((1,time_interval[1] - time_interval[0] + 1))
    
    index = 0

    for i in range(time_interval[0], time_interval[1]+1):
        
        Q_proj_t.append(Q_t[i] * Q_t[i].T)
        Q_proj_norm[0, index] =  norm(Q_proj_t[-1])
        
        Vr_proj_t.append(Vr_t[i] * Vr_t[i].T)
        Vr_proj_norm[0, index] =  norm(Vr_proj_t[-1])
        
        diff_t.append(Vr_proj_t[-1] - Q_proj_t[-1])
        diff_norm[0, index] =  norm(diff_t[-1])
        
        index += 1

    return   diff_norm,  Q_proj_norm , Vr_proj_norm, diff_t, Q_proj_t, Vr_proj_t 
    

#if __name__ == '__main__' :
 #   norm_var, cum_diff_var = diff_search((465,473), 'Q_t',1)
    