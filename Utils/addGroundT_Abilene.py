# -*- coding: utf-8 -*-
"""
Created on Mon Mar 07 04:14:06 2011

@author: musselle
"""

import scipy.io as sio
from numpy import array

AbileneMat = sio.loadmat('/Users/Main/DataSets/Abilene/Abilene.mat')

P_g_truths = array([[647, 2],
                    [708, 3],
                    [777, 4],
                    [1138, 1],
                    [1100, 2],
                    [1270, 2],
                    [1456, 6],
                    [1615, 5],
                    [1723, 4],
                    [1907, 3],
                    [1952, 3]])
                    
P_g_truths_alt = array([[411, 3],   
                        [708,  3],   
                        [777,  4],  
                        [1100, 2],  
                        [1158, 2],   
                        [1270, 1],   
                        [1456, 5],   
                        [1615, 9],   
                        [1723, 2],   
                        [1907, 3],   
                        [1952, 3]])   
                    
AbileneMat['P_g_truth_tab'] = P_g_truths
AbileneMat['P_g_truth_tab_alt'] = P_g_truths_alt

A = []
for line in P_g_truths :
    for n in range(0,line[1]) :
        A.append(line[0] + n)       
        
AbileneMat['P_g_truth_vec'] = A          
        
#####################################

F_g_truths = array([[708, 2],
                    [862, 6],
                    [1100, 2],
                    [1160, 2],
                    [1181, 1],
                    [1384, 1],
                    [1434, 6],
                    [1546, 3],
                    [1618, 3],
                    [1625, 1],
                    [1723, 3],
                    [1869, 1],
                    [1906, 4],
                    [1968, 1]])
                    
                    
F_g_truths_alt = array([[28, 1],   
                        [206, 10],  
                        [247, 7],   
                        [294, 4],   
                        [300, 5],   
                        [331, 3],   
                        [578, 2],   
                        [708, 2],   
                        [862, 21],  
                        [1100, 2],   
                        [1160, 2],   
                        [1181, 1],   
                        [1384, 1],   
                        [1434, 6],   
                        [1546, 3],   
                        [1618, 3],   
                        [1625, 1],   
                        [1723, 3],   
                        [1896, 1],   
                        [1906, 4],   
                        [1968, 1]])   
                    
                    
AbileneMat['F_g_truth_tab'] = F_g_truths
AbileneMat['F_g_truth_tab_alt'] = F_g_truths_alt
              
B = []
for line in F_g_truths :
    for n in range(0,line[1]) :
        B.append(line[0] + n)     
        
AbileneMat['F_g_truth_vec'] = B

sio.savemat('/Users/Main/DataSets/Abilene/Abilene.mat', AbileneMat)

