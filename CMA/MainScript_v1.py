# -*- coding: utf-8 -*-
"""
Created on Tue May 10 19:36:29 2011

Master Script for CMA with synthetic data

@author: - Musselle
"""

from numpy import array, vstack, eye, zeros, atleast_2d, arange,  \
transpose, concatenate, empty 
from numpy.linalg import norm 
from Frahst_single_itter import FRAHST_itter
import numpy as np
import os
import pickle as pk
from MyNodes import FrahstNode, MA_Node
from time import time
from animatorClass import animator

# Generate / load data #########################################

# Load OD network dataset from folder.
os.chdir('/Users/Chris/DataSets/Python/Network_ODsmall')

# The packet flows vector
with open('P_t_data.pk', 'r') as data_file :
    P_t = pk.load(data_file)

# The Link flows vector
with open('Y_t_data.pk', 'r') as data_file :
    Y_t = pk.load(data_file)
       
# Create data Iterators
P_t_iter = iter(P_t)
Y_t_iter = iter(Y_t)

# Initalise Frahst Parameters ##################################

# t_steps = Y_t.shape[0] 
t_steps = 500
MA_output = [10,50,90,130]

F1 = FrahstNode(0.96, 0.96, 0.98)   
# create 1 MA node per link
MA = {}
for i in range(P_t.shape[1]):
    MA[i] = MA_Node(0.98)
      
start_time = time()      
 
# Link to monitour
link_num = 1
     
for t in range(t_steps): 
    
    # Input data for time t
    Y_data_vec = Y_t_iter.next()
    P_data_vec = P_t_iter.next()

    # Input to Frahst Node 
    r, hidden, recon, E_x, E_y, anomaly = F1._execute(Y_data_vec)

    # Input to Sequence buffer
    for i in range(P_t.shape[1]):
        MA[i]._execute(P_data_vec[i])
        
        #TODO: Increment Plot of CMA and EMA for chosen link
        # set up animator
        if t == 0 and i == link_num - 1 : 
            MAtitles = ['CMA', 'EMA'] 
            MAPlotter = animator(2, [len(MA[i].MAs), len(MA[i].MAs)], 5, MAtitles)
            tstart = time()  # for profiling
        # only for chosen link 
        if i == link_num - 1 :
            CMA = []
            EMA = []
            for k,v in MA[i].MAs.iteritems():
                CMA.append(v['CMA'])
                EMA.append(v['EMA'])
            MAPlotter.ani_plot(np.array(CMA), np.array(EMA))   # Run animated plot.
    
    # Incremental plot of rank and hidden variables
    if t == 0 :
        Ftitles = ['Rank', 'Hidden Variables']
        FrahstPlotter = animator(2, [1, hidden.shape[1]], 5, Ftitles)
        tstart = time()  # for profiling
       
    FrahstPlotter.ani_plot(np.array([r]), hidden)   # Run animated plot.

    print t, 'th time step complete'
 
run_time = time() - start_time
print 'Total runtime = ', run_time
# Current Runtime is approx 30 sec
print 'Average FPS: ' , FrahstPlotter.plot_count / (time()-tstart)  

FrahstPlotter.finalise()
MAPlotter.finalise()