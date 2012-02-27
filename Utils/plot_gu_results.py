#!/usr/bin/env python
#coding:utf-8
# Author:  C Musselle --<>
# Purpose: script for plotting of Gus results 
# Created: 02/06/12

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sys
import os 

from plot_utils import adjust_spines, adjust_ticks

# Define Data 
data = np.loadtxt('../icarisrets2011.csv', delimiter=',')
LC = data[:,0]
DCA1 = data[:,1]
DCA2 = data[:,2]
DMOV1 = data[:,3]
DMOV2 = data[:,4]
SMOV = data[:,5]

# Plot Data 
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(LC, 'bs:')
ax1.plot(DCA1, 'r*:')
ax1.plot(DMOV1, 'ko:')
ax1.plot(SMOV, 'g^:')

# Shift spines and renumber ticks 
adjust_spines(ax1, ['left', 'bottom'])

# Set limits and labels 
ax1.set_ylim(-0.01, 0.6)
ax1.set_xlim(-1, 100)

ax1.set_xlabel('Euclidian Distance', size= 16)  
ax1.set_ylabel('Error Rate', size = 16)  

plt.legend(ax1.lines, ['Linear SVM', 'DCA', 'Static Window', 'Dynamic Window'], shadow = True)