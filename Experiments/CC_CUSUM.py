# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 14:53:55 2010

@author: musselle
"""

from ControlCharts import Tseries
from CUSUM import cusum 
import numpy as np
import matplotlib.pyplot as plt

#===============================================================================
# Runscript 
#===============================================================================

#============
# Initialise
#============

'Create time series string'

series_1 = Tseries(0)

# NORMAL - normalEt(self, size, base=0, noise, noise_type = 'gauss')
series_1.normalEt(50, 0, 1,'gauss')
series_1.normalEt(50, 3, 1,'gauss')

# CYCLIC - cyclicEt(self, size, base=0, noise=2, amp=10, period=25, \
#                 noise_type = 'gauss'):
series_1.cyclicEt(100, 3, 1, 5, 50, 'gauss')

# NORMAL - 
series_1.normalEt(50, 3, 1,'gauss')
series_1.normalEt(50, -3, 1,'gauss')

# UP - upEt(self,size,base=0,noise=1,gradient=0.2, noise_type = 'gauss')
series_1.upEt(100, -3, 1, 0.2, 'gauss')

# Normal 
series_1.normalEt(50, 17, 1,'gauss')

# Dowm - downEt(self,size,base=0,noise=1,gradient=0.2, noise_type = 'gauss')
series_1.downEt(50, 17, 1, 0.4, 'gauss')

# Normal 
series_1.normalEt(50, 0, 1, 'gauss' )


stream = series_1

# Window sizes to try out
win_size = [10, 20, 40, 60]  

# Pre allocate memory 
score = []
tag = np.zeros((len(stream)))

sMaxScore = []  # Alternative for matlab Cell, a list 
tagMetric = []   
dataStr = [] # I use Similar to Matlab Structure 

#=======================#
# Sliding window Method #
#=======================#

# for each window size
for j in np.arange(len(win_size)):  # (0,...,N-1) --> 1:size(win_size,2)
    
    # slide along input stream 
    for i in np.arange(len(stream)-win_size[j]+1): #(1:size(stream,2)-win_size(j)+1)
        
        myInputs = stream[i:i + win_size[j]]
        
        #Need to store, win_size[j], start point i, masxS, row , col. 
        #Possible to store as list[[ i ],....,[ I ]] index by start point 
        #Add win_size later
        
        d = {}
        d['Start'] = i
        d['Win_size'] = win_size[j]
        d['Smax'], d['row'], d['col'] = cusum(myInputs)
        
        dataStr.append(d)
        score.append(d['Smax'])
        
        index = i + d['col'] - 1 # May need to add 1 here 
        tag[index] = tag[index] + d['Smax']   # Cummulative sum of Smax

    sMaxScore.append(score) # list of Smax for each win_size
    score = []                          # reset score
    tagMetric.append(tag)   # list of cumSUm of Smax
    tag = np.zeros((len(stream)))          # reset tage

# Plot results 

x = range(1, len(stream))

plt.figure(1)
plt.subplot(311)
plt.plot(stream)
plt.title('Input Stream')

plt.subplot(312)
for i in range(len(win_size)):                    #   i = 1:size(win_size,2)   
    plt.plot(sMaxScore[i])                           #'Color', ColOrd(i,:))
plt.title('Smax Score')

plt.subplot(313)
for i in range(len(win_size)):
    plt.plot(tagMetric[i])
plt.title('Tag')