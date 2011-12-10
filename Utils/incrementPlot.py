# -*- coding: utf-8 -*-
"""
Created on Sat May 28 16:33:36 2011

Incremental Plotting Function using MacOSX backend in matplotlib  

@author: - Musselle
"""

import time
import numpy as np
import matplotlib as mpl

mpl.use('TKAgg')

import matplotlib.pyplot as plt
import scipy.io as sio


def ani_plot(new_data, background, rate):  
    # restore the clean slate background
    fig.canvas.restore_region(background)
    
    # update the data
    ani_plot.ydata = np.vstack((ani_plot.ydata, new_data))   
    ani_plot.xdata = np.arange(ani_plot.cnt + 2)    
    
    # Adjust xlim
    xmin, xmax = ax.get_xlim()
    if ani_plot.cnt >= xmax:
        ax.set_xlim(xmin, 1.5*xmax)
        fig.canvas.draw()
        background = fig.canvas.copy_from_bbox(ax.bbox)
        
    # Adjust ylim         
    ymin, ymax = ax.get_ylim()
    data_max = max(new_data)
    if data_max >= ymax :
        ax.set_ylim(ymin, data_max)
        fig.canvas.draw()
        background = fig.canvas.copy_from_bbox(ax.bbox)

    # Set the new data for each line and draw
    for idx in range(len(lines)):
        # Set data 
        lines[idx].set_data(ani_plot.xdata, ani_plot.ydata[:,idx])
        ax.draw_artist(lines[idx])
    
    # just redraw the axes rectangle
    if ani_plot.cnt % rate == 0 :
        fig.canvas.blit(ax.bbox)

    ani_plot.cnt += 1
    
    return background


if __name__ == '__main__' : 

    # Data setup 
    AbileneMat = sio.loadmat('/Users/Main/DataSets/Abilene/Abilene.mat')
    data = AbileneMat['P'][:,:3]
    data_iter = iter(data)
    ani_plot.ydata = np.zeros((1,data.shape[1]))  
    
    # Figure Setup 
    fig = plt.figure()
    ax = fig.add_subplot(111)
    lines = ax.plot(ani_plot.ydata, animated=True, lw=2)
    
    ax.set_xlim(0, 50)
    ax.grid()
    fig.canvas.draw()
    background = fig.canvas.copy_from_bbox(ax.bbox)

    # Draw figure and show
    fig.show()
    time.sleep(0.1)          
    
    ani_plot.cnt = 0      # conter for number of frames plotted  
    tstart = time.time()  # for profiling
    rate = 5
    for new_data in data_iter: 
        background = ani_plot(new_data, background, rate)   # Run animated plot.
        
    print 'FPS:' , ani_plot.cnt / (time.time()-tstart)  
        
        