# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 14:52:49 2011

@author: -
"""
import matplotlib as mpl
import matplotlib.pyplot as plt 
import pickle as pk
import numpy as np

def adjust_spines(ax,spines):
    for loc, spine in ax.spines.iteritems():
        if loc in spines:
            spine.set_position(('outward',10)) # outward by 10 points
            spine.set_smart_bounds(False)
        else:
            spine.set_color('none') # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])


def adjust_ticks(ax, axis, num_ticks):
    ''' Adjust the number tof ticks to num_ticks'''
    from matplotlib.ticker import MaxNLocator
    
    if axis == 'x' :
        ax.xaxis.set_major_locator(MaxNLocator(num_ticks))
    elif axis == 'y' :
        ax.yaxis.set_major_locator(MaxNLocator(num_ticks))
    

def plot_2x1(data1, data2, ylab = '', xlab = '', main_title = '', ylims = 0):
    ''' Nice format to plot two subplotted time series'''
            
    # Plot data 
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.plot(data1)
    #plt.grid(True)
    if ylims != 0:
        if ylims[0] == 1 :
            plt.ylim(ylims[1], ylims[2])        
    
    ax2 = fig.add_subplot(212, sharex = ax1)
    ax2.plot(data2)
    #plt.grid(True)
    if ylims != 0:
        if ylims[0] == 2 :
            plt.ylim(ylims[1], ylims[2])
    
    # Shift spines and renumber ticks 
    adjust_spines(ax1, ['left', 'bottom'])
    adjust_spines(ax2, ['left', 'bottom'])        
    
    adjust_ticks(ax1, 'y', 6)
    adjust_ticks(ax2, 'y', 6)

    # Labels
    fig.suptitle( main_title, size = 18, verticalalignment = 'top' )

    ax2.set_xlabel(xlab)        
    ax1.text(-0.11, 0.5, ylab[0], transform=ax1.transAxes, rotation=90,
                                     ha='right', va='center')        
    ax2.text(-0.11, 0.5, ylab[1], transform=ax2.transAxes, rotation=90,
                                     ha='right', va='center')        
    # Remove superflous ink 
    plt.setp(ax1.get_xticklabels(), visible = False)

    fig.show()

    
def plot_3x1(data1, data2, data3, ylab = '', xlab = '', main_title = ''):
            
    # Plot data 
    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax1.plot(data1)
    #plt.grid(True)
    
    ax2 = fig.add_subplot(312, sharex = ax1)
    ax2.plot(data2)
    #plt.grid(True)
    
    ax3 = fig.add_subplot(313, sharex = ax1)
    ax3.plot(data3)
    #plt.grid(True)
    
    # Shift spines and renumber ticks 
    adjust_spines(ax1, ['left', 'bottom'])
    adjust_spines(ax2, ['left', 'bottom'])        
    adjust_spines(ax3, ['left', 'bottom'])
    
    adjust_ticks(ax1, 'y', 5)
    adjust_ticks(ax2, 'y', 5)
    adjust_ticks(ax3, 'y', 5)
                    
    # Labels
    plt.suptitle( main_title, size = 18, verticalalignment = 'top' )

    ax3.set_xlabel(xlab)        
    ax1.text(-0.09, 0.5, ylab[0], transform=ax1.transAxes, rotation=90,
                                     ha='right', va='center')        
    ax2.text(-0.09, 0.5, ylab[1], transform=ax2.transAxes, rotation=90,
                                     ha='right', va='center')        
    ax3.text(-0.09, 0.5, ylab[2], transform=ax3.transAxes, rotation=90,
                                     ha='right', va='center', multialignment = 'center')
    # Remove superflous ink 
    plt.setp(ax1.get_xticklabels(), visible = False)
    plt.setp(ax2.get_xticklabels(), visible = False)    

    fig.show()
    
def plot_4x1(data1, data2, data3, data4, ylab = '', xlab = '', main_title = '', ylims = 0):

    # Plot data 
    fig = plt.figure()
    ax1 = fig.add_subplot(411)
    ax1.plot(data1)
    if ylims != 0:
        if ylims[0] == 1 :
            plt.ylim(ylims[1], ylims[2])  
    
    ax2 = fig.add_subplot(412, sharex = ax1)
    ax2.plot(data2)
    if ylims != 0:
        if ylims[0] == 2 :
            plt.ylim(ylims[1], ylims[2])  
    
    ax3 = fig.add_subplot(413, sharex = ax1)
    ax3.plot(data3)
    if ylims != 0:
        if ylims[0] == 3 :
            plt.ylim(ylims[1], ylims[2])  
    
    ax4 = fig.add_subplot(414, sharex = ax1)
    ax4.plot(data4)
    if ylims != 0:
        if ylims[0] == 4 :
            plt.ylim(ylims[1], ylims[2])  
    
    # Shift spines
    adjust_spines(ax1, ['left', 'bottom'])
    adjust_spines(ax2, ['left', 'bottom'])        
    adjust_spines(ax3, ['left', 'bottom'])
    adjust_spines(ax4, ['left', 'bottom'])
    
    adjust_ticks(ax1, 'y', 5)
    adjust_ticks(ax2, 'y', 5)
    adjust_ticks(ax3, 'y', 5)
    adjust_ticks(ax4, 'y', 5)
    
    ax1.set_xlim([0, data1.shape[0]])
                
    # Labels
    plt.suptitle( main_title, size = 18, verticalalignment = 'top' )
    
    ax4.set_xlabel(xlab)        
    ax1.text(-0.09, 0.5, ylab[0], transform=ax1.transAxes, rotation=90,
             ha='right', va='center')        
    ax2.text(-0.09, 0.5, ylab[1], transform=ax2.transAxes, rotation=90,
             ha='right', va='center')        
    ax3.text(-0.09, 0.5, ylab[2], transform=ax3.transAxes, rotation=90,
             ha='right', va='center', multialignment = 'center')    
    ax4.text(-0.09, 0.5, ylab[3], transform=ax3.transAxes, rotation=90,
             ha='right', va='center', multialignment = 'center')    
    
    # Remove superflous ink 
    plt.setp(ax1.get_xticklabels(), visible = False)
    plt.setp(ax2.get_xticklabels(), visible = False)    
    plt.setp(ax3.get_xticklabels(), visible = False) 
    
    fig.show()    