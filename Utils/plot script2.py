# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 13:55:23 2011

@author: -
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

# Plot data 
fig = plt.figure()
ax1 = fig.add_subplot(411)
ax1.plot(streams)
#plt.grid(True)

ax2 = fig.add_subplot(412, sharex = ax1)
ax2.plot(data['hidden'])
#plt.grid(True)

ax3 = fig.add_subplot(413, sharex = ax1)
ax3.plot(data['e_ratio'])
#plt.grid(True)
plt.axhline(y=e_high, ls = '--')
plt.axhline(y=e_low, ls = '--')

ax4 = fig.add_subplot(414, sharex = ax1)
ax4.plot(data['orthog_error'])


# Shift spines
adjust_spines(ax1, ['left', 'bottom'])
adjust_spines(ax2, ['left', 'bottom'])        
adjust_spines(ax3, ['left', 'bottom'])
adjust_spines(ax4, ['left', 'bottom'])


adjust_ticks(ax1, 'y', 5)
adjust_ticks(ax2, 'y', 5)
adjust_ticks(ax3, 'y', 5)
adjust_ticks(ax4, 'y', 5)

ax1.set_xlim([0, streams.shape[0]])
            
# Axis Labels
plt.suptitle('Error Analysis of SPIRIT', size = 18, verticalalignment = 'top' )

ylabx = -0.13
ax1.set_ylabel('Input Data', horizontalalignment = 'center', transform = ax1.transAxes )
ax1.yaxis.set_label_coords(ylabx, 0.5)
ax2.set_ylabel('Hidden\nVariables', horizontalalignment = 'center')
ax2.yaxis.set_label_coords(ylabx, 0.5)
ax3.set_ylabel('Energy\nRatio', multialignment='center',
           horizontalalignment = 'center')
ax3.yaxis.set_label_coords(ylabx, 0.5)           
ax4.set_ylabel('Orthonormality\nError (dB)', multialignment='center',
           horizontalalignment = 'center')
ax4.yaxis.set_label_coords(ylabx, 0.5)

ax4.set_xlabel('Time Steps')        

# Must draw canvas before editing text
fig.canvas.draw()
for ax in fig.axes :
    fmat = ax.yaxis.major.formatter
    if fmat.orderOfMagnitude >= fmat._powerlimits[1] \
                or fmat.orderOfMagnitude <= fmat._powerlimits[0] :
                            
        offset_text = str(ax.yaxis.offsetText._text) 
        ax.yaxis.offsetText.set_visible(False)              
        string = offset_text.split('e')        
        ax.set_ylabel(ax.get_ylabel() + ' $(10^{%s})$' % string[1])

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
plt.setp(ax3.get_xticklabels(), visible = False) 

fig.show()