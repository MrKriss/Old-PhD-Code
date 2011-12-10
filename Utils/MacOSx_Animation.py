# -*- coding: utf-8 -*-
"""
Created on Sat May 28 16:33:36 2011

@author: -
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 26 18:23:06 2011

@author: -
"""

import time, sys
import numpy as np
import matplotlib as mpl

import matplotlib.pyplot as plt

def data_gen_decay():
    t = data_gen_decay.t
    data_gen_decay.t += 0.05
    return np.sin(2*np.pi*t) * np.exp(-t/10.)

def ani_plot(*args):

    ani_plot.cnt = 0    
    
    background = fig.canvas.copy_from_bbox(ax.bbox)
    # for profiling
    tstart = time.time()

    while 1:
        # restore the clean slate background
        fig.canvas.restore_region(background)
        # update the data
        t = data_gen_decay.t
        y = data_gen_decay()
        xdata.append(t)
        ydata.append(y)
        xmin, xmax = ax.get_xlim()
        if t>=xmax:
            ax.set_xlim(xmin, 2*xmax)
            fig.canvas.draw()
            background = fig.canvas.copy_from_bbox(ax.bbox)

        line.set_data(xdata, ydata)

        # just draw the animated artist
        ax.draw_artist(line)
        # just redraw the axes rectangle
        fig.canvas.blit(ax.bbox)

        if ani_plot.cnt==1000:
            # print the timing info and quit
            print 'FPS:' , 1000/(time.time()-tstart)
            break
        ani_plot.cnt += 1

data_gen_decay.t = 0


# Figure Setup 
fig = plt.figure()
ax = fig.add_subplot(111)
line, = ax.plot([], [], animated=True, lw=2)
ax.set_ylim(-1.1, 1.1)
ax.set_xlim(0, 5)
ax.grid()
xdata, ydata = [], []
# Draw figure and show
fig.canvas.draw()
fig.show()

time.sleep(0.1)
ani_plot()   # Run animated plot.