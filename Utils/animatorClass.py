# -*- coding: utf-8 -*-
"""
Created on Sat May 28 16:33:36 2011

Incremental Plotting CLASS using MacOSX backend in matplotlib  

@author: - Musselle
"""

import time
import numpy as np
import matplotlib as mpl

mpl.use('TKAgg')

import matplotlib.pyplot as plt
import scipy.io as sio

class animator():

    def __init__(self, num_plots, lines_per_plot, rate_of_draw, titles = ['1','2','3']):
        """
        num_plots - number of subplot axes in figure (x11) to (x1x)
        lines_per_plot- number of lines to be animates in each subplot        
        
        """
        if num_plots != len(lines_per_plot): 
            print 'number of plots != number of line containers defined in plots'

        self.num_plots = num_plots
        self.lines_per_plot = lines_per_plot
        self.rate_of_draw = rate_of_draw
        self.plot_count = 0
                
        # Figure Setup 
        if num_plots == 1:            
            self.ydata1 = np.zeros((1,lines_per_plot[0]))  
            self.fig = plt.figure()
            self.ax1 = self.fig.add_subplot(111)
            self.lines1 = self.ax1.plot(self.ydata1, animated=True, lw=2)
            self.ax1.set_xlim(0, 50)
            self.ax1.grid()
            self.fig.canvas.draw()
            self.background1 = self.fig.canvas.copy_from_bbox(self.ax1.bbox)
            self.ax1.set_title(titles[0])
                    
        elif num_plots == 2:
            self.ydata1 = np.zeros((1,lines_per_plot[0]))  
            self.ydata2 = np.zeros((1,lines_per_plot[1]))
            
            self.fig = plt.figure()

            self.ax1 = self.fig.add_subplot(211)
            self.ax1.set_title(titles[0])
            self.ax2 = self.fig.add_subplot(212,  sharex=self.ax1)
            self.ax2.set_title(titles[1])
            
            self.lines1 = self.ax1.plot(self.ydata1, animated=True, lw=2)
            self.lines2 = self.ax2.plot(self.ydata2, animated=True, lw=2)
            
            self.ax1.set_xlim(0, 50)
            self.ax2.set_xlim(0, 50)
            
            self.ax1.grid()
            self.ax2.grid()
            
            self.fig.canvas.draw()
            self.background1 = self.fig.canvas.copy_from_bbox(self.ax1.bbox)
            self.background2 = self.fig.canvas.copy_from_bbox(self.ax2.bbox)

        elif num_plots == 3:
            self.ydata1 = np.zeros((1,lines_per_plot[0]))  
            self.ydata2 = np.zeros((1,lines_per_plot[1]))
            self.ydata3 = np.zeros((1,lines_per_plot[2]))
            
            self.fig = plt.figure()

            self.ax1 = self.fig.add_subplot(211)
            self.ax1.set_title(titles[0])
            self.ax2 = self.fig.add_subplot(212,  sharex=self.ax1)
            self.ax2.set_title(titles[1])
            self.ax3 = self.fig.add_subplot(313,  sharex=self.ax1)
            self.ax3.set_title(titles[2])
            
            self.lines1 = self.ax1.plot(self.ydata1, animated=True, lw=2)
            self.lines2 = self.ax2.plot(self.ydata2, animated=True, lw=2)
            self.lines3 = self.ax3.plot(self.ydata3, animated=True, lw=2)            
            
            self.ax1.set_xlim(0, 50)
            self.ax2.set_xlim(0, 50)
            self.ax3.set_xlim(0, 50)
            
            self.ax1.grid()
            self.ax2.grid()
            self.ax3.grid()            
            
            self.fig.canvas.draw()
            self.background1 = self.fig.canvas.copy_from_bbox(self.ax1.bbox)
            self.background2 = self.fig.canvas.copy_from_bbox(self.ax2.bbox)
            self.background3 = self.fig.canvas.copy_from_bbox(self.ax3.bbox)            
        
        # Draw figure and show
        self.fig.show()
        time.sleep(0.1)  

    def ani_plot(self, *args):
        """
        args is a list of the new data vectors/ points to be plotted.
        args length = len(lines_per_plot)
        """
        
        if len(args) != len(self.lines_per_plot): 
            print 'not enough input data for number of lines defined in plots'
    
        # update the data
        count = 0
        for arg in args:
            """ arg = new data vecor """
            count += 1
            
            # restore the clean slate background
            var_back = 'background' + str(count)
            self.fig.canvas.restore_region(vars(self)[var_back])            
            
            # Update Data    
            var_ydata = 'ydata' + str(count)
            vars(self)[var_ydata] = np.vstack((vars(self)[var_ydata], arg)) 
            
            # Adjust xlim
            var_ax = 'ax' + str(count)
            xmin, xmax = vars(self)[var_ax].get_xlim()
            if self.plot_count >= xmax :
                vars(self)[var_ax].set_xlim(xmin, 1.5 * xmax)
                self.fig.canvas.draw()
                # Save new background
                vars(self)[var_back] = self.fig.canvas.copy_from_bbox(vars(self)[var_ax].bbox)
                
            # Adjust ylim         
            ymin, ymax = vars(self)[var_ax].get_ylim()        
            data_max = np.nanmax(np.atleast_2d(arg), axis = 1)
            if data_max >= ymax :
                vars(self)[var_ax].set_ylim(ymin, data_max)
                
                # Redraw everything
                # set lines to not be animated                
#                var_lines = 'lines' + str(count)
#                for idx in range(len(vars(self)[var_lines])):
#                    vars(self)[var_lines][idx].set_animated(False)
                # Redraw Draw all 
                self.fig.canvas.draw() 
                
#                # Set lines to be animated again
#                for idx in range(len(vars(self)[var_lines])):
#                    vars(self)[var_lines][idx].set_animated(True)                
                
                # Save new background
                vars(self)[var_back] = self.fig.canvas.copy_from_bbox(vars(self)[var_ax].bbox)
            
            # Set the new data for each line and draw
            var_lines = 'lines' + str(count)
            for idx in range(len(vars(self)[var_lines])):
                # Set data 
                vars(self)[var_lines][idx].set_data(np.arange(self.plot_count + 2),
                                             vars(self)[var_ydata][:,idx])
                                             
                vars(self)[var_ax].draw_artist(vars(self)[var_lines][idx])

        # redder every 'rate_of_draw' loops
        if self.plot_count % self.rate_of_draw == 0 :
            for cnt in range(1, self.num_plots+1) :
                var_ax = 'ax' + str(cnt)
                vars(self)[var_ax].figure.canvas.blit(vars(self)[var_ax].bbox)        

#        elif flag == 1 : 
#            for cnt in range(1, self.num_plots+1):
#                var_ax = 'ax' + str(cnt)
#                vars(self)[var_ax].figure.canvas.blit(vars(self)[var_ax].bbox)
#                
        self.plot_count += 1

    def finalise(self):
        count = 0
        for plot_num in range(self.num_plots):
            count += 1
            var_lines = 'lines' + str(count)
            for line_num in range(self.lines_per_plot[plot_num]):
                vars(self)[var_lines][line_num].set_animated(False)
            

if __name__ == '__main__' : 

    # Data setup 
    AbileneMat = sio.loadmat('/Users/Main/DataSets/Abilene/Abilene.mat')
    data1 = AbileneMat['P'][:500, :3]
    data2 = AbileneMat['P'][:500, 3:8]
    data_iter1 = iter(data1)
    data_iter2 = iter(data2)    
    
    myplotter = animator(2, [data1.shape[1], data2.shape[1]], 5) 
        
    tstart = time.time()  # for profiling
    for i in range(data1.shape[0]) : 
        new_data1 = data_iter1.next()
        new_data2 = data_iter2.next()
        myplotter.ani_plot(new_data1, new_data2)   # Run animated plot.
        if i % 5 == 0 :
            print 'FPS: ' , myplotter.plot_count / (time.time()-tstart)
    print 'Average FPS: ' , myplotter.plot_count / (time.time()-tstart)
    
    myplotter.finalise()