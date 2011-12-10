# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 13:05:27 2010

@author: musselle
"""

from ControlCharts import Tseries 
# import numpy as np
# from matplotlib.pyplot import plot, figure
import scipy

# Initialise stream 
series1 = Tseries(0)
series2 = Tseries(0)

series3 = Tseries(0)

series1.makeSeries([3],[5],[2], gradient = 10, noise = 0.000000001)
series2.makeSeries([3],[4],[2], gradient = 10, noise = 0.000000001)
series3.makeSeries([3],[8],[2], gradient = 5, noise = 0.000000001)   

series1.makeSeries([2,1,2],[95, 100, 100],[50, 50, 50], amp = 10,
                   noise = 0.00000001)
                   
series2.makeSeries([2],[296],[40], amp = 10,
               noise = 0.000000001, period = 10, phase = 5)
               
series3.makeSeries([2,4,2],[142, 10, 140],[40, 40, 20], gradient = 2,
                   amp = 10, noise = 0.00000001)  
                   
streams = scipy.c_[series1, series2, series3]   