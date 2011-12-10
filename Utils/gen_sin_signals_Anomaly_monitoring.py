#!/usr/bin/env python
#coding:utf-8
# Author:  C Musselle --<>
# Purpose: Generate Synthetic data streams to test for anomaly detection and monitouring applications
# Created: 11/18/11

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sys
import os 
from ControlCharts import Tseries

"""
Code Description:
  .
"""

s0 = Tseries(0)
s1 = Tseries(0)
s2 = Tseries(0)
s3 = Tseries(0)
s4 = Tseries(0)
s5 = Tseries(0)
s6 = Tseries(0)
s7 = Tseries(0)
s8 = Tseries(0)
s9_lin = Tseries(0)
s9_sin = Tseries(0)

s0.makeSeries([2], [1000], amp = 1, noise = 0.25)
s1.makeSeries([2], [1000], amp = 1, noise = 0.25)
s2.makeSeries([2], [1000], amp = 1, noise = 0.25)
s3.makeSeries([2], [1000], amp = 1, noise = 0.25)
s4.makeSeries([2], [1000], amp = 1, noise = 0.25)
s5.makeSeries([2], [1000], amp = 1, noise = 0.25)
s6.makeSeries([2], [1000], amp = 1, noise = 0.25)
s7.makeSeries([2], [1000], amp = 1, noise = 0.25)
s8.makeSeries([2], [1000], amp = 1, noise = 0.25)

s9_lin.makeSeries([1,4,1,3,1] , [400,10,290,10,290], base = [0, 0, -2, -2, 0], gradient = 0.2, noise_type = 'none') 
s9_sin.makeSeries([2] ,[1000], base = [0], amp = 1, noise = 0.25)

s9 = np.array(s9_lin) + np.array(s9_sin)

data = sp.r_['1,2,0', s0,s1,s2,s3,s4,s5,s6,s7,s8,s9]
