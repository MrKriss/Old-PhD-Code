#!/usr/bin/env python
#coding:utf-8
# Author:   --<>
# Purpose: 
# Created: 12/07/11

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sys
import os 

"""
Code Description:
  .
"""

def gen_anom_peak()

#s1 = Tseries(0)
        s2 = Tseries(0)
        #s1.makeSeries([2,1,2], [300, 300, 300], noise = 0.5, period = 50, amp = 5)
        #s2.makeSeries([2], [900], noise = 0.5, period = 50, amp = 5)
        #data = sp.r_['1,2,0', s1, s2]
        
        s0lin = Tseries(0)
        s0sin = Tseries(0)
        s2lin = Tseries(0)
        s2sin = Tseries(0)
        s3 = Tseries(0)
        s4 = Tseries(0)
        
        interval_length = 300
        l = 10
        m = 10
        baseLine = 0
        amp = 5
        period = 50 
        s0lin.makeSeries([1,3,4,1], [interval_length, l/2, l/2, 2 * interval_length - l], 
                        [baseLine, baseLine, baseLine + m, baseLine], 
                        gradient = float(m)/float(l/2), noise = 0.5)
        s0sin.makeSeries([2], [3 * interval_length], [0.0], 
                        amp = amp, period = period, noise = 0.5)

        # sum sin and linear components to get data stream                         
        s1 = np.array(s0lin) + np.array(s0sin)   

        s2lin.makeSeries([1,4,3,1], [interval_length * 2, l/2, l/2, interval_length - l], 
                        [baseLine, baseLine, baseLine - m, baseLine], 
                        gradient = float(m)/float(l/2), noise = 0.5)
        s2sin.makeSeries([2], [3 * interval_length], [0.0], 
                        amp = amp, period = period, noise = 0.5)
        s2 = np.array(s2lin) + np.array(s2sin)   
        
        s3.makeSeries([2], [3 * interval_length], [0.0], 
                        amp = amp, period = period, noise = 0.5)
        s4.makeSeries([2], [3 * interval_length], [0.0], 
                        amp = amp, period = period, noise = 0.5)

        data = sp.r_['1,2,0', s1, s2, s3, s4]





if __name__=='__main__':
  