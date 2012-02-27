#!/usr/bin/env python
#coding:utf-8
# Author:   --<>
# Purpose: 
# Created: 01/05/12

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sys
import os 
import time

"""
Code Description:
  .
"""
iterations = 100
start = time.time()
for i in xrange(iterations): 
  execfile('/Users/chris/Dropbox/Work/MacSpyder/Algorithms/Frahst_class.py')

fin = time.time() - start
print 'Finished %i iterations in %f seconds' % (iterations, fin)
print 'Average of %f seconds per iteration' % (fin / iterations)

  