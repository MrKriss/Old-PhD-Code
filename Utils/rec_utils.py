#!/usr/bin/env python
#coding:utf-8
# Author:   --<C Musselle>
# Created: 12/14/11

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sys
import os 

"""
Code Description: Utility functions to be used with record arrays
  .
"""

def pprec(rec):
  """ Function to pretty print a single record array entry that 
  has field lengths > 1 """
  
  for r in range(len(rec)):
  
    fields = list(rec[r].dtype.names)
    
    # Add print titles 
    print fields 
    
    if len(rec) > 1:
      for i in range(rec[r][fields[0]].size):
        s = ''
        for f in fields:
          s += str(rec[r][f][:,i]) + ' '
        print s
    else:
      for i in range(rec[r][fields[0]].size):
        s = ''
        for f in fields:
          s += str(rec[r][f][i]) + ' '
        print s

if __name__=='__main__':
  pass
  