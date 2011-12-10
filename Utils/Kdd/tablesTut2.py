# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 11:45:20 2010

@author: musselle
"""

from tables import *

h5file = openFile("tutorial1.h5", "a")

# Use of walkgroups and list nodes methods
for group in h5file.walkGroups("/"):
      for array in h5file.listNodes(group, classname='Array'):
          print array

# A shorter way using the walkNodes method
for array in h5file.walkNodes("/", "Array"):
      print array

