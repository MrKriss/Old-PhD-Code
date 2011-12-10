# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 15:19:12 2010

@author: musselle
"""

from tables import * 

# initialise used variables
numDataSets = 50 

# Define descritions classes 

class sigTableDesc(IsDescription):
    Time_Step = Int64Col()
    Danger_Signal = Float32Col()
    Safe_Signal = Float32Col()
    Total_Signal = Float32Col()
    
class scoreTable(IsDescription):
    Antigen = Int64Col()
    MCAV = Float64Col()
    K_alpha = Float 
    

# Open a file in "w"rite mode
dfile = openFile("DataFile1.h5", mode = "w")
# Get the HDF5 root group
root = dfile.root

# Create new groups for each dataSet:
for n in range(numDataSets):
    group = 'set'+ str(n)
    vars()[group] = dfile.createGroup(root, group)
    group_name = '/'+'set'+ str(n)
    antigen = dfile.createGroup(group_name, "antigen")
    sigTable = dfile.createTable(group_name, "Signal_Table", sigTableDesc)

# Finally, close the file (this also will flush all the remaining buffers!)
dfile.close()