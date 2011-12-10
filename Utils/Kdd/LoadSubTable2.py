# -*- coding: utf-8 -*-
"""
Created on Tue Oct 05 14:24:41 2010

@author: musselle
"""

import tables as tb
# import numpy as np


def getSubTableDsc(tableObject, targetValue = 'float32'):
    ''' Function to create a sub Table description from target tableObject
    Output is dic with {column name : data type} 
    later will be used to construct the sub table     
    '''
    
    seq = []
    # Dic of all column descriptions
    tabDescs = tableObject.coldescrs
    
    if type(targetValue) == str : # if str input 
        
        for k,v in tabDescs.iteritems():
            if v == targetValue:
                seq.append((k,v))
        
    elif type(targetValue) == list : 
        
        names = tableObject.colnames
        index = range(len(names))

        if type(targetValue[0]) == int : # if list of position indexes  
        
            # Create a dic linking col names to position
            names = tableObject.colnames
            index = range(len(names))
            zipped = zip(index,names)
            indexnameDic = dict(zipped)
            
            for item in targetValue:
                seq.append((indexnameDic[item], tabDescs[indexnameDic[item]]))
            
        elif type(targetValue[0]) == str : # if list of column names 
        
            for item in targetValue:
                seq.append((item, tabDescs[item]))
    
    targetDic = dict(seq)
    return  targetDic


# Open File
f = tb.openFile('C:\DataSets\Kdd.h5','r+')

# Refs to tables
mytab10 = f.root.RawData.KDD_10_Tab
mytabFull = f.root.RawData.KDD_Full_Tab

# Create Master index for column names 
names = mytab10.colnames
index = range(len(names))
zipped = zip(index,names)
indexnameDic = dict(zipped)

# Get Table Dsc for sub table 
##################
targetValue = [1,2,3,4,5,6,7,8,9,11,12,13] # 
##################
subTabDsc = getSubTableDsc(mytab10, targetValue)

#subsubTabDsc = dtype(subTabDsc.items())

# Create new table
group1 = f.root.FeatureSets
subTable = f.createTable(group1, 'SubTable1', subTabDsc) 

row = subTable.row
subNames = subTable.colnames

numRows = len(mytab10) # Change depending on dataSet

for i in range(numRows): 
    for elem in subNames:
        row[elem] = mytab10[i][elem]
        print i
    row.append()


subTable.flush()

