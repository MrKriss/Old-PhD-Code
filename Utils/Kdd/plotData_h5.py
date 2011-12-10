# -*- coding: utf-8 -*-
"""
Created on Thu Oct 07 11:40:52 2010

@author: musselle
"""

import tables as tb
# import numpy as np
import pdb

def getSubTableDsc(tableObject, targetValue = 'float32'):
    ''' Function to create a sub Table description from target tableObject
    Output is dic with {column name : data type} 
    later will be used to construct the sub table     
    '''
    
    seq = []
    # Dic of all column descriptions
    tabDescs = tableObject.coldescrs
    tabTypes = tableObject.coltypes
    
    if type(targetValue) == str : # if str input 
        
        for k,v in tabTypes.iteritems():
            if v == targetValue:
                seq.append((k, tabDescs[k])) 
        
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
    

def createSubTable(targetValue, tableName, datasetFlag = '10'):
    ''' 
    Writes a sub table from the original data file 
    ''' 

    # Open File
    f = tb.openFile('C:\DataSets\Kdd.h5','a')
    root = f.root
    
    # Create Master index for column names 
    names = root.RawData.KDD_10_Tab.colnames
    index = range(len(names))
    zipped = zip(names,index)
    nameindexDic = dict(zipped)
    
    
    if datasetFlag == '10':  # Fill 10 percent table 

        mytab10 = f.root.RawData.KDD_10_Tab
        
        # Get description dictionary
        subTabDsc = getSubTableDsc(mytab10, targetValue)
    
        pdb.set_trace()
    
        # Create Sub Table if it doesnt exist
        
        kids = f.root.FeatureSubSets._v_children
        
        
        if not kids.has_key(tableName):
            
            print 'Before'
            subTab = f.createTable(root.FeatureSubSets, tableName, subTabDsc)
        
        # open file 
        kddfile = open('C:\DataSets\kdd_10_percent_corr.txt', 'r')   
        
        
        row = subTab.row
        names = subTab.colnames
        types = subTab.coltypes
        
        for line in kddfile:
            # Read csv into a list 
            temp = line.rstrip('.\n')
            temp = temp.split(',')
            
            for elem in names:
                index = nameindexDic[elem] # get position index 
                
                if types[elem] == 'bool':
                    row[elem] = int(temp[index])
                else:
                    row[elem] = temp[index]
            row.append()
    
        subTab.flush()
        
    elif datasetFlag == 'full':

        mytabFull = f.root.RawData.KDD_Full_Tab
        # Get description dictionary
        subTabDsc = getSubTableDsc(mytabFull, targetValue)
        
        # Create Sub Table  if not present 
        
        # Create Sub Table if it doesnt exist
        kids = f.root.FeatureSubSets._v_children
        if not kids.has_key(tableName):
            subTab = f.createTable(root.FeatureSubSets, tableName, subTabDsc)
        
        # open file 
        kddfile = open('C:\DataSets\kdd_Full_corr.txt', 'r')   
        
        row = subTab.row
        names = subTab.colnames
        types = subTab.coltypes
        
        for line in kddfile:
            # Read csv into a list 
            temp = line.rstrip('.\n')
            temp = temp.split(',')
            
            for elem in names:
                index = nameindexDic[elem] # get position index 
                
                if types[elem] == 'bool':
                    row[elem] = int(temp[index])
                else:
                    row[elem] = temp[index]
            row.append()
    
        subTab.flush()
        


targetValue = [1,2,3,4,5,6,7,8,9,11,12,13] 

createSubTable('float32', "FloatTab1", datasetFlag = '10')

