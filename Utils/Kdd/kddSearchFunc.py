# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 14:06:28 2010

@author: musselle
"""

from tables import *
import numpy as np

myfile = openFile('Kdd.h5', 'r')

fullTab = myfile.root.RawData.KDD_Full_Tab
trainTab = myfile.root.RawData.KDD_10_Tab

trainTab.colnames
names = trainTab.colnames
typesDic = trainTab.coltypes

mySetDic = {}

for elem in names: 
    mySetDic[elem] = set(fullTab.col(elem))
    print 'Done ' , elem 


def gen_set(mySetDic, names):
    for elem in names:
        yield mySetDic[elem]
    return

it = gen_set(mySetDic, names)

def max_str(a):
    m = 0
    for elem in a:
        temp = len(elem)
        if temp > m:
            m = temp
    return m

