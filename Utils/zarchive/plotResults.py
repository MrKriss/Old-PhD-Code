# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 13:22:22 2011

@author: musselle
"""

import numpy as np
import os
from matplotlib.pyplot import plot, title, figure

def plotSummaryMat(toplot, BatchName, mainPath = 'C:/DataSets/Results/Fraust'):
    """
    Plots the e_qq/f_qq and or g_qq of Batch in Batch Name 
    """
    # Change Directory 
    os.chdir(mainPath + '/' + BatchName)
    
    out = []
    
    # plot e_qq if desired
    if toplot.find('e') != -1:
        eqq_mat = np.fromfile('e_qq_mat.dat')
        eqq_mat = eqq_mat.reshape(10000,10)
        figure()
        plot(eqq_mat)
        title(BatchName + ' e_qq')
        out.append(eqq_mat)

    # plot f_qq if desired
    if toplot.find('f') != -1 : 
        fqq_mat = np.fromfile('f_qq_mat.dat')
        fqq_mat = fqq_mat.reshape(10001,10)
        figure()
        plot(fqq_mat)
        title(BatchName + ' f_qq')
        out.append(fqq_mat)        
        
    # plot g_qq if desired
    if toplot.find('g') != -1:
        gqq_mat = np.fromfile('g_qq_mat.dat')
        gqq_mat = gqq_mat.reshape(10000,10)
        figure()
        plot(gqq_mat)
        title(BatchName + ' g_qq')
        out.append(gqq_mat)        
        
    return out
#===============================================================================
# If main method 
#===============================================================================
if __name__ == '__main__':
        
        
        BatchNameList = ['NewCosData, a=996, sci= -1',
                         'NewCosData, a=996, sci= 0',
                         'NewCosData, a=996, sci= +1',
                         'NewCosData, a=96, sci= -1',
                         'NewCosData, a=96, sci= 0',
                         'NewCosData, a=96, sci= +1']
        toplot = 'efg'
        
        for BatchName in BatchNameList:
            plotSummaryMat(toplot,BatchName)
            
        
    