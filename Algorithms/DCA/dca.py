# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 12:40:11 2010
The deterministic DCA - Julies Version

@author: musselle
"""

import tables as tb
from numpy import dtype, zeros
import pdb

def dDCA(dcaP,dataSet):
    """ Function to carry out deterministic DCA
    
    INPUTS 
    dcaP - Parameters for dDCA
        .wm - wieght matrix -- > Structure for storage of parameters for DCA
        .numDC - number of virtual DCs
        .min - Minimum threshold of DCs
        .range - how far the range of threshold extends above min threshold 
        .strategy - 4 only - The dDCA 

    dataSet - Group in h5file for input data
           ._v_attrib - seed
           .SignalTable - Table( Time_Step = Int64Col(pos = 0)
                               Danger_Signal = Float32Col(pos = 1)
                               Safe_Signal = Float32Col(pos = 2)
                               Total_Signal = Float32Col(pos = 3)
                               Class = BoolCol(pos = 4))
           .Antigen - Group
                   .Kvalues - VLarray(FloatAtom)
                   .Sequence - VLarray(IntAtom)
                   .Types - Array
                   .Scores - Table(Antigen = Int64Col(pos = 0),
                                   MCAV = Float64Col(pos = 1),
                                   K_alpha = FloatCol(pos = 2) 
    
    OUTPUTS
    
    dataSet - write values into Kvalues and Scores 
    
    """
    
    # Initialise Values 
    
    seed = dataSet._v_attrs.seed
    
    # Set DC logs
    dtLogs = dtype([('DC_Num', int), 
            ('Threshold', float), 
            ('Total_Itts', int),
            ('Sum_Num_Antigen', int),
            ('Ave_Ant/Itt', float)])     
    logs = zeros(dcaP['numDc'], dtLogs)

    # Set Temp Record Structure
    dtRec = dtype([('Antigen', int), 
            ('Sum_k', float), 
            ('Num_Normal', int),
            ('Num_Anomalous', int), 
            ('Times_Sampled', int),
            ('MCAV', float),
            ('K_alpha', float)]) 
    record = zeros(len(dataSet.Antigen.Types), dtRec)
    
    

    numAntSampledFreq = []

    # Setup DCA structure dcStr
    dtDca = dtype([('ID', int), 
            ('Threshold', float), 
            ('K', float), 
            ('Lifetime', float),
            ('Antigen', object),
            ('Iteration', int)])
    dcStr = zeros(dcaP['numDc'], dtDca)
    increment = (dcaP['range']) / dcaP['numDc']
    
    for i in range(dcaP['numDc']):
        dcStr[i]['ID'] = i
        dcStr[i]['Threshold'] = dcaP['min'] + increment * (i)
        dcStr[i]['K'] = 0
        dcStr[i]['Antigen'] = []
        dcStr[i]['Lifetime'] = dcStr[i]['Threshold']
        dcStr[i]['Iteration'] = 0
        logs[i]['DC_Num'] = i
        logs[i]['Threshold'] = dcStr[i]['Threshold']
    
    #---------------------------------------------------------------
    # DCA main body 
    #---------------------------------------------------------------
    
    antCount = 0
    timeStepCounter = len(dataSet.SignalTable) #  how many entries of data
    
    for z in range(timeStepCounter):
        # OPTIMISATION - only process signals once per loop
        # calculate Signal Processing % Sample Signals at timestep Z
        k = (dcaP['wm'][1][0] * dataSet.SignalTable.col('Danger_Signal')[z] +dcaP['wm'][1][2] * dataSet.SignalTable.col('Safe_Signal')[z]) * 100
        csm = (dcaP['wm'][0][0] * dataSet.SignalTable.col('Danger_Signal')[z] +dcaP['wm'][0][2] * dataSet.SignalTable.col('Safe_Signal')[z]) * 100
        
        
        # Antigen Sampling 
        sequence = dataSet.Antigen.Sequence
        antPoolSize = len(sequence[z])
        if antPoolSize != 0:
            for a in range(antPoolSize):            
                DCindex = antCount % dcaP['numDc'] 
                dcStr[DCindex]['Antigen'].append(sequence[z][a]) 
                antCount += 1 # antigen counter increment
                
        for i in range(dcaP['numDc']) :
            dcStr[i]['K'] += k # sample signals
            dcStr[i]['Lifetime'] -= csm # Update life time
            # output if necessary
            if dcStr[i]['Lifetime'] <= 0:
                # update iteration count 
                logs[i]['Total_Itts'] += 1
                # Antigen Sampling Frequency
                numAntSampledFreq.append(len(dcStr[i]['Antigen']))
                # update record str array 
                for b in dcStr[i]['Antigen']:
                    b = b - 1 # convert antigen to index
                    
                    record[b]['Times_Sampled'] += 1
                    record[b]['Sum_k'] += dcStr[i]['K']
                    logs[i]['Sum_Num_Antigen'] += 1
                    if dcStr[i]['K'] <= 0:
                        record[b]['Num_Normal'] += 1
                    elif dcStr[i]['K'] > 0:
                        record[b]['Num_Anomalous'] += 1
                # reset DC
                dcStr[i]['Lifetime'] = dcStr[i]['Threshold']
                dcStr[i]['K'] = 0
                dcStr[i]['Antigen']= []


    # Metric Processing
    # MCAV 
    for i in range(len(record)):
        record[i]['MCAV'] = (float(record[i]['Num_Anomalous']) / 
                            float(record[i]['Times_Sampled']))
        record[i]['K_alpha'] = (float(record[i]['Sum_k']) / 
                            float(record[i]['Times_Sampled']))
        record[i]['Antigen'] = i
    #totalSampled = sum(record[:]['Times_Sampled'])

    for i in range(dcaP['numDc']):
        logs[i]['Ave_Ant/Itt'] = (float(logs[i]['Sum_Num_Antigen']) / 
                                    float(logs[i]['Total_Itts']))

    # x = 1:max(numAntigenSampled);
    # figure
    # hist(numAntigenSampled)
    # title('Frequncy of Antigen Sample size by DCs')
    
    return [record,logs,numAntSampledFreq]

#-----------------------------------------------------------------------------#
#DCA PARAMETERS 

# Wieghts Matrix
#      PAMP | Danger | Safe
#  CSM   1  |    1   |   1
#   k    1  |    1   |  -2
#

dcaP = {}
dcaP['wm'] = [[1, 1, 1], [1, 1, -2]]
dcaP['strategy'] = 4 # The dDCA implementation
dcaP['numDc'] = 100
dcaP['range'] = 100 * 1.25  #%AveTotal * 1.25; % may change later
dcaP['min'] = 50 # some alalysis of wm needs to be done to assess min and max thresh
 
# Input Checks

#Initalise

# Open data file
infile = tb.openFile('C:\DataSets\InputDataFile1.h5','r')

dataSet = infile.root.set0

record,logs,numAntSampledFreq = dDCA(dcaP,dataSet)

# Close File
infile.close()