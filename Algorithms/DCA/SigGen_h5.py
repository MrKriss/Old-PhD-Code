# -*- coding: utf-8 -*-
"""
Created on Fri Jul 02 12:32:27 2010

Script to generate and store the signal and antigen data files 

@author: musselle
"""

from numpy import array, zeros, dtype, ceil, uint8
from numpy.random import beta, rand, seed
from tables import * 

# openFile, IsDescription, Int32Col, Float32Col, FloatCol , BoolCol 

class SigTableDesc(IsDescription):
    Time_Step = Int32Col(pos = 0)
    Danger_Signal = Float32Col(pos = 1)
    Safe_Signal = Float32Col(pos = 2)
    Total_Signal = Float32Col(pos = 3)
    Class = BoolCol(pos = 4)
    
class ScoreTableDesc(IsDescription):
    Antigen = Int32Col(pos = 0)
    MCAV = FloatCol(pos = 1)
    K_alpha = FloatCol(pos = 2) 

def signal_antigen_generator(sigP,antP,s):
    """ Generates signals and antigens according to passed parameters
    
    OUTPUTS
    
    signalTable - record array length (antigenSequence)
                - dtype([('ID', int), 
                        ('Danger', float), 
                        ('Safe', float), 
                        ('Total', float),
                        ('Class', bool)])
    
    antigen - dictionary 
    {'types' : types, 'sequence' : antigenSequence, 'score' : antigenKtable}
        antigenSequence  -- > vector of states/events
        types     -- > Vector of unique antigen types
        antigenKTable - record array length (types)
                      - dtype([('Antigen',int),
                        ('K value', float),
                        ('MCAV', float),
                        ('K Alpha', float)])
    """
    
    # Initailise
    seed(s)
    state = 'N' # % True aka Normal state to start with
    #allocate memory
    safeSig = zeros((sigP['runtime']))
    danger1Sig = zeros((sigP['runtime']))
    danger2Sig = zeros((sigP['runtime']))
    
    # Ok, so how about just using a list?
    antigenSequence = [[] for j in range(sigP['runtime'])]
    antigenSequence = tuple(antigenSequence)
    stateString = []
    
    #Generate Signals and Antigen
    for i in range(sigP['runtime']):  
            
        for k in range(antP['antPerStep']):
        
            if 'N' == state : #  if state = normal
            
                # Generate random number from beta distribution
                r = beta(antP['beta_N'][0], antP['beta_N'][1])
                antigenSequence[i].append(uint8(ceil(r * (antP['numAntNorm']))))
                
            elif 'A' == state :   # if state = anomalous
                # Generate random number from beta distribution
                r = beta(antP['beta_A'][0],antP['beta_A'][1])
                antigenSequence[i].append(uint8(ceil((r * antP['numAntAnom'] + \
                                (antP['alphabetSize'] - antP['numAntAnom'])))))
                                
        # Generate signals # 
        if 'N' == state :
            
            #generate signals from normal beta parameters
            safeSig[i] = beta(sigP['beta_S_N'][0], sigP['beta_S_N'][1])
            danger1Sig[i] = beta(sigP['beta_D1_N'][0], sigP['beta_D1_N'][1])
            
            if sigP['numSig'] == 3:
                danger2Sig[i] = beta(sigP['beta_D2_N'][0], sigP['beta_D2_N'][1])
      
            # record states in string
            stateString.append(1)
        
            if rand() <= sigP['T'][0,1]: # transition to abnormal
                state = 'A'
                
        elif 'A' == state:
            
            # generate signals from Abnormal beta parameters
            safeSig[i] = beta(sigP['beta_S_A'][0], sigP['beta_S_A'][1])
            danger1Sig[i] = beta(sigP['beta_D1_A'][0], sigP['beta_D1_A'][1])
            
            if sigP['numSig'] == 3:
                danger1Sig[i] = beta(sigP['beta_D2_A'][0], sigP['beta_D2_A'][1])
    
            # record states in string
            stateString.append(0)
            
            if rand() <= sigP['T'][1,0]: # transition to normal
                state = 'N'
    
    if antP['timeDelay'] != 0:
        """ shift all entries by time delay where 
    
        %     Original :  ABCDEFGH
        %                    |
        %         -ve  <---  |  ---> +ve
        %                    |
        %     
        %      i.e +ve is where events lag behind and -ve is when events are 
        %          shifted forward in time 
        %      
        % -ve = forward in time. i.e event happens x
        % steps before the associated signal is seen. 
        """
             
           
        # shift = antP.timeDelay
        
        
        if antP['timeDelay'] < 0:
            # take the first x from the start and put them at the back
            temp = antigenSequence[antP['timeDelay']:]
            temp = temp + antigenSequence[:antP['timeDelay']]
        
        elif antP['timeDelay'] > 0:
            # take the last x from the end and put them at the start
            temp = antigenSequence[-antP['timeDelay']:]
            temp = temp + antigenSequence[:-antP['timeDelay']]
    
        antigenSequence = temp
    
    # Store Input tables
    
    # 
    # Setup to work with Hdf5 data format #
    # Keep antigen and signals separate
    
    # Store as           1        2        3         4       5        6
    # Array columns {'Antigen', 'Pamp', 'Danger', 'Safe', 'Total', 'Class'}
    
    dtTable = dtype([('ID', int), 
              #  ('Antigen', object), 
                ('Danger', float), 
                ('Safe', float), 
                ('Total', float),
                ('Class', bool)])
    
    signalTable = zeros(len(antigenSequence), dtTable)
    
    #% if multiple antigen, put them all in the cell ignoring NaNs
#    if antP['method'] == 3 or antP['method'] == 4 :
#        for n in range(len(antigenSequence)):
#            signalTable[n]['Antigen'] = antigenSequence[n]
    
    signalTable[:]['ID'] = range(sigP['runtime'])
    signalTable[:]['Danger'] = danger1Sig
    signalTable[:]['Safe'] = safeSig
    signalTable[:]['Total'] = danger1Sig + safeSig 
    signalTable[:]['Class'] = [bool(x) for x in stateString]
    
    # To finish later if decide to use 3 or more signals
    if sigP['numSig'] == 3:
        signalTable[:]['Danger'] = danger2Sig
    
    types = []
    for i in range(len(antigenSequence)):
        types = types + antigenSequence[i]
    types = set(types)
    
    dtKtable = dtype([('Antigen',int),
                ('K value', float),
                ('MCAV', float),
                ('K Alpha', float)])
                
    
    antigenKtable = zeros(len(types), dtype = dtKtable)
    antigenKtable[:]['Antigen'] = list(types)
    antigen = {'types' : types, 'sequence' : antigenSequence, 'score' : antigenKtable}

    return signalTable, antigen

###############################
# Script to setup experiment  #
###############################

# Initialise Signal parameters
numDataSets = 10
Pan = 0.1
Pna = 0.1

sigP = {'numSig': 2, 
        'runtime': 500,
        'beta_S_N': (10,4),
        'beta_D1_N': (4,10),
        'beta_S_A': (4,10), 
        'beta_D1_A': (10,4),
        'T': array([[1-Pna, Pna], [Pan, 1-Pan]])}
                

# Initialise Antigen Parameters

antP = {'method': 4,         #  Method 4 now used as standard..... 
        'timeDelay': 0,   #  delay between antigen and signals  # (-ve = shift forward in time) 
        'antPerStep': 100,  
        'percentageRemoved': 0, # used to vary amount of ants per step (outdated)
        'alphabetSize': 10, 
        'beta_N': (3,3),
        'beta_A': (3,3)}

if antP['method'] == 4:
    antP['overlap'] = 0 
    antP['ratio'] = (0.5, 0, 0.5) 
    
    # Set number of normal and anomalous antigens
    antP['numAntNorm'] = round(antP['alphabetSize'] * antP['ratio'][0])
    antP['numAntAnom'] = round(antP['alphabetSize'] * \
                            (antP['ratio'][2] + antP['overlap']))
    antP['numAntBoth'] = round(antP['alphabetSize'] * antP['overlap'])

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Setup HDF5 data file to hold data #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# Open a file in "w"rite mode
dfile = openFile("C:\DataSets\InputDataFile1.h5", mode = "w")
# Get the HDF5 root group
root = dfile.root

# Create new groups for each dataSet:
for n in range(numDataSets):
    
    # Set group for data sets 1:n with seed attribute
    group = 'set' + str(n)
    group_loc = '/' + group
    tempGroup = dfile.createGroup(root, group)
    tempGroup._v_attrs.seed = n

    # Set antigen sub group + leaves 
    antigen = dfile.createGroup(tempGroup, "Antigen")
    sequence = dfile.createVLArray(antigen, 'Sequence', IntAtom(), 'Antigen pool per timestep')
    k_values = dfile.createVLArray(antigen, 'K_values', FloatAtom(), 'Logged k_values')
    
    # Set tables 
    sigTable = dfile.createTable(tempGroup, 'SignalTable', SigTableDesc)
    scores = dfile.createTable(antigen, 'ScoresTable', ScoreTableDesc)
    
    # Run Exp
    out_signals, out_antigens = signal_antigen_generator(sigP,antP, n)
    
    # Store Values
    types = dfile.createArray(antigen, 'Types', list(out_antigens['types']) , 'Unique Antigen Types')
    
    # VLArrays 
    for row in out_antigens['sequence']:
        sequence.append(row)

    tab_row = sigTable.row

    # Signal Data
    for row in out_signals:
        tab_row['Time_Step'] = row['ID']
        tab_row['Danger_Signal'] = row['Danger']
        tab_row['Safe_Signal'] = row['Safe']
        tab_row['Total_Signal'] = row['Total']
        tab_row['Class'] = row['Class']
        tab_row.append()
    
    sigTable.flush()

# Finally, close the file (this also will flush all the remaining buffers!)
dfile.close()