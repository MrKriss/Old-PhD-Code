# -*- coding: utf-8 -*-
"""
Created on Fri Jul 02 12:32:27 2010

Script to setup and run the signal and antigen generation side of things 

@author: musselle
"""

from numpy import array, zeros



def signal_antigen_generator(sigP,antP,s):
    """ Generates signals and antigens according to passed parameters
    Outputs
    antigen.sequence  -- > vector of states/events
           .types     -- > Vector of unique antigen types
           
    """
    
    # Initailise
    seed(s)
    state = 'N' # % True aka Normal state to start with
    #allocate memory
    safeSig = zeros((sigP['runtime']))
    danger1Sig = zeros((sigP['runtime']))
    danger2Sig = zeros((sigP['runtime']))
    
    # Ok, so how about just using a list?
    antigenSequence = [[0] for j in range(sigP['runtime'])]
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
    # Store as           1        2        3         4       5        6
    # Array columns {'Antigen', 'Pamp', 'Danger', 'Safe', 'Total', 'Class'}
    
    dtTable = dtype([('ID', int), 
                ('Antigen', object), 
                ('Danger', float), 
                ('Safe', float), 
                ('Total', float),
                ('Class', bool)])
    
    signalTable = zeros(len(antigenSequence), dtTable)
    
    #% if multiple antigen, put them all in the cell ignoring NaNs
    if antP['method'] == 3 or antP['method'] == 4 :
        for n in range(len(antigenSequence)):
            signalTable[n]['Antigen'] = antigenSequence[n]
    
    signalTable[:]['Danger'] = danger2Sig
    signalTable[:]['Safe'] = safeSig
    signalTable[:]['Total'] = danger2Sig + safeSig 
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
        'beta_S_N': (3,2),
        'beta_D1_N': (2,3),
        'beta_S_A': (2,3), 
        'beta_D1_A': (3,2),
        'T': array([[1-Pna, Pna], [Pan, 1-Pan]])}
                

# Initialise Antigen Parameters

antP = {'method': 4,         #  Method 4 now used as standard..... 
        'timeDelay': 5,   #  delay between antigen and signals  # (-ve = shift forward in time) 
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

# setup dataset structure
dtds = np.dtype([('signalTable', object), 
               ('antigen', object), 
               ('seed', int8)])
               
dataSet = np.zeros(numDataSets, dtype = dtds)

# run sigGen using 50 different random seeds.
for i in range(numDataSets):
    dataSet['seed'][i] = i
    # signal_antigen_generator function 
    s , a = signal_antigen_generator(sigP,antP, dataSet[i]['seed'])
    dataSet[i]['signalTable'] = s
    dataSet[i]['antigen'] = a

