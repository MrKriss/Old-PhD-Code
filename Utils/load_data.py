# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 10:47:55 2011

Defines the function to selectively load any data set. 
If ran, will load all data sets

#################
# Data Set INFO #
#################

ABELINE NETWORK DATA 

We use data collected from the 11 core routers in the Abilene backbone network 
for 1 week (December 15 to December 21, 2003).  It is comprised of a multivariate 
timeseries consisting of the number of packets (P), the number of individual IP 
flows (F), and the number of bytes (B) in each of the Abilene backbone flows 
(the traffic entering at one core router and exiting at another).  The statistics 
are collected (binned) at 5 minute intervals.  All three datasets are of dimension 
T x F, where T is the number of timesteps (in our case = 2016) and F is the number
 of backbone flows (in our case = 121). At each subsequent timestep t, we may then
use the tth row of the relevant timeseries (P, F or B) as the input vector xt.


PEDROS DATA

The measurements are collected every 6 minutes via SNMP and each file stores 
only the last 600 intervals. The data center operations team kindly provided us 
the following two set of files with annotated anomalies from real incidents that 
were not alarmed by the current monitoring solution based on na ̈ıve thresholding.

Routers #######

Contains statistics exported by the main data center router, which is 
connected to five telecom operators by many redundant network links. All the 
Internet traffic that comes in or out of the data center passes through one these
gateway links. 

The dataset contains the number of bits per second in 15 links for both directions
of communication averaged at 6 minute intervals. The measurements
were taken from 16/04/2009 to 19/04/2009, where we managed to merge two RRD files
to obtain over three days of data. There was a communication failure with one of
the operators, which caused two of the links to malfunction. The large failure 
was preceded by smaller loss of connectivity.

Servers #######
Contains CPU (idle, usr) and memory (used, available) usage statistics 
from each of the six machines within a cluster serving a specific application. 
The data was collected from 12/06/2009 to 13/06/2009, and contains anomalous events 
due to an unoptimized application that gained sudden popularity on the Internet and 
there was unexpected heavy load in two of the machines which had to be better balanced
to avoid high latencies for the end-users.


Motes ######

Light sensors around a lab. 

Chlorine #######

Chlorine sensors in simulated water network.

IC Model #######

Traffic Data from simple network simulation 


@author: - Musselle
"""

import numpy as np
import csv
import os

def load_data(file_name):
    
    names = ['isp_routers', 'isp_servers', 'abilene', 'motes_l', 
             'motes_h','motes_t','motes_v', 'chlorine', 'IC_model']     
    
    assert file_name in names, 'Dataset "%s" does not exist' % file_name
    
    if file_name == 'isp_routers' :
        
        # PEDROS DATA ROUTERS datasets
        # Bits per second in 15 links, for both forward and backward traffic
        current_dir = os.getcwd()

        routers = {}
        routers['ground_thruths'] = np.array([[321, 2], [352, 2], [532, 3]])
        routers['name'] = 'ds3b'
        routers['title'] = 'Operadoras 300s 4/16/2009'
        routers['date'] = '20091604_0000'
         
        os.chdir('/Users/chris/DataSets/ISP Pedro/Routers/')
          
        with open('isp_routers_data.csv', 'Ur') as dfile:
            routers['data'] = np.loadtxt(dfile, delimiter=",", skiprows=1)
            
        with open('isp_routers_original.csv', 'Ur') as dfile:
            routers['original'] = np.loadtxt(dfile, delimiter=",", skiprows=1)
        
        # Data Headers
        reader=csv.reader(open("isp_routers_data.csv","Ur"), delimiter=',')
        headers = reader.next() # List of all headers 
        
        short_headers = []
        
        for s in headers:
            temp = s.split('//')
            short_headers.append(temp[1])
        
        routers['data_headers'] = short_headers # First half of string removed, superfluous.
        
        # Original Headers
        reader=csv.reader(open("isp_routers_original.csv","Ur"), delimiter=',')
        headers = reader.next() # List of all headers 
        
        short_headers = []
        
        for s in headers:
            temp = s.split('//')
            short_headers.append(temp[1])
        
        routers['original_headers'] = short_headers # First half of string removed, superfluous.

        os.chdir(current_dir)        
        
        return routers
        
    if file_name == 'isp_servers' :
        
        # Pedros DATA Servers 
        # Contains CPU (idle, usr) and memory (used, available) usage statistics 
        # from each of the six machines within a cluster
        current_dir = os.getcwd()
        
        os.chdir('/Users/chris/DataSets/ISP Pedro/Servers/')
        
        servers = {}
        servers['name'] = 'ds1'
        servers['title'] = '360s 12/06/2009'
        servers['comment'] = 'Problema com NET. Inclui CPU e mem de riolf334 e riolf355.'
        servers['ground_truths'] = np.array([[114,4],[406,20], [421,4], [436,16], 
                                    [469,1], [500,5], [513,1], [528,2]]) 
        
        servers['original_names'] = ['riolf334_CPU.rrd.xml-scale360.data_1',
                                    'riolf334_CPU.rrd.xml-scale360.data_2',
                                    'riolf334_CPU.rrd.xml-scale360.data_3',
                                    'riolf334_CPU.rrd.xml-scale360.data_4',
                                    'riolf334_Memory.rrd.xml-scale360.data_1',
                                    'riolf334_Memory.rrd.xml-scale360.data_2',
                                    'riolf334_Memory.rrd.xml-scale360.data_3',
                                    'riolf334_Memory.rrd.xml-scale360.data_4',
                                    'riolf334_Memory.rrd.xml-scale360.data_5',
                                    'riolf334_Memory.rrd.xml-scale360.data_6',
                                    'riolf334_Memory.rrd.xml-scale360.data_7',
                                    'riolf334_Memory.rrd.xml-scale360.data_8',
                                    'riolf355_CPU.rrd.xml-scale360.data_1',
                                    'riolf355_CPU.rrd.xml-scale360.data_2',
                                    'riolf355_CPU.rrd.xml-scale360.data_3',
                                    'riolf355_CPU.rrd.xml-scale360.data_4',
                                    'riolf355_Memory.rrd.xml-scale360.data_1',
                                    'riolf355_Memory.rrd.xml-scale360.data_2',
                                    'riolf355_Memory.rrd.xml-scale360.data_3',
                                    'riolf355_Memory.rrd.xml-scale360.data_4',
                                    'riolf355_Memory.rrd.xml-scale360.data_5',
                                    'riolf355_Memory.rrd.xml-scale360.data_6',
                                    'riolf355_Memory.rrd.xml-scale360.data_7',
                                    'riolf355_Memory.rrd.xml-scale360.data_8']
                                    
        with open('isp_servers_data.csv', 'Ur') as dfile:
            servers['data'] = np.loadtxt(dfile, delimiter=",", skiprows=1)
        
        with open('isp_servers_original.csv', 'Ur') as dfile:
            servers['original'] = np.loadtxt(dfile, delimiter=",", skiprows=1)
        
        os.chdir(current_dir)
        
        return servers
        
    if file_name == 'abilene' :
        
        import scipy.io as sio
        
        # Abilene Network: 
        # Nodes = 11
        # Links = 121
        # Duration = 1 week 
        # ==> 5 min per time step t 
        AbileneMat = sio.loadmat('/Users/chris/DataSets/Abilene/Abilene.mat')
        
        # Number of Packets 
        packet_data = AbileneMat['P']
        
        # Number of indervidual IP flows
        IPflow_data = AbileneMat['F']
        
        # Number of bytes in each flow
        byte_flow_data = AbileneMat['B']    
        
        return packet_data, IPflow_data, byte_flow_data
        
    if file_name == 'motes_l' :
        motes_streams = np.loadtxt('/Users/chris/DataSets/Motes/q8calibLight.dat')
        return motes_streams
        
    if file_name == 'motes_h' :
        motes_streams = np.loadtxt('/Users/chris/DataSets/Motes/q8calibHumid.dat')
        return motes_streams
        
    if file_name == 'motes_t' :
        motes_streams = np.loadtxt('/Users/chris/DataSets/Motes/q8calibHumTemp.dat')
        return motes_streams
        
    if file_name == 'motes_v' :
        motes_streams = np.loadtxt('/Users/chris/DataSets/Motes/q8calibVolt.dat')
        return motes_streams
        
    if file_name == 'chlorine' :
        
        chlorine = np.loadtxt('/Users/chris/DataSets/Chlorine/cl2fullLarge.dat')
        
        return chlorine
        
    if file_name == 'IC_model' :
        
        import pickle as pk
        
        current_dir = os.getcwd()
        # Load OD network dataset from folder.
        os.chdir('/Users/chris/DataSets/Synthetic/Network_ODsmall')
        
        # The packet flows vector
        with open('P_t_data.pk', 'r') as data_file :
            P_t = pk.load(data_file)
        
        # The Link flows vector
        with open('Y_t_data.pk', 'r') as data_file :
            Y_t = pk.load(data_file)

        with open('other_data.pk', 'r') as data_file:
            packet_numbers = pk.load(data_file)
            R = pk.load(data_file)
            G = pk.load(data_file)

        os.chdir(current_dir)
        
        return P_t, Y_t, packet_numbers, R, G 
        

def load_ts_data(name, cleaning):
    """ Load specified data set and apply cleaning/preprocessing """    
    
    if name == 'isp_routers':
        data = load_data(name)['data']
        if cleaning == 'full':
            mask = data.max(axis = 0) > 150
            # Also remove Time series 18
            mask[22] = False
            data = data[:, mask]
        elif cleaning == 'mid':
            mask = data.max(axis = 0) > 50
            data = data[:, mask]
        elif cleaning == 'min':
            mask = data.max(axis = 0) > 5
            data = data[:, mask]
            
    return data  
    
    
    
if __name__ == '__main__' :
    
    motes = load_data('motes')
    chlorine = load_data('chlorine')
    packet_data, IPflow_data, byte_flow_data = load_data('abilene')
    routers = load_data('isp_routers')
    servers = load_data('isp_servers')
   # P_t, Y_t, packet_numbers, R, G = load_data('IC_model')
    
    
    