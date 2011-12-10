# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 17:41:36 2011

Script to extract pedros data from csvs.


@author: -
"""
import numpy as np
import csv
import os


current_dir = os.getcwd()


routers = {}
routers['ground_thruths'] = np.array([[321, 2], [352, 2], [532, 3]])
routers['name'] = 'ds3b'
routers['title'] = 'Operadoras 300s 4/16/2009'
routers['date'] = '20091604_0000'
 
os.chdir('/Users/Main/DataSets/ISP Pedro/Routers/')
  
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


# PEDROS SERVER DATA

# or 

#import csv
#import numpy
#reader=csv.reader(open("test.csv","rb"),delimiter=',')
#x=list(reader)
#result=numpy.array(x).astype('float')


# Pedros DATA Servers 
# Contains CPU (idle, usr) and memory (used, available) usage statistics 
# from each of the six machines within a cluster
 

os.chdir('/Users/Main/DataSets/ISP Pedro/Servers/')

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







