# -*- coding = utf-8 -*-
"""
Created on Thu Jul 15 11 =53 =53 2010

Script to import KDD data into HDF5 data file format

@author = musselle
"""

from tables import *
import time

start = time.time()

class TabDesc(IsDescription):
    duration = Float32Col(pos = 0)
    protocol_type = StringCol(4, pos = 1)
    service = StringCol(10, pos = 2)
    flag = StringCol(10, pos = 3)
    src_bytes = Float32Col(pos = 4)
    dst_bytes = Float32Col(pos = 5)
    land = BoolCol(pos = 6)
    wrong_fragment = Float32Col(pos = 7)
    urgent = Float32Col(pos = 8)
    hot = Float32Col(pos = 9)
    num_failed_logins = Float32Col(pos = 10)
    logged_in = BoolCol(pos = 11)
    num_compromised = Float32Col(pos = 12)
    root_shell = BoolCol(pos = 13)
    su_attempted = BoolCol(pos = 14)
    num_root = Float32Col(pos = 15)
    num_file_creations = Float32Col(pos = 16)
    num_shells = Float32Col(pos = 17)
    num_access_files = Float32Col(pos = 18)
    num_outbound_cmds = Float32Col(pos = 19)
    is_host_login = BoolCol(pos = 20)
    is_guest_login = BoolCol(pos = 21)
    count = Float32Col(pos = 22)
    srv_count = Float32Col(pos = 23)
    serror_rate = Float32Col(pos = 24)
    srv_serror_rate = Float32Col(pos = 25)
    rerror_rate = Float32Col(pos = 26)
    srv_rerror_rate = Float32Col(pos = 27)
    same_srv_rate = Float32Col(pos = 28)
    diff_srv_rate = Float32Col(pos = 29)
    srv_diff_host_rate = Float32Col(pos = 30)
    dst_host_count = Float32Col(pos = 31)
    dst_host_srv_count = Float32Col(pos = 32)
    dst_host_same_srv_rate = Float32Col(pos = 33)
    dst_host_diff_srv_rate = Float32Col(pos = 34)
    dst_host_same_src_port_rate = Float32Col(pos = 35)
    dst_host_srv_diff_host_rate = Float32Col(pos = 36)
    dst_host_serror_rate = Float32Col(pos = 37)
    dst_host_srv_serror_rate = Float32Col(pos = 38 )
    dst_host_rerror_rate = Float32Col(pos = 39)
    dst_host_srv_rerror_rate = Float32Col(pos = 40)
    Label = StringCol(20, pos = 41)

# set Directory 


# Open a file in "w"rite mode
fileh = openFile("C:\DataSets\Kdd.h5", mode = "w")
root = fileh.root

# Create the groups:
group1 = fileh.createGroup(root, "RawData", title = 'Full list of features')

# Create 2 new tables in group1
table1 = fileh.createTable(group1, "KDD_10_Tab", TabDesc)
table2 = fileh.createTable(group1, "KDD_Full_Tab", TabDesc)

# Fill 10 percent table 
row = table1.row
kddfile = open('C:\DataSets\kdd_10_percent_corr.txt', 'r')   
names = table1.colnames
types = table1.coltypes


for line in kddfile:
    temp = line.rstrip('.\n')
    temp = temp.split(',')
    for index in range(len(temp)):
        
        if types[names[index]] == 'bool':
            row[names[index]] = int(temp[index])
        else:
            row[names[index]] = temp[index]
    row.append()
    
table1.flush()

# Fill full data Table
row = table2.row
kddfile2 = open('C:\DataSets\kdd_Full_corr.txt', 'r')  

for line in kddfile2:
    temp = line.rstrip('.\n')
    temp = temp.split(',')
    for index in range(len(temp)):
        if types[names[index]] == 'bool':
            row[names[index]] = int(temp[index])
        else:
            row[names[index]] = temp[index]
    row.append()

table2.flush()

# Finally, close the file (this also will flush all the remaining buffers!)
fileh.close()

print 'Execution Time:', time.time() - start