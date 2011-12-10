# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 12:35:13 2011


Function to search Through Text results and find only those matching the search
criteria. Returns those in new text file.

@author: -
"""
import os


def searchRes(filename, target, statement, value):
    ''' Search through massive txt file and filter results ''' 
    
    hit_count = 0   
    count = 0
    with open('filtereded_results.txt', 'w') as f_out: 
        with open(filename, 'r') as f:
            
            breakout = 0
            while 1:
                writefile = 0
                block = []            
                for i in range(14):
                    line = f.readline()
                    if not line: 
                        breakout = 1
                        break 
                    block.append(line)             
     
                if breakout : break  

                count += 1

                for ln in block:
                    st = ln.split()            
                    if st:
                        if st[0] == target:                
                            if statement == '>=' :
                                median_value = float(st[-1][:-1])
                                if median_value >= value:
                                    writefile = 1      
                                    hit_count += 1
                                
                                
                            if statement == '<=' :
                                median_value = float(st[-1][:-1])
                                if median_value <= value:
                                    writefile = 1
                                    hit_count += 1
                                
                            if statement == '=' :
                                median_value = float(st[-1][:-1])
                                if median_value == value:
                                    writefile = 1
                                    hit_count += 1
                                    
                if writefile == 1:
                    f_out.writelines(block)

    print '{0} Results were found in total out of {1}'.format(hit_count, count )

                            