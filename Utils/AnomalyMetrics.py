# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 12:17:57 2011

SCript to analise performance of Frahst on Control Charts (shift) batch of data sets 


@author: - Musselle
"""

from numpy import dot, array, zeros, zeros_like, nan
from numpy.linalg import qr
import numpy.matlib as npm
from matplotlib.pyplot import plot, figure, subplot, title, xlim, ylim, axvline, show, axhline
import scipy.stats as stats

# Data Stores
#res = {'hidden' :  zeros((timeSteps, numStreams)) * nan,  # Array for hidden Variables
#       'E_t' : zeros([timeSteps, 1]),                     # total energy of data 
#       'E_dash_t' : zeros([timeSteps, 1]),                # hidden var energy
#       'e_ratio' : zeros([timeSteps, 1]),              # Energy ratio 
#       'RSRE' : zeros([timeSteps, 1]),           # Relative squared Reconstruction error 
#       'recon' : zeros([timeSteps, numStreams]),  # reconstructed data
#       'r_hist' : zeros([timeSteps, 1]), # history of r values 
#       'anomalies' : []}      

def fmeasure(B, hits, misses, falses) :
    """ General formular for F measure 
    
    Uses TP(hits), FN(misses) and FP(falses)
    """
    x = ((1 + B**2) * hits) / ((1 + B**2) * hits + B**2 * misses + falses)
    return x

def analysis(res, ground_truths_tab, timesteps, epsilon = 0 , ignoreUpTo = 0, return_sets_n_tab = 0):
    ''' Calculate all anomally detection Metrics 

    # epsilon: used to allow for lagged detections: if Anomaly occurs in time window
    anom_start - anom_end + eplsilon it is considered a TP
    
    # ignoreUpTo: does not count FPs before this time step
    '''    
          
    # Detections  
    D = array(res['anomalies'])
    index =  D > ignoreUpTo 
    D = set(list(D[index]))        
    
    # Total Neg 
    total_negative = timesteps - ignoreUpTo - len(D)    
    
    # initalise metrics     
    metric = { 'TP' : 0.0 ,
               'FP' : 0.0 ,
               'FN' : 0.0 ,
               'precision' : 0.0 ,
               'recall' : 0.0 ,
               'F1' : 0.0,
               'TP_delay' : 0.0}
               
    # set of point anomalies detected as true
    anom_TP = set()
    
    # Set of anomalous segments detected           
    anom_segments_detected_set  = set()  
    # Table to record frequency of anomalous segment detections
    anomalies_detected_tab  = zeros_like(ground_truths_tab)
    anomalies_detected_tab[:,0] = ground_truths_tab[:,0]
    
    # TRUE POSITIVES
    # Run through ground truths 
    idx = 0
    for truth in ground_truths_tab:
        count = 0
        
        # Run through the list of detections    
        for d in D :
            if d >= truth[0] and d <= truth[0] + truth[1] + epsilon:
                # if set does not yet contain the anomaly, add it and increment TP
                if not anom_segments_detected_set.issuperset(set([truth[0]])):
                    
                    anom_segments_detected_set.add(truth[0])
                    anom_TP.add(d)
                    metric['TP'] += 1
                    count += 1
                    
                    # Calculate minimum Detection Delay 
                    metric['TP_delay'] = metric['TP_delay'] + (d - truth[0])                     
                    
                    
                else: # if multiple detections in anomalous segment 
                    count += 1 
                    anom_TP.add(d)                    
                    
        anomalies_detected_tab[idx,1] = count   
        idx += 1     
    
    # Average minimum detection delay
    if metric['TP'] == 0 :
        metric['TP_delay'] = nan
    else:
        metric['TP_delay'] = metric['TP_delay'] / len(ground_truths_tab[0])
    
    # FALSE Pos 
    anom_FP = D - anom_TP    
    metric['FP'] = len(anom_FP)
    
    # FALSE Neg     
    anom_FN = set(ground_truths_tab[:,0]) - anom_segments_detected_set
    metric['FN'] = len(anom_FN)
    
    if metric['TP'] == 0 and metric['FP'] == 0 :
        metric['precision'] = 0.0
    else:
        metric['precision'] = metric['TP'] / (metric['TP'] + metric['FP'])
    
    if metric['TP'] == 0 and metric['FN'] == 0 :
        metric['recall'] = 0.0
    else:
        metric['recall'] = metric['TP'] / (metric['TP'] + metric['FN'])    
    
    metric['FPR'] = metric['FP'] / total_negative
  #  metric['FDR'] = metric['FP'] / (metric['FP'] + metric['TP'])    
  #  metric['ACC'] = (metric['TP'] + total_negative - metric['FP'] )/  \
  #                 ( metric['TP'] + metric['FN'] + total_negative )
    metric['F1'] = fmeasure(1, metric['TP'], metric['FN'], metric['FP'])
    metric['F2'] = fmeasure(2, metric['TP'], metric['FN'], metric['FP'])
    metric['F05'] = fmeasure(0.5, metric['TP'], metric['FN'], metric['FP']) 
    
    if return_sets_n_tab == 1:
        sets = {'TP' : anom_TP,
            'anom_seg_detected' : anom_segments_detected_set,
            'FN' : anom_FN,
            'FP' : anom_FP}     
        return metric, sets, anomalies_detected_tab
    else:
        return metric


def aveMetrics(metricsList):
    ''' Averages metrics over all initial conditions 
        Calculates averages and standard deviation 
    '''
    
    # Initialise Lists    
    TP_list = []    
    FP_list = []    
    FN_list = []    
    precision_list = []    
    recall_list = []    
    F05_list = []    
    F1_list = []    
    F2_list = []    
    TP_delay_list = []    

    # append metrics for all initial conditions     
    for metric in metricsList:
        
        TP_list.append(metric['TP'])  
        FP_list.append(metric['FP'])     
        FN_list.append(metric['FN'])
        precision_list.append(metric['precision']) 
        recall_list.append(metric['recall'])
        F05_list.append(metric['F05'])
        F1_list.append(metric['F1'])
        F2_list.append(metric['F2'])
        TP_delay_list.append(metric['TP_delay'])
        
    # Calculate stats and save as dictionary of dictionaries
    AllMetrics = { 'TP' : {'25th' : stats.scoreatpercentile(TP_list, 25), 
                           'Median' : stats.scoreatpercentile(TP_list, 50),                           
                           '75th' : stats.scoreatpercentile(TP_list, 50) 
                           },
                           
                   'FP' : {'25th' : stats.scoreatpercentile(FP_list, 25), 
                           'Median' : stats.scoreatpercentile(FP_list, 50),                           
                           '75th' : stats.scoreatpercentile(FP_list, 50)   
                           },
                           
                   'FN' : {'25th' : stats.scoreatpercentile(FN_list, 25), 
                           'Median' : stats.scoreatpercentile(FN_list, 50),                           
                           '75th' : stats.scoreatpercentile(FN_list, 50)   
                           },
                           
            'precision' : {'25th' : stats.scoreatpercentile(precision_list, 25), 
                           'Median' : stats.scoreatpercentile(precision_list, 50),                           
                           '75th' : stats.scoreatpercentile(precision_list, 50)  
                           },
                           
              'recall' : {'25th' : stats.scoreatpercentile(recall_list, 25), 
                           'Median' : stats.scoreatpercentile(recall_list, 50),                           
                           '75th' : stats.scoreatpercentile(recall_list, 50)
                          },
                           
                'F05' : {'25th' : stats.scoreatpercentile(F05_list, 25), 
                           'Median' : stats.scoreatpercentile(F05_list, 50),                           
                           '75th' : stats.scoreatpercentile(F05_list, 50)    
                          },

                'F1' : {'25th' : stats.scoreatpercentile(F1_list, 25), 
                           'Median' : stats.scoreatpercentile(F1_list, 50),                           
                           '75th' : stats.scoreatpercentile(F1_list, 50)    
                          },                          
                          
                'F2' : {'25th' : stats.scoreatpercentile(F2_list, 25), 
                           'Median' : stats.scoreatpercentile(F2_list, 50),                           
                           '75th' : stats.scoreatpercentile(F2_list, 50)    
                          },                          
                
                'TP_delay' : {'25th' : stats.scoreatpercentile(TP_delay_list, 25), 
                           'Median' : stats.scoreatpercentile(TP_delay_list, 50),                           
                           '75th' : stats.scoreatpercentile(TP_delay_list, 50)    
                          }}
                          
    return AllMetrics
