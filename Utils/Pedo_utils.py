# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 09:51:28 2011

Pedros Utility Functions

@author: Musselle
"""

from numpy import zeros, mean, inf, isnan, median, array
import scipy.io as sio
from Frahst_v3 import FRAHST_V3
from utils import analysis
from PedrosFrahst import frahst_pedro

def fmeasure(B, hits, misses, falses) :
    x = ((1 + B**2) * hits) / ((1 + B**2) * hits + B**2 * misses + falses)
    return x

def filterConsecutive(detections) : 
    ''' 
    Return a filtered list where consecutive detections are discounted. 
    '''    
    
    filtered = []
    
    if len(detections) > 0 :
        
        lastPoint = -inf
        for i in range(len(detections)):
            
            if not isnan(detections[i]):
                if lastPoint + 1 != detections[i] : 
                    filtered.append(detections[i])            
                lastPoint = detections[i]
            else :
                lastPoint = inf
                        
    return(filtered)

def scoreEvaluation(n, detections, truth, epsilon = 1, ignoreUpTo=0):

#Score considering only if anomaly segment was detected, not counting actual duration
# n -> is total data points
# detections -> vector of ticks indicating begining of anomalies 
# truth -> list of anomalies (tick and duration)
##

    print "Evaluating data with total %s ticks and %s detections of %s true anomalies" % (n, len(detections), len(truth))

    #apply here 
    
    # Hmmm Function to consolidate consecutive detections? -- Done
    detections = filterConsecutive(detections)
    

    print 'Detections' , detections
    print 'Truths    ' , truth[:,0]    
    
    detections = array(detections)    
    
    if ignoreUpTo > 0 : 
        #filtering        
        detections  = detections[detections > ignoreUpTo ]           
        truth[truth[:,0]  > ignoreUpTo]
        print 'After Filtering'
        print 'Detections' , detections 
        print 'Truths    ' , truth[:,0] 
        
    hits = 0
    misses = 0
    falses = 0
    
    total_positive = []    
    for item in truth:
        ma = max(item[0] - epsilon,0)
        mi = min(item[0] + item[0],n)
        total_positive.append(range(ma, mi + 1))

    total_negative = n - ignoreUpTo - len(total_positive)
    
    detection_lags = [] #distance to start of anomaly
    
    if len(truth) > 0 :  
        #give little epsilon    
        expandedDetections = zeros(n)
        idx = []
        for elem in detections:
            ma = max(elem - epsilon, 0)
            mi = min(elem, n)
            idx.extend(range(ma, mi + 1)) 
        expandedDetections[idx] = 1        
        
        
        for i in range(len(truth)) :  
            tick = truth[i, 0]
            duration = truth[i, 1]
            
            # Section of overlap between truth and detections            
            idx = expandedDetections[tick:min(tick + duration,n)] == 1
            intercept = expandedDetections[tick:min(tick + duration,n)][idx]
            
            if len(intercept) > 0 :
                hits = hits + 1
                
                detection_lags.append(intercept[0])
            else:
                print "miss at %s" % tick
                misses = misses + 1


    if len(detections) > 0 :
        expandedAnomalies = zeros(n)
        
        idx = []
        for elem in truth:
            ma = max(elem[0] - epsilon,0)
            mi = min(elem[0] + elem[1] , n)
            idx.extend(range(ma,mi+1)) 
        expandedAnomalies[idx] = 1  
        
         
        for i in range(len(detections)) : 
            if expandedAnomalies[detections[i]] == 0 : 
                print "false at %s" % detections[i]
                falses = falses + 1            
    
    print "detections = %s    misses = %s    falses = %s\n" % (hits, misses, falses)    
    
    print "N = %s" % total_negative

    # Convert to floats    
    hits = float(hits)
    misses = float(misses)
    falses = float(falses)
    total_negative = float(total_negative)
    
    # Calculate Metrics 
    metrics = {}
    metrics['recall'] = hits / ( hits + misses)
    metrics['precision'] = hits / (hits + falses)
    metrics['FPR'] = falses / total_negative
    metrics['FDR'] = falses/(falses + hits)
    metrics['ACC'] = (hits + total_negative - falses )/( hits + misses + total_negative )
    metrics['F1'] = fmeasure(1, hits, misses, falses)
    metrics['F2'] = fmeasure(2, hits, misses, falses)
    metrics['F05'] = fmeasure(0.5, hits, misses, falses)    

    print " TPR/Recall = %s" % metrics['recall'] 
    print " Precision = %s" % metrics['precision']    
    print " FPR = %s " % metrics['FPR']
    print " FDR = %s" % metrics['FDR']
    print " ACC = %s" % metrics['ACC']
    print " F1 = %s   " % metrics['F1'] 
    print " F2 = %s   " % metrics['F2'] 
    print " F0.5 = %s   " % metrics['F05']         
    
    if hits > 0 :
        print "mean detection lag = %s" % mean(detection_lags)
        print "max detection lag = %s" % max(detection_lags)
        print "median detection lag = %s" % median(detection_lags)   

    metrics['hits'] = hits
    metrics['misses'] = misses
    metrics['falses'] = falses    
    metrics['lags'] = detection_lags    
    
    return metrics
    
if __name__ == '__main__' : 
    
       
    AbileneMat = sio.loadmat('/Users/Main/DataSets/Abilene/Abilene.mat')
    data = AbileneMat['P']
    
    n = data.shape[0]    
    
    e_high = 0.98
    e_low = 0.96
    alpha = 0.96
    sci = -1

    holdOFF = 50

    # My version 
    res_me = FRAHST_V3(data, alpha=0.96, e_low=0.96, e_high=0.98, sci = -1, \
    holdOffTime=holdOFF, r = 1, evalMetrics = 'F') 

    metric_me, sets_me, anom_det_tab_me = analysis(res_me, AbileneMat['P_g_truth_tab_alt'], n)

    new_metrics_me = scoreEvaluation(n, res_me['anomalies'], 
                                  AbileneMat['P_g_truth_tab_alt'], ignoreUpTo = 400)
                                  
    # Pedros Version
    res_ped = frahst_pedro(data, r = 1, alpha=0.96, energy_low=0.96, energy_high=0.98,  \
    holdOffTime=holdOFF, evaluateTruth='FALSE')
    
    metric_ped, sets_ped, anom_det_tab_ped = analysis(res_ped, AbileneMat['P_g_truth_tab_alt'], n)
    
    new_metrics_ped = scoreEvaluation(n, res_ped['anomalies'], 
                                  AbileneMat['P_g_truth_tab_alt'], ignoreUpTo = 400)

    
    
    
    
    