# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 05:36:13 2011

@author: musselle
"""

import scipy.io as sio
from utils import analysis
from PedrosFrahst import frahst_pedro
from Frahst_v3 import FRAHST_V3
from artSigs import genCosSignals_no_rand , genCosSignals

#AbileneMat = sio.loadmat('C:\DataSets\Abilene\Abilene.mat')
#data = AbileneMat['F']

data = genCosSignals_no_rand()

data = genCosSignals(0,-3.0)

e_high = 0.98
e_low = 0.96
alpha = 0.96


holdTime = 0
# My version  
res_me = FRAHST_V3(data, alpha=0.96, e_low=0.96, e_high=0.98, sci = -1, \
holdOffTime = holdTime, r = 1, evalMetrics = 'T') 
#metric_me_5, sets_me, anom_det_tab_me = analysis(res_me, AbileneMat['F_g_truth_tab'])

# Pedros Version
res_ped = frahst_pedro(data, r=1, alpha=0.96, energy_low=0.96, energy_high=0.98,  \
holdOffTime = holdTime, evaluateTruth='T')
#metric_ped_5, sets_ped, anom_det_tab_ped = analysis(res_ped, AbileneMat['F_g_truth_tab'])