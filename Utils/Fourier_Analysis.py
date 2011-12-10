#!/usr/bin/env python
#coding:utf-8
# Author:  Musselle --<>
# Purpose: Example of Fourier analysis of a signal
# Created: 08/24/11
  
import numpy as np
from scipy import fftpack
import pylab as P
from load_data import load_data


def fourier_analysis(sig, time_step = 1.0, top_freqs = 5, zoomed_num = 15):

  time_vec = np.arange(0, len(sig), time_step)
  
  sample_freq = fftpack.fftfreq(sig.size, d=time_step)
  
  sig_fft = fftpack.fft(sig)
  
  # Only take first half of sampling frequency 
  pidxs = np.where(sample_freq > 0)
  freqs, power = sample_freq[pidxs], np.abs(sig_fft)[pidxs]
  
  # plot with zoomed in sub plot
  P.figure()
  P.plot(freqs, power)
  P.ylabel('Power')
  P.xlabel('Frequency [Hz]')
  axes = P.axes([0.3, 0.3, 0.5, 0.5])
  P.title('Peak frequency')
  P.stem(freqs[:zoomed_num], power[:zoomed_num])
  #P.setp(axes, yticks=[])
  #P.savefig('source/fftpack-frequency.png')
  
  # Find top x frequencies to use in reconstruction 
  full_sort_idx = power.argsort()
  
  find_idx = full_sort_idx >= top_freqs 
  
  
  sorted_power = power[sort_idx][::-1] # Sort decending
  component_freqs = freqs[sort_idx[:top_freqs]]
  
  # copy fft
  reduced_sig_fft = sig_fft.copy()
  
  # set all values not in component freqs to 0 
  L = np.array(reduced_sig_fft, dtype = bool)
  L[sort_idx[:top_freqs]] = False  
  reduced_sig_fft[L] = 0
  
  # Reconstruct signal
  reconstruct_sig = fftpack.ifft(reduced_sig_fft)
  
  # Plot original and reconstructed signal 
  P.figure()
  P.plot(time_vec, sig)
  P.plot(time_vec, reconstruct_sig, linewidth=2)
  P.ylabel('Amplitude')
  P.xlabel('Time [s]')
  
  return sig_fft, reduced_sig_fft, reconstruct_sig, freqs, power, component_freqs
  
def matlab_fourier_anal(data, sample_rate = 1.0):
 
  n = data.shape[0] 
  t = np.arange(n)

  P.figure()
  P.plot(t/sample_rate,data)
  P.xlabel('Time Units')
  P.ylabel('Data Magnitude')

  Y = fftpack.fft(data);
  freqs = np.arange(n/2.0) * sample_rate / n  
  power = np.abs(Y[:n/2.0])

  P.figure()
  markerline, stemlines, baseline = P.stem(freqs, power, '--')
  P.setp(markerline, 'markerfacecolor', 'b')
  # P.setp(baseline, 'color','r', 'linewidth', 2)
  P.xlabel('Cycles/ Unit Time')
  P.ylabel('Power')
  P.title('Periodogram')

  period = 1. / freqs
  
  k = np.arange(50)
  f = k / float(n)
  power2 = power[k];

  P.figure()
  markerline, stemlines, baseline = P.stem(f, power2, '--')
  P.setp(markerline, 'markerfacecolor', 'b')
  P.setp(baseline, 'color','r', 'linewidth', 2)
  P.xlabel('Unit Time / Cycle')
  P.ylabel('Power')
  P.title('Periodogram: First 50 periods')
  
#axis([0 max(f) 0 pmax])
#k = 2:3:41;
#f = k/n;
#period = 1./f;
#periods = sprintf('%5.1f|',period);
#set(gca,'xtick',f)
#set(gca,'xticklabel',periods)
#xlabel('years/cycle')
#ylabel('power')
#title('Periodogram detail')


#----------------------------------------------------------------------
def fft_analysis(sig):
  """"""
  N = 8 # number of points
  t = np.arange(N)/N  # define time
  f = np.sin(2*np.pi*t) # define function
  p = np.abs(fftpack.fft(f)) / (N/2) # absolute value of the fft
  p = p[:N/2] ** 2 # take the positve frequency half, only


if __name__=='__main__':
  
  # Approximate Basis Functions 
  # 7 days
  p1 = 7 * 24 * 60 / 5.
  f1 = 1. /p1
  # 5 days
  p2 = 5 * 24 * 60 / 5.  
  f2 = 1. /p2
  # 3 days 
  p3 = 3 * 24 * 60 / 5.  
  f3 = 1. /p3
  # 24 hours 
  p4 = 24 * 60 / 5.  
  f4 = 1. /p4  
  # 12 hours
  p5 = 12 * 60 / 5.
  f5 = 1. /p5
  # 6 hours 
  p6 = 6 * 60 / 5.
  f6 = 1. /p6
  # 3 hours 
  p7 = 3 * 60 / 5.
  f7 = 1. /p7
  # 1.5 hours 
  p8 = 90 / 5.
  f8 = 1. /p8  
  

  # Generate signal 
  time_step = 0.1
  period = 5.
  time_vec = np.arange(0, 20, time_step)
  sig = np.sin(2 * np.pi / period * time_vec) + np.cos(10 * np.pi * time_vec)
  
  Packets, Flows, Bytes = load_data('abilene')
  
  data = np.hstack((Packets[:,1], np.zeros((38,))))
  
  sig_fft, reduced_sig_fft, recon_sig, freqs, power, comp_freqs = fourier_analysis(data, time_step = 1)
  # matlab_fourier_anal(data)