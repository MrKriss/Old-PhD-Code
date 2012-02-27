#!/usr/bin/env python
#coding:utf-8
# Author:  C Musselle --<>
# Purpose: Symbollic Aggregate Approximation 
# Created: 11/14/11

import numpy as np
import numpy.random as npr
import scipy.spatial.distance as distance
import scipy.cluster.hierarchy as sph
import matplotlib.pyplot as plt
import sys
import os 
import string
from normalisationFunc import zscore, zscore_win
from load_data import load_ts_data, load_data
from gen_anom_data import gen_a_step_n_back
from itertools import permutations 
#import ete2 as ete 

#import hcluster

"""
Code Description:
  Implimentation of SAX.
"""

def SAX(data, alphabet_size, word_size, minstd = 1.0, pre_normed = False):
  """ Returns one word for each data stream 
  
  word_size == Number of segments data is split into for PAA
  alphabet_size == Number of symbols used
  """
    
  num_streams = data.shape[1]    
    
  # Need to insert check here for stationary segemnts
  mask = data.std(axis=0) < minstd
  passed = np.invert(mask)
  if np.any(mask):    
    # Scale data to have a mean of 0 and a standard deviation of 1.
    if pre_normed == False:
      data[:,passed] = zscore(data[:, passed])
    symbol4skips = string.ascii_letters[int(np.ceil(alphabet_size/2.))]
  else:
    # Scale data to have a mean of 0 and a standard deviation of 1.
    if pre_normed == False:
      data = zscore(data)
  
  # Calculate our breakpoint locations.
  breakpoints = bp_lookup(alphabet_size)
  breakpoints = np.concatenate((breakpoints, np.array([np.Inf])))

  # Split the data into a list of word_size pieces.
  data = np.array_split(data, word_size, axis=0)
  
  # Predifine Matrices 
  segment_means = np.zeros((word_size,num_streams))
  #segment_symbol = np.zeros((word_size,num_streams), dtype = np.str)
  p_array = np.zeros((num_streams,),dtype = ('a1,' * word_size + 'i2'))
  p_dict = {}
  
  # Calculate the mean for each section.  
  for i in range(word_size):
    segment_means[i,passed] = data[i][:,passed].mean(axis = 0) 
    
  # Figure out which break each section is in based on the section_means and
  # calculated breakpoints. 
  for i in range(num_streams): 
    for j in range(word_size):
      if passed[i]:
        idx = int(np.where(breakpoints > segment_means[j,i])[0][0])
        # Store in phrase_array 
        p_array[i][j] = string.ascii_letters[idx]
      else:
        p_array[i][j] = symbol4skips
    
    # Store in phrase_dict
    phrase  = ''.join(tuple(p_array[i])[:word_size])
    if p_dict.has_key(phrase):
      p_dict[phrase].append(i)
    else:
      p_dict[phrase] = [i]

  # Put frequency of pattern in p_array
  for vals in p_dict.itervalues():
    count = len(vals)
    for i in range(count):
      p_array[vals[i]][-1] = count  
  
  return p_array, p_dict, segment_means




def occurances(string, sub):
    count, start = 0,0
    while True:
        start = string.find(sub, start) + 1
        if start > 0:
            count+=1
        else:
            return count

def bitmap(p_dict, level, mode = 'indervidual' ):
  """ Make 2x2 Time series bitmap for the list of phrases"""
  
  phrases = p_dict.keys()
  
  dim = 2 ** level
  
  if level == 1:
    syms = ['a', 'b', 'c', 'd']
  elif level == 2:
    syms =  ['aa', 'ab', 'ba', 'bb', 
             'ac', 'ad', 'bc', 'bd',
              'ca', 'cb', 'da', 'db', 
              'cc', 'cd', 'dc', 'dd']
              
  elif level == 3:
    syms = ['aaa', 'aab', 'aba', 'abb', 'baa', 'bab', 'bba', 'bbb', 
            'aac', 'aad', 'abc', 'abd', 'bac', 'bad', 'bbc', 'bbd', 
            'aca', 'aca', 'ada', 'adb', 'bca', 'bca', 'bda', 'bdb',
            'acb', 'acd', 'adc', 'add', 'bcb', 'bcd', 'bdc', 'bdd', 
            'caa', 'cab', 'cba', 'cbb', 'daa', 'dab', 'dba', 'dbb', 
            'cac', 'cad', 'cbc', 'cbd', 'dac', 'dad', 'dbc', 'dbd', 
            'cca', 'cca', 'cda', 'cdb', 'dca', 'dca', 'dda', 'ddb',
            'ccb', 'ccd', 'cdc', 'cdd', 'dcb', 'dcd', 'ddc', 'ddd']
  
  # Storage for bitmaps 
  dtype_string = '({0},{0})float32'.format(dim)
  bitmaps= np.zeros(len(phrases), dtype = dtype_string)

  # for each phrase
  for i in range(len(phrases)): 
    counts = []
    #for each subsring 
    for j in range(dim**2):
      counts.append(occurances(phrases[i], syms[j]))
    
    # Update Bitmap 
    bitmaps[i] = np.array(counts).reshape(dim,dim)
    # Normalise
    bitmaps[i] = bitmaps[i] / bitmaps[i].max()
  
  return bitmaps

def bitmap2(p_dict, alphaSize, level):
  """ Make 2x2 or 3x3 Time series bitmap for the list of phrases 
  """
  
  # Define symbol alphabet 
  alphabet = string.ascii_lowercase[:alphaSize]    
  
  # filter out stationary phrase
  phrases = p_dict.keys()
  wordLength = len(phrases[0])  
  symbol4skips = alphabet[int(np.ceil(alphabet_size/2.))-1]
  if symbol4skips * wordLength in phrases:
    phrases.remove(symbol4skips * wordLength)


  # Find all permutations of alphabet to required level depth 
  syms = []

  if level == 2:
    for i in alphabet:
      for j in alphabet:
        syms.append(i+j)
    dim = 2 ** level
  
  elif level == 3:
    for i in alphabet:
      for j in alphabet:
        for k in alphabet:
          syms.append(i+j+j)    
    dim = 3 ** level
  
  # Storage for bitmaps 
  dtype_string = '({0},{0})float32'.format(dim)
  bitmaps= np.zeros(len(phrases), dtype = dtype_string)

  # for each phrase
  for i in range(len(phrases)): 
    counts = []
    # for each subsring 
    for j in range(dim**2):
      counts.append(occurances(phrases[i], syms[j]))
    
    # Update Bitmap 
    bitmaps[i] = np.array(counts).reshape(dim,dim)
    
    # Dont think there is any reason to normalise
    # Normalise: min-max
    #bitmaps[i] = bitmaps[i] - bitmaps[i].min() 
    #bitmaps[i] = bitmaps[i] / bitmaps[i].max()
  
  return bitmaps


def min_dist(str1, str2, alphabet_size, compression_ratio):
  """ Return minimum distance between two SAX phrases """
  
  # Error Check 
  assert len(str1) == len(str2), 'error: the strings must have equal length!'
  assert (len(set(str1)) <= alphabet_size) or (len(set(str2)) <= alphabet_size) ,\
  'error: some symbol(s) in the string(s) exceed(s) the alphabet size!'
  
  # Get breakpoints
  breakpoints = bp_lookup(alphabet_size)
  
  # Build distance table
  dist_matrix = np.zeros((alphabet_size, alphabet_size))

  for i in range(alphabet_size):
    # the min_dist for adjacent symbols are 0, so we start with i+2
    for j in range(i+2 , alphabet_size):
      # square the distance now for future use
      dist_matrix[i,j]= (breakpoints[i] - breakpoints[j-1]) ** 2
      # the distance matrix is symmetric
      dist_matrix[j,i] = dist_matrix[i,j]

  dist = 0.
  for k in range(len(str1)):
    i = int_str_map(str1[k])
    j = int_str_map(str2[k])
    dist = dist + dist_matrix[i,j] ** 2
  
  return np.sqrt(compression_ratio * dist)

def dist_mat(p_dict, alphabet_size, compression_ratio):
  """ Calculate distance matrix using min_dist function 
  
  Ignores any Stationary phrases.
  """
  
  # filter out stationary phrase
  word_list = p_dict.keys()
  wordLength = len(word_list[0])  
  symbol4skips = string.ascii_lowercase[int(np.ceil(alphabet_size/2.))]
  if symbol4skips * wordLength in word_list:
    word_list.remove(symbol4skips * wordLength)  
  
  N = len(word_list)
  dist_array = np.zeros((N,N))
  
  for i in range(N):
    for j in range(N):
      dist_array[i,j] = min_dist(word_list[i], word_list[j], alphabet_size, compression_ratio)
  
  return dist_array


def bitmapDistMat(bitmapList):
  
  N = len(bitmapList)
  distArray = np.zeros((N,N))
    
  for i in range(N):
    for j in range(N):
      distArray[i,j] = np.sum(np.abs(bitmapList[i] - bitmapList[j]))
    
  return distArray  


def save_tree(word_list, calc_distances = 0):
  """ Build and store a tree for the list of phrases """
  
  num_words = len(word_list)
  word_length = len(word_list[0])

  T = ete.Tree()
  T.add_feature('level',-1)
  
  # build tree bredth first 
  for idx in range(word_length): # for each character in word. 
    
    # Run through list of parent nodes at that level
    for current_node in T.search_nodes(level = idx-1):  
      
      matched_words = []
      # For current node, get list of words that filter down to it  
      if current_node.name != 'NoName' : 
        for word in word_list:
          if word[:idx] == current_node.name :        
            matched_words.append(word) 
      else:
        # If level -1 use all words 
        matched_words = word_list

      # Reset added names list 
      added_child_names = []
      
      for word in matched_words: # go through matching words 
        # Check if node already exists in tree 
        if word[:idx+1] in added_child_names:
          # increment count and pass to next loop
          (current_node&word[:idx+1]).count += 1
        else:
          # Other wise Add character as child to node 
          child = current_node.add_child(name = word[:idx+1])
          child.add_feature('count' , 1)
          child.add_feature('level' , idx)
          added_child_names.append(word[:idx+1])

  return T

def cluster(dist_mat):
  """ Cluster based on difference matrix """
  
  DM = np.ma.masked_array(dist_mat + np.eye(N) * 1*10**8)
  DM.mask = np.tril(np.eye(N))
  num_ts = dist_mat.shape[0] 
  cluster_levels = []

  ts = []
  for i in range(num_ts):
    ts.append(set(i))
  
  count = 0
  while run:
    #XXXXXX broke
    # Find min in DM and join sets
    indexes = np.ma.where(DM == DM.min())
    i_set = set(indexes[0])
    j_set = set(indexes[1])
    cluster_levels[count] = set.union(i_set, j_set)
    count += 1 
    #Exclude using mask 
    DM.mask[indexes] = 1
  
  
  
  for i in indexes[0]:
    for j in indexes[1]:      
      cluster_levels[0] = set.union(ts[i], ts[j])

  
def int_str_map(input_var):
  
  """ Map a char to int or int to char for 'a-z' """ 
 
  str2int = { 'a' : 0,
              'b' : 1,
              'c' : 2,
              'd' : 3,
              'e' : 4,
              'f' : 5,
              'g' : 6,
              'h' : 7,
              'i' : 8,
              'j' : 9,}
 
  int2str = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
  
  if type(input_var) == str:
    output = str2int[input_var]
  elif type(input_var) == int:
    output = int2str[input_var]
  
  return output 

def bp_lookup(alphabet_size):

  # Lookup Table
  LT = { '2' : np.array([0]),
         '3' : np.array([-0.43, 0.43]),
         '4' : np.array([-0.67, 0, 0.67]),
         '5' : np.array([-0.84, -0.25, 0.25, 0.84]),
         '6' : np.array([-0.97, -0.43, 0, 0.43, 0.97]),
         '7' : np.array([-1.07, -0.57, -0.18, 0.18, 0.57, 1.07]),
         '8' : np.array([-1.15, -0.67, -0.32, 0, 0.32, 0.67, 1.15]),
         '9' : np.array([-1.22, -0.76, -0.43, -0.14, 0.14, 0.43, 0.76, 1.22]),
         '10' : np.array([-1.28, -0.84, -0.52, -0.25, 0., 0.25, 0.52, 0.84, 1.28])}
  
  break_points = LT[str(alphabet_size)]
  
  return break_points

def rand_proj(p_array, num_dim, itter = 50):
  
  N = len(p_array)
  W = len(p_array[0]) - 1
  
  # construct collision matrix
  col_mat = np.zeros((N, N))
  rand_dim = np.zeros(num_dim)
  fields = [0] * num_dim
  
  for i in xrange(itter):
    # Create random projection 
    # gen rand numbers
    rand_i = []
    num_left = float(W)
    num_needed = float(num_dim)
    for i in xrange(W):
      # probability of selection = (number needed)/(number left)
      p = num_needed / num_left
      if npr.rand() <= p:
        rand_i.append(i) 
        num_needed -= 1
        num_left -=1
      else:
        num_left -=1   
    
    # create list of fields 
    for i,r in enumerate(rand_i):
      fields[i] = 'f' + str(r) 
    
    # Create projection 
    proj = p_array[fields]
    
    # Search for collisions
    for i in xrange(len(proj)):
      for j in xrange(i+1,len(proj)):
        
        if proj[i] == proj[j]:
          # increment colliosns matrix
          col_mat[i,j] +=1
  
  return col_mat
  
  


def plot_SAX(segMeans, alphaSize, compRatio):
  
  # Stored originally as wordSize x numStreams. Becaule plot func plots columsn by default 
  numStreams = segMeans.shape[1]
  wordSize = segMeans.shape[0]
  bpList = bp_lookup(alphaSize)  
  
  PAAStreams = np.zeros((wordSize*compRatio, numStreams)) 
  
  for stream in xrange(numStreams):
    temp = []
    for mean in segMeans[:,stream]: 
      temp.extend([mean]*int(compRatio)) 
    
    PAAStreams[:,stream] = np.array(temp)
  
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.plot(PAAStreams)
  for bp in bpList:
    ax.axhline(y=bp, xmin=0, xmax=PAAStreams.shape[0], ls = '--')
  
  plt.show()  
  
  return PAAStreams

  
if __name__== '__main__':
  
  # Generate some data
  #data = np.array([np.sin(np.linspace(0,2*np.pi,500)), np.cos(np.linspace(0,2*np.pi, 500)), np.cos(np.linspace(0,2*np.pi, 500))])
  #data = np.array([np.sin(np.linspace(0,2*np.pi,500))], ndmin = 2)
  
  # Real Data 
  #data = load_ts_data('isp_routers', 'full')  
  #data = data[500:601, :]
  
  #d2 = gen_a_step_n_back(50,1000,10,5,0.1, L2 = 100)
  #data = d2['data']
  #data = zscore_win(data, 100)

  # Scale data to have a mean of 0 and a standard deviation of 1.
  #data -= np.split(np.mean(data, axis=1), data.shape[0])
  #data *= np.split(1.0/data.std(axis=1), data.shape[0])

  #data = data.T
  
  data = np.load('SAXTestData.npy')
  
  dataSeg = data[400:600,:]  
  
  alphabet_size = 9
  word_size = 50
  compression_ratio = float(dataSeg.shape[0]) / float(word_size)
  
  w_array, w_dic, segMeans = SAX(dataSeg, alphabet_size, word_size)
  D = dist_mat(w_dic, alphabet_size, compression_ratio)
  PAA = plot_SAX(segMeans, alphabet_size, compression_ratio)
  
  B = bitmap2(w_dic, alphabet_size, level = 2)
  BD = bitmapDistMat(B)
  
  L = w_dic.keys()
  stationaryWord = string.ascii_lowercase[int(np.ceil(alphabet_size/2.))] * word_size
  if stationaryWord in L:
    L.remove(stationaryWord)

  # Plot Dendogram
  # condensed distance metric 
  BDvec = distance.squareform(BD)
  Z = sph.linkage(BDvec, method = 'complete')
  plt.figure()
  # something is wrong with the dendogram mapping 
  Dgram = sph.dendrogram(Z)
  plt.show()
  #T = save_tree(d.keys())
  
  #distVec = hcluster.squareform(M)
  
  #L = ['abccd', 'aaddb', 'abddb', 'abccb', 'aaabb', 'baaad']
  #T = save_tree(L)
  #M = dist_mat(L, alphabet_size, compression_ratio)
  #distVec = hcluster.squareform(M)
  