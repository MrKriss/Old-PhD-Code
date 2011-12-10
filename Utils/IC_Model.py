# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 10:57:31 2011

Functions for Creating Simulated Data Using the Independent Connections (IC) 
model for Traffic Matrix (TM) generation.

@author: - Musselle
"""

from numpy import zeros, cos, arange, pi, ones, dot, atleast_2d, eye, squeeze, \
transpose, array 
from numpy.random import lognormal, uniform, normal, shuffle
import numpy as np
import networkx as nx
from pylab import show, figure
from random import randrange
import pickle
import os

def gen_TM_PM(nodes, f, time_steps):
    """ Generator for TM and PM using the stable f and P IC model 
    
    Function creates TM for each single time step. A  
    varies with time, according to a simple periodic pattern.
    
    P_i = vector [1,...,nodes] drawn from lognormal dist with u = -4.2 sigma = 1.7
    A_i = vector [1,...,nodes] varies with t over a repeating sinosoidal period. 
    Indervidual A_i are scaled to this sinosoid.
    
    
    """    

    max_activity = 100000
    
    # Simple Periodic signal, with uniform random means
    A_scale = uniform(low = 1000, high = max_activity, size = nodes)

    # temperary sinosoidal pattern

    h0 = 0.771
    h1 = -0.771
    h2 = -0.067
    h3 = 0.159
    h4 = 0.024
    h5 = -0.034

    time_vec = arange(time_steps) + 1

    period = 2016./7.

    sig = ((h0 * cos(2*pi* 0 * time_vec / period)) + 
          (h1 * cos(2*pi* 1 * time_vec / period)) +  
          (h2 * cos(2*pi* 2 * time_vec / period)) +
          (h3 * cos(2*pi* 3 * time_vec / period)) +
          (h4 * cos(2*pi* 4 *time_vec / period)) +
          (h5 * cos(2*pi* 5 *time_vec / period)))
          
    # Normalise
    sig = sig / max(sig)

    A = zeros((time_steps, nodes))

    # Generate A matrix time_stemps x nodes    
    for i in range(nodes):
        for t in range(time_steps):
            A[t,i] = max(sig[t] * A_scale[i] + normal(0, A_scale[i] * 0.1), 0)

        
    # Draw Pi from lognormal 
    P = lognormal(-4.3, 1.7, nodes)    
    sum_P = sum(P) 

    # Traffic Matrix initialise
    X = zeros((nodes,nodes,time_steps))     
    
    # Packet Matrix initialise + parameters
    Z = zeros((nodes,nodes,time_steps), dtype = np.object)
    fraction = 0.01 # Fraction of total packets sampled
    
    for t in range(time_steps):
        for i in range(nodes):
            for j in range(nodes):
                
                forward_T = (f * A[t,i] * P[j]) / sum_P
                backward_T = ((1-f) * A[t,j] * P[i]) / sum_P                 
                X[i,j,t] = forward_T + backward_T 
                
                forward_P = [i] * max(int(round(forward_T * fraction)), 1) 
                backward_P = [j] * max(int(round(backward_T * fraction)), 1)
                temp = []
                temp.extend(forward_P)
                temp.extend(backward_P)
                shuffle(temp)
                # Store node ID packets in a 1D array 
                Z[i,j,t] = temp
                
    return X , Z


def all_shortest_paths(G,a,b): 
    """ Return a list of all shortest paths in graph G between nodes a 
and b """ 
    ret = [] 
    pred = nx.predecessor(G,b) 
    if not pred.has_key(a):  # b is not reachable from a 
        return [] 
    pth = [[a,0]] 
    pthlength = 1  # instead of array shortening and appending, which are relatively 
    ind = 0        # slow operations, we will just overwrite array elements at position ind 
    while ind >= 0: 
        n,i = pth[ind] 
        if n == b: 
            ret.append(map(lambda x:x[0],pth[:ind+1])) 
        if len(pred[n]) > i: 
            ind += 1 
            if ind == pthlength: 
                pth.append([pred[n][i],0]) 
                pthlength += 1 
            else: 
                pth[ind] = [pred[n][i],0] 
        else: 
            ind -= 1 
            if ind >= 0: 
                pth[ind][1] += 1 
    return ret 


def genR(n, l, s):
    """ Generate a Random graph with network x package 
    
    returns Routing marix R for random graph G based on shortest paths     
    """  
    import networkx as nx

    # Generate directed graph with n nodes and l links
    G = nx.generators.random_graphs.gnm_random_graph(n,l, seed = s)

    # Edges to Links dictionary    
    edg2link = {}
    ind = 0    
    for edg in G.edges_iter():
        edg2link[str(edg)] = ind
        opposite_edg = tuple((edg[1],edg[0]))
        edg2link[str(opposite_edg)] = ind
        ind += 1
    
    # Calculate Routing matrix 
    # R is an L x P matrix
    # L is number of edges
    # P dimention goes from s0d0,...s0dn,...,snd0,...sndn
    R = zeros((l, n**2))

    # initialise array        
    short_paths = zeros((n, n), dtype = np.object)
    # short_paths[:] = []   
    
    for i in range(n):
        for j in range(n):  
            # Pair index for R matrix 
            pair = i * n + j
            # Array containing lists of all the shortest paths
            short_paths[i,j] = all_shortest_paths(G, i, j)

            num_paths = len(short_paths[i,j])
            for k in range(num_paths): # for each path
                
                path = short_paths[i,j][k]
                for p_idx in range(len(path)- 1): # for each edge                                        
                    edg = tuple([path[p_idx], path[p_idx + 1]])                     
                    link = edg2link[str(edg)]                                        
                    # Update R matrix                    
                    R[link, pair] = 1. / num_paths 
        
    return R, G, edg2link, short_paths

    

if __name__ == '__main__' : 

    # Define parameters 
    nodes = 8
    links = 15
    f = 0.25
    time_steps = 2016
    
    my_seed = 111    
    
    # Generate Traffic Matrix and Packet Matrix
    X_t, Z_t = gen_TM_PM(nodes, f, time_steps)
    
    # Link flows matrix
    Y_t = zeros((time_steps,links))
    
    # Generate Routing matrix from random graph 
    R, G, edg2link, s_paths = genR(nodes,links, my_seed)    
    
    # Calculate link Flows 
    for i in range(time_steps):
        X_temp = atleast_2d(X_t[:,:,i].flatten())
        Y_t[i,:] = squeeze(transpose(dot(R, X_temp.T)))
        
    # Packet flows matrix P
    P_t = zeros((time_steps, links), dtype = np.object) 
    
    # Calculate Packet Flows
    for t in range(time_steps): # for each row of P_t
        
        # Z_t as a flattened 1 by ODpairs vector
        Z_temp = atleast_2d(Z_t[:,:,t].flatten())
        
        for p_link in range(links):  # for each link at time t     
            
            # initalise P_t object as a list
            P_t[t, p_link] = []        
        
            for r in range(len(R[p_link,:])): # Run through each OD pair
                
                if R[p_link,r] > 0: 
                    # R[t,r] = fraction of packets in OD pair r that go along 
                    # link P_link, pop that fraction form the total 
                
                    num_2_pop =  int(round(R[p_link,r] * len(Z_temp[0,r])))                    
                    
                    for k in range(num_2_pop):
                        if len(Z_temp[0, r]) > 0: # if there as still packets to pop
                            packet = Z_temp[0, r].pop(randrange(len(Z_temp[0,r])))                        
                            P_t[t, p_link].append(packet)
    

    packet_numbers = zeros((time_steps, links))
    # Claculate number of antigen
    for i in range(time_steps):
        for j in range(links):
            packet_numbers[i,j] = len(P_t[i,j]) 
    
    figure()    
    nx.draw(G)
    show()
    
    os.chdir('/Users/Main/DataSets/Python/Network_ODsmall')    
    
    with open('P_t_data.pk', 'w') as output:
        pickle.dump(P_t, output, pickle.HIGHEST_PROTOCOL)
        
    with open('Y_t_data.pk', 'w') as output:
        pickle.dump(Y_t, output, pickle.HIGHEST_PROTOCOL)
        
    with open('other_data.pk', 'w') as output:
        pickle.dump(packet_numbers, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(R, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(G, output, pickle.HIGHEST_PROTOCOL)
