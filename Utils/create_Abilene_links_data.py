#!/usr/bin/env python
#coding:utf-8
# Author:  Musselle --<>
# Purpose: Creation of Abilene Link flow data from Abilene OD flow data.
# Created: 08/24/11

import numpy as np
import networkx as nx
import os
import scipy.io as sio

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

def genR(G):
    """ 
    Returns Routing marix R for graph G based on shortest paths     
    """  

    l = G.number_of_edges()
    n = G.number_of_nodes()
    
    # Generate directed graph with n nodes and l links
    # G = nx.generators.random_graphs.gnm_random_graph(n,l, seed = s)

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
    R = np.zeros((l, n**2))

    # initialise array        
    short_paths = np.zeros((n, n), dtype = np.object)
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
        
    return R, edg2link, short_paths

#----------------------------------------------------------------------
def graph_from_adjM(links):
    """Generates graph from an adjacency matrix and returns it"""
    
    G = nx.Graph()    
    nodes = range(links.shape[0]) 
    G.add_nodes_from(nodes)
    
    edge_list = []
    E = np.triu(links)
    for i in nodes:
        for j in nodes:            
            if E[i,j] == 1:
                edge_list.append((i,j))
    
    G.add_edges_from(edge_list)
    
    return G
    
def create_abilene_links_data(data_source = 'packets'):
    '''Load data''' 
    current_dir = os.getcwd()
    os.chdir('/Users/chris/DataSets/Abilene')
    AbileneMat = sio.loadmat('/Users/chris/DataSets/Abilene/Abilene.mat')
    
    if data_source == 'packets':
        # Number of Packets 
        data = AbileneMat['P']
    elif data_source == 'flows':
        # Number of indervidual IP flows
        data = AbileneMat['F']
    elif data_source == 'bytes':
        # Number of bytes in each flow
        data = AbileneMat['B']
    
    # Links Matrix 
    links = np.load('link_matrix.npy')
    os.chdir(current_dir)
    
    '''Create Graph and Y links count matrix'''
    G = graph_from_adjM(links)
    R, edg2link, short_paths = genR(G)
    # Final Links Data     
    Y = np.zeros((data.shape[0], G.number_of_edges()))
    for t in range(Y.shape[0]): # for each time step
        X = np.atleast_2d(data[t,:])
        Y[t,:] = np.dot(R , X.T).T

    return Y, G

if __name__=='__main__':

    Y, G = create_abilene_links_data()
