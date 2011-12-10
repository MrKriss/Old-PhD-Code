#!/usr/bin/env python
#coding:utf-8
# Author:  C Musselle --<>
# Purpose: Get QR decomposition + eigenvalues estimation working together for X.T 
# Created: 08/16/11

import numpy as np
import matplotlib.pyplot as plt
import sys
import os 
import time 

# Fast Back Substitution in Matlab
#n = length( b );
#x = zeros( n, 1 );
#for i=n:-1:1
#   x(i) = ( b(i) - U(i, :)*x )/U(i, i);
#end


def QRsolve_eigV(A,Z,h, Ut_1): # For arrays
    '''Estimates the eigenvalues of A and then solves equations of the type 
            Ax = b by using A = QR ==> R * x = Q^T b ''' 
   
    W = np.dot(A , Ut_1)
    
    U, R = np.linalg.qr(W)
    
    r = np.sqrt(Z) * np.dot(U.T , h) 
    
    # Now Rs = r ----> solve for s
    # Compact and faster back substitution 
    n = R.shape[0]
    s = np.zeros((n,1))
    for i in range(n-1,-1,-1):  # i=n:-1:-1  
        s[i] = (r[i] - np.dot(R[i, :],s)) / R[i, i]

    # The solution for Ax = b
    x = np.dot(Ut_1,s)
    # The eignevalue estimates for A
    e_values = R.diagonal()   
        
    return x, e_values, U

def QRsolveA1(A,b): # For arrays
    '''Solves equations of the type Ax = b by using A = QR ==> R * x = Q^T b ''' 
    Q, R = np.linalg.qr(A)
    cc = np.dot(Q.T , b)
    n = Q.shape[0]
    x = np.zeros((n,1))
    for j in range(n-1, -1, -1): # n-1, ....., 0
        if j != n-1:
            sum_rjk_xk = 0
            for k in range(j+1, n):
                sum_rjk_xk =  sum_rjk_xk + (R[j,k] * x[k])
            x[j] = (cc[j] - sum_rjk_xk) / R[j,j] 
        else:    
            x[j] = cc[j] / R[j,j]         
    return x

def QRsolveA2(A,b): # For arrays
    '''Solves equations of the type Ax = b by using A = QR ==> R * x = Q^T b ''' 
    Q, R = np.linalg.qr(A)
    cc = np.dot(Q.T , b)
    n = Q.shape[0]
    x = np.zeros((n,1))
    for j in range(n-1, -1, -1): # n-1, ....., 0
        x[j] = (cc[j] - np.dot(R[j, :],x)) / R[j, j]
    return x


if __name__=='__main__':
    
    # Test example
    # x --> 2, 5, 7
    A = np.array([[ 2, 1, -1], [ 1, 4, 1], [ 1, -1, 1]])
    b = np.array([[2], [29], [4]])
    
    x1 = QRsolveA1(A,b)
    x2 = QRsolveA2(A,b)
    
    start_x1 = time.time()
    for i in xrange(100000):
        x1 = QRsolveA1(A,b)

    x1_time = time.time() - start_x1

    start_x2 = time.time()
    for i in xrange(100000):
        x2 = QRsolveA2(A,b)

    x2_time = time.time() - start_x2
    
    