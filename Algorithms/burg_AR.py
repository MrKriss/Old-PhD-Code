"""
file              CollombBurg.py

author/translator Ernesto P. Adorio,Ph.D.
                  UPDEPP (UP Clark, Pampanga)
                  ernesto.adorio@gmail.com

Version           0.0.1 jun 11, 2010 # first release.

References        Burg's Method, Algorithm and Recursion, pp. 9-11
"""
from math import cos
import numpy as np 

def  burg_AR(m,  x):
    """
    Based on Collomb's C++ code, pp. 10-11
    Burgs Method, algorithm and recursion
      m - number of lags in autoregressive model.
      x  - data vector to approximate.
    """
    N = len(x)-1
    coeffs = np.zeros(m)
    
    # initialize Ak
    Ak    = np.zeros(m+1)
    Ak[0] = 1.0 
    # initialize f and b.
    f  = x.copy()
    b = x.copy()
    # Initialize Dk
    Dk = 0.0
    for j in range(N+1):
        Dk += 2.0 * f[j] ** 2 
    Dk -= (f[0] ** 2) + (b[ N ] ** 2) 

    #Burg recursion
    for k in range(m):
        # compute mu
        mu = 0.0;
        for n in range(N-k):
            mu += f[ n + k + 1 ] * b[ n ]
        mu *= -2.0 / Dk
        
        # update Ak
        maxn = (k+1)/2 + 1 # rounds down 
        for n in range(maxn):
            t1 = Ak[ n ] + mu * Ak[ k + 1 - n ]
            t2 = Ak[ k + 1 - n ] + mu * Ak[ n ]
            Ak[ n ] = t1
            Ak[ k + 1 - n ] = t2
        #update f and b
        for n in range(N-k):
            t1 = f[ n + k + 1 ] + mu * b[n]
            t2 = b[ n ] + mu * f[ n + k + 1]
            f[ n + k + 1 ] = t1
            b[ n ] = t2
            
        #update Dk
        Dk = ( 1.0 - mu ** 2) * Dk - (f[ k + 1 ] ** 2) - (b[ N - k - 1 ] ** 2)
    
    # assign coefficients.
    coeffs[:] = Ak[1:]
    
    return coeffs
    
    

if __name__ == "__main__":
    # data to 
    original = [0.0]* 128
    for i in range(128):
        t  = cos( i * 0.01 ) + 0.75 *cos( i * 0.03 )\
            + 0.5 *cos( i * 0.05 ) + 0.25 *cos( i * 0.11 )
        original[i] = t
        print i,  t
        
    #get linear prediction coefficients.
    # using BurgAlgorithm( m, original )
    coeffs = burg_AR(4,  np.array(original))
    
    # Linear Predict Data
    predicted = [orig for orig in original]
    m = len(coeffs)
    for i in range(m,  len(predicted)):
        predicted[i] =0.0
        for j in range(m):
            predicted[ i ] -= coeffs[ j ] * original[i-1-j]

    #Calculate and display error.
    error = 0.0
    for i in range(len(predicted)):
       print "Index: %2d / Original: %.6f / Predicted: %.6f" % (i, original[i], predicted[i] )
       delta = predicted[i] - original[i]
       error += delta * delta;
    print "Burg Approximation Error: %f\n" % error
