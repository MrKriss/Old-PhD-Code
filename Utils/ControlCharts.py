# file : ControlCharts.py
"""
Created on Sun Jun 20 21:42:28 2010

Contains the 'Tseries' Class and  genTSdic Function 

@author: musselle
"""
# IMPORTS
from itertools import permutations
from numpy import zeros, sin, pi, array
from numpy.random import normal, uniform
from matplotlib.pyplot import plot, figure, show 
from copy import copy


#===============================================================================
# # Time-series Class
#===============================================================================
class Tseries(list): 
    """Time series Class 

    Initiates an empty list of length N

    Defines following methods 
    1. normalCC - add time series with uniform variation 
    2. cyclicCC - add time 
    3. upCC
    4. downCC   

    """
    
    def __init__(self,N):
        self = zeros([1,N])
        self.tolist()
    # Extend methods 
    
    def normalEt(self, size, base=0, noise = 1, noise_type = 'gauss'):
        """ Extend Time series by size using normal control chart pattern
        
        noise = rang in uniform method and sigma in normal method
        """
        
        if noise_type == 'gauss':
            return self.extend(normal(base,noise,size))
        elif noise_type == 'uni':
            return self.extend(uniform((0-noise),noise,size) + base)
        elif noise_type == 'none' :
            return self.extend([base] * size)            
            
    def cyclicEt(self, size, base=0, noise=2, amp=10, period=25, \
                 phase = 0, noise_type = 'gauss'):
        """ Extend Time series by size using cyclic control chart pattern """    
        # Designe function to take parameters (pattern_type, number)
        if noise_type == 'gauss':
            series = normal(base,noise,size)
        elif noise_type == 'uni' :
            series = uniform(0-noise,noise,size) + base 
        elif noise_type == 'none' :
            series =  array([base] * size)

        
        for i in range(len(series)):  
            series[i] = series[i] + amp * sin((2 * pi * (i+1) / float(period)) + phase)
        
        return self.extend(series)

    def upEt(self, size, base=0, noise=1, gradient=0.2, noise_type = 'gauss'):
        """ Extend time series by size using increasing trend Control chart """
        
        if noise_type == 'gauss':
            series = normal(base,noise,size)
        elif noise_type == 'uni':
            series = uniform(0-noise,noise,size) + base 
        elif noise_type == 'none' :
            series =  array([base] * size)
                
        for i in range(len(series)):  
            series[i] = series[i] + gradient * i
        
        return self.extend(series)


    def downEt(self, size, base=0, noise=1, gradient=0.2, noise_type = 'gauss'):
        """ Extend time series by using decreasing trend Control chart """
        
        if noise_type == 'gauss':
            series = normal(base,noise,size)
        elif noise_type == 'uni' :
            series = uniform(0-noise,noise,size) + base 
        elif noise_type == 'none' :
            series =  array([base] * size)
        
        for i in range(len(series)):  
            series[i] = series[i] - gradient*i
        return self.extend(series)

  
  # Replace Methods

    def normalRp(self,myslice,base=0,noise=2, noise_type = 'gauss'):
        """ Replaces Time series myslice using normal control chart pattern
     
        myslice is a tuple (i,j) of the start and end point. Uses Python syntax 
        i <= x < j 
        All values edited are those of x
        """       
        if noise_type == 'gauss':
            series = normal(base,noise,myslice[1]-myslice[0])
        elif noise_type == 'uni' :
            series = uniform(0-noise,noise,myslice[1]-myslice[0]) + base 
        elif noise_type == 'none' :
            series =  array([base] * myslice)
        
        # series = uniform(0-noise,noise,myslice[1]-myslice[0]) + base 
        
        a = self[:myslice[0]]
        b = self[myslice[1]:]
        
        self = a.extend(series)
        self = self.extend(b)
        
        return self

    def cyclicRp(self,myslice,base=0,noise=1,amp=10,period=15, \
                phase = 0, noise_type = 'gauss'):
        """ Replace Time series with cyclic control chart pattern 
        
        myslice is a tuple (i,j) of the start and end point. Uses Python syntax 
        i <= x < j 
        All values edited are those of x
        """    
        
        # series = uniform(0-noise,noise,myslice[1]-myslice[0]) + base 
        
        if noise_type == 'gauss':
            series = normal(base,noise,myslice[1]-myslice[0])
        elif noise_type == 'uni' :
            series = uniform(0-noise,noise,myslice[1]-myslice[0]) + base 
        elif noise_type == 'none' :
            series =  array([base] * myslice)
        
        for i in range(len(series)):  
            series[i] = series[i] + amp * sin((2 * pi * i / period) + phase)
        
        a = self[:myslice[0]]
        b = self[myslice[1]:]
        
        self = a.extend(series)
        self = self.extend(b)
        
        return self

    def upRp(self,myslice,base=0,noise=1,gradient=0.2, noise_type = 'gauss'):
        """ Replace Time series myslice with upward trend control chart pattern 
        
        myslice is a tuple (i,j) of the start and end point. Uses Python syntax 
        i <= x < j 
        All values edited are those of x
        """    
        
        if noise_type == 'gauss':
            series = normal(base,noise,myslice[1]-myslice[0])
        elif noise_type == 'uni' :
            series = uniform(0-noise,noise,myslice[1]-myslice[0]) + base 
        elif noise_type == 'none' :
            series =  array([base] * myslice)
            
        for i in range(len(series)):  
            series[i] = series[i] + gradient*i
        
        a = self[:myslice[0]]
        b = self[myslice[1]:]
        
        self = a.extend(series)
        self = self.extend(b)
        
        return self
    
    def downRp(self,myslice,base=0,noise=1,gradient=0.2, noise_type = 'gauss'):
        """Replace Time series myslice with downward trend control chart pattern 
        
        myslice is a tuple (i,j) of the start and end point. Uses Python syntax 
        i <= x < j 
        All values edited are those of x
        """    
                 
        if noise_type == 'gauss':
            series = normal(base,noise,myslice[1]-myslice[0])
        elif noise_type == 'uni' :
            series = uniform(0-noise,noise,myslice[1]-myslice[0]) + base 
        elif noise_type == 'none' :
            series =  array([base] * myslice)
            
        series = uniform(0-noise,noise,myslice[1]-myslice[0]) + base 
        
        for i in range(len(series)):  
            series[i] = series[i] - gradient*i
        
        a = self[:myslice[0]]
        b = self[myslice[1]:]
        
        self = a.extend(series)
        self = self.extend(b)
        
        return self
    
    # Plot
    def p(self):
        figure(1)
        plot(self)
    
    def makeSeries(self, pattern_type, number, base=None ,**kwargs):
        """function to take parameters (pattern_type, number), both of which are
        sequences. 
        
        patterns = {1: 'normal', 2: 'cyclic', 3: 'up', 4: 'down'}         
        
        Will generate output as follows:
        series = pattern_type[1] * number[1], ..., pattern_type[n] * number[n] 
        all with parametern defined in kwargs dic.
                    
            Parameter key words are:
                base=0,
                noise=2,
                noise_type = 'gauss',
                amp=10,
                period=25, 
                gradient=0.2
        """

        if base == None :
            base = zeros(len(number))        
        
        message = 'length of inputs not equal'
        assert len(number) == len(pattern_type), message         
        
        patterns = {1: 'normal', 2: 'cyclic', 3: 'up', 4: 'down'}      
        inputs = zip(pattern_type, number, base)        
        
        for ppp, nn, bb in inputs:
            
            kwargs_passed = copy(kwargs)
            
            # Remove key works that are not used in the method
            if ppp == 1 :  # Normal 
                if kwargs_passed.has_key('gradient'):       
                    del(kwargs_passed['gradient'])      
                if kwargs_passed.has_key('amp'):
                    del(kwargs_passed['amp'])   
                if kwargs_passed.has_key('period'):
                    del(kwargs_passed['period'])
                if kwargs_passed.has_key('phase'):
                    del(kwargs_passed['phase'])
            elif ppp == 2: # Cyclic
                if kwargs_passed.has_key('gradient'):       
                    del(kwargs_passed['gradient'])        
            elif ppp == 3: # Up trend
                if kwargs_passed.has_key('amp'):
                    del(kwargs_passed['amp'])
                if kwargs_passed.has_key('period'):
                    del(kwargs_passed['period'])
                if kwargs_passed.has_key('phase'):
                    del(kwargs_passed['phase'])
            elif ppp == 4 : # Down Trend
                if kwargs_passed.has_key('amp'):
                    del(kwargs_passed['amp'])
                if kwargs_passed.has_key('period'):
                    del(kwargs_passed['period'])
                if kwargs_passed.has_key('phase'):
                    del(kwargs_passed['phase'])
            
            # Generate Time-Series Patterns
            pattern = patterns[ppp] + 'Et'        
            getattr(self, pattern)(nn, base = bb , **kwargs_passed )
            
        return self
#            
        
        

#===============================================================================
#  TSdictionary Generation Function 
#===============================================================================

def genTSdic(num1 = 100, num2 = 100):
    """ Generates a dict of time series for each non-repeating permutation 

    100 ticks * pattern1   +  100 ticks pattern2

    where  patterns = {1: 'normal', 
                       2: 'cyclic', 
                       3: 'up', 
                       4: 'down'}
    """
    patterns = {1: 'normal', 2: 'cyclic', 3: 'up', 4: 'down'}
    
    perms = permutations(range(1,5),2)
    perms = list(perms)
    
    ccc = 0
    
    output = {}
    
    for aaa, bbb in perms:  
        ccc += 1 # permutation counter
        
        # Generate variable name for time series 
        series_name = 's' + str(ccc)    
        vars()[series_name] = Tseries(0)
        series = vars()[series_name]    
            
        # Patterns to use + suffix
        pattern1 = patterns[aaa] + 'Et'
        pattern2 = patterns[bbb] + 'Et'
            
        # Generate Time-Series Patterns
        getattr(series, pattern1)(num1)    
        getattr(series, pattern2)(num2)    
        
        # Store series in dictionary
        output[ccc] = array(series)

    return output

        
if __name__ == '__main__':
    # TSdic = genTSdic()
    series = Tseries(0)
    series.makeSeries([1,2,3,4] , [500,500,500,500], noise = 4)
    plot(series)
    show()

