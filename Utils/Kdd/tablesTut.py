# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 10:50:34 2010

@author: musselle
"""

from tables import *

class Particle(IsDescription):
    name      = StringCol(16)   # 16-character String
    idnumber  = Int64Col()      # Signed 64-bit integer
    ADCcount  = UInt16Col()     # Unsigned short integer
    TDCcount  = UInt8Col()      # unsigned byte
    grid_i    = Int32Col()      # 32-bit integer
    grid_j    = Int32Col()      # 32-bit integer
    pressure  = Float32Col()    # float  (single-precision)
    energy    = Float64Col()    # double (double-precision)

h5file = openFile("tutorial1.h5", mode = "w", title = "Test file")

group = h5file.createGroup("/", 'detector', 'Detector information')

table = h5file.createTable(group, 'readout', Particle, "Readout example")

particle = table.row

for i in xrange(10):
    particle['name']  = 'Particle: %6d' % (i)
    particle['TDCcount'] = i % 256
    particle['ADCcount'] = (i * 256) % (1 << 16)
    particle['grid_i'] = i
    particle['grid_j'] = 10 - i
    particle['pressure'] = float(i*i)
    particle['energy'] = float(particle['pressure'] ** 4)
    particle['idnumber'] = i * (2 ** 34)
    # Insert a new particle record
    particle.append()

table.flush()

table = h5file.root.detector.readout

# Querry method 1
pressure = [ x['pressure'] for x in table.iterrows()
            if x['TDCcount'] > 3 and 20 <= x['pressure'] < 50 ]

# Querry Method 2 - in-kernal, much faster
names = [ x['name'] for x in table.where(
        """(TDCcount > 3) & (20 <= pressure) & (pressure < 50)""") ]

# New group with pressure array 
gcolumns = h5file.createGroup(h5file.root, "columns", "Pressure and Name")
h5file.createArray(gcolumns, 'pressure', array(pressure),
                    "Pressure column selection")
h5file.createArray(gcolumns, 'name', names, "Name column selection")

h5file.close()
