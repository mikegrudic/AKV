#!/usr/bin/python

import glob
import numpy as np
from numpy import exp,arange
from optparse import OptionParser
import matplotlib.pyplot as plt

p = OptionParser()
p.add_option("--f", type="string")
p.add_option("--xCol", default=0)
p.add_option("--yCol", default=1)

(opts,args) = p.parse_args()

if opts.f==None: dataFiles = glob.glob("*.dat")
else: dataFiles = glob.glob(opts.f)

for f in dataFiles:
#    print f
    data = np.loadtxt(f)
    name = f.split(".")[0]

    x, y = data[:,opts.xCol], data[:,opts.yCol]

    std_dev = np.std(y)
    spike_index = (np.abs(y) > 2*std_dev)
    spike_points = x[spike_index]
    spike_values = y[spike_index]
#    np.savetxt("D"+name+".dat", np.column_stack((data[:,opts.xCol],np.gradient(data[:,opts.yCol]))))
    if spike_points.size != 0:
        print name, spike_points[0]
