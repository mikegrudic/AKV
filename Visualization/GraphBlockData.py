#!/usr/bin/python

import glob
import numpy as np
from numpy import exp,arange
from optparse import OptionParser
import matplotlib.pyplot as plt

p = OptionParser()
p.add_option("--f", type="string")
p.add_option("--xCol", type="int", default=0)
p.add_option("--yCol", type="int", default=1)
p.add_option("--xLabel", type="string", default="")
p.add_option("--yLabel", type="string", default="")
p.add_option("--MultipleGraphs", default=True)

(opts,args) = p.parse_args()

if opts.f==None: dataFiles = glob.glob("*.dat")
else: dataFiles = glob.glob(opts.f)

for f in dataFiles:
    print f
    data = np.loadtxt(f)
    name = f.split(".")[0]
    plt.plot(data[:,opts.xCol],data[:,opts.yCol])
    plt.xlabel(opts.xLabel)
    plt.ylabel(opts.yLabel)
    plt.title(name)
    plt.savefig(name+".png")
    if opts.MultipleGraphs==True:
        plt.clf()

#if opts.MultipleGraphs==False:
#    plt.savefig("graph.png")
