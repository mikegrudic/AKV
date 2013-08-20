#!/usr/bin/python                                                                                             
import glob
import numpy as np
from scipy.optimize import leastsq
from optparse import OptionParser
import matplotlib.pyplot as plt

p = OptionParser()
p.add_option("--f", type="string")
p.add_option("--xCol", type="int", default=0)
p.add_option("--yCol", type="int", default=1)

(opts,args) = p.parse_args()

if opts.f==None: dataFiles = glob.glob("*.dat")
else: dataFiles = glob.glob(opts.f)

dataFiles.sort()

out = open("PowerLaw.dat","w")

for f in dataFiles:
    dat = np.loadtxt(f)
    X, Y = dat[:,opts.xCol], dat[:,opts.yCol]
#    error = lambda p, y, x: p*x**2.0 - y
#    lsq = leastsq(error, [-1e-3], args=(Y,X))
    print f, np.polyfit(X,Y,8)
#    residual = np.std(error(lsq, Y, X))/np.max(np.abs(Y))
#    print f, lsq[0], residual
