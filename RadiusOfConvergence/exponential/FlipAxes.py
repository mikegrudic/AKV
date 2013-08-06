#!/usr/bin/python                                                                                             
import glob
import numpy as np
from optparse import OptionParser
import matplotlib.pyplot as plt

p = OptionParser()
p.add_option("--f", type="string")
p.add_option("--xCol", type="int", default=0)
p.add_option("--yCol", type="int", default=1)

(opts,args) = p.parse_args()

if opts.f==None: dataFiles = glob.glob("*.dat")
else: dataFiles = glob.glob(opts.f)

for f in dataFiles:
    dat = np.loadtxt(f)
    dat[:,[opts.xCol,opts.yCol]] = dat[:,[opts.yCol,opts.xCol]]
    np.savetxt(f, dat)
