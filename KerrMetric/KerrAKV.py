#!/usr/bin/env python                                                                                                                                
import time
import scipy
import numpy as np
from optparse import OptionParser
import SphericalGrid
import KerrMetric
import AKV

#######################################################################################
# Parse options

p=OptionParser()
p.add_option("--mNorm", type="string", help="Which norm to use for AKV: either 'Cook-Whiting' or 'Owen'", default="Owen")
p.add_option("--KerrNorm", default = True)
p.add_option("--Lmax", type ="int", default = 15,
             help = "Maximum degree of spherical harmonic expansion")
p.add_option("--M", type="float", default=1.0)
p.add_option("--J", type="float", default=0.0)
(opts,args) = p.parse_args()

########################################################################################

#create black hole object to get functions for metric, curvatures
a = opts.J/opts.M
blackHole = KerrMetric.KerrMetric(opts.M,a)
t=time.time()
AKV.AKV(blackHole.HorizonMetric, blackHole.HorizonRicci, Lmax=opts.Lmax, KerrNorm=opts.KerrNorm, mNorm=opts.mNorm)
print time.time()-t
