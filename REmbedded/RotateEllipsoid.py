#!/usr/bin/env python                                                                                                           
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from optparse import OptionParser
import SphericalGrid
import AKV
from R3EmbeddedSurface import *

pi=np.pi

#######################################################################################
# Parse options

p=OptionParser()
p.add_option("--mNorm", type="string", help="Which norm to use for AKV: either 'Cook-Whiting' or 'Owen'", default="Owen")
p.add_option("--KerrNorm", default = False)
p.add_option("--Lmax", type ="int", default = 15,
             help = "Maximum degree of spherical harmonic expansion")
(opts,args) = p.parse_args()
########################################################################################

n = 8

for i, angle in enumerate(np.linspace(0,pi/2,n+1)):
    name = str(i)
    print name
    grid = Ellipsoid(opts.Lmax,opts.Lmax,2,1, angle)
    AKV.AKV(grid=grid, KerrNorm=opts.KerrNorm, name=name, mNorm=opts.mNorm)
