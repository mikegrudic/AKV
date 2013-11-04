#!/usr/bin/python

import glob
import numpy as np
from numpy import exp,arange
from optparse import OptionParser
from pylab import *

p = OptionParser()
p.add_option("--f", type="string")
p.add_option("--Lmax", type="int", default = 15)

(opts,args) = p.parse_args()

if opts.f is None:
    vecFiles = glob.glob("*vec*.dat")
else:
    vecFiles = glob.glob(opts.f)

for vecFile in vecFiles:
    print vecFile
    figure()
    name, extension = vecFile.split("vec")

    vec = np.loadtxt(vecFile)

    gridShape = (opts.Lmax+1,len(vec)/(opts.Lmax+1))
    theta = vec[:,0].reshape(gridShape)
    phi = vec[:,1].reshape(gridShape)
    v_theta = vec[:,2].reshape(gridShape)
    v_phi = vec[:,3].reshape(gridShape)

    quiver(phi,theta,v_phi*np.sin(theta), v_theta, pivot = 'middle')

    x, y = phi, theta

    title(vecFile.split(".dat")[0])
    xlim([0,2*np.pi])
    ylim([0,np.pi])
    xlabel("Phi")
    ylabel("Theta")
    savefig(vecFile.split(".dat")[0]+".png")
