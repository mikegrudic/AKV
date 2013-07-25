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

files = glob.glob(opts.f)

for f in files:
    figure()
    name, extension = f.split(".")
#    print f.split(".dat")[0]
#    vecFile = name+"_vec1"+extension

    s = np.loadtxt(f)

    gridShape = (opts.Lmax+1,len(s)/(opts.Lmax+1))
    print gridShape
    theta = s[:,0].reshape(gridShape)
    phi = s[:,1].reshape(gridShape)
    scalar = s[:,2].reshape(gridShape)

    smax, smin = np.max(scalar), np.min(scalar)

    x, y = phi, theta

    im = imshow(scalar,cmap=cm.RdBu, extent=[0,2*np.pi,0,np.pi]) # drawing the function
    # adding the Contour lines with labels
#    cset = contour(scalar,arange(smin,smax,(smax-smin)/8),linewidths=2,cmap=cm.Set2)
#    cset = contour(phi,theta,scalar,linewidths=1)
#    clabel(cset,inline=True,fmt='%1.1f',fontsize=10)
#    colorbar(im) # adding the colobar on the right
    # latex fashion title
    title(name)
    xlabel("Phi")
    ylabel("Theta")
    savefig(name+".png")
