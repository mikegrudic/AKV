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
#    print f.split(".dat")[0]
#    vecFile = name+"_vec1"+extension

    vec = np.loadtxt(vecFile)

    gridShape = (opts.Lmax+1,len(vec)/(opts.Lmax+1))
    theta = vec[:,0].reshape(gridShape)
    phi = vec[:,1].reshape(gridShape)
    v_theta = vec[:,2].reshape(gridShape)
    v_phi = vec[:,3].reshape(gridShape)

    quiver(phi,theta,v_phi*np.sin(theta), v_theta, pivot = 'middle')

#    smax, smin = np.max(scalar), np.min(scalar)

    x, y = phi, theta

#    im = imshow(scalar,cmap=cm.RdBu, extent=[0,2*np.pi,0,np.pi]) # drawing the function
    # adding the Contour lines with labels
#    cset = contour(scalar,arange(smin,smax,(smax-smin)/8),linewidths=2,cmap=cm.Set2)
#    cset = contour(phi,theta,scalar,linewidths=1)
#    clabel(cset,inline=True,fmt='%1.1f',fontsize=10)
#    colorbar(im) # adding the colobar on the right
    # latex fashion title
    title(vecFile.split(".dat")[0])
    xlabel("Phi")
    ylabel("Theta")
    savefig(vecFile.split(".dat")[0]+".png")
