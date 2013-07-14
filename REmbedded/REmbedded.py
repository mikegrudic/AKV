#!/usr/bin/env python                                                                                                           
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy
import numpy as np
from optparse import OptionParser
import SphericalGrid
import AKV
import cProfile
from R3EmbeddedSurface import *
import time

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

#grid = R3EmbeddedSurface(opts.Lmax, opts.Lmax)

#a, b = 0.1, 4
#grid.R = np.ones(grid.extents) + a*np.exp(-b*grid.theta**2)
#grid.UpdateR()

#InitR3EmbeddedSurface(grid,R)
grid = Ellipsoid(opts.Lmax,opts.Lmax,1.1,1)
#cProfile.run("AKV.AKV(grid=grid, KerrNorm=opts.KerrNorm)")
t = time.time()
AKV.AKV(grid=grid, KerrNorm=opts.KerrNorm)
print time.time() - t
R = grid.R
np.savetxt("R.dat", np.column_stack((grid.theta.flatten(), grid.phi.flatten(), R.flatten())))

# Visualization
fig = plt.figure()
ax = fig.gca(projection='3d', aspect='equal')
X = R*np.cos(grid.phi)*grid.sintheta
Y = R*np.sin(grid.phi)*grid.sintheta
Z = R*grid.costheta
#X.resize(X.shape[0],X.shape[1]+1)
#X[:,-1] = X[:,0]
#Y.resize(Y.shape[0],Y.shape[1]+1)
#Y[:,-1] = Y[:,0]
#Z.resize(Z.shape[0],Z.shape[1]+1)
#Z[:,-1] = Z[:,0]

surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
        linewidth=0, antialiased=True)

for direction in (-1, 1):
    for point in np.diag(direction * np.max(R) * np.array([1,1,1])):
        ax.plot([point[0]], [point[1]], [point[2]], 'w')
plt.show()
plt.savefig("Shape.png")
