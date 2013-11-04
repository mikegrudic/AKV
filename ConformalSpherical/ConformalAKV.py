#!/usr/bin/env python                                                                                                                                
import scipy
import numpy as np
from optparse import OptionParser
import SphericalGrid
import AKV

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

grid = SphericalGrid.SphericalGrid(opts.Lmax, opts.Lmax)

#Conformal factor will be of form exp(2H) - hardcode H here:
H = np.cos(grid.theta)

grid.gthth, grid.gphph = np.exp(2*H), np.exp(2*H)*np.sin(grid.theta)**2
grid.UpdateMetric()

grid.ricci = -2.0/grid.gthth * (-1.0 + grid.D2(H,1)/grid.sintheta**2 + grid.D(H,0)*grid.costheta/grid.sintheta + grid.D2(H,0))

np.savetxt("ConformalFactor.dat", np.column_stack((grid.theta.flatten(), grid.phi.flatten(), grid.gthth.flatten())))
np.savetxt("RicciScalar.dat", np.column_stack((grid.theta.flatten(), grid.phi.flatten(), grid.ricci.flatten())))

AKV.AKV(grid=grid, KerrNorm=opts.KerrNorm)
