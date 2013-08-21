#!/usr/bin/env python                                                                                                                                
import scipy
import numpy as np
from optparse import OptionParser
import SphericalGrid
import RotateCoords
import AKV

#######################################################################################
# Parse options

p=OptionParser()
p.add_option("--mNorm", type="string", 
             help="Which method to use: either 'Cook-Whiting' (solves eqn A13) or 'Owen' (solves eqn A11)", 
             default="Owen")
p.add_option("--f", type="string", default="YlmCoeffs.dat")
p.add_option("--KerrNorm", default = False, 
             help="Whether to use integral norm good for Kerr-like metric as described in Lovelace et al")
p.add_option("--Lmax", type ="int", default = 15)
(opts,args) = p.parse_args()

pi=np.pi
########################################################################################

grid = SphericalGrid.SphericalGrid(opts.Lmax, opts.Lmax)

coeffs = np.loadtxt(opts.f)[:,2]
H = grid.SpecToPhys(coeffs)

grid.gthth, grid.gphph = np.exp(2*H), np.exp(2*H)*np.sin(grid.theta)**2
grid.UpdateMetric()
grid.ricci = 2.0/grid.gthth * (1.0 - grid.D2(H,1)/grid.sintheta**2 - grid.D(H,0)*grid.costheta/grid.sintheta - grid.D2(H,0))

np.savetxt("ConformalFactor.dat", np.column_stack((grid.theta.flatten(), grid.phi.flatten(), grid.gthth.flatten())))
np.savetxt("RicciScalar.dat", np.column_stack((grid.theta.flatten(), grid.phi.flatten(), grid.ricci.flatten())))

AKV.AKV(grid=grid, KerrNorm=opts.KerrNorm, mNorm=opts.mNorm, Lmax=opts.Lmax)
