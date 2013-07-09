#!/usr/bin/env python                                                                                                                                
import scipy
import numpy as np
from optparse import OptionParser
from RotateCoords import RotateCoords
from subprocess import call
import RealSH
import KerrMetric
import AKV

#######################################################################################
# Parse options

p=OptionParser()
p.add_option("--mNorm", type="string")
#p.add_option("--AssumeZSpin", default = True)
#p.add_option("--out", type ="string")
p.add_option("--KerrNorm", default = True)
p.add_option("--Lmax", type ="int", default = 15)
(opts,args) = p.parse_args()

if opts.mNorm is None:
    opts.mNorm = "Owen"

pi=np.pi
########################################################################################

grid = RealSH.SphericalGrid(opts.Lmax, opts.Lmax)
blackHole = KerrMetric.KerrMetric(1,0.0)

for alpha in np.linspace(0,pi/2,9):
    print alpha
#create black hole object to get functions for metric, curvatures
    thetaPrime, phiPrime = RotateCoords(grid.theta, grid.phi, np.array([0,1.0,0]), -alpha)
    grid.gthth, grid.gphph = blackHole.HorizonMetric(thetaPrime, phiPrime)
    grid.ricci = blackHole.HorizonRicci(thetaPrime,phiPrime)
    np.savetxt("alpha"+"%3.3f"%(alpha)+"_ricci.dat", np.column_stack((grid.theta.flatten(), grid.phi.flatten(), grid.ricci.flatten())))
    grid.detg = grid.gthth*grid.gphph
    grid.dA = np.sqrt(grid.detg)
    grid.ComputeMetricDerivs()
    AKV.AKV(grid=grid, KerrNorm=opts.KerrNorm, name="alpha"+"%3.3f"%(alpha))
