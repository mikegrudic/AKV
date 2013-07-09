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
p.add_option("--KerrNorm", default = False)
p.add_option("--Lmax", type ="int", default = 15)
(opts,args) = p.parse_args()

if opts.mNorm is None:
    opts.mNorm = "Owen"

pi=np.pi
########################################################################################

grid = RealSH.SphericalGrid(opts.Lmax, opts.Lmax)


for alpha in np.linspace(0,pi/2,9):
    print alpha

    coeffs = np.zeros(grid.numTerms)
    coeffs[RealSH.YlmIndex(1,0)] = 1.0
    coeffs = grid.ShtnsToStandard(grid.grid.Yrotate(grid.StandardToShtns(coeffs),-alpha))
    H = grid.SpecToPhys(coeffs)

    grid.gthth = np.exp(2*H)
    grid.gphph = grid.gthth*np.sin(grid.theta)**2
    grid.detg = grid.gthth*grid.gphph
    grid.dA = np.sqrt(grid.detg)
    grid.ComputeMetricDerivs()
    grid.ricci = -2.0/grid.gthth * (-1.0 + grid.D2(H,1)/grid.sintheta**2 + grid.D(H,0)*grid.costheta/grid.sintheta + grid.D2(H,0))   

    AKV.AKV(grid=grid, KerrNorm=opts.KerrNorm, name="alpha"+"%3.3f"%(alpha))

    np.savetxt("alpha"+"%3.3f"%(alpha)+"_conformal.dat",np.column_stack((grid.theta.flatten(), grid.phi.flatten(), grid.gthth.flatten())))
    np.savetxt("alpha"+"%3.3f"%(alpha)+"_ricci.dat", np.column_stack((grid.theta.flatten(), grid.phi.flatten(), grid.ricci.flatten())))
