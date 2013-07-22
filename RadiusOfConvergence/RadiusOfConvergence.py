#!/usr/bin/env python                                                                                                                                
import scipy
import numpy as np
from optparse import OptionParser
import SphericalGrid
import AKV
import time

pi=np.pi

def Error(grid, v1, v2):
    return grid.Integrate(grid.VecLength(v1-v2))/grid.Integrate(np.ones(grid.extents))

def ConformalError(grid, l, m, coefficient):
    coeffs = np.zeros(grid.numTerms)
    coeffs[SphericalGrid.YlmIndex(l,m)] = coefficient
    H = grid.SpecToPhys(coeffs)
    grid.gthth, grid.gphph = np.exp(2*H), np.exp(2*H)*np.sin(grid.theta)**2
    grid.UpdateMetric()
    grid.ricci = -2.0/grid.gthth * (-1.0 + grid.D2(H,1)/grid.sintheta**2 + grid.D(H,0)*grid.costheta/grid.sintheta + grid.D2(H,0))   
    new_akv = np.array(AKV.AKV(grid=grid, IO=False))
    min_error = min([min([min(Error(grid, newvec, spherevec),Error(grid, -newvec, spherevec)) for newvec in new_akv]) for spherevec in sphere_akv])

    return min_error


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

H = np.zeros(grid.extents)
grid.gthth, grid.gphph = np.exp(2*H), np.exp(2*H)*np.sin(grid.theta)**2
grid.UpdateMetric()
grid.ricci = -2.0/grid.gthth * (-1.0 + grid.D2(H,1)/grid.sintheta**2 + grid.D(H,0)*grid.costheta/grid.sintheta + grid.D2(H,0))

sphere_akv = np.array(AKV.AKV(grid=grid, IO=False))

amax = 1
coeff_range = np.linspace(0,amax,301)

for i in xrange(4,grid.numTerms):
    l, m = grid.l[i], grid.m[i]
    print l, m
    error = np.array([ConformalError(grid,l,m,a) for a in coeff_range])
    np.savetxt("Ylm%d%dError.dat"%(l,m), np.column_stack((coeff_range, error, np.gradient(error))))
