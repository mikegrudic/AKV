#!/usr/bin/env python                                                                                                                                
import scipy
import sys
import numpy as np
from optparse import OptionParser
import SphericalGrid
from AKVSolution import AKVSolution
import AKV
import time

pi=np.pi

def VecError(grid, v1, v2):
    return np.sqrt(grid.Integrate(grid.VecLength(v1-v2)**2)/grid.Integrate(np.ones(grid.extents)))
def PotentialError(grid,pot1,pot2):
    return np.sqrt(grid.Integrate((pot1-pot2)**2)/grid.Integrate(np.ones(grid.extents)))

def ConformalAKVSol(grid, l, m, coefficient):
    coeffs = np.zeros(grid.numTerms)
    coeffs[SphericalGrid.YlmIndex(l,m)] = coefficient
    H = grid.SpecToPhys(coeffs)
    grid.gthth, grid.gphph = np.exp(2*H), np.exp(2*H)*np.sin(grid.theta)**2
    grid.UpdateMetric()
    grid.ricci = -2.0/grid.gthth * (-1.0 + grid.D2(H,1)/grid.sintheta**2 + grid.D(H,0)*grid.costheta/grid.sintheta + grid.D2(H,0))
    return AKVSolution(grid=grid,IO=False)

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
sphere_solution = AKVSolution(grid=grid,IO=False)
sphere_akv = sphere_solution.GetAKV()
sphere_pot = sphere_solution.GetPotentials()

amax = 1
nPoints = 301
coeff_range = np.linspace(0,amax,nPoints)

da = float(amax)/nPoints

AKV_all = []
Eigenvalue_all = []
Potential_all = []
AKV_error_all = []
pot_error_all = []

for i in xrange(4,grid.numTerms):
    l, m = grid.l[i], grid.m[i]
    if m < 0: continue
    if l > 9: break
    print l, m
    sols = np.array([ConformalAKVSol(grid,l,m,a) for a in coeff_range])

    akvs = [sol.GetAKV() for sol in sols]
    eigs = [sol.GetEigs() for sol in sols]
    potentials = [sol.GetPotentials() for sol in sols]
    pot_errors = []
    vec_errors = []
    for i in xrange(nPoints):
        pot_error1 = min([min([PotentialError(grid, sph, pot) for pot in potentials[i]]) for sph in sphere_pot])
        pot_error2 = min([min([PotentialError(grid, sph, -pot) for pot in potentials[i]]) for sph in sphere_pot])
        pot_error = min(pot_error1, pot_error2)
        pot_errors.append(pot_error)
        vec_error1 = min([min([VecError(grid, sph, vec) for vec in akvs[i]]) for sph in sphere_akv])
        vec_error2 = min([min([VecError(grid, sph, -vec) for vec in akvs[i]]) for sph in sphere_akv])
        vec_error = min(vec_error1, vec_error2)
        vec_errors.append(vec_error)

    AKV_all.append(akvs)
    Eigenvalue_all.append(eigs)
    Potential_all.append(potentials)
    AKV_error_all.append(vec_errors)
    pot_error_all.append(pot_errors)

np.save("AKV", np.array(AKV_all))
np.save("Eigenvalues", np.array(Eigenvalue_all))
np.save("Potentials", np.array(Potential_all))
np.save("AKV_Error", np.array(AKV_error_all))
np.save("Potential_error", np.array(pot_error_all))
