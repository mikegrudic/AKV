#!/usr/bin/env python                                                                                                           
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt               
import scipy
import numpy as np
from optparse import OptionParser
import SphericalGrid
import AKV
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

grid = SphericalGrid.SphericalGrid(opts.Lmax, opts.Lmax)

R = np.ones(grid.extents) + 0.1*np.exp(-4*grid.theta**2)

dR_dth, dR_dph = grid.D(R)
d2R_dth, d2R_dph = grid.D2(R)
d2R_dthdph = grid.D(dR_dth,1)

grid.gthth, grid.gphph, grid.gthph = R**2 + dR_dth**2, R**2*grid.sintheta**2 + dR_dph**2, 2*dR_dph*dR_dth
grid.UpdateMetric()

#grid.ricci = (2*R**2*grid.sintheta**4 + dR_dph**2 - 
#     3*np.cos(2*grid.theta)*dR_dph**2 - 2*R*grid.sintheta**2*d2R_dph - 
 #    2*grid.costheta*R*grid.sintheta**3*dR_dth - 
#     (8*grid.costheta*grid.sintheta*dR_dph**2*dR_dth)/R + 
#     4*grid.sintheta**4*dR_dth**2 - 
#     (4*grid.sintheta**2*d2R_dph*dR_dth**2)/R - 
#     (4*grid.costheta*grid.sintheta**3*dR_dth**3)/R + 
#     4*grid.costheta*grid.sintheta*dR_dph*d2R_dthdph + 
#     (8*grid.sintheta**2*dR_dph*dR_dth*d2R_dthdph)/R - 
#     2*grid.sintheta**2*d2R_dthdph**2 - 
#     2*R*grid.sintheta**4*d2R_dth - 
#     (4*grid.sintheta**2*dR_dph**2*d2R_dth)/R + 
#     2*grid.sintheta**2*d2R_dph*d2R_dth + 
#     2*grid.costheta*grid.sintheta**3*dR_dth*d2R_dth)/(dR_dph**2 + grid.sintheta**2*(R**2+dR_dth**2))**2

grid.ricci = (2*R**3*grid.sintheta**4 - 4*grid.sintheta*dR_dth*
              (2*grid.costheta*dR_dph**2 + 
               grid.sintheta*dR_dth*
               (d2R_dph + grid.costheta*grid.sintheta*dR_dth) - 
               2*grid.sintheta*dR_dph*d2R_dthdph) - 
              4*grid.sintheta**2*dR_dph**2*d2R_dth - 
              2*R**2*grid.sintheta**2*(d2R_dph + 
              grid.sintheta*(grid.costheta*dR_dth + grid.sintheta*d2R_dth)) + 
              R*((1 - 3*np.cos(2*grid.theta))*dR_dph**2 + 
             4*grid.costheta*grid.sintheta*dR_dph*d2R_dthdph + 
              2*grid.sintheta**2*(2*grid.sintheta**2*dR_dth**2 - d2R_dthdph**2 + 
             (d2R_dph + grid.costheta*grid.sintheta*dR_dth)*
              d2R_dth)))/R/(dR_dph**2 + grid.sintheta**2*(R**2+dR_dth**2))**2

#rExact =  (200*np.exp(8*grid.theta**2)*(9 + 64*grid.theta**2 + 20*np.exp(4*grid.theta**2)*(5 + 5*np.exp(4*grid.theta**2) - 32*grid.theta**2))*(1 + 10*np.exp(4*grid.theta**2) + 8*grid.theta/np.tan(grid.theta)))/((1 + 10*np.exp(4*grid.theta**2))*(1 + 20*np.exp(4*grid.theta**2) + 100*np.exp(8*grid.theta**2) + 64*grid.theta**2)**2)

#print np.std(grid.ricci-rExact)
t = time.time()
AKV.AKV(grid=grid, KerrNorm=opts.KerrNorm)
print time.time() -t
