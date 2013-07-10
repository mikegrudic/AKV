import numpy as np
import SphericalGrid
import RotateCoords

#class R3EmbeddedSurface(SphericalGrid):
#    def __init__(self, Lmax, Mmax, fmetric=None, fricci=None):
#        SphericalGrid.__init__(self,Lmax,Mmax)
def R3EmbeddedSurface(Lmax, MMax, Rfunc):
    grid = SphericalGrid.SphericalGrid(Lmax,MMax)
    R = Rfunc(grid.theta, grid.phi)
    InitR3EmbeddedSurface(grid,R)
    return grid

def InitR3EmbeddedSurface(grid, R):
    dR_dth, dR_dph = grid.D(R)
    d2R_dth, d2R_dph = grid.D2(R)
    d2R_dthdph = grid.D(dR_dth,1)

    grid.gthth, grid.gphph, grid.gthph = R**2 + dR_dth**2, R**2*grid.sintheta**2 + dR_dph**2, 2*dR_dph*dR_dth
    grid.UpdateMetric()

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

def Ellipsoid(Lmax, MMax, a, b, offset_angle):
    grid = SphericalGrid.SphericalGrid(Lmax,MMax)
    R = a*b/np.sqrt((b*grid.costheta)**2 + (a*grid.sintheta)**2)
    R = RotateCoords.RotateScalarY(grid,R,offset_angle)
    InitR3EmbeddedSurface(grid,R)

    return grid
