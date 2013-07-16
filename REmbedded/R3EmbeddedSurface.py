import numpy as np
from SphericalGrid import *
import RotateCoords

class R3EmbeddedSurface(SphericalGrid):
    def __init__(self, Lmax, Mmax, Rfunc=None):
        SphericalGrid.__init__(self, Lmax, Mmax)
        if Rfunc != None:
            self.R = Rfunc(grid.theta, grid.phi)
            self.UpdateR()

    def UpdateR(self):
        dR_dth, dR_dph = self.D(self.R)
        d2R_dth, d2R_dph = self.D2(self.R)
        d2R_dthdph = self.D(dR_dth,1)

        self.gthth, self.gphph, self.gthph = self.R**2 + dR_dth**2, self.R**2*self.sintheta**2 + dR_dph**2, dR_dph*dR_dth
        self.UpdateMetric() #computes derivatives redundantly... can optimize
        
#        self.gthth_th = 2*dR_dth*(self.R + d2R_dth)
#        self.gthth_ph = 2*(self.R*dR_dph + dR_dth*d2R_dthdph)
#        self.gphph_ph = 2*dR_dph*(self.R*self.sintheta**2 + d2R_dph)
#        self.gphph_th = 2*self.R*self.sintheta*(self.costheta*self.R + self.sintheta*dR_dth) + 2*dR_dph*d2R_dthdph
#        self.gthph_th = dR_dth*d2R_dthdph + dR_dph*d2R_dth
#        self.gthph_ph = d2R_dph*dR_dth + dR_dph*d2R_dthdph
#        self.gamma_ph = (-(self.gthth**2*self.gphph_ph) + self.gthth*(self.gthph*(2*self.gthth_ph + self.gphph_th) + self.gphph*(self.gthth_ph - 2*self.gthph_th)) + self.gthph*(-2*self.gthph*self.gthth_ph + self.gphph*self.gthth_th))/2.0/self.detg**2
#        self.gamma_th = (-2*self.gthph**2*self.gphph_th + self.gthph*(self.gthth*self.gphph_ph + self.gphph*(self.gthth_ph + 2*self.gthph_th)) + self.gphph*(self.gthth*(-2*self.gthth_ph + self.gphph_th) - self.gphph*self.gthth_th))/2.0/self.detg**2

        self.ricci = (2*self.R**3*self.sintheta**4 - 4*self.sintheta*dR_dth*
              (2*self.costheta*dR_dph**2 +
               self.sintheta*dR_dth*
               (d2R_dph + self.costheta*self.sintheta*dR_dth) -
               2*self.sintheta*dR_dph*d2R_dthdph) -
              4*self.sintheta**2*dR_dph**2*d2R_dth -
              2*self.R**2*self.sintheta**2*(d2R_dph +
              self.sintheta*(self.costheta*dR_dth + self.sintheta*d2R_dth)) +
              self.R*((1 - 3*np.cos(2*self.theta))*dR_dph**2 +
             4*self.costheta*self.sintheta*dR_dph*d2R_dthdph +
              2*self.sintheta**2*(2*self.sintheta**2*dR_dth**2 - d2R_dthdph**2 +
             (d2R_dph + self.costheta*self.sintheta*dR_dth)*
              d2R_dth)))/self.R/(dR_dph**2 + self.sintheta**2*(self.R**2+dR_dth**2))**2

class Ellipsoid(R3EmbeddedSurface):
    def __init__(self, Lmax, Mmax, a, b, offset_angle=0):
        R3EmbeddedSurface.__init__(self,Lmax, Mmax)

        self.R = a*b/np.sqrt((b*self.costheta)**2 + (a*self.sintheta)**2)
        self.R = RotateCoords.RotateScalarY(self,self.R,offset_angle)
        self.UpdateR()
#        self.ricci = (2*(a*b**2*self.costheta**2 + a**3*self.sintheta**2)**2)/(b**4*self.costheta**2 + a**4*self.sintheta**2)**2
#        self.ricci = RotateCoords.RotateScalarY(self, self.ricci, offset_angle)
