#!/usr/bin/env python                
import scipy
from scipy import special, misc, linalg, optimize
import numpy as np
import shtns

pi = np.pi

###### YlmIndex #########################################
# Map between l and m indices and position in a 1D array
# NOTE: l starts at 1
##########################################################

def YlmIndex(n1, n2 = None):
    if n2 is None:
        l = np.int_(np.sqrt(n1))
        return l , n1 - (l*l+l)
    else:
        if n2>n1: n2 = n1
        return n1*n1+n1+n2


######### SphericalGrid ############################################
# Contains coordinates of points on theta-phi grid, function values,
# and spectral methods acting on real space and spectral arrays
###################################################################

class SphericalGrid:
    def __init__(self, Lmax, Mmax, fmetric=None, fricci=None):
        self.grid = shtns.sht(Lmax,Mmax, norm=shtns.SHT_REAL_NORM)
        self.Lmax = Lmax
        self.nTheta, self.nPhi = self.grid.set_grid()
        self.nlm = self.grid.nlm
        self.numTerms = (self.Lmax+1)*(self.Lmax+1)
        self.extents = np.array([self.nTheta,self.nPhi])
        self.l, self.m = YlmIndex(np.arange(self.numTerms))

#grid coordinates and properties:
        self.theta, self.phi = np.meshgrid(
            np.arccos(np.polynomial.legendre.leggauss(self.nTheta)[0])[::-1], 
            np.linspace(0,2*pi*(1-1./self.nPhi),self.nPhi),
            sparse=False)
        self.theta, self.phi = self.theta.T, self.phi.T
        self.gaussian_weights = np.polynomial.legendre.leggauss(self.nTheta)[1]
        self.costheta = np.cos(self.theta)
        self.sintheta = np.sin(self.theta)

        #Metric on grid points:
        if fmetric == None:
            self.gthth = np.zeros(self.extents)
            self.gphph = np.zeros(self.extents)
            self.gthph = np.zeros(self.extents)
        else:
            self.gthth, self.gphph, self.gthph = fmetric(self.theta, self.phi)
            self.UpdateMetric()

        #construct normal matrix for least squares spectral analysis
        self.A = np.zeros((self.nTheta*self.nPhi, self.numTerms))
        for i in xrange(self.numTerms):
#            l, m = YlmIndex(i)
            coeffs = np.zeros(self.numTerms)
            coeffs[i] = 1.0
            self.A[:,i] = self.SpecToPhys(coeffs).flatten()
        self.ATA = linalg.cholesky(self.A.T.dot(self.A))

        # Compute ricci scalar, or not...
        if fricci==None:
            self.ricci = None
        else:
            self.ricci = fricci(self.theta, self.phi)

    def ComputeMetricDerivs(self):
        self.gthth_th, self.gthth_ph = self.D(self.gthth)
        self.gphph_th, self.gphph_ph = self.D(self.gphph)        
        self.gthph_th, self.gthph_ph = self.D(self.gthph)

        #These terms show up when computing Laplacians
        self.gamma_ph = -0.5*self.gphph_th*self.ginvthph*self.ginvphph - 0.5*self.gphph_ph*self.ginvphph*self.ginvphph - self.ginvphph*self.ginvthth*self.gthph_th - 0.5*self.ginvthth*self.ginvthph*self.gthth_th + 0.5*self.ginvthth*self.ginvphph*self.gthth_ph - self.ginvthph*self.ginvphph*self.gthph_ph - self.ginvthph*self.ginvthph*self.gthth_ph
        self.gamma_th = -0.5*self.gphph_ph*self.ginvthph*self.ginvphph - self.gphph_th*self.ginvthph*self.ginvthph + 0.5*self.gphph_th*self.ginvthth*self.ginvphph - self.ginvphph*self.ginvthth*self.gthph_ph - self.ginvthth*self.ginvthph*self.gthph_th - 0.5*self.ginvthth*self.ginvthph*self.gthth_ph - 0.5*self.ginvthth*self.ginvthth*self.gthth_th

#### ComputeMetric ##########################################################
# Given a function returning metric functions computes metric functions and
# area element on the grid
##############################################################################
    def ComputeMetric(self, fmetric):
        self.gthth, self.gphph, self.gthph = fmetric(self.theta, self.phi)
        self.UpdateMetric()

    def UpdateMetric(self):
        self.detg = self.gthth*self.gphph - self.gthph*self.gthph
        self.ginvthth, self.ginvphph, self.ginvthph = self.gphph/self.detg, self.gthth/self.detg, -self.gthph/self.detg
        self.dA = np.sqrt(self.detg)
        self.ComputeMetricDerivs()

#### ComputeRicci ############################################################
# Computes Ricci scalar from metric using spectral differentiation
# CAVEAT EMPTOR, SOME METRIC FUNCTIONS HAVE BAD CONVERGENCE IN Ylm BASIS - BEST
# NOT TO USE GENERAL FORM - eg use conformal spherical form
##############################################################################
    def ComputeRicci(self):
        self.ComputeMetricDerivs()
        gthth_ph2 = self.D2(self.gthth,1)
        gphph_th2 = self.D2(self.gphph,0)
        self.ricci = (self.gphph*(gphph_th * gthth_th + gthth_ph**2) + self.gthth*(gphph_ph*gthth_ph + gphph_th**2 - 2*self.gphph*(gphph_th2 + gthth_ph2)))/2.0/(self.detg**2)

######### SpecToPhys ##########################################################
#   IN:
#   coeffs - spherical harmonic expansion coefficients
#
#   OUT:
#   scalar - set of values defined on the spherical grid
##############################################################################

    def SpecToPhys(self, coeffs):
        if len(coeffs) < self.numTerms:
            coeffs = np.hstack((coeffs,np.zeros(self.numTerms-len(coeffs))))
#        shtns_spec = np.zeros(self.nlm, dtype=np.complex128)
#        for i in xrange(self.numTerms):
#            l, m = YlmIndex(i)
#            index = self.grid.idx(int(self.l[i]),int(abs(self.m[i])))
#            if self.m[i]>=0:
#                shtns_spec.real[index] = coeffs[i]
#            else:
#                shtns_spec.imag[index] = coeffs[i]
        return self.grid.synth(self.StandardToShtns(coeffs))

######### PhysToSpec ##########################################################
#   IN:
#   scalar - set of values defined on the spherical grid
#
#   OUT:
#   coeffs - spherical harmonic expansion coefficients
##############################################################################

    def PhysToSpec(self, scalar):
#        coeffs = np.zeros(self.numTerms)
        shtns_spec = self.grid.analys(scalar)
#        for i in xrange(self.numTerms):
#            l, m = YlmIndex(i)
#            index = self.grid.idx(int(self.l[i]),int(abs(self.m[i])))
#            if self.m[i]>=0:
#                coeffs[i] = shtns_spec[index].real
#            else:
#                coeffs[i] = shtns_spec[index].imag       
        return self.ShtnsToStandard(shtns_spec)

####### PhysToSpecLS #####################################################
# Computes spectral analysis using least squares fit - usually faster
# and slightly more accurate when function values are reconstructed
###########################################################################

    def PhysToSpecLS(self, scalar):
        return linalg.cho_solve((self.ATA, False), np.dot(self.A.T, scalar.flatten()))

### Integrate ################################################################
# Integrates a function over the manifold
##############################################################################
    def Integrate(self, scalar):
        return 2*pi/self.extents[1]*np.sum(np.inner(self.gaussian_weights, (scalar*self.dA/self.sintheta).T))

####### D #################################################
# Gives the differential of a scalar
#
# IN: scalar s defined on SphericalGrid
#     var - if 0 or 1 returns just theta or phi derivative
# OUT: (ds/dtheta, ds/dphi)
# 
############################################################

    def D(self, scalar, var = None):
        coeffs = self.grid.analys(scalar)
        grad = self.grid.synth_grad(coeffs)

        if var==0:
            return grad[0]
        elif var==1:
            return grad[1]*self.sintheta
        else:
            return np.array((grad[0], grad[1]*self.sintheta))

    def D2(self, scalar, var = None):
        coeffs = self.PhysToSpec(scalar)
        d2dphi = self.SpecToPhys(-self.m*self.m*coeffs)
        ddtheta = self.grid.synth_grad(self.StandardToShtns(coeffs))[0]
        laplacian = self.SpecToPhys(-self.l*(self.l+1)*coeffs)
        if var == 0:
            return laplacian - d2dphi/(self.sintheta*self.sintheta) - ddtheta*self.costheta/self.sintheta
        if var == 1:
            return d2dphi
        else: 
            return laplacian - d2dphi/(self.sintheta*self.sintheta) - ddtheta*self.costheta/self.sintheta, d2dphi


    def StandardToShtns(self,coeffs):
        shtns_spec = np.zeros(self.nlm, dtype=np.complex128)
        for i in xrange(self.numTerms):
#            l, m = YlmIndex(i)
            index = self.grid.idx(int(self.l[i]),int(abs(self.m[i])))
            if self.m[i]>=0:
                shtns_spec.real[index] = coeffs[i]
            else:
                shtns_spec.imag[index] = coeffs[i]
        return shtns_spec

    def ShtnsToStandard(self,shtns_spec):
        coeffs = np.zeros(self.numTerms)
        for i in xrange(self.numTerms):
#            l, m = YlmIndex(i)
            index = self.grid.idx(int(self.l[i]),int(abs(self.m[i])))
            if self.m[i]>=0:
                coeffs[i] = shtns_spec[index].real
            else:
                coeffs[i] = shtns_spec[index].imag
        return coeffs
            

###### RaiseForm ##################################
# Raises the index of a 1-form
# IN: form - 1-form a_i
# OUT: vector g^{ij} a_j
################################################### 
    def Raise(self, form):
        return np.array((form[0]*self.ginvthth + form[1]*self.ginvthph, form[1]*self.ginvphph + form[0]*self.ginvthph))

##### Div ##########################################
# Computes the divergence of a vector
# IN: vec - 2-component vector on grid
# OUT: div - divergence of vector
#####################################################

    def Div(self, vec):
        return 1/self.dA * (self.D(self.dA*vec[0])[0]+ self.D(self.dA*vec[1])[1])

##### Hodge ####################################################
# Computes the (raised) Hodge dual of a 1-form
################################################################
    def Hodge(self, form):
        return np.array((form[1]/self.dA, -form[0]/self.dA))

##### Laplacian ##########################################
# Computes div grad s for a scalar function s
# IN: scalar
# OUT: div grad scalar
##########################################################

    def Laplacian(self, scalar): 
        coeffs = self.PhysToSpec(scalar)
        d2phph = self.SpecToPhys(-self.m*self.m*coeffs)
        dth,dph = self.grid.synth_grad(self.StandardToShtns(coeffs))
        dthph = self.D(dth,0)
        sph_laplacian = self.SpecToPhys(-self.l*(self.l+1)*coeffs)
        d2thth = sph_laplacian - d2phph/(self.sintheta*self.sintheta) - dth*self.costheta/self.sintheta
        return d2thth*self.ginvthth + 2.0*self.ginvthph*dthph + d2phph*self.ginvphph + dth*self.gamma_th + dph*self.gamma_ph

#### InterpolateToPoint #############################################
# Computes function value at a point using spectral interpolation
#####################################################################
    def InterpolateToPoint(self, scalar, theta, phi):
        return self.grid.SH_to_point(x.grid.analys(scalar), theta, phi)

#### EvalAtPoint ##################################################
# Computes function value at a point given spectral coefficients
###################################################################

    def EvalAtPoint(self, coeffs, theta, phi):
        return self.grid.SH_to_point(self.StandardToShtns(coeffs), np.cos(theta), phi)

    EvalAtPoints = np.vectorize(EvalAtPoint, excluded = ['self','coeffs'])

#### Minimize ########################################################
# Find a local minimum of a function using spectral interpolation
######################################################################

    def Minimize(self, scalar):
        coeffs = self.PhysToSpecLS(scalar)
        gridMin = np.min(scalar)
        gridMinIndex = np.unravel_index(np.argmin(scalar), self.extents)
        gridMinTh, gridMinPh = self.theta[gridMinIndex], self.phi[gridMinIndex]
        
        #lower and upper bounds on phi and theta: min 
        phi_lower = self.phi[0,(gridMinIndex[1]-1)%self.extents[1]]
        phi_upper = self.phi[0,(gridMinIndex[1]+1)%self.extents[1]]
        if phi_upper<phi_lower:
            phi_upper += 2*pi 

        if gridMinIndex[0] !=0 and gridMinIndex[0] != (self.extents[0]-1):
            theta_lower = self.theta[gridMinIndex[0]-1,0]
            theta_upper = self.theta[gridMinIndex[0]+1,0]
        elif gridMinIndex[0] == 0:
            theta_lower = 0
            theta_upper = self.theta[1,0]
        else:
            theta_lower = self.theta[self.extents[0]-2,0]
            theta_upper = pi

        bnds = ((theta_lower, theta_upper), (phi_lower, phi_upper))
        function = lambda pt: self.EvalAtPoint(coeffs, pt[0], pt[1]%(2*pi))
        return optimize.minimize(function,(gridMinTh,gridMinPh), method='SLSQP', bounds = bnds, tol = 1e-12).fun

#x = SphericalGrid(15,15)
#x.gthth = np.ones(x.extents)
#x.gthph = 0*x.gthth
#x.gphph = x.sintheta**2
#x.UpdateMetric()

#for i in xrange(3,4):
#    l, m = YlmIndex(i)
#    coeffs = np.zeros(x.numTerms)
#    coeffs[i] = 1.0
#    s = x.SpecToPhys(coeffs)
#    np.std(x.Laplacian(s) + l*(l+1)*s)
