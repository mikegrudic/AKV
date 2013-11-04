#!/usr/bin/env python                
import scipy
from scipy import special, misc, linalg, optimize, weave
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
        self.grid = shtns.sht(Lmax,Mmax, norm=shtns.SHT_REAL_NORM, nthreads=4)
        self.Lmax = Lmax
        self.nTheta, self.nPhi = self.grid.set_grid()
        self.nlm = self.grid.nlm
        self.numTerms = (self.Lmax+1)*(self.Lmax+1)
        self.extents = np.array([self.nTheta,self.nPhi])
        self.l, self.m = YlmIndex(np.arange(self.numTerms))
        self.index = np.array([self.grid.idx(int(self.l[i]),int(abs(self.m[i]))) for i in xrange(self.numTerms)])

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
#        self.A = np.zeros((self.nTheta*self.nPhi, self.numTerms))
#        for i in xrange(self.numTerms):
#            l, m = YlmIndex(i)
#            coeffs = np.zeros(self.numTerms)
#            coeffs[i] = 1.0
#            self.A[:,i] = self.SpecToPhys(coeffs).flatten()
#        self.ATA = linalg.cholesky(self.A.T.dot(self.A))

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
        self.gamma_ph = (-(self.gthth**2*self.gphph_ph) + self.gthth*(self.gthph*(2*self.gthth_ph + self.gphph_th) + self.gphph*(self.gthth_ph - 2*self.gthph_th)) + self.gthph*(-2*self.gthph*self.gthth_ph + self.gphph*self.gthth_th))/2.0/self.detg**2
        self.gamma_th = (-2*self.gthph**2*self.gphph_th + self.gthph*(self.gthth*self.gphph_ph + self.gphph*(self.gthth_ph + 2*self.gthph_th)) + self.gphph*(self.gthth*(-2*self.gthth_ph + self.gphph_th) - self.gphph*self.gthth_th))/2.0/self.detg**2

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
        d2gthth_dph = self.D2(self.gthth,1)
        d2gphph_dth = self.D2(self.gphph,0)
        d2gthph_dthdph = self.D(self.D(self.gthph,0),1)

        self.ricci = ((self.gthth*self.gphph_ph*self.gthth_ph)/2. - 
        self.gthph*self.gthph_ph*self.gthth_ph + 
        (self.gphph*self.gthth_ph**2)/2. + 
        self.gthph**2*d2gthth_dph - 
        self.gphph*self.gthth*d2gthth_dph - 
        (self.gthph*self.gthth_ph*self.gphph_th)/2. + 
        (self.gthth*self.gphph_th**2)/2. - 
        self.gthth*self.gphph_ph*self.gthph_th + 
        2*self.gthph*self.gthph_ph*self.gthph_th - 
        self.gthph*self.gphph_th*self.gthph_th + 
        (self.gthph*self.gphph_ph*self.gthth_th)/2. - 
        self.gphph*self.gthph_ph*self.gthth_th + 
        (self.gphph*self.gphph_th*self.gthth_th)/2. - 
        2*self.gthph**2*d2gthph_dthdph + 
        2*self.gphph*self.gthth*d2gthph_dthdph + 
        self.gthph**2*d2gphph_dth - 
        self.gphph*self.gthth*d2gphph_dth)/self.detg**2

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
        return self.grid.synth(self.StandardToShtns(coeffs))

######### PhysToSpec ##########################################################
#   IN:
#   scalar - set of values defined on the spherical grid
#
#   OUT:
#   coeffs - spherical harmonic expansion coefficients
##############################################################################

    def PhysToSpec(self, scalar):
        shtns_spec = self.grid.analys(scalar)
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

### SphereIntegrate
    def SphereIntegrate(self, scalar):
        return self.PhysToSpec(scalar)[0]/np.sqrt(4.0*np.pi)

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
        index, m, re, im, n = self.index, self.m, np.zeros(self.nlm), np.zeros(self.nlm), self.numTerms
        code = """
        int i;
        for (i = 0; i < n; ++i){
            if (M1(i)>=0) RE1(INDEX1(i)) = COEFFS1(i);
            else IM1(INDEX1(i)) = COEFFS1(i);
        }
        """
        weave.inline(code,['index', 'm', 're', 'im', 'n','coeffs'])
        return re + 1j*im

    def ShtnsToStandard(self,shtns_spec):
        index, m, coeffs, n, re, im = self.index, self.m, np.zeros(self.numTerms), self.numTerms, shtns_spec.real, shtns_spec.imag
        code = """
        int i;
        for(i=0; i<n; ++i){
            if (M1(i)>=0) COEFFS1(i) = RE1(INDEX1(i));
            else COEFFS1(i) = IM1(INDEX1(i));
            }
        """
        weave.inline(code,['index','m','re','im','n','coeffs'])
        return coeffs

###### RaiseForm ##################################
# Raises the index of a 1-form
# IN: form - 1-form a_i
# OUT: vector g^{ij} a_j
################################################### 
    def Raise(self, form):
        return np.array((form[0]*self.ginvthth + form[1]*self.ginvthph, form[1]*self.ginvphph + form[0]*self.ginvthph))

    def VecLength(self, vector):
        return np.sqrt(self.gthth*vector[0]**2 + 2*self.gthph*vector[0]*vector[1] + self.gphph*vector[1]**2)

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

    def Laplacian(self, scalar, dth=None, dph=None): 
        coeffs = self.PhysToSpec(scalar)
        d2phph = self.SpecToPhys(-self.m*self.m*coeffs)
        if dth==None:
            dth,dph = self.grid.synth_grad(self.StandardToShtns(coeffs))
            dph *= self.sintheta
        dthph = self.D(dth,0)
        sph_laplacian = self.SpecToPhys(-self.l*(self.l+1)*coeffs)
#        print np.std(self.gamma_th), np.std(self.gamma_ph)
        return sph_laplacian/self.gthth
#        d2thth = sph_laplacian - d2phph/(self.sintheta*self.sintheta) - dth*self.costheta/self.sintheta
#        return d2thth*self.ginvthth + 2.0*self.ginvthph*dthph + d2phph*self.ginvphph + dth*self.gamma_th + dph*self.gamma_ph
#        return sph_laplacian*self.ginvthth

    def SphereLaplacian(self, scalar):
        return self.SpecToPhys(-self.l*(self.l+1)*self.PhysToSpec(scalar))

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
#        coeffs = self.PhysToSpecLS(scalar)
        coeffs = self.PhysToSpec(scalar)
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

    def CalcLaplaceEig(self):
        coeffs = np.zeros(self.numTerms)
        matrix = np.zeros((self.numTerms-1, self.numTerms-1))

        for i in xrange(1,self.numTerms):
            coeffs = np.zeros(self.numTerms)
            coeffs[i] = 1.0
            Lf_s = self.PhysToSpec(self.Laplacian(self.SpecToPhys(coeffs)))
            matrix[i-1] = Lf_s[1:]
           
        matrix[np.abs(matrix) < 1e-13] = 0

        self.lap_eig, self.lap_vec = scipy.linalg.eig(matrix.T)

        sort_index = np.abs(self.lap_eig).argsort()
        self.lap_eig, self.lap_vec = self.lap_eig[sort_index], self.lap_vec[:,sort_index]

        # Must deal with case of complex vector components - construct a real basis by combining degenerate complex conjugates
        complex_index = self.lap_vec.imag.nonzero()

        for vector in self.lap_vec.T[complex_index]:
#            vector, lap
            print vector

        I = np.identity(self.numTerms, dtype=np.complex128)
        I[1:,1:] = self.lap_vec
        self.lap_vec = I
        self.lap_eig = np.insert(self.lap_eig,0,0.0)
        self.lap_basis = np.array([self.SpecToPhys(self.lap_vec[:,i].real) for i in xrange(self.numTerms)])

