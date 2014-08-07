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

        # Compute ricci scalar, or not...
        if fricci==None:
            self.ricci = None
        else:
            self.ricci = fricci(self.theta, self.phi)

    def ComputeMetricDerivs(self):
        self.gthth_th, self.gthth_ph = self.D(self.gthth)
        self.gphph_th, self.gphph_ph = self.D(self.gphph)        
        self.gthph_th, self.gthph_ph = self.D(self.gthph)

        psi = 0.5*np.log(self.gthth)
        dpsi = self.D(psi)
        self.gamma = np.zeros((2,2,2,self.nTheta, self.nPhi))
        self.gamma[0,0,:] = dpsi
        self.gamma[0,1,0] = dpsi[1]
        self.gamma[0,1,1] = -self.sintheta*(self.costheta + self.sintheta*dpsi[0])
        self.gamma[1,0,0] = -dpsi[1]/self.sintheta**2
        self.gamma[1,1,0] = self.costheta/self.sintheta + dpsi[0]
        self.gamma[1,0,1] = self.gamma[1,1,0]
        self.gamma[1,1,1] = dpsi[1]

    def SetConformalFactor(self, psi_coeffs):
        psi = self.SpecToPhys(psi_coeffs)

        self.gthth, self.gphph = np.exp(2*psi), np.exp(2*psi)*self.sintheta**2
        self.UpdateMetric()
        self.ComputeMetricDerivs()

        self.ricci = 2/self.gthth*(1 - self.SphereLaplacian(psi))


    def CovariantDeriv(self, form):
        if form.shape == (2,self.nTheta, self.nPhi):
            partial = np.array([self.D(form[0]), self.D(form[1])])
            return partial + self.gamma[0]*form[0] + self.gamma[1]*form[1]
        
        

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

    def SpecToPhys(self, coeffs, coeffs2=None):
        if len(coeffs) < self.numTerms:
            coeffs = np.hstack((coeffs,np.zeros(self.numTerms-len(coeffs))))
        if coeffs2 is not None:
            if len(coeffs2) < self.numTerms:
                coeffs2 = np.hstack((coeffs2,np.zeros(self.numTerms-len(coeffs2))))
            vtheta, vphi = self.grid.synth(self.StandardToShtns(coeffs), self.StandardToShtns(coeffs2))
            vphi /= self.sintheta
            return np.array([vtheta, vphi])
        
        else:
            return self.grid.synth(self.StandardToShtns(coeffs))

######### PhysToSpec ##########################################################
#   IN:
#   scalar - set of values defined on the spherical grid
#
#   OUT:
#   coeffs - spherical harmonic expansion coefficients
##############################################################################

    def PhysToSpec(self, f1, f2=None):
        if f2 is not None:
            s1, s2 = self.grid.analys(f1, f2*self.sintheta)
            return np.array([self.ShtnsToStandard(s1), self.ShtnsToStandard(s2)])
        else:
            s1 = self.grid.analys(f1) 
            return self.ShtnsToStandard(s1)

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
        return self.PhysToSpec(scalar)[0]*np.sqrt(4.0*np.pi)

    def AllDerivs(self, f):
        df = self.D(f)
        d2 = self.D2(f)
        return df[0], df[1], self.D(df[0],1), d2[0], d2[1]
    
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
        double derp = 0.0;
        for (i = 0; i < n; ++i){
            if (M1(i)>=0) RE1(INDEX1(i)) = COEFFS1(i);
            else IM1(INDEX1(i)) = COEFFS1(i);
        }
        """
        weave.inline(code,['index', 'm', 're', 'im', 'n','coeffs'], extra_compile_args =['-O3 -mtune=native -march=native -ffast-math -msse3 -fomit-frame-pointer -malign-double -fstrict-aliasing'])
        return re + 1j*im

    def ShtnsToStandard(self,shtns_spec):
        index, m, coeffs, n, re, im = self.index, self.m, np.zeros(self.numTerms), self.numTerms, shtns_spec.real, shtns_spec.imag
        code = """
        int i;
        double derp = 0.0;
        for(i=0; i<n; ++i){
            if (M1(i)>=0) COEFFS1(i) = RE1(INDEX1(i));
            else COEFFS1(i) = IM1(INDEX1(i));
            }
        """
        weave.inline(code,['index','m','re','im','n','coeffs'],extra_compile_args =['-O3 -mtune=native -march=native -ffast-math -msse3 -fomit-frame-pointer -malign-double -fstrict-aliasing'])
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
        sph_laplacian = self.SpecToPhys(-self.l*(self.l+1)*coeffs)
        return sph_laplacian/self.gthth

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

    def CalcLaplaceEig(self, N = None):
        if N is None:
            N = self.numTerms - 2
        coeffs = np.zeros(self.numTerms)
        matrix = np.zeros((self.numTerms-1, self.numTerms-1))
        ll1 = -self.l*(self.l+1)

        Ylm = []
        for i in xrange(1,self.numTerms):
            coeffs = np.zeros(self.numTerms)
            coeffs[i] = 1.0
            ylm = self.SpecToPhys(coeffs)
            Ylm.append(ylm)

        for i in xrange(1,self.numTerms):
            Lf_s = [self.Integrate(Ylm[i-1]*Ylm[j-1]) for j in xrange(1,self.numTerms)]
            matrix[i-1] = Lf_s

        matrix[np.abs(matrix) < 1e-12] = 0.0
        print "Matrix population: %g"%(float(np.count_nonzero(matrix))/matrix.size)

        self.lap_eig, self.lap_vec = scipy.sparse.linalg.eigsh(np.diag(-self.l[1:]*(1.0*self.l[1:]+1)), N, M=(matrix.T + matrix)/2,  which='SM')

        sort_index = np.abs(self.lap_eig).argsort()
        self.lap_eig, self.lap_vec = self.lap_eig[sort_index].real, self.lap_vec.T[sort_index].real
        self.lap_vec = np.column_stack((np.zeros(len(self.lap_vec)), self.lap_vec))        
#        self.lap_vec[np.abs(self.lap_vec) < 1e-12] = 0.0
        self.lap_basis = np.array([self.SpecToPhys(vec) for vec in self.lap_vec])
        norms = np.array([np.sqrt(self.Integrate(f**2)) for f in self.lap_basis])
        self.lap_basis = np.array([f/norm for f, norm in zip(self.lap_basis, norms)])

    def TestLaplaceEigs(self):
        self.CalcLaplaceEig()
        
        return np.array([[eig, self.Integrate((eig*f - self.Laplacian(f))**2), self.Integrate(f**2)] for eig, f in zip(self.lap_eig, self.lap_basis)])


    def KillingLaplacian(self, v):
        psi = 0.5*np.log(self.gthth)
        vtheta10, vtheta01, vtheta11, vtheta20, vtheta02 = self.AllDerivs(v[0])
        vphi10, vphi01, vphi11, vphi20, vphi02 = self.AllDerivs(v[1])
        psi10, psi01, psi11, psi20, psi02 = self.AllDerivs(psi)
        SLvtheta = self.SphereLaplacian(v[0])
        cott = self.costheta/self.sintheta
        lvtheta = 2*((cott**2 + cott*psi01 - psi20)*v[0] - 2*psi10*vtheta10 - vtheta01/self.sintheta**2 - SLvtheta - 2*psi01*vphi10 + (cott + psi10)*vphi01 - 0.5*vphi11)
        lvtheta /= self.gthth
        lvphi = 2*(psi02*v[1]/self.sintheta**2 - (1.5*cott + psi10)*vphi10 - 2*psi01*vphi01/self.sintheta**2 - 0.5*vphi20 - vphi02/self.sintheta**2 + (psi01*vtheta10 -(1.5*cott + 2*psi10)*vtheta01 - 0.5*vtheta11)/self.sintheta**2)
        
        return lvtheta, lvphi

    def VecFieldNorm(self, vec):
        return self.Integrate(self.gthth*(vec[0]**2 + self.sintheta**2 * vec[1]**2))
