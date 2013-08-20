#!/usr/bin/env python                                                                                                                                
import scipy
import numpy as np
import SphericalGrid

pi = np.pi

class PerturbAKVSol:
    """ PerturbAKVSol
    Calculates the 3 minimum shear approximate killing vectors of a 2-manifold with
    (theta, phi) coordinates, and the SphericalGrid containing all the pseudospectral grid information

    OPTIONS:
    Metric - function which returns the metric functions g_{\theta\theta} and
    g_{\phi\phi} with arguments (theta, phi)
    RicciScalar - function returning the Ricci scalar of the 2-manifold at a point (theta, phi)
    grid - already-created SphericalGrid - must supply either this or Metric and Ricci
    Lmax - maximum spherical harmonic degree on the grid
    KerrNorm - whether to use the integral normalization as in Lovelace 2008. By default
    extrema normalization is used
    mNorm - either "Owen" or "CookWhiting" - default Owen
    return_eigs - whether to return the 3 minimum eigenvalues
    name - leading characters in output files - default 'AKV'

    OUTPUT FILES:
    <name>potx.dat - eigenvector potential of x'th smallest shear - (theta, phi, z(theta,phi))
    <name>Ylmx.dat - Spherical harmonic coefficients of potential - (l, m, Q_lm)
    <name>vecx.dat - actual eigenvector's components: (theta, phi, vector_theta, vector_phi)
    """
    def __init__(self, grid, h, epsilon, KerrNorm=False, name='pertAKV', IO=True):

        sphere_L_s = np.diag(-grid.l*(grid.l+1))
        sphere_H_s = sphere_L_s**2 + 2.0*sphere_L_s

        sphere_L = np.array([grid.SpecToPhys(sphere_L_s[i]) for i in xrange(grid.numTerms)])

        coeffs = np.eye(grid.numTerms)
        sph_harmonics = np.array([grid.SpecToPhys(coeffs[i]) for i in xrange(grid.numTerms)])
        
        Lh = grid.SphereLaplacian(h)

        L_1 = np.array([-2 * h*sphere_L[i] for i in xrange(grid.numTerms)])
        D4_1 = np.array([-2 * h*grid.SpecToPhys(sphere_L_s[i]**2) + grid.SphereLaplacian(L_1[i]) for i in xrange(grid.numTerms)])
        RD2_1 = np.array([-2 * (4*h + Lh)*sphere_L[i] for i in xrange(grid.numTerms)])
        gradRdf_1 = np.array([-2 * (grid.D(2*h + Lh, 0)*grid.D(sph_harmonics[i],0) + grid.D(2*h + Lh,1)*grid.D(sph_harmonics[i],1)/grid.sintheta**2) for i in xrange(grid.numTerms)])
        H_1 = np.array([grid.PhysToSpec(D4_1[i] + RD2_1[i] + gradRdf_1[i]) for i in xrange(grid.numTerms)])

        

'''            
        zeroth_order_vecs = 
        first_order_vecs = np.loadtxt("firstorder.dat").T
        if len(first_order_vecs) < grid.numTerms:
            first_order_vecs = np.hstack((first_order_vecs,np.zeros(3,self.numTerms-len(first_order_vecs))))

        l, m = SphericalGrid.YlmIndex(np.arange(grid.numTerms))

        sorted_index = np.abs(eigenvals).argsort()

        eigenvals, vRight, vLeft = eigenvals[sorted_index], vRight[:,sorted_index], vLeft[:,sorted_index]
        self.minEigenvals = eigenvals[:3]

        self.vecs = [np.zeros(grid.numTerms) for i in xrange(3)]

        self.potentials = [grid.SpecToPhys(vec) for vec in self.vecs]

        Area = grid.Integrate(np.ones(grid.extents))

        for i in xrange(3):    
            if KerrNorm == True:
                potential_avg = grid.Integrate(self.potentials[i])
                normint = grid.Integrate((self.potentials[i]-potential_avg)**2)
                norm = np.sqrt(Area**3/(48.0*pi**2*normint))
            else:
                min, max = grid.Minimize(self.potentials[i]), -grid.Minimize(-self.potentials[i])
                norm = Area/(2*pi*(max-min))

#            self.vecs[i] = self.vecs[i] * norm
#            self.vecs[i] = self.vecs[i]*np.sign(np.argmax(np.abs(self.vecs[i])))/np.linalg.norm(self.vecs[i])
            self.vecs[i] /= np.linalg.norm(self.vecs[i][:4])
            print self.vecs[i][:4]

            self.potentials[i] = self.potentials[i] * norm

        self.AKVs = [grid.Hodge(grid.D(pot)) for pot in self.potentials]

        if IO==True:
            np.savetxt(name+"_Eigenvalues.dat", eigenvals)
            for i in xrange(3):
                np.savetxt(name+"_pot"+str(i+1)+".dat", np.column_stack((grid.theta.flatten(),grid.phi.flatten(),self.potentials[i].flatten())))
                np.savetxt(name+"_Ylm"+str(i+1)+".dat", np.column_stack((l,m,self.vecs[i])),fmt="%d\t%d\t%g")
                np.savetxt(name+"_vec"+str(i+1)+".dat", np.column_stack((grid.theta.flatten(), grid.phi.flatten(), self.AKVs[i][0].flatten(), self.AKVs[i][1].flatten())))

        
    def GetAKV(self):
        return self.AKVs
        
    def GetPotentials(self):
        return self.potentials

    def GetYlm(self):
        return self.vecs

    def GetEigs(self):
        return self.minEigenvals

    def GetMatrixNorms(self):
        deltaM = scipy.linalg.norm(self.M - self.sphere_M,ord='fro')
        deltaL = scipy.linalg.norm(self.B - self.sphere_L_s, ord='fro')
        return deltaM, deltaL
'''
