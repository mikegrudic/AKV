#!/usr/bin/env python                                                                                                                                
import scipy
import numpy as np
import SphericalGrid

pi = np.pi

def AKV(Metric=None, RicciScalar=None, grid = None, Lmax=15, KerrNorm=False, mNorm = "Owen", return_eigs=False, name='AKV', use_sparse_alg=False, IO=True):
    """ AKV 
    Returns the 3 minimum shear approximate killing vectors of a 2-manifold with
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

    #Initialize grid
    if grid==None:
        grid = SphericalGrid.SphericalGrid(Lmax, Lmax, Metric, RicciScalar)
        if Metric == None:
            "Must supply either metric function or a SphericalGrid with metric defined!"
            exit()

    MM = LL = grid.extents[0]

    lmax = grid.Lmax
#    l, m = grid.grid.l, grid.grid.m

    #number of terms in spectral expansion to solve for (SpEC code goes up to Lmax-2)
    numpoints = (LL-2)*(LL-2)-1

    #Real space quantities
    f = np.zeros(grid.extents)
    Lf = np.zeros(grid.extents)
    LLf = np.zeros(grid.extents)
    RLf = np.zeros(grid.extents)
    gradRgradf = np.zeros(grid.extents)
    Hf = np.zeros(grid.extents)
    CWBf = np.zeros(grid.extents)

    #Spectral coefficient arrays
    f_s = np.zeros(grid.numTerms)
    Lf_s = np.zeros(grid.numTerms)
    Hf_s = np.zeros(grid.numTerms)
    CWBf_s = np.zeros(grid.numTerms)

    #Matrices - M for H, B for Laplacian
    M = np.zeros((numpoints, numpoints))
    B = np.zeros((numpoints, numpoints))

    dR = grid.D(grid.ricci)
    gradR = grid.Raise(dR)

    l, m = SphericalGrid.YlmIndex(np.arange(grid.numTerms))

    for i in xrange(1,numpoints+1):
        #generate the spherical harmonic of index i
        coeffs = np.zeros(grid.numTerms)
        coeffs[i] = 1.0
        f = grid.SpecToPhys(coeffs)

        #act operators to construct H Ylm in real space
        Df = grid.D(f)

        Lf = grid.Laplacian(f, Df[0], Df[1])
        Lf_s = grid.PhysToSpec(Lf)

        LLf = grid.Laplacian(Lf)
        RLf = grid.ricci*Lf
        gradRgradf = gradR[0]*Df[0] + gradR[1]*Df[1]
        Hf = LLf + RLf + gradRgradf
        Hf_s = grid.PhysToSpec(Hf)

        if mNorm == "CookWhiting":
            CWBf = RLf + gradRgradf;
            CWBf_s = grid.PhysToSpec(CWBf);

        #Populate the matrices
        M[:,i-1] = Hf_s[1:numpoints+1]
        if mNorm == "Owen":
            B[:,i-1] = Lf_s[1:numpoints+1]
        else:
            B[:,i-1] = CWBf_s[1:numpoints+1]

    # Solve the generalized eigenvalue problem
    if use_sparse_alg:
#  Truncate all "small" values to 0 to make matrix sparse
        M[np.abs(M) < 1e-12] = 0.0
        B[np.abs(B) < 1e-12] = 0.0

        invB = scipy.linalg.inv(B)
        invB = scipy.sparse.csr_matrix(invB)
        M = scipy.sparse.csr_matrix(M)
        eigensol = scipy.sparse.linalg.eigs(invB*M, 3, which='SM')
    else:
        eigensol = scipy.linalg.eig(M, B, left=True)

    eigenvals, vLeft, vRight = eigensol#[0], eigensol[1]

#    vRight /= np.sqrt(np.sum(np.abs(vRight)**2, axis=0))
#    vLeft /= np.sqrt(np.sum(np.abs(vLeft)**2, axis=0))

    sorted_index = np.abs(eigenvals).argsort()

    eigenvals, vRight, vLeft = eigenvals[sorted_index], vRight[:,sorted_index], vLeft[:,sorted_index]
    minEigenvals = eigenvals[:3]

    firstVec = np.zeros(grid.numTerms)
    secondVec = np.zeros(grid.numTerms)
    thirdVec = np.zeros(grid.numTerms)

    vecs = [np.zeros(grid.numTerms) for i in xrange(3)]

    for i, vec in enumerate(vecs):
        vec[1:numpoints+1] = vRight[:,i].T.real

    potentials = [grid.SpecToPhys(vec) for vec in vecs]

    Area = grid.Integrate(np.ones(grid.extents))

    for i in xrange(3):    
        if KerrNorm == True:
            potential_avg = grid.Integrate(potentials[i])
            normint = grid.Integrate((potentials[i]-potential_avg)**2)
            norm = np.sqrt(Area**3/(48.0*pi**2*normint))
        else:
            min, max = grid.Minimize(potentials[i]), -grid.Minimize(-potentials[i])
            norm = Area/(2*pi*(max-min))

        vecs[i] = vecs[i] * norm
        potentials[i] = potentials[i] * norm

    AKVs = [grid.Hodge(grid.D(pot)) for pot in potentials]


    if IO==True:
        np.savetxt(name+"_Eigenvalues.dat", eigenvals)
        for i in xrange(3):
            np.savetxt(name+"_pot"+str(i+1)+".dat", np.column_stack((grid.theta.flatten(),grid.phi.flatten(),potentials[i].flatten())))
            np.savetxt(name+"_Ylm"+str(i+1)+".dat", np.column_stack((l,m,vecs[i])),fmt="%d\t%d\t%g")
            np.savetxt(name+"_vec"+str(i+1)+".dat", np.column_stack((grid.theta.flatten(), grid.phi.flatten(), AKVs[i][0].flatten(), AKVs[i][1].flatten())))


    if return_eigs==True:
        return AKVs, minEigenvals
    else:
        return AKVs
