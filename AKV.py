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

        Lf = grid.Laplacian(f)
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
        M[i-1] = Hf_s[1:numpoints+1]

        if mNorm == "Owen":
            B[i-1] = Lf_s[1:numpoints+1]
        else:
            B[i-1] = CWBf_s[1:numpoints+1]

    # Solve the generalized eigenvalue problem
    if use_sparse_alg:
#  Truncate all "small" values to 0 to make matrix sparse
        M[np.abs(M) < 1e-12] = 0.0
        B[np.abs(B) < 1e-12] = 0.0

        invB = scipy.linalg.inv(B.T)
        invB = scipy.sparse.csr_matrix(invB)
        M = scipy.sparse.csr_matrix(M.T)
        eigensol = scipy.sparse.linalg.eigs(invB*M, 3, which='SM')
    else:
        eigensol = scipy.linalg.eig(M.T, B.T)

    eigenvals, vRight = eigensol[0], eigensol[1]
    sorted_index = np.abs(eigenvals).argsort()
    eigenvals, vRight = eigenvals[sorted_index],vRight[:,sorted_index]
#    print eigenvals[:3]
#    print vRight[:,:3]
#    exit()
    minEigenvals = eigenvals[sorted_index][:3]

#    for vec in vRight.T[:3]:
#        complex_index = vec.imag.nonzero()
#        print complex_index

    firstVec = np.zeros(grid.numTerms)
    secondVec = np.zeros(grid.numTerms)
    thirdVec = np.zeros(grid.numTerms)

#    index1 = np.argmin(np.abs(eigenvals))
    #Set smallest eigenval to max, find next smallest
#    eigenvals[index1] = eigenvals[np.argmax(np.abs(eigenvals))]*100
#    index2 = np.argmin(np.abs(eigenvals))
    #set second smallest to max to find the third smallest
#    eigenvals[index2] = eigenvals[np.argmax(np.abs(eigenvals))]*100
#    index3 = np.argmin(np.abs(eigenvals))
#    print np.std(vRight.imag)

    firstVec[1:numpoints+1] = vRight[:,0].T.real
    secondVec[1:numpoints+1] = vRight[:,1].T.real
    thirdVec[1:numpoints+1] = vRight[:,2].T.real

    first_pot = grid.SpecToPhys(firstVec)
    second_pot = grid.SpecToPhys(secondVec)
    third_pot = grid.SpecToPhys(thirdVec)

    Area = grid.Integrate(np.ones(grid.extents))
    
    if KerrNorm == True:
        pot1avg = grid.Integrate(first_pot)/Area
        pot2avg = grid.Integrate(second_pot)/Area
        pot3avg = grid.Integrate(third_pot)/Area

        normint1 = grid.Integrate((first_pot-pot1avg)**2)
        normint2 = grid.Integrate((second_pot-pot2avg)**2)
        normint3 = grid.Integrate((third_pot-pot3avg)**2)
        norm1 = np.sqrt(Area**3/(48.0*pi**2*normint1))
        norm2 = np.sqrt(Area**3/(48.0*pi**2*normint2))
        norm3 = np.sqrt(Area**3/(48.0*pi**2*normint3))
    else:
        min1, max1 = grid.Minimize(first_pot), -grid.Minimize(-first_pot)
        min2, max2 = grid.Minimize(second_pot), -grid.Minimize(-second_pot)
        min3, max3 = grid.Minimize(third_pot), -grid.Minimize(-third_pot)
        norm1 = Area/(2*pi*(max1-min1))
        norm2 = Area/(2*pi*(max2-min2))
        norm3 = Area/(2*pi*(max3-min3))

    first_pot = first_pot * norm1
    firstVec = firstVec * norm1
    second_pot = second_pot * norm2
    secondVec = secondVec * norm2
    third_pot = third_pot * norm3
    thirdVec = thirdVec * norm3
    AKV1 = grid.Hodge(grid.D(first_pot))
    AKV2 = grid.Hodge(grid.D(second_pot))
    AKV3 = grid.Hodge(grid.D(third_pot))

    if IO==True:
        np.savetxt(name+"_Eigenvalues.dat", eigenvals)
        np.savetxt(name+"_pot1.dat",np.column_stack((grid.theta.flatten(),grid.phi.flatten(),first_pot.flatten())))
        np.savetxt(name+"_pot2.dat",np.column_stack((grid.theta.flatten(),grid.phi.flatten(),second_pot.flatten())))
        np.savetxt(name+"_pot3.dat",np.column_stack((grid.theta.flatten(),grid.phi.flatten(),third_pot.flatten())))
        np.savetxt(name+"_Ylm1.dat",np.column_stack((l,m,firstVec)),fmt="%d\t%d\t%g")
        np.savetxt(name+"_Ylm2.dat",np.column_stack((l,m,secondVec)),fmt="%d\t%d\t%g")
        np.savetxt(name+"_Ylm3.dat",np.column_stack((l,m,thirdVec)),fmt="%d\t%d\t%g")
        np.savetxt(name+"_vec1.dat", np.column_stack((grid.theta.flatten(), grid.phi.flatten(), AKV1[0].flatten(), AKV1[1].flatten())))
        np.savetxt(name+"_vec2.dat", np.column_stack((grid.theta.flatten(), grid.phi.flatten(), AKV2[0].flatten(), AKV2[1].flatten())))
        np.savetxt(name+"_vec3.dat", np.column_stack((grid.theta.flatten(), grid.phi.flatten(), AKV3[0].flatten(), AKV3[1].flatten())))

    if return_eigs==True:
        return AKV1, AKV2, AKV3, minEigenvals
    else:
        return AKV1, AKV2, AKV3
