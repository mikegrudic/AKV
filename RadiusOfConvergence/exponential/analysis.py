import SphericalGrid
import numpy as np
a = np.load("a.npy")
dH = np.load("delta_H.npy")
dL = np.load("delta_L.npy")
pot = np.load("Potential_error.npy")
eigs = np.load("Eigenvalues.npy")
lm = np.load("lm.npy")
vec = np.load("AKV_Error.npy")
a=np.load("a.npy")

nPoints = dH
delta = np.sqrt(dH**2 + dL**2)

dDelta = (delta[:,1:]-delta[:,:-1])
da = a[1:]-a[:-1]
dEigs = eigs[:,1:]-eigs[:,:-1]
dError = (pot[:,1:]-pot[:,:-1])

print a.shape

for i in xrange(len(lm)):
    l, m = lm[i]
#    np.savetxt("%d%d_vec.dat"%(l,m),np.column_stack((delta[i],vec[i])))
#    np.savetxt("%d%d_eig1.dat"%(l,m),np.column_stack((delta[i],eigs[i,:,0].real)))
#    np.savetxt("%d%d_eig2.dat"%(l,m),np.column_stack((delta[i],eigs[i,:,1].real)))
#    np.savetxt("%d%d_eig3.dat"%(l,m),np.column_stack((delta[i],eigs[i,:,2].real)))
    np.savetxt("%d%d_eig1.dat"%(l,m),np.column_stack((a,eigs[i,:,0].real)))
    np.savetxt("%d%d_eig2.dat"%(l,m),np.column_stack((a,eigs[i,:,1].real)))
    np.savetxt("%d%d_eig3.dat"%(l,m),np.column_stack((a,eigs[i,:,2].real)))
#l, m = SphericalGrid.YlmIndex(np.arange(4,
#print dError_dDelta[:,i]
