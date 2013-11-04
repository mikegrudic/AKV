import SphericalGrid
import numpy as np
a = np.load("a.npy")
dH = np.load("delta_H.npy")
dL = np.load("delta_L.npy")
pot = np.load("Potential_error.npy")
eigs = np.load("Eigenvalues.npy")
lm = np.load("lm.npy")
vec = np.load("AKV_Error.npy")
ylm = np.load("Ylm.npy")
a = np.load("a.npy")

nPoints = dH
delta = np.sqrt(dH**2 + dL**2)

dDelta = (delta[:,1:]-delta[:,:-1])
da = a[1:]-a[:-1]
dEigs = eigs[:,1:]-eigs[:,:-1]
dError = (pot[:,1:]-pot[:,:-1])

fits = np.array([[np.polyfit(a[40:], np.abs(ylm[i,40:,j]),10) for i in xrange(len(lm))] for j in xrange(3)])

eig_fit = np.array([[np.polyfit(a[40:], np.abs(eigs[i,40:,j]),10) for i in xrange(len(lm))] for j in xrange(3)])

for i in xrange(len(lm)):
    for j in xrange(3):
        l, m = lm[i]
    
#    np.savetxt("%d%d_vec.dat"%(l,m),np.column_stack((delta[i],vec[i])))
#    np.savetxt("%d%d_eig1.dat"%(l,m),np.column_stack((delta[i],eigs[i,:,0].real)))
#    np.savetxt("%d%d_eig2.dat"%(l,m),np.column_stack((delta[i],eigs[i,:,1].real)))
#    np.savetxt("%d%d_eig3.dat"%(l,m),np.column_stack((delta[i],eigs[i,:,2].real)))
        np.savetxt("%d%d_fit%d.dat"%(l,m,j), fits[j,i].T, fmt="%g")
        np.savetxt("%d%d_eig%d.dat"%(l,m,j),np.column_stack((a,eigs[i,:,j].real)))
#        np.savetxt("%d%d_eig%d.dat"%(l,m,j), eig_fit[j,i].T, fmt="%g")
#l, m = SphericalGrid.YlmIndex(np.arange(4,
#print dError_dDelta[:,i]
