import numpy as np
import SphericalGrid as sg

def RotationMatrix(axis,theta):
    axis = axis/np.sqrt(np.dot(axis,axis))
    a = np.cos(theta/2)
    b,c,d = -axis*np.sin(theta/2)
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]]).T

def RotateCoords(theta, phi, axis, angle):
    rot = RotationMatrix(axis, angle)
    x = np.cos(phi)*np.sin(theta)
    y = np.sin(phi)*np.sin(theta)
    z = np.cos(theta)
    pos = np.inner(rot,np.array([x,y,z]).T)
    return np.arccos(pos[2].T), np.arctan2(pos[1].T,pos[0].T)%(2*np.pi)

def RotateScalarY(grid, scalar, angle):
    coeffs = grid.grid.analys(scalar)
    return grid.grid.synth(grid.grid.Yrotate(coeffs, angle))

def RotateGridY(grid, angle):
    theta2, phi2 = RotateCoords(grid.theta, grid.phi, np.array([0,1.0,0]), -angle)
    zprime = np.cos(angle)*np.cos(grid.theta) - np.cos(grid.phi)*np.sin(angle)*np.sin(grid.theta)
    denom1 = np.sqrt(1-zprime**2)
    denom2 = 1 + (np.cos(angle)/np.tan(grid.phi) + np.sin(angle)/np.sin(grid.phi)/np.tan(grid.theta))**2
    dth2_dth = (np.cos(grid.phi)*grid.costheta*np.sin(angle) - np.cos(alpha)*grid.sintheta)/denom1
    dth2_dph = np.sin(angle)*np.sin(grid.phi)*grid.sintheta/denom1
    dph2_dth = np.sin(angle)/grid.sintheta**2/np.sin(grid.phi)/denom2
    dph2_dph = 

grid = sg.SphericalGrid(15,15)

coeffs =  np.zeros(grid.numTerms)
coeffs[2] = 1.0
#coeffs2[1] = 1.0
for i in xrange(8):
    np.savetxt("scalar"+str(i)+".dat", np.column_stack((grid.theta.flatten(), grid.phi.flatten(), RotateScalarY(grid, grid.SpecToPhys(coeffs), i*np.pi/8).flatten())))
#for i, theta in enumerate(grid.theta):
#    print i, theta
