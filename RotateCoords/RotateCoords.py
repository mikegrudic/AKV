import numpy as np

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

def RotateGridY(grid, angle):
    theta2, phi2 = RotateCoords(grid.theta, grid.phi, np.array([0,1.0,0]), -alpha)
    gthth_s, gphph_s, gthph_s = grid.PhysToSpec(grid.gthth), grid.PhysToSpec(grid.gphph), grid.PhysToSpec(grid.gthph)
    ricci_s = grid.PhysToSpec(grid.ricci)
    
    gthth_rot = grid.EvalAtPoint
