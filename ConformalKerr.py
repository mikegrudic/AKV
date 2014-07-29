import numpy as np
import scipy
import math
import SphericalGrid
from scipy import weave
from matplotlib import pyplot as plt

code ="""
double x, xnew, error, t2;
t2 = x0;
x = x0;
do {
    xnew = 2*atan(exp(-a*a*cos(x)/(1+a*a + sqrt(1-a*a))) * tan(t2/2));
    error = fabs(x - xnew);
    x = xnew;
   } while (error > 1e-16);
   return_val = xnew;
"""

def theta1(x0, a):
    return weave.inline(code, ['x0','a'], extra_compile_args =['-O3 -mtune=native -march=native -ffast-math -msse3 -fomit-frame-pointer -malign-double -fstrict-aliasing'])

def ConformalKerr(a, grid=None):
    r = 1 + math.sqrt(1-a**2)

    if grid == None: grid = SphericalGrid.SphericalGrid(15,15)

    t1 = np.array([theta1(t2, a) for t2 in grid.theta[:,0]])

    H = 0.5*np.log((r**2+a**2)**2/(r**2 + a**2*np.cos(t1)**2) * np.sin(t1)**2/np.sin(grid.theta[:,0])**2)
    Hgrid = (np.ones(grid.extents).T * H).T

    coeffs = grid.PhysToSpec(Hgrid)
    coeffs[np.abs(coeffs) < 1e-16] = 0.0
    return coeffs
