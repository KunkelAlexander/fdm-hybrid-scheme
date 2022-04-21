import scipy.optimize as sco
import numpy as np 

#Define solutions for hyperbolic conservation laws that we can then compare against ENO scheme
#

def charEq1(y, x, t, alpha):
    return y - alpha - np.pi*np.sin(np.pi*(x - y*t))

def dcharEq1(y, x, t, alpha):
    return 1 + t*np.pi**2*np.cos(np.pi*(x - y*t))

def charEq2(y, x, t, alpha):
    return y - np.exp(-(x - y*t-3)**2)

def solveCharEq1(x, t, alpha):
    return sco.newton(charEq1, 0, args=(x, t, alpha), tol=1e-15, maxiter=5000)

def solveCharEq2(x, t, alpha):
    return sco.newton(charEq2, 0, args=(x, t, alpha), tol=1e-4, maxiter=5000)

def eq1AnalyticalSolution(xx, dx, t, x0 = 2, alpha = 0):
    y = np.zeros(xx.shape)
    for i, x in enumerate(xx):
        y[i] = solveCharEq1(x - x0, t, alpha) - alpha

    return y, xx * 0


def eq1(xx, dx, t, x0 = 2):
    return - np.cos(np.pi * (xx-x0)), xx * 0

def eq2(xx, yy, dx, t, x0 = 2):
    return - np.cos(np.pi * ((xx-x0 + yy-x0)/2)), xx * 0

def charEq3(r, x, t):
    return x - np.cos(r)*t - r

def solveCharEq3(x, t):
    return sco.newton(charEq3, 0, args=(x, t), tol=1e-5, maxiter=2000)

def rf(xx, dx, t):
    y = np.zeros(xx.shape)
    for i, x in enumerate(xx):
        y[i] = solveCharEq3(x, t)

    return y

def eq3AnalyticalSolution(xx, dx, t):
    r = rf(xx, dx, t)
    return np.cos(r)**2/2*t + np.sin(r), xx * 0


def eq3(xx, dx, dt):
    return np.sin(xx), xx * 0