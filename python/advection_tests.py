import numpy as np
import math 

"""""" """""" """""" """""" """""" ""
"""   Initial conditions     """
"""""" """""" """""" """""" """""" ""

def gaussian1D(xx, dx, t=0, x0 = 1.2, v = 1, alpha = 1/5):
    psi = np.exp(-(xx-x0-t*v)**2/alpha)
    return psi, xx * v

def tophat1D(xx, dx, t=0, x0 = .6, v = 1, L = 4, w = 1, h = 1):
    x0c = (x0 + v * t)%L
    x0l, x0r = x0c - w/2, x0c + w/2
    psi = h * (xx > x0l) * (xx < x0r)
    return psi, xx * v

def periodic1D(xx, dx, t, B = 1, omega = 1):
    x0   = 2 * np.arctan(np.tan(xx/2)*np.exp(-B/omega * np.sin(omega*t)))
    q0 = np.zeros(xx.shape)
    q  = q0 + np.log((1 + np.tan(x0/2)**2 * np.exp(2*B/omega*np.sin(omega*t)))/(1 + np.tan(x0/2)**2)) - B/omega*np.sin(omega*t)
    u    = -B * np.cos(xx) * np.cos(omega * t)
    return np.exp(q), -B * np.cos(xx) * np.cos(omega * t)

def periodic2D(xx, yy, dx, t, B = 1, omega = 1):
    u = -B * np.cos(xx) * np.cos(omega * t)
    v = -B * np.cos(yy) * np.cos(omega * t)
    x0   = 2 * np.arctan(np.tan(xx/2)*np.exp(-B/omega * np.sin(omega*t)))
    y0   = 2 * np.arctan(np.tan(yy/2)*np.exp(-B/omega * np.sin(omega*t)))
    q0 = np.zeros(xx.shape)
    q  = q0
    q  = q + np.log((1 + np.tan(x0/2)**2 * np.exp(2*B/omega*np.sin(omega*t)))/(1 + np.tan(x0/2)**2))
    q  = q + np.log((1 + np.tan(y0/2)**2 * np.exp(2*B/omega*np.sin(omega*t)))/(1 + np.tan(y0/2)**2))
    q  = q - 2*B/omega*np.sin(omega*t)
    return np.exp(q), -B * np.cos(omega * t) * (np.cos(xx) + np.cos(yy))