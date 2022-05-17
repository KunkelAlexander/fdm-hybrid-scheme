import numpy as np 

ROLL_R = -1
ROLL_L = 1

#Quantum Pressure divided by dimension
def quantumPressure(density, dx):
    sr = 0.5 * np.log(density)
    qp = np.zeros(density.shape)

    for i in range(density.ndim):
        srr = np.roll(sr, ROLL_R, axis=i)
        srl = np.roll(sr, ROLL_L, axis=i)
        qp += -1 / 2 * (srr - 2 * sr + srl) / (dx ** 2) - 1 / 2 * ((srr - srl) / (2 * dx)) ** 2

    return qp/density.ndim 

#Convective term in HJ-equation divided by dimension
def velocitySquare(phase, dx):
    si = phase
    velocitySquare = np.zeros(phase.shape)
    
    for i in range(phase.ndim):
        sir = np.roll(si, ROLL_R, axis=i)
        sil = np.roll(si, ROLL_L, axis=i)

        velocitySquare += - 1 / 2 * ((sir - sil) / (2 * dx)) ** 2

    return velocitySquare/phase.ndim

#Laplacian of phase
def phaseCurvature(phase, dx):
    si = phase
    result = np.zeros(phase.shape)
    
    for i in range(phase.ndim):
        sir = np.roll(si, ROLL_R, axis=i)
        sil = np.roll(si, ROLL_L, axis=i)

        result += ((sir - 2 * si + sil) / (dx**2))

    return result/phase.ndim 
 
#\partial_t S / dim
def deltaSi(density, phase, dx):
    sr = 0.5 * np.log(density)

    si = phase
    dsi = np.zeros(sr.shape)
    dim = sr.ndim
    for i in range(dim):
        sir = np.roll(si, ROLL_R, axis=i)
        sil = np.roll(si, ROLL_L, axis=i)
        srr = np.roll(sr, ROLL_R, axis=i)
        srl = np.roll(sr, ROLL_L, axis=i)

        dsi += (
            -1 / 2 * (srr - 2 * sr + srl) / (dx ** 2)
            + ((sir - sil) / (2 * dx)) ** 2
            - 1 / 2 * (((sir - sil) / (2 * dx)) ** 2 + ((srr - srl) / (2 * dx)) ** 2)
        )
    return dsi/dim

# sum_i f_i^2 / dim
def getVelocity(f, dx):
    res = np.zeros(f.shape)
    for i in range(f.ndim):
        res += fd.getCenteredGradient(f, dx, axis=i)**2
    return res/f.ndim  

# lap f / dim
def getCurvature(f, dx):
    res = np.zeros(f.shape)
    for i in range(f.ndim):
        res += fd.getCenteredLaplacian(f, dx, axis=i)
    return res/f.ndim
import fd

def getPecletSr(density, dx, eta):
    vr = np.zeros(density.shape)
    sr = 0.5 * np.log(density)
    for i in range(sr.ndim):
        vr += fd.getCenteredGradient(sr, dx, axis=i)**2
    vr = np.sqrt(vr)
    D = eta/(2*dx)
    F = vr/sr.ndim  
    return F/D

def getPecletSi(phase, dx, eta):
    vi = np.zeros(phase.shape)
    si = phase
    for i in range(si.ndim):
        vi += fd.getCenteredGradient(si, dx, axis=i)**2
    vi = np.sqrt(vi)
    D = -eta/(2*dx)
    F = vi/si.ndim  
    return F/D

conditions = {
    "qp": lambda density, phase, dx, eta: quantumPressure(density, dx),
    "qp dx^2": lambda density, phase, dx, eta: quantumPressure(density, dx) * dx**2,
    "lap S": lambda density, phase, dx, eta: phaseCurvature(phase, dx),
    "lap S dx^2": lambda density, phase, dx, eta: phaseCurvature(phase, dx) * dx**2,
    "dS": lambda density, phase, dx, eta: deltaSi(density, phase, dx),
    "dS dx^2": lambda density, phase, dx, eta: deltaSi(density, phase, dx) * dx**2,
    "peclet sr": lambda density, phase, dx, eta: getPecletSr(density, dx, eta),
    "peclet si": lambda density, phase, dx, eta: getPecletSi(phase, dx, eta),
} 