import numpy as np
import math 
import numpy.polynomial.hermite as Herm
import scipy.special as sc 
from numba import njit 

"""""" """""" """""" """""" """""" ""
"""   Initial conditions     """
"""""" """""" """""" """""" """""" ""

#coefficients_1d = (
#    np.random.rand(2, 10) - 0.5
#)    # Order of magnitude of initial density perturbations
coefficients_1d = np.load("data/1d_coefficients.npy")

#coefficients_2d = (
#    np.random.rand(4, 10, 10) - 0.5
#) 
coefficients_2d = np.load("data/2d_coefficients.npy")

#coefficients_3d = (
#    np.random.rand(8, 10, 10, 10) - 0.5
#)  # Order of magnitude of initial density perturbations
coefficients_3d = np.load("data/3d_coefficients.npy")

@njit
def normalise(psi):
    norm = np.mean(np.abs(psi)**2)
    return psi/np.sqrt(norm + (norm == 0) * 1e-12)

def generate1DUniform(x, dx, t):
    psi = np.ones(x.shape, dtype=complex)
    return normalise(psi)

# Generate analytical solution for the 1D free Schrödinger equation
# Li test 1
def generate1DGaussian(x0, x, t, m = 1, hbar = 1, alpha=1.0 / 10):
    psi = np.sqrt(1 / (alpha + 1.0j * t * hbar/m)) * np.exp(
        -((x - x0) ** 2) / (2 * (alpha + 1.0j * t * hbar/m))
    )
    return psi

def li1(x, dx, t, m = 1, hbar = 1, x0=0.5, alpha= 1/20, eps=0):
    psi = eps + 0j
    psi += generate1DGaussian(x0, x, t, m, hbar, alpha)
    return normalise(psi)

def periodicLi1(x, dx, t, m = 1, hbar = 1,  x0=0.5, alpha= 1/20, eps=0, L=1, N=100):
    psi = eps + 0j
    N = 20
    for i in range(-N, N + 1):
        psi += generate1DGaussian(x0, x + L * i, t, m, hbar, alpha)
    return normalise(psi)


def li2(x, dx, t, m = 1, hbar = 1, x0 = 1, A=np.sqrt(5)-np.sqrt(6), B=np.sqrt(6)):
    C, S = sc.fresnel((x-x0)*np.sqrt(1/(np.pi*t * hbar/m)))
    psi = A/2 + B - A/2*(1+1.j)*C - A/2*(1-1.j)*S
    return psi

def periodiLi2(x, dx, t, m = 1, hbar = 1, x0 = 1, A=np.sqrt(5)-np.sqrt(6), B=np.sqrt(6)):
    psi = 0
    N = 20
    L = 1
    for i in range(-N, N + 1):
      psi += li2(x + L * i, dx, t, m, hbar, x0, A, B)
    return psi


# Li test 3
def li3(x, dx, t, m = 1, hbar = 1, alpha= 1/500, k= 20*np.pi, x0=.5, x1=.1):
    C = np.sqrt(alpha / (alpha + 1.0j * t * hbar/m))
    psi = (
        C
        * np.exp(-((x + x1 - x0 - 1.0j * k * alpha) ** 2) / (2 * (alpha + 1.0j * t * hbar/m)))
        * np.exp(-(alpha * k ** 2) / 2)
    )
    psi += (
        C
        * np.exp(-((x - x1 - x0 + 1.0j * k * alpha) ** 2) / (2 * (alpha + 1.0j * t * hbar/m)))
        * np.exp(-(alpha * k ** 2) / 2)
    )
    return normalise(psi)


def periodicLi3(x, dx, t, m = 1, hbar = 1, alpha=1 / 50, k=1 * np.pi, x0=5, x1=2, eps=2):
    psi = eps + 0j
    N = 500
    L = 10
    for i in range(- N, N + 1):
        psi += li3(x + L * i, dx, t, m, hbar, alpha, k, x0, x1)

    psisq_mean = np.mean(np.abs(psi) ** 2)
    psi /= np.sqrt(psisq_mean) + 1e-8

    return normalise(psi)


# Li test 3
def travellingWavePacket(x, dx, t, m = 1, hbar = 1, alpha= 1/500, k= 20*np.pi, x0=.2):
    C = np.sqrt(alpha / (alpha + 1.0j * t * hbar/m))
    psi = (
        C
        * np.exp(-((x - x0 - 1.0j * k * alpha) ** 2) / (2 * (alpha + 1.0j * t * hbar/m)))
        * np.exp(-(alpha * k ** 2) / 2)
    )
    return normalise(psi)


def standingWave(xx, dx, t, m = 1, hbar = 1,  k = 1):
    omega = hbar/(2*m) * k**2
    return np.exp(1j*(k*xx - omega * t))

def hermite(x, n, m = 1, w = 1, hbar = 1,):
    xi = np.sqrt(m*w/hbar)*x
    herm_coeffs = np.zeros(n+1)
    herm_coeffs[n] = 1
    return Herm.hermval(xi, herm_coeffs)
  
def oscillatorEigenstate1D(xx, dx, t, m = 1, hbar = 1,  n = 0, x0 = 3, w = 1):
    x = (xx-x0)
    xi = np.sqrt(m*w/hbar)*x
    prefactor = 1./math.sqrt(2.**n * math.factorial(n)) * (m*w/(np.pi*hbar))**(0.25)
    psi = prefactor * np.exp(- xi**2 / 2) * hermite(x,n)
    psi = psi * np.exp(-1j*hbar*w*t)
    return normalise(psi)

  
def oscillatorCoherentState1D(xx, dx, t, m = 1, hbar = 1, x0 = 7, alpha = 1, w = 1):
    alpha = alpha * np.exp(-1j*w*t)
    
    #a1 = np.real(alpha)
    a2 = np.imag(alpha)
    psi = np.exp(-a2**2) * (m * w / np.pi / hbar)**(0.25) * np.exp(-0.5 * m * w / hbar * (xx - x0 - np.sqrt(2) * alpha)**2)
    return normalise(psi)

def oscillatorPotential1D(xx, m, x0 = 7, w = 1):
    return m * w/2 * (xx - x0)**2

def infiniteWell1D(xx, dx, t, m = 1, hbar = 1,  n = 0, x0 = 0, L = 1, xl=0, xr=1, w = 1):
    kn = (n+1) * np.pi / L

    if n % 2 == 0:
        psi = np.sin(kn * (xx - x0))
    else:
        psi = np.cos(kn * (xx - x0))

    psi[xx < xl] = 0
    psi[xx > xr] = 0

    return normalise(psi)

def wellPotential1D(xx, m, xl = 0, xr = 1, V = 1e10):
    return (xx < xl) * V + (xx > xr) * V

def cosmological1D(x, dx, t, m = 1, hbar = 1, Lx=1, N=3, eps = 5e-3):
    psi = 1 + 0j
    for n in range(N):
        kx = 2 * np.pi * (1 + n) / Lx
        psi += eps * coefficients_1d[0, n] * np.cos(kx * x) * np.exp(-1j * t * hbar/m * kx ** 2 / 2)
        psi += eps * coefficients_1d[1, n] * np.sin(kx * x) * np.exp(-1j * t * hbar/m * kx ** 2 / 2)

    return normalise(psi)

def cosmological2D(x, y, dx, t, m = 1, hbar = 1, Lx=25, Ly=25, N=10, eps=5e-3):
    psi = np.ones(x.shape, dtype=np.complex)
    for n in range(N):
        for m in range(N):
            kx = 2 * np.pi * (1 + n) / Lx
            ky = 2 * np.pi * (1 + m) / Ly
            psi += (
                eps
                * coefficients_2d[0, n, m]
                * np.cos(kx * x)
                * np.cos(ky * y)
                * np.exp(-1j * t * (kx ** 2 + ky ** 2) / 2)
            )
            psi += (
                eps
                * coefficients_2d[1, n, m]
                * np.sin(kx * x)
                * np.cos(ky * y)
                * np.exp(-1j * t * (kx ** 2 + ky ** 2) / 2)
            )
            psi += (
                eps
                * coefficients_2d[2, n, m]
                * np.cos(kx * x)
                * np.sin(ky * y)
                * np.exp(-1j * t * (kx ** 2 + ky ** 2) / 2)
            )
            psi += (
                eps
                * coefficients_2d[3, n, m]
                * np.sin(kx * x)
                * np.sin(ky * y)
                * np.exp(-1j * t * (kx ** 2 + ky ** 2) / 2)
            )

    return normalise(psi)

# Generate analytical solution for the 1D free Schrödinger equation
# Li test 1
def generate2DGaussian(x0, y0, xx, yy, t, m = 1, hbar = 1,  alpha=1.0 / 10):
    psi = np.sqrt(1 / (alpha + 1.0j * t)) * np.exp(
        -((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * (alpha + 1.0j * t))
    )
    return normalise(psi)

def periodic2DGaussian(xx, yy, dx, t, m = 1, hbar = 1,  x0=5, y0=5, alpha=1.0 / 10, eps=2, L=10, N=10):
    psi = eps + 0j
    for i in range(-N, N + 1):
        for j in range(-N, N + 1):
            psi += generate2DGaussian(x0, y0, xx + L * i, yy + L * j, t, alpha)
    return normalise(psi)

def twoWavePackets2D(xx, yy, dx, t, m = 1, hbar = 1,  alpha = 1/10, eps = 2):
    psi = eps + 0j
    for i in range(-N, N + 1):
        for j in range(-N, N + 1):
            psi += generate2DGaussian(4, 5, xx + L * i, yy + L * j, t, alpha)
            psi += generate2DGaussian(6, 5, xx + L * i, yy + L * j, t, alpha)
    return normalise(psi)

  
#Generate analytical solution for the 1D free Schrödinger equation
#Li test 1
def generate3DGaussian(x0, y0, z0, x, y, z, t = 0, alpha = 1./10, eps = 10):
  psi = np.sqrt(1/(alpha + 1.j*t))*np.exp(-((x-x0)**2+ (y-y0)**2+ (z-z0)**2)/(2*(alpha + 1.j*t)))
  return normalise(psi)

def generateLiTest13D(x, y, z, dx, t, m = 1, hbar = 1, x0 = 5, alpha = 1./50, eps = 10):
  psi = eps
  psi += generate3DGaussian(x0, x0, x0, x, y, z, t, alpha, eps)
  return normalise(psi)


def twoWavePackets3D(x, y, z, dx, t, m = 1, hbar = 1, x0 = 3, x1 = 7, alpha = 1./2, eps = 10):
  psi = eps
  psi += 10*generate3DGaussian(x0, x0, x0, x, y, z, t, alpha, eps)
  psi += 10*generate3DGaussian(x1, x1, x1, x, y, z, t, alpha, eps)
  return normalise(psi)



@njit
def cosmological3D(x, y, z, dx, t, m = 1, hbar = 1, Lx=1, Ly=1, Lz=1, N=10, eps = 5e-3):
    psi = np.ones(x.shape, dtype=np.complex)
    for n in range(N):
        for m in range(N):
          for l in range(N):
            kx = 2 * np.pi * (1 + n) / Lx
            ky = 2 * np.pi * (1 + m) / Ly
            kz = 2 * np.pi * (1 + l) / Lz
            d1 = np.cos(kx * x)
            d2 = np.cos(ky * y)
            d3 = np.cos(kz * z)
            d4 = np.sin(kx * x)
            d5 = np.sin(ky * y)
            d6 = np.sin(kz * z)

            dpsi  = eps * (
                    coefficients_3d[0, n, m, l] * d1 * d2 * d3  \
                  + coefficients_3d[1, n, m, l] * d4 * d2 * d3  \
                  + coefficients_3d[2, n, m, l] * d1 * d5 * d3  \
                  + coefficients_3d[3, n, m, l] * d1 * d2 * d6  \
                  + coefficients_3d[4, n, m, l] * d4 * d5 * d3  \
                  + coefficients_3d[5, n, m, l] * d1 * d5 * d6  \
                  + coefficients_3d[6, n, m, l] * d4 * d2 * d6  \
                  + coefficients_3d[7, n, m, l] * d4 * d5 * d6
            )

            dpsi = dpsi * np.exp(-1j * t * (kx ** 2 + ky ** 2 + kz ** 2) / 2)
            psi  = psi + dpsi

    return normalise(psi)