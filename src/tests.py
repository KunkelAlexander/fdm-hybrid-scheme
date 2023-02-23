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
    N = 10 
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


def cosmological1D(x, dx, t, m = 1, hbar = 1, Lx=1, N=10, eps = 5e-3):
    return cosmological3D(x, 0, 0, dx, t, m, hbar, Lx, Lx, Lx, N, eps)


def cosmological2D(x, y, dx, t, m = 1, hbar = 1, Lx=1, Ly=1, N=10, eps = 5e-3):
    return cosmological3D(x, y, 0, dx, t, m, hbar, Lx, Ly, Lx, N, eps)


# Calculate the gradient by Richardson extrapolation (with periodic boundary condition)
def GRAD(field, axis, h, order):
    dim = list(field.shape)
    dim.insert(0, order)
    grad = np.zeros(tuple(dim))
    for o in range(order):
        interval = 2**(order-1-o)
        grad[o] = (np.roll(field, -interval, axis=axis) - np.roll(field, interval, axis=axis)) / (2*interval)
    for o in range(1,order):
        grad[o:] = (4.**o*grad[o:]-grad[o-1:-1]) / (4.**o-1.)
    return grad[-1]/h
    
# Physical Constant
phidm_mass = 2e-23 #eV
h_bar = 1.0545718e-34
eV_to_kg = 1.78266192162790e-36
Mpc_to_m = 3.0856776e22
#####################################################################################################################


def antisymmetricMode(x, y, dx, t, m = 1, hbar = 1, L = 1, N=10, eps = 0.2):

    H0 = 100
    box_length = L
    factor = box_length*1000.
    N = 2/dx
    h = 1./N
    k_factor = 2.*np.pi/N
    box_length *= Mpc_to_m/(H0/100.) # meter
    a = 1/t**2
    a_c = 0.13
    delta = 2 * a**(3/2) / a_c 
    delta_dot = -3 * a**(1/2) / a_c * (-2) * t**(3)
    rho_bar = 1 
    kp = 2*np.pi/L
    ka = 4 * np.pi /L
    chi = x + eps / ( 2 * np.pi * ka ) * kp / ka * np.cos ( 2 * np.pi * kp * eps * Cx * Sy**2 )
    Cy = np.cos(2 * np.pi * ka * y) 
    Cx = np.cos(2 * np.pi * kp * chi) 
    Sy = np.sin(2 * np.pi * ka * y) 
    Sx = np.sin(2 * np.pi * kp * chi)
    vx = + delta_dot / (2 * np.pi * kp) * Sx
    vy = - delta_dot / (2 * np.pi * ka) * Sx * Sy * eps
    rho = rho_bar / ( 1 + delta * Cx - delta * eps * ( Sx * Cy - 2 * np.pi * kp * eps * Cx * Sy**2))

    # Calculate div(v)
    vx_x = GRAD(vx, axis = 0, h=dx ,order=3)
    vy_y = GRAD(vy, axis = 1, h=dx ,order=3)
    v_div = vx_x + vy_y
    v_div *= a*phidm_mass*eV_to_kg/h_bar
    
    # Do forward DFT
    v_div_k = np.fft.rfftn(v_div)
    # Do inverse Laplacian
    kx, ky = np.arange(N), np.arange(N//2+1.)
    kxx, kyy = np.meshgrid(kx, ky)
    v_div_k /= 2.*(np.cos(k_factor*kxx)+np.cos(k_factor*kyy)-2.)
    v_div_k[0,0,0] = 0.
    # Do inverse DFT
    phi_fft = np.fft.irfftn(v_div_k)
    # Rescale to correct unit
    phi_fft *= box_length/N**2
    phi_fft -= phi_fft.min()
  
    return np.sqrt(rho) * np.exp(1 * phi_fft)

