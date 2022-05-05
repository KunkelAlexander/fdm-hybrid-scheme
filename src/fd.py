import numpy as np
import scipy.sparse
import math 
from   numba import njit 
import matplotlib.pyplot as plt  
import scipy

import sys

"""""" """""" """""" """""" """"""
"""    GLOBAL CONSTANTS    """
"""""" """""" """""" """""" """"""

# directions for np.roll()
ROLL_R = -1  # right
ROLL_L = 1  # left


# options for finite difference functions
MODE_FORWARD = 0
MODE_CENTERED = 1
MODE_BACKWARD = 2
MODE_CUSTOM = 3


""" CONTINUITY """

def make_1d_continuous(f):
    for i in range(len(f) - 1):
        while (f[i] - f[i + 1]) > np.pi:
            f[i + 1 :] += 2 * np.pi
        while (f[i] - f[i + 1]) < -np.pi:
            f[i + 1 :] -= 2 * np.pi
    return f


def make_2d_continuous(f):
    for i in range(f.shape[0]):
        make_1d_continuous(f[i, :])
    for i in range(f.shape[1]):
        make_1d_continuous(f[:, i])
    return f


def make_3d_continuous(f):
    for i in range(f.shape[0]):
        make_2d_continuous(f[i, :, :])
    for i in range(f.shape[1]):
        make_2d_continuous(f[:, i, :])
    for i in range(f.shape[2]):
        make_2d_continuous(f[:, :, i])
    return f


def make_continuous(f):
    if f.ndim == 1:
        return make_1d_continuous(f)
    elif f.ndim == 2:
        return make_2d_continuous(f)
    elif f.ndim == 3:
        return make_3d_continuous(f)
    else:
        raise ValueError()


def make_boundary_continuous(f, boundaryThickness):
    if (np.array(f.shape) < boundaryThickness).any():
        raise ValueError("Boundary thicker than f itself")

    if f.ndim == 1:
        return make_1d_boundary_continuous(f, boundaryThickness)
    elif f.ndim == 2:
        return make_2d_boundary_continuous(f, boundaryThickness)
    elif f.ndim == 3:
        return make_3d_boundary_continuous(f, boundaryThickness)
    else:
        raise ValueError()


@njit
def make_1d_boundary_continuous(f, boundaryThickness):
    make_1d_continuous(f[:boundaryThickness])
    make_1d_continuous(f[-boundaryThickness:])
    return f


#def make_2d_boundary_continuous(f, boundaryThickness):
#    Nx, Ny = np.array(f.shape) - 1
#    bt = boundaryThickness
#    
#    for i in range(bt):
#        #Continuity along faces (4 * faces)
#        make_1d_continuous(f[i   , :   ])
#        make_1d_continuous(f[Nx-i, :   ])
#        make_1d_continuous(f[:   , i   ])
#        make_1d_continuous(f[:   , Ny-i])
#
#        #Continuity perpendicular to faces (4 * faces)
#        #make_1d_continuous(f[i          , :bt])
#        #make_1d_continuous(f[i          , Ny - bt: Ny])
#        #make_1d_continuous(f[:bt        , i   ])
#        #make_1d_continuous(f[Nx-bt:Nx   , i])
#
#    return f

def make_2d_boundary_continuous(f, boundaryThickness=7):
    if (f.shape[0] < boundaryThickness) or (f.shape[1] < boundaryThickness):
        raise ValueError("Boundary thicker than f itself")

    for i in range(boundaryThickness):
        make_1d_continuous(f[i, :])

    for i in range(f.shape[0] - boundaryThickness, f.shape[0]):
        make_1d_continuous(f[i, :])

    for i in range(boundaryThickness):
        make_1d_continuous(f[:, i])

    for i in range(f.shape[1] - boundaryThickness, f.shape[1]):
        make_1d_continuous(f[:, i])

    return f


def getXGradient(f, dx):
	f_dx = ( np.roll(f, ROLL_R,axis=0) - np.roll(f, ROLL_L,axis=0) ) / (2*dx)
	return f_dx

def getYGradient(f, dx):
	f_dy = ( np.roll(f, ROLL_R,axis=1) - np.roll(f, ROLL_L,axis=1) ) / (2*dx)
	return f_dy


# Return array with finite difference coefficients for approximations of order (stencil_length - derivative_order)
def getFiniteDifferenceCoefficients(
    derivative_order, accuracy, mode, stencil=None, debug=False
):
    stencil_length = derivative_order + accuracy

    if mode == MODE_FORWARD:
        stencil = np.array(range(0, stencil_length), dtype=int)
    elif mode == MODE_BACKWARD:
        stencil = np.array(range(-stencil_length + 1, 1), dtype=int)
    elif mode == MODE_CENTERED:
        if accuracy % 2 != 0:
            raise ValueError(
                "Centered stencils only available with even accuracy orders"
            )
        if (stencil_length % 2 == 0) and stencil_length >= 4:
            stencil_length -= 1
        half_stencil_length = int((stencil_length - 1) / 2)
        stencil = np.array(
            range(-half_stencil_length, half_stencil_length + 1), dtype=int
        )
    elif mode == MODE_CUSTOM:
        if stencil is None:
            raise ValueError("Need to provide custom stencil in MODE_CUSTOM")
        stencil_length = len(stencil)
        if derivative_order >= stencil_length:
            raise ValueError("Derivative order must be smaller than stencil length")

    A = np.zeros((stencil_length, stencil_length))
    b = np.zeros(stencil_length)

    for i in range(stencil_length):
        A[i, :] = stencil ** i

    b[derivative_order] = math.factorial(derivative_order)

    if debug:
        print("A", A)
        print("b", b)

    coefficients = np.linalg.solve(A, b)
    return stencil, coefficients


def getDerivative(f, dx, stencil, coeff, axis, derivative_order=1, debug=False):
    # directions for np.roll()
    f_dx = np.zeros(f.shape, dtype=f.dtype)
    for i, shift in enumerate(stencil):
        if debug:
            print(
                "Derivative order",
                derivative_order,
                "Order = ",
                len(stencil) - 1,
                "shift ",
                shift,
                " coefficient = ",
                coeff[i],
            )
        f_dx += np.roll(f, shift * ROLL_R, axis=axis) * coeff[i]
    return f_dx / dx ** derivative_order





def getHOSqrtQuantumPressure(rho, dx, eta, c1_stencil, c1_coeff, c2_stencil, c2_coeff):
    rho[rho<0] = 1e-8
    logrho = np.log(rho)
    if 0:#np.isnan(logrho).any():
        print("!NAN in QP")
        plt.title("NAN in QP")
        plt.imshow(logrho)
        plt.colorbar()
        plt.show()
        np.save("logrho.npy", logrho)
        np.save("rho.npy", rho)
        #sys.exit("Study this")

    result = np.zeros(rho.shape)

    for i in range(rho.ndim):
        f_di = getDerivative(logrho, dx, c1_stencil, c1_coeff, axis=i, derivative_order=1)
        f_ddi = getDerivative(logrho, dx, c2_stencil, c2_coeff, axis=i, derivative_order=2)

        result += -1 / 4 * f_ddi - 1 / 8 * f_di ** 2

    return result * eta ** 2


def getC4QuantumPressure(rho, dx, c1_stencil, c1_coeff):
    logrho = np.log(rho)

    result = np.zeros(rho.shape)

    for i in range(rho.ndim):
        f_di = getDerivative(logrho, dx, c1_stencil, c1_coeff, axis=i, derivative_order=1)
        result -= 1 / 8 * f_di ** 2

    L = 1
    R = -1
    lr = np.roll(logrho, L, axis=1)
    ulr = np.roll(lr, L, axis=0)
    llr = np.roll(lr, R, axis=0)
    rr = np.roll(logrho, R, axis=1)
    urr = np.roll(rr, L, axis=0)
    lrr = np.roll(rr, R, axis=0)
    ur = np.roll(logrho, L, axis=0)
    dr = np.roll(logrho, R, axis=0)
    lap = (-20 * logrho + 4 * (lr + rr + ur + dr) + 1 * (ulr + llr + urr + lrr)) / (
        6 * dx ** 2
    )

    result -= 1 / 4 * lap

    return result


def computePotential(rho, G, kSq, workers = None):
  Vhat = -scipy.fft.fftn(4.0*np.pi*G*(rho - 1.), workers=workers) / ( kSq  + (kSq==0))
  V = np.real(scipy.fft.ifftn(Vhat, workers=workers))
  return V


# Solve Laplace x = f with Dirichlet boundary conditions x_0 = alpha, x_1 = beta
# f is vector with number of cells - 1 components
def computeDirichletPotential(rho, V0, V1, G, D2):
    u = 4.0 * np.pi * G * (rho - 1.0)
    u[0] = V0
    u[-1] = V1
    # print(f"Computing the Dirichlet potential with u 0 to 5 {u[0:5]} with N = {len(u)}")
    V2 = scipy.sparse.linalg.spsolve(D2, u)
    return V2




# Solve diffusion operator using forward in time, centered in space, 1st order explicit method
def solvePeriodicFTCSDiffusion(psi, dt, dx, coeff, stencil):
    dpsi = np.zeros(psi.shape, dtype=complex)

    # Create five-point stencil
    for i in range(psi.ndim):
        dpsi += 1j * dt/2 * getDerivative(psi, dx, stencil, coeff, axis = i, derivative_order=2)

    return dpsi




def deltaSi(density, phase, dx):
    sr = 0.5 * np.log(density)
    if np.isnan(sr).any():
        print("NAN in sr with shape ", sr.shape)
        plt.title("NANI")
        plt.imshow(density)
        plt.colorbar()
        plt.show()

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
    return dsi


def getLaxFriedrichsFlux(rho_a, rho_b, v_a, v_b, dx, alpha):
    #a = np.maximum(np.abs(v_a), np.abs(v_b)) + alpha/dx
    result = rho_a * v_a + rho_b * v_b - alpha * (rho_b - rho_a)
    return 0.5 * result


#Osher-Sethian flux for H = v^2 for updating phase
def getOsherSethianFlux(up, um):
    osf = (np.minimum(up, 0)**2 + np.maximum(um, 0)**2)/2
    return osf

def getModifiedOsherSethianFlux(up, um, vp, vm):
    osf = (np.minimum(up*vp, 0) + np.maximum(vm*um, 0))
    return osf

def getPhaseLaxFriedrichsFlux(up, um, dx, alpha):
    h = ((up + um)/2)**2/2
    h -= 1/2 * alpha/dx * (up - um)
    return h


def getBackwardGradient(f, dx, axis):
    f_dx = (f - np.roll(f, ROLL_L, axis=axis))/dx
    return f_dx

def getForwardGradient(f, dx, axis):
    f_dx = (np.roll(f, ROLL_R, axis=axis) - f)/dx
    return f_dx

def getB2Gradient(f, dx, axis):
  f_dx = (3*f - 4*np.roll(f, ROLL_L, axis=axis) + np.roll(f, 2*ROLL_L, axis=axis)) / (2*dx)
  return f_dx

def getF2Gradient(f, dx, axis):
  f_dx = (-np.roll(f, 2*ROLL_R, axis=axis) + 4*np.roll(f, ROLL_R, axis=axis) - 3*f) / (2*dx)
  return f_dx


def getCenteredGradient(f, dx, axis):
    f_dx = (np.roll(f, ROLL_R, axis=axis) - np.roll(f, ROLL_L, axis=axis) )/(2*dx)
    return f_dx

def getCenteredLaplacian(f, dx, axis):
    f_ddx = (np.roll(f, ROLL_R, axis=axis) - 2 * f + np.roll(f, ROLL_L, axis=axis) )/(dx**2)
    return f_ddx

# eta = hbar / m
def getQuantumPressure(rho, dx, eta):
    logrho = np.log(rho)
    result = np.zeros(rho.shape)
    for i in range(rho.ndim):
        f_di =  getCenteredGradient(logrho, dx, axis=i)
        f_ddi = getCenteredLaplacian(logrho, dx, axis=i)
        result += -1 / 4 * f_ddi - 1 / 8 * f_di ** 2

    return result * eta ** 2


#Upwind fluxes for updating density
def getUpwindFlux(up, um, rho, dx, axis):
  #phi_i + 1/2, j
  phi_u_p = np.maximum(up, 0) * rho                          + np.minimum(up, 0) * np.roll(rho, ROLL_R, axis=axis)
  #phi_i - 1/2, j
  phi_u_m = np.maximum(um, 0) * np.roll(rho, ROLL_L, axis=axis) + np.minimum(um, 0) * rho
  
  phi =  (phi_u_p - phi_u_m) / dx
  return phi


#Create stencil matrices for laplace operator of kinetic term in 2nd order Cranck Nicolson scheme
def createKineticLaplacian(nx, dt, dx, debug = False, periodicBC = False):
  # Set up tridiagonal coefficients
  A = np.empty((3, nx), dtype=complex)
  A[1, :] =  1 + 1j*dt/dx**2 * 1/2
  A[0, :] = -1j*dt/(2*dx**2) * 1/2
  A[2, :] = -1j*dt/(2*dx**2) * 1/2

  #Dirichlet boundary conditions
  if not periodicBC:
    A[1, 0]  = 1
    A[1, -1] = 1
    A[0, 1]  = 0
    A[2, -2] = 0

  bdiag = np.empty(nx, dtype=complex)
  bsup  = np.empty(nx, dtype=complex)
  bsub  = np.empty(nx, dtype=complex)
  bdiag.fill(1 - 1j*dt/dx**2 * 1/2)
  bsup.fill(1j * dt/(2*dx**2) * 1/2)
  bsub.fill(1j * dt/(2*dx**2) * 1/2)

  A = scipy.sparse.spdiags([A[1,:], A[0,:], A[2,:]], [0, 1, -1], nx, nx, format = 'csr')
  b = scipy.sparse.spdiags([bdiag, bsup, bsub], [0, 1, -1], nx, nx, format = 'csr')

  if periodicBC:
    Ad = A.todense()
    bd = b.todense()
    Ad[0, -1] = -1j*dt/(2*dx**2) * 1/2
    Ad[-1, 0] = -1j*dt/(2*dx**2) * 1/2
    bd[0, -1] =  1j*dt/(2*dx**2) * 1/2
    bd[-1, 0] =  1j*dt/(2*dx**2) * 1/2
    A = scipy.sparse.csr_matrix(Ad)
    b = scipy.sparse.csr_matrix(bd)

  if debug:
    print("Kinetic Laplacian A", A.todense())
    print("Kinetic Laplacian b", b.todense())

  # Construct tridiagonal matrix
  return A, b


def get_norm(rho, dx):
    # Normalise psi
    psisq_norm = np.sum(rho * dx)
    return np.sqrt(psisq_norm)