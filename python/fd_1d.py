import numpy as np 
import fd

from numba import njit 
import scipy.linalg
import scipy.sparse.linalg
import scipy.sparse


''' SIMPLE FIRST AND SECOND ORDER FD '''


@njit
def getBackwardGradient(f, dx):
    f_dx = (f - np.roll(f, fd.ROLL_L))/dx
    return f_dx

@njit
def getForwardGradient(f, dx):
    f_dx = (np.roll(f, fd.ROLL_R) - f)/dx
    return f_dx

@njit
def getB2Gradient(f, dx):
    f_dx = (3*f - 4*np.roll(f, fd.ROLL_L) + np.roll(f, 2*fd.ROLL_L)) / (2*dx)
    return f_dx

@njit
def getF2Gradient(f, dx):
    f_dx = (-np.roll(f, 2*fd.ROLL_R) + 4*np.roll(f, fd.ROLL_R) - 3*f) / (2*dx)
    return f_dx

@njit
def getPeriodicForwardGradient(f, dx, N):
    p = np.sin(f/N)
    q = np.cos(f/N)
    q_dx = (np.roll(q, fd.ROLL_R) - q)/dx
    p_dx = (np.roll(p, fd.ROLL_R) - p)/dx
    return (q*p_dx - p*q_dx)*N

@njit
def getPeriodicBackwardGradient(f, dx, N):
    p = np.sin(f/N)
    q = np.cos(f/N)
    q_dx = (q - np.roll(q, fd.ROLL_L))/dx
    p_dx = (p - np.roll(p, fd.ROLL_L))/dx
    return (q*p_dx - p*q_dx)*N

@njit
def getPeriodicF2Gradient(f, dx, N):
    p = np.sin(f/N)
    q = np.cos(f/N)
    q_dx = (-np.roll(q, 2*fd.ROLL_R) + 4*np.roll(q, fd.ROLL_R) - 3*q) / (2*dx)
    p_dx = (-np.roll(p, 2*fd.ROLL_R) + 4*np.roll(p, fd.ROLL_R) - 3*p) / (2*dx)
    return (q*p_dx - p*q_dx)*N

@njit
def getPeriodicB2Gradient(f, dx, N):
    p = np.sin(f/N)
    q = np.cos(f/N)
    q_dx = (3*q - 4*np.roll(q, fd.ROLL_L) + np.roll(q, 2*fd.ROLL_L)) / (2*dx)
    p_dx = (3*p - 4*np.roll(p, fd.ROLL_L) + np.roll(p, 2*fd.ROLL_L)) / (2*dx)
    return (q*p_dx - p*q_dx)*N

@njit
def getCenteredGradient(f, dx):
    # f_dx = 1/12*np.roll(f, 2*fd.ROLL_L, axis=0)-2/3*np.roll(f, fd.ROLL_L, axis=0) + 2/3*np.roll(f, fd.ROLL_R, axis=0) - 1/12*np.roll(f, 2*fd.ROLL_R, axis=0)#(np.roll(f, fd.ROLL_R, axis=0) - np.roll(f, fd.ROLL_L, axis=0)) / (2*dx)
    return (np.roll(f, fd.ROLL_R) - np.roll(f, fd.ROLL_L)) / (2*dx)  # f_dx/dx

@njit
def getC2Gradient(f, dx):
    f_dx = 1/12*np.roll(f, 2*fd.ROLL_L) - 2/3*np.roll(f, fd.ROLL_L) + 2 / \
        3*np.roll(f, fd.ROLL_R) - 1/12*np.roll(f, 2*fd.ROLL_R)
    return f_dx/dx

@njit
def getCenteredAverage(f):
    avg = (np.roll(f, fd.ROLL_L) + np.roll(f, fd.ROLL_R)) / 2
    return avg


@njit
def getCenteredLaplacian(f, dx):
    return (np.roll(f, fd.ROLL_R) + np.roll(f, fd.ROLL_L) - 2*f) / (dx**2)

@njit
def getC2Laplacian(f, dx):
    result = -1/12 * np.roll(f, 2*fd.ROLL_L) + 4/3 * np.roll(f, fd.ROLL_L) - \
        5/2 * f + 4/3 * np.roll(f, fd.ROLL_R) - 1/12*np.roll(f, 2*fd.ROLL_R)
    return result/dx**2

@njit
def getCenteredQuantumPressure(rho, dx):
  logrho = np.log(rho)
  f_dx = getCenteredGradient(logrho, dx)
  result = 0.5*getCenteredLaplacian(logrho, dx)+0.25*(f_dx**2)
  return -0.5*result

def createKineticLaplacian(nx, dt, dx):
    # Stencil for laplace operator of kinetic term
    # Set up tridiagonal coefficients
    Adiag = np.empty(nx, dtype=np.complex128)
    Asup = np.empty(nx, dtype=np.complex128)
    Asub = np.empty(nx, dtype=np.complex128)
    bdiag = np.empty(nx, dtype=np.complex128)
    bsup = np.empty(nx, dtype=np.complex128)
    bsub = np.empty(nx, dtype=np.complex128)
    Adiag.fill(1 + 1j*dt/dx**2 * 1/2)
    Asup.fill(-1j*dt/(2*dx**2) * 1/2)
    Asub.fill(-1j*dt/(2*dx**2) * 1/2)
    bdiag.fill(1 - 1j*dt/dx**2 * 1/2)
    bsup.fill(1j*dt/(2*dx**2) * 1/2)
    bsub.fill(1j*dt/(2*dx**2) * 1/2)

    # Dirichlet boundary conditions
    Adiag[0] = 1
    Adiag[-1] = 1
    Asup[1] = 0
    Asub[-2] = 0

    # Construct tridiagonal matrix
    A = scipy.sparse.spdiags([Adiag, Asup, Asub], [
                             0, 1, -1], nx, nx, format='csr')
    b = scipy.sparse.spdiags([bdiag, bsup, bsub], [
                             0, 1, -1], nx, nx, format='csr')
    return A, b

def createPotentialLaplacian(nx, dx):
    # Stencil for laplace operator used in calculating gravitational potential
    # Set up tridiagonal coefficients
    Cdiag = np.empty(nx)
    Csup = np.empty(nx)
    Csub = np.empty(nx)
    Cdiag.fill(-2/dx**2)
    Csup.fill(1/dx**2)
    Csub.fill(1/dx**2)

    # Dirichlet boundary conditions
    Cdiag[0] = 1
    Cdiag[-1] = 1
    Csup[1] = 0
    Csub[-2] = 0

    # Construct tridiagonal matrix
    return scipy.sparse.spdiags([Cdiag, Csup, Csub], [0, 1, -1], nx, nx, format='csr')


#Solve diffusion operator using Cranck Nicolson scheme with Dirichlet boundary conditions provided by leb (left boundary), lob (lower boundary), upb (upper boundary) and rib (right boundary)
def solveDirichletCNDiffusion(psi, leb, rib, A, b):
    rhs     = b.dot(psi)
    rhs[0]  = leb
    rhs[-1] = rib

    psi = scipy.sparse.linalg.spsolve(A, rhs)
    return psi 


#Solve diffusion operator using Cranck Nicolson scheme using periodic boundary conditions
def solvePeriodicCNDiffusion(psi, A, b):
    rhs = b.dot(psi)
    psi = scipy.sparse.linalg.spsolve(A, rhs)
    return psi

#Solve diffusion operator using forward in time, centered in space, 1st order explicit method
def solveDirichletFTCSDiffusion(psi, leb, rib, dt, dx):
  s1 = 1j*dt/(2*dx**2)
  
  psi[0]  = leb
  psi[-1] = rib

  #Create five-point stencil
  c = psi
  left = np.roll(psi,  1, axis = 0)
  right = np.roll(psi, -1, axis = 0)

  psi = psi  + s1 * (left + right - 2 * c)

  psi[0]  = leb
  psi[-1] = rib
  
  return psi


  # Solve Laplace x = f with Dirichlet boundary conditions x_0 = alpha, x_1 = beta
  # f is vector with number of cells - 1 components
def computeDirichletPotential(rho, leb, rib, G, D):
    #Potential
    f       = -4.0*np.pi*G*(rho - 1.0)
    #Boundary terms
    f[0]    = leb
    f[-1]   = rib

    return scipy.sparse.linalg.spsolve(D, f) 