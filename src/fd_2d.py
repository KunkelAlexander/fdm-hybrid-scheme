import numpy as np 
import scipy.linalg
import scipy.sparse

import src.fd as fd


#f_i - f_i-1
def getBackwardGradient(f, dx):
  # directions for np.roll() 
  R = -1   # right
  L = 1    # left
  f_dx = (f - np.roll(f, L, axis = 0))/dx
  f_dy = (f - np.roll(f, L, axis = 1))/dx
  return f_dx, f_dy

#f_i+1 - f_i
def getForwardGradient(f, dx):
  # directions for np.roll() 
  R = -1   # right
  L = 1    # left
  f_dx = (np.roll(f, R, axis = 0) - f)/dx
  f_dy = (np.roll(f, R, axis = 1) - f)/dx
  return f_dx, f_dy

#f_i - f_i-1
def getB2Gradient(f, dx):
  # directions for np.roll() 
  R = -1   # right
  L = 1    # left
  f_dx = (3*f - 4*np.roll(f, L, axis=0) + np.roll(f, 2*L, axis=0)) / (2*dx)
  f_dy = (3*f - 4*np.roll(f, L, axis=1) + np.roll(f, 2*L, axis=1)) / (2*dx)
  return f_dx, f_dy

#f_i+1 - f_i
def getF2Gradient(f, dx):
  # directions for np.roll() 
  R = -1   # right
  L = 1    # left
  f_dx = (-np.roll(f, 2*R, axis=0) + 4*np.roll(f, R, axis=0) - 3*f) / (2*dx)
  f_dy = (-np.roll(f, 2*R, axis=1) + 4*np.roll(f, R, axis=1) - 3*f) / (2*dx)
  return f_dx, f_dy


#f_i+1 - f_i-1
def getCenteredGradient(f, dx):
  # directions for np.roll() 
  R = -1   # right
  L = 1    # left
  f_dx = (np.roll(f, R, axis=0) - np.roll(f, L, axis=0)) / (2*dx)
  f_dy = (np.roll(f, R, axis=1) - np.roll(f, L, axis=1)) / (2*dx)
  return f_dx, f_dy


def getC2Gradient(f, dx):
  # directions for np.roll() 
  R = -1   # right
  L = 1    # left
  f_dx = (1/12*np.roll(f, 2*L, axis=0) - 2/3*np.roll(f, L, axis=0) + 2/3*np.roll(f, R, axis=0) - 1/12*np.roll(f, 2*R, axis=0))/dx
  f_dy = (1/12*np.roll(f, 2*L, axis=1) - 2/3*np.roll(f, L, axis=1) + 2/3*np.roll(f, R, axis=1) - 1/12*np.roll(f, 2*R, axis=1))/dx
  return f_dx, f_dy


def getCenteredLaplacian(f, dx):
  R = -1   # right
  L = 1    # left
  result = (np.roll(f,R,axis=0) + np.roll(f,L,axis=0) + np.roll(f,R,axis=1) +  np.roll(f,L,axis=1) - 4*f) / (dx**2)
  return result


def getC2Laplacian(f, dx):
	R = -1   # right
	L = 1    # left
	result  = -1/12 * np.roll(f, 2*L, axis=0) + 4/3 * np.roll(f, L, axis=0) - 5/2 * f + 4/3 * np.roll(f, R, axis=0) - 1/12*np.roll(f, 2*R, axis=0)
	result += -1/12 * np.roll(f, 2*L, axis=1) + 4/3 * np.roll(f, L, axis=1) - 5/2 * f + 4/3 * np.roll(f, R, axis=1) - 1/12*np.roll(f, 2*R, axis=1)
	return result/dx**2

def getCenteredQuantumPressure(rho, dx):
  logrho = np.log(rho)
  f_dx, f_dy = getCenteredGradient(logrho, dx)
  result = 0.5*getCenteredLaplacian(logrho, dx)+0.25*(f_dx**2 + f_dy**2)
  return -0.5*result

def getC2QuantumPressure(rho, dx):
  logrho = np.log(rho)
  f_dx, f_dy = getC2Gradient(logrho, dx)
  result = 0.5*getC2Laplacian(logrho, dx)+0.25*(f_dx**2 + f_dy**2)
  return -0.5*result
  

#Solve diffusion operator using Cranck Nicolson scheme with Dirichlet boundary conditions provided by leb (left boundary), lob (lower boundary), upb (upper boundary) and rib (right boundary)
def solveDirichletCNDiffusion(psi, leb, upb, lob, rib, A_row, b_row, A_col, b_col):
  psi[0 ,  :]  = np.copy(upb)
  psi[-1,  :]  = np.copy(lob)
  psi[: ,  0]  = np.copy(leb)
  psi[: , -1]  = np.copy(rib)

  for row in range(1, psi.shape[0] - 1):
    RHS         = b_row.dot(psi[row, :])
    RHS[ 0]     = leb[row]
    RHS[-1]     = rib[row]
    psi[row, :] = scipy.sparse.linalg.spsolve(A_row, RHS)

  for col in range(1, psi.shape[1] - 1):
    RHS         = b_col.dot(psi[:, col])
    RHS[ 0]     = upb[col]
    RHS[-1]     = lob[col]
    psi[:, col] = scipy.sparse.linalg.spsolve(A_col, RHS)

  return psi

#Solve diffusion operator using Cranck Nicolson scheme using periodic boundary conditions
def solvePeriodicCNDiffusion(psi, A_row, b_row, A_col, b_col):
  for row in range(psi.shape[0]):
    RHS         = b_row.dot(psi[row, :])
    psi[row, :] = scipy.sparse.linalg.spsolve(A_row, RHS)

  for col in range(psi.shape[1]):
    RHS         = b_col.dot(psi[:, col])
    psi[:, col] = scipy.sparse.linalg.spsolve(A_col, RHS)

  #print("Left and right column", np.abs(psi[:, 0] - psi[:, -1]))
  return psi

#Solve diffusion operator using forward in time, centered in space, 1st order explicit method
def solveDirichletFTCSDiffusion(psi, leb, upb, lob, rib, dt, dx):
  s1 = 1j*dt/(2*dx**2)
  
  psi[0 ,  :]  = upb.copy()
  psi[-1,  :]  = lob.copy()
  psi[: ,  0]  = leb.copy()
  psi[: , -1]  = rib.copy()

  #Create five-point stencil
  ce = psi
  up = np.roll(psi,  1, axis = 0)
  do = np.roll(psi, -1, axis = 0)
  le = np.roll(psi,  1, axis = 1)
  ri = np.roll(psi, -1, axis = 1)

  psi = psi  + s1 * (up + do + le + ri - 4 * ce)

  psi[0 ,  :]  = upb.copy()
  psi[-1,  :]  = lob.copy()
  psi[: ,  0]  = leb.copy()
  psi[: , -1]  = rib.copy()
  
  return psi

#Solve diffusion operator using forward in time, centered in space, 1st order explicit method
def solveHOPeriodicFTCSDiffusion(psi, dt, dx, stencil, coeff):
  s1 = 1j*dt/2

  for i in range(psi.ndim):
      psi  += s1 * fd.getDerivative(psi, dx, stencil, coeff, axis = i, derivative_order = 2)

  return psi

# Five-point stencil matrix for laplace operator used in calculating gravitational potential
def createPotentialLaplacian(nx, dx):
  # Set up tridiagonal coefficients
  C = np.zeros((3, nx))
  C[0, :] =  1/dx**2
  C[1, :] = -4/dx**2
  C[2, :] =  1/dx**2

  #Dirichlet boundary conditions
  C[1, 0]  = 1
  C[1,-1]  = 1
  C[0, 1]  = 0
  C[2,-2]  = 0

  # Construct tridiagonal matrix
  return C

#Iterative algorithm to solver 2D poisson equation, converges very slowly
def computeDirichletPotential(V, rho, G, leb, upb, lob, rib, dx, D2_row, D2_col, N_iterations = 20):
  V[0 ,  :]  = upb.copy()
  V[-1,  :]  = lob.copy()
  V[: ,  0]  = leb.copy()
  V[: , -1]  = rib.copy()

  for i in range(N_iterations):
    #Iterate over rows
    for row in range(1, rho.shape[0] - 1):
      u     = 4*np.pi*G*(rho[row, :] - 1) - (V[row - 1, :] + V[row + 1, :])/dx**2
      u[0]  = leb[row]
      u[-1] = rib[row]
      V[row, :] = scipy.linalg.solve_banded((1,1), D2_row, u)

    for col in range(1, rho.shape[1] - 1):
      u     = 4*np.pi*G*(rho[:, col] - 1) - (V[:, col - 1] + V[:, col + 1])/dx**2
      u[0]  = upb[col]
      u[-1] = lob[col]
      V[:, col] = scipy.linalg.solve_banded((1,1), D2_col, u)
  return V

