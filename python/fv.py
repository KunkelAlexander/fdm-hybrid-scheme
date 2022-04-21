import numpy as np
import fd_2d
import fd 

""""""""""""""""""""""""""""""
"""   FINITE VOLUME        """
""""""""""""""""""""""""""""""

#Note that we want 1/rho grad P in Euler equation, but 
#Quantum pressure is given as off-diagonal element of stress tensor
#1/rho del i P_ij in Euler equation
def get1DPressureTensor(rho, dx):
  logrho = np.log(rho)
  result = -1/4*rho*fd.getCenteredLaplacian(logrho, dx, axis=0)
  return result
  
#Note that we want 1/rho grad P in Euler equation, but 
#Quantum pressure is given as off-diagonal element of stress tensor
#1/rho del i P_ij in Euler equation
def get2DPressureTensor(rho, dx, i, j):
    if i != j:
        rho_x, rho_y   = fd_2d.getC2Gradient(rho, dx)
        return 1/4 * (1/rho * rho_x * rho_y)
    elif i == 0:
        rho_x, rho_y   = fd_2d.getC2Gradient(rho, dx)
        return 1/4 * (1/rho * rho_x**2 - fd_2d.getC2Laplacian(rho, dx))
    else:
        return 1/4 * (1/rho * rho_y**2 - fd_2d.getC2Laplacian(rho, dx))


def getConserved( rho, vx, vy, vol ):
	"""
    Calculate the conserved variable from the primitive
	rho      is matrix of cell densities
	vx       is matrix of cell x-velocity
	vy       is matrix of cell y-velocity
	vol      is cell volume
	Mass     is matrix of mass in cells
	Momx     is matrix of x-momentum in cells
	Momy     is matrix of y-momentum in cells
	"""
	Mass   = rho * vol
	Momx   = rho * vx * vol
	Momy   = rho * vy * vol
	
	return Mass, Momx, Momy


def getPrimitive( Mass, Momx, Momy, vol ):
	"""
    Calculate the primitive variable from the conservative
	Mass     is matrix of mass in cells
	Momx     is matrix of x-momentum in cells
	Momy     is matrix of y-momentum in cells
	vol      is cell volume
	rho      is matrix of cell densities
	vx       is matrix of cell x-velocity
	vy       is matrix of cell y-velocity
	"""
	rho = Mass / vol
	vx  = Momx / rho / vol
	vy  = Momy / rho / vol

	return rho, vx, vy




def slopeLimit(f, dx, f_dx, f_dy):
	"""
    Apply slope limiter to slopes
	f        is a matrix of the field
	dx       is the cell size
	f_dx     is a matrix of derivative of f in the x-direction
	f_dy     is a matrix of derivative of f in the y-direction
	"""
	
	f_dx = np.maximum(0., np.minimum(1., ( (f-np.roll(f, fd.ROLL_L,axis=0))/dx)/(f_dx + 1.0e-8*(f_dx==0)))) * f_dx
	f_dx = np.maximum(0., np.minimum(1., (-(f-np.roll(f, fd.ROLL_R,axis=0))/dx)/(f_dx + 1.0e-8*(f_dx==0)))) * f_dx
	f_dy = np.maximum(0., np.minimum(1., ( (f-np.roll(f, fd.ROLL_L,axis=1))/dx)/(f_dy + 1.0e-8*(f_dy==0)))) * f_dy
	f_dy = np.maximum(0., np.minimum(1., (-(f-np.roll(f, fd.ROLL_R,axis=1))/dx)/(f_dy + 1.0e-8*(f_dy==0)))) * f_dy
	
	return f_dx, f_dy


def extrapolateInSpaceToFace(f, f_dx, f_dy, dx):
	"""
    Calculate the gradients of a field
	f        is a matrix of the field
	f_dx     is a matrix of the field x-derivatives
	f_dy     is a matrix of the field y-derivatives
	dx       is the cell size
	f_XL     is a matrix of spatial-extrapolated values on `left' face along x-axis 
	f_XR     is a matrix of spatial-extrapolated values on `right' face along x-axis 
	f_YR     is a matrix of spatial-extrapolated values on `left' face along y-axis 
	f_YR     is a matrix of spatial-extrapolated values on `right' face along y-axis 
	"""
	f_XL = f - f_dx * dx/2
	f_XL = np.roll(f_XL, fd.ROLL_R,axis=0)
	f_XR = f + f_dx * dx/2
	
	f_YL = f - f_dy * dx/2
	f_YL = np.roll(f_YL, fd.ROLL_R,axis=1)
	f_YR = f + f_dy * dx/2
	
	return f_XL, f_XR, f_YL, f_YR
	


def applyFluxes(F, flux_F_X, flux_F_Y, dx, dt):
    """
    Apply fluxes to conserved variables
    F        is a matrix of the conserved variable field
    flux_F_X is a matrix of the x-dir fluxes
    flux_F_Y is a matrix of the y-dir fluxes
    dx       is the cell size
    dt       is the timestep
    """

    # update solution
    F += - dt * dx * flux_F_X
    F +=   dt * dx * np.roll(flux_F_X, fd.ROLL_L,axis=0)
    F += - dt * dx * flux_F_Y
    F +=   dt * dx * np.roll(flux_F_Y, fd.ROLL_L,axis=1)

    return F


def getFlux(rho_L, rho_R, vx_L, vx_R, vy_L, vy_R, dx, direction, maxSpeed):
    """
    Calculate fluxed between 2 states with local Lax-Friedrichs/Rusanov rule 
    rho_L        is a matrix of left-state  density
    rho_R        is a matrix of right-state density
    vx_L         is a matrix of left-state  x-velocity
    vx_R         is a matrix of right-state x-velocity
    vy_L         is a matrix of left-state  y-velocity
    vy_R         is a matrix of right-state y-velocity
    flux_Mass    is the matrix of mass fluxes
    flux_Momx    is the matrix of x-momentum fluxes
    flux_Momy    is the matrix of y-momentum fluxes
    flux_Energy  is the matrix of energy fluxes
    """


    # compute star (averaged) states
    rho_star  = 0.5*(rho_L + rho_R)
    momx_star = 0.5*(rho_L * vx_L + rho_R * vx_R)
    momy_star = 0.5*(rho_L * vy_L + rho_R * vy_R)


    print("rho star, px star, py star ", rho_star, momx_star, momy_star)

    if direction ==  0:
        P1 = getC2PressureTensor(rho_star, dx, 0, 0)
        P2 = getC2PressureTensor(rho_star, dx, 1, 0)
    else:
        P1 = getC2PressureTensor(rho_star, dx, 1, 1)
        P2 = getC2PressureTensor(rho_star, dx, 0, 1)

    # compute fluxes (local Lax-Friedrichs/Rusanov)
    flux_Mass   = momx_star
    flux_Momx   = momx_star**2/rho_star + P1
    flux_Momy   = momx_star * momy_star/rho_star + P2


    print("flux mass, flux momx, flux momy ", flux_Mass, flux_Momx, flux_Momy)

    # find wavespeeds
    C = 4

    print("rho_l rho_r ", rho_L, rho_R)
    print("vx_l vx_r ", vx_L, vx_R)
    print("vy_l vy_r ", vy_L, vy_R)

    # add stabilizing diffusive term
    flux_Mass   -= C * 0.5 * (rho_L - rho_R)
    flux_Momx   -= C * 0.5 * (rho_L * vx_L - rho_R * vx_R)
    flux_Momy   -= C * 0.5 * (rho_L * vy_L - rho_R * vy_R)

    print("stabilised flux mass, flux momx, flux momy ", flux_Mass, flux_Momx, flux_Momy)

    return flux_Mass, flux_Momx, flux_Momy

def addSourceTerm( Mass, Momx, Momy, Vx, Vy, dt ):
	"""
    Add gravitational source term to conservative variables
	Mass     is matrix of mass in cells
	Momx     is matrix of x-momentum in cells
	Momy     is matrix of y-momentum in cells
	dt       is timestep to progress solution
	"""
	
	Momx -= dt * Mass * Vx
	Momy -= dt * Mass * Vy
	
	return Mass, Momx, Momy
