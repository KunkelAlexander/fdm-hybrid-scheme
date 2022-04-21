import numpy as np
import fd_2d
import fd 

def getCenteredGradient(f, dx):
    gradient = np.zeros((f.ndim, *f.shape))
    for i in range(f.ndim):
        gradient[i] = fd.getCenteredGradient(f, dx, axis = i)
    return gradient


def extrapolateInSpaceToFace(f, f_grad, dx):
	extrapolatedf = np.zeros((f.ndim, 2, *f.shape))

	for i in range(f.ndim):
		f_iL = f - f_grad[i] * dx/2
		f_iL = np.roll(f_iL, fd.ROLL_R, axis=i)
		f_iR = f + f_grad[i] * dx/2

		extrapolatedf[i, 0] = f_iL
		extrapolatedf[i, 1] = f_iR
	
	return extrapolatedf

def getFlux(density_ext, velocities_ext, dx, axis, maxSpeed):
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
    rho_star     = 0.5 * (density_ext[0] + density_ext[1])
    momenta_star = np.zeros((rho_star.ndim, *rho_star.shape))

    for i in range(rho_star.ndim):
        momenta_star[i] = 0.5 * (density_ext[0] * velocities_ext[i][0] + density_ext[0] * velocities_ext[i][1])


    # compute fluxes (local Lax-Friedrichs/Rusanov)
    flux_Mass   = momx_star

    flux_Momenta = np.zeros((rho_star.ndim, *rho_star.shape))

    flux_Momx   = momx_star**2/rho_star + P1
    flux_Momy   = momx_star * momy_star/rho_star + P2

    # add stabilizing diffusive term
    flux_Mass   -= maxSpeed * 0.5 * (density_ext[0] - density_ext[1])

    for i in range(rho_star.ndim):
        flux_Momenta[i]   -= maxSpeed * 0.5 * (density_ext[0] * velocities_ext[i][0] - density_ext[0] * velocities_ext[i][1])

    return flux_Mass, flux_Momenta