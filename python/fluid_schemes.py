import cosmology
import fv 
import my_fv 

import fd
import fd_1d
import fd_2d
import numpy as np
import tree
import interpolation 
import config as configuration
import integration 
import schemes 


from enum import Enum
import matplotlib.pyplot as plt

class FluidScheme(schemes.SchroedingerScheme):
    def __init__(self, config, generateIC):
        super().__init__(config, generateIC)

        #Fluid scheme-specific settings
        self.fluidMode        = config["fluidMode"]
        self.useSlopeLimiting = True
        self.vol              = self.dx**2

        if self.dimension != 2:
            raise ValueError("Fluid scheme only implemented in 2D")

        self.density = np.abs(self.psi) ** 2
        self.phase   = fd.make_continuous(np.angle(self.psi))

        if self.fluidMode == configuration.INTEGRATE_V:
            self.x_i, self.y_i            = ((np.array(self.config["integrationOrigin"]) * self.boxWidth)/self.dx).astype(int)
            self.integrationConstant      = self.phase[self.y_i, self.x_i]
        self.vx, self.vy                = fd_2d.getC2Gradient(self.phase, self.dx)
        self.P                          = fd.getQuantumPressure(self.density, self.dx, 1)
        self.Mass, self.Momx, self.Momy = fv.getConserved(self.density, self.vx, self.vy, self.vol)

        self.vmax = np.maximum(np.max(np.abs(self.vx)), np.max(np.abs(self.vy)))
        self.cfl  = .4


    def step(self, dt):
        if self.outputTimestep:
            print(self.t, dt)
        dx = self.dx
        Mass, Momx, Momy, vol, maxSpeed = self.Mass, self.Momx, self.Momy, self.vol, self.vmax

        density, vx, vy = fv.getPrimitive(Mass, Momx, Momy, vol)

        ###KICK###
        # Add Source (half-step)

        #Here V is the updated V at t = t, maybe this introduces an error of order dt?
        Vx, Vy = fd_2d.getCenteredGradient(self.potential, dx)

        Mass, Momx, Momy = fv.addSourceTerm(Mass, Momx, Momy, Vx, Vy, dt/2)

        ###DRIFT###
        density, vx, vy    = fv.getPrimitive( Mass, Momx, Momy, vol )
        P              = fd_2d.getC2QuantumPressure(density, dx)
        density_dx, density_dy = fd_2d.getCenteredGradient(density, dx)
        vx_dx,  vx_dy  = fd_2d.getCenteredGradient(vx,  dx)
        vy_dx,  vy_dy  = fd_2d.getCenteredGradient(vy,  dx)
        P_dx,   P_dy   = fd_2d.getCenteredGradient(P,   dx)

        self.vmax = np.maximum(np.max(np.abs(vx)), np.max(np.abs(vy)))

        # slope limit gradients
        if self.useSlopeLimiting:
            density_dx, density_dy = fv.slopeLimit(density, dx, density_dx, density_dy)
            vx_dx,  vx_dy  = fv.slopeLimit(vx , dx, vx_dx,  vx_dy )
            vy_dx,  vy_dy  = fv.slopeLimit(vy , dx, vy_dx,  vy_dy )
            P_dx,   P_dy   = fv.slopeLimit(P  , dx, P_dx,   P_dy  )


        # extrapolate half-step in time
        density_prime = density - 0.5*dt * ( vx * density_dx + density * vx_dx + vy * density_dy + density * vy_dy)
        vx_prime  = vx  - 0.5*dt * ( vx * vx_dx + vy * vx_dy + P_dx )
        vy_prime  = vy  - 0.5*dt * ( vx * vy_dx + vy * vy_dy + P_dy )


        # Update global constant using the midpoint method
        if self.fluidMode == configuration.EVOLVE_PHASE:
          V_prime =  self.computePotential(density_prime)
          self.phase = self.phase - dt * ((vx_prime + vy_prime)**2 / 2 + fd_2d.getC2QuantumPressure(density_prime, dx) + V_prime)
        elif self.fluidMode == configuration.INTEGRATE_V:
          V_prime =  self.computePotential(density_prime)
          self.integrationConstant -=  dt * ((vx_prime[self.y_i, self.x_i] + vy_prime[self.y_i, self.x_i])**2 / 2 + fd_2d.getC2QuantumPressure(density_prime, dx)[self.y_i, self.x_i] + V_prime[self.y_i, self.x_i])

        # extrapolate in space to face centers
        density_XL, density_XR, density_YL, density_YR = fv.extrapolateInSpaceToFace(density_prime, density_dx, density_dy, dx)
        vx_XL,  vx_XR,  vx_YL,  vx_YR  = fv.extrapolateInSpaceToFace(vx_prime,  vx_dx,  vx_dy,  dx)
        vy_XL,  vy_XR,  vy_YL,  vy_YR  = fv.extrapolateInSpaceToFace(vy_prime,  vy_dx,  vy_dy,  dx)

        # compute fluxes (local Lax-Friedrichs/Rusanov)
        flux_Mass_X, flux_Momx_X, flux_Momy_X = fv.getFlux(density_XL, density_XR, vx_XL, vx_XR, vy_XL, vy_XR, dx, 0, self.vmax)
        flux_Mass_Y, flux_Momy_Y, flux_Momx_Y = fv.getFlux(density_YL, density_YR, vy_YL, vy_YR, vx_YL, vx_YR, dx, 1, self.vmax)

        # update solution
        Mass   = fv.applyFluxes(Mass, flux_Mass_X, flux_Mass_Y, dx, dt)
        Momx   = fv.applyFluxes(Momx, flux_Momx_X, flux_Momx_Y, dx, dt)
        Momy   = fv.applyFluxes(Momy, flux_Momy_X, flux_Momy_Y, dx, dt)

        ####KICK###

        ##Get primitive variables
        density, vx, vy = fv.getPrimitive(Mass, Momx, Momy, vol)

        ##Calculate gravitational potential
        self.potential =  self.computePotential(density)
        Vx, Vy = fd_2d.getCenteredGradient(self.potential, dx)

        ## Add Source (half-step)
        Mass, Momx, Momy = fv.addSourceTerm(Mass, Momx, Momy, Vx, Vy, dt/2 )

        ##Get primitive variables
        self.density, self.vx, self.vy = fv.getPrimitive(Mass, Momx, Momy, vol)

        if self.fluidMode == configuration.INTEGRATE_V:
          self.phase = integration.antiderivative2D(self.vy, self.vx, dx = dx, C0=self.integrationConstant, x0=self.x_i, y0=self.y_i, debug=False)

        self.t += dt 

    def getDensity(self):
        return self.density

    def getPhase(self):
        return self.phase 


    def setFields(self, density, phase):
        if self.fluidMode == configuration.INTEGRATE_V:
            self.x_i, self.y_i            = ((np.array(self.config["integrationOrigin"]) * self.boxWidth)/self.dx).astype(int)
            self.integrationConstant      = self.phase[self.y_i, self.x_i]
        self.vx, self.vy                  = fd_2d.getC2Gradient(self.phase, self.dx)
        self.P                            = fd.getC2QuantumPressure(self.density, self.dx)
        self.Mass, self.Momx, self.Momy   = fv.getConserved(self.density, self.vx, self.vy, self.vol)
        self.potential                    = self.computePotential(density)

    def setPhase(self, phase):
        self.phase                         = phase
        if self.fluidMode == configuration.INTEGRATE_V:
           self.x_i, self.y_i             = ((np.array(self.config["integrationOrigin"]) * self.boxWidth)/self.dx).astype(int)
           self.integrationConstant       = self.phase[self.y_i, self.x_i]
        self.vx, self.vy                  = fd_2d.getC2Gradient(self.phase, self.dx)
        self.Mass, self.Momx, self.Momy   = fv.getConserved(self.density, self.vx, self.vy, self.vol)
        
    def setDensity(self, density):
        self.density                      = density 
        self.P                            = fd.getC2QuantumPressure(self.density, self.dx)
        self.Mass, self.Momx, self.Momy   = fv.getConserved(self.density, self.vx, self.vy, self.vol)
        self.potential                    = self.computePotential(density)


    def getAdaptiveTimeStep(self):
        # Combination of 
        # CFL-condition for advection: dt < CFL * dx / (sum |v_i|)
        # CFL-condition for diffusion: dt < CFL * hbar/m * dx^2
        return self.cfl*self.eta*self.dx*self.dx/(1.0+self.eta*self.dx*self.vmax)

    def getName(self):
        return "fluid scheme"


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


def getPressureTensor(rho, dx, j, hbar = 1, m = 1):
    logrho = np.log(rho)
    return -hbar**4 / (4 * m**2) * rho * getCenteredGradient(fd.getCenteredGradient(logrho, dx, j), dx)


def getFlux(density_ext, velocities_ext, dx, component):
    # compute star (averaged) states
    rho_star     = 0.5 * (density_ext[component, 0] + density_ext[component, 1])
    momenta_star = 0.5 * (density_ext[component, 0] * velocities_ext[:, component, 0] + density_ext[component, 1] * velocities_ext[:, component, 1])

    flux_Mass    = momenta_star[component]
    flux_Momenta = momenta_star * momenta_star[component] / rho_star + getPressureTensor(rho_star, dx, j = component)

    # find wavespeeds
    maxSpeed = 0.5 / dx + np.max(np.abs(velocities_ext))

    # add stabilizing diffusive term
    flux_Mass    -=  maxSpeed * 0.5 * (density_ext[component, 0] - density_ext[component, 1])
    flux_Momenta -= maxSpeed * 0.5 * (density_ext[component, 0] * velocities_ext[:, component, 0] - density_ext[component, 1] * velocities_ext[:, component, 1])

    return flux_Mass, flux_Momenta



def applyFluxes(F, flux_F, dx, dt):
    # update solution
    for i in range(F.ndim):
        F += - dt * dx * flux_F[i]
        F +=   dt * dx * np.roll(flux_F[i], fd.ROLL_L,axis=i)

    return F


class MUSCLHancock(schemes.SchroedingerScheme):
    def __init__(self, config, generateIC):
        super().__init__(config, generateIC)

        #Fluid scheme-specific settings
        self.fluidMode        = config["fluidMode"]

        self.density            = np.abs(self.psi) ** 2
        self.potential          = self.computePotential(self.density)
        self.potentialGradient  = getCenteredGradient(self.potential, self.dx)
        self.phase              = fd.make_continuous(np.angle(self.psi))
        self.velocities         = getCenteredGradient(self.phase, self.dx)

        self.volume             = self.dx**self.dimension 

        self.vmax               = 1/self.dx 

    def limitGradient(self, f, fGradient, dx):
        #limitGradients
        for i in range(f.ndim):
            fGradient[i] = self.limiter(( (f-np.roll(f, fd.ROLL_L,axis=i))/dx)/(fGradient[i] + 1.0e-8*(fGradient[i]==0))) * fGradient[i]
            fGradient[i] = self.limiter((-(f-np.roll(f, fd.ROLL_R,axis=i))/dx)/(fGradient[i] + 1.0e-8*(fGradient[i]==0))) * fGradient[i]

        return fGradient

    def step(self, dt):
        if self.outputTimestep:
            print(self.t, dt)


        velocities, density, potentialGradient = self.velocities, self.density, self.potentialGradient

        #for extrapolation to face
        ddensity          = np.zeros(density.shape)
        dvelocities       = np.zeros((self.dimension, *density.shape))

        #after extrapolation
        velocities_prime  = np.zeros((self.dimension,                    *density.shape))
        velocityGradients = np.zeros((self.dimension, self.dimension,    *density.shape))
        massFluxes        = np.zeros((self.dimension,                    *density.shape))
        momentaFluxes     = np.zeros((self.dimension, self.dimension,    *density.shape))
        velocities_ext    = np.zeros((self.dimension, self.dimension, 2, *density.shape))

        ###KICK###

        # Add Source (half-step)
        velocities -= dt/2 * potentialGradient

        self.vmax = np.max(np.abs(velocities)) * np.sqrt(2)

        ###DRIFT###
        #extrapolate to face centers (half step in space using centered gradients (MUSCL) and half step in time (Hancock))
        pressure          = fd.getQuantumPressure(density, self.dx, eta = self.eta)
        pressureGradient  = getCenteredGradient(pressure, self.dx)
        densityGradient   = getCenteredGradient(density, self.dx)

        #compute spatial gradients
        #self.limitGradient(density, densityGradient, self.dx)
        #self.limitGradient(pressure, pressureGradient, self.dx)

        for i in range(self.dimension):
            velocityGradients[i] = getCenteredGradient(velocities[i], self.dx)
            #self.limitGradient(velocities[i], velocityGradients[i], self.dx)


        #extrapolate
        for i in range(self.dimension):
            ddensity += velocities[i] * densityGradient[i] + density * velocityGradients[i][i]
            for j in range(self.dimension):
                dvelocities[i] += velocities[j] * velocityGradients[i][j]

            dvelocities[i] += pressureGradient[i]


        # extrapolate half-step in time
        density_prime    = density    - 0.5 * dt * ddensity
        velocities_prime = velocities - 0.5 * dt * dvelocities

        # extrapolate in space to face centers
        density_ext    = extrapolateInSpaceToFace(density_prime, densityGradient, self.dx)

        for i in range(self.dimension):
            velocities_ext[i] = extrapolateInSpaceToFace(velocities_prime[i],  velocityGradients[i],  self.dx)

        
        # compute fluxes (Lax-Friedrichs)
        for i in range(self.dimension):
            massFluxes[i], momentaFluxes[i] = getFlux(density_ext, velocities_ext, self.dx, i)

        # update solution
        momenta   = velocities * density * self.volume
        mass      = density * self.volume

        #momentaFluxes[0] contains momx_X, momy_X, momz_X
        #momentaFluxes[1] contains momx_Y, momy_Y, momz_Y

        if self.debug:
            print("Before swap: ")

            plt.title("density 0")
            plt.imshow(massFluxes[0])
            plt.show()


            plt.title("density 1")
            plt.imshow(massFluxes[1])
            plt.show()

    #
            plt.title("0 0")
            plt.imshow(momentaFluxes[0, 0])
            plt.show()
            plt.title("0 1")
            plt.imshow(momentaFluxes[0, 1])
            plt.show()
            plt.title("1 0")
            plt.imshow(momentaFluxes[1, 0])
            plt.show()
            plt.title("1 1")
            plt.imshow(momentaFluxes[1, 1])
            plt.show()
    #
            print("After swarp: ", )

            momentaFluxes = np.swapaxes(momentaFluxes, axis1=0, axis2=1)
    #
            plt.title("0 0")
            plt.imshow(momentaFluxes[0, 0])
            plt.show()
            plt.title("0 1")
            plt.imshow(momentaFluxes[0, 1])
            plt.show()
            plt.title("1 0")
            plt.imshow(momentaFluxes[1, 0])
            plt.show()
            plt.title("1 1")
            plt.imshow(momentaFluxes[1, 1])
            plt.show()
            momentaFluxes = np.swapaxes(momentaFluxes, axis1=0, axis2=1)

        momentaFluxes = np.swapaxes(momentaFluxes, axis1=0, axis2=1)
        
        mass      = applyFluxes(mass, massFluxes, self.dx, dt)
        for i in range(self.dimension):
            momenta[i] = applyFluxes(momenta[i], momentaFluxes[i], self.dx, dt)

        density    = mass / self.volume 
        velocities = momenta / density / self.volume

        ####KICK###

        ##Calculate gravitational potential
        self.potential     = self.computePotential(density)
        potentialGradient  = getCenteredGradient(self.potential, self.dx)

        ## Add Source (half-step)
        velocities -= dt/2 * potentialGradient[i]

        self.potentialGradient  = potentialGradient
        self.velocities         = velocities
        self.density            = density

        self.t += dt

    def getDensity(self):
        return self.density

    def getPhase(self):
        return self.density * 0

    def getAdaptiveTimeStep(self):
        t1 = 1/6 * self.eta*self.dx*self.dx
        t2 = 0.5 * self.dx/(self.dimension*(self.vmax + 1e-8))
        #print("Advection: ", t2, " Diffusion: ", t1, " Acceleration: ", t3)
        return np.min([t1, t2])

    def getName(self):
        return "MUSCL-Hancock"