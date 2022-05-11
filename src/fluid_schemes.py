
import numpy as np
import matplotlib.pyplot as plt

import src.fd as fd
import src.integration as integration
import src.schemes as schemes 
import src.config as c 

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
    maxSpeed = 0.15 / dx + np.max(np.abs(velocities_ext))

    # add stabilizing diffusive term
    flux_Mass    -= maxSpeed * 0.5 * (density_ext[component, 0] - density_ext[component, 1])
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
        self.fluidMode          = config["fluidMode"]

        self.density            = np.abs(self.psi) ** 2
        self.potential          = self.computePotential(self.density)
        self.potentialGradient  = getCenteredGradient(self.potential, self.dx)
        self.phase              = fd.make_continuous(np.angle(self.psi))


        if self.fluidMode == c.INTEGRATE_V:
            if "integrationOrigin" in self.config:
                self.integrationOrigin        = tuple(((np.array(self.config["integrationOrigin"]) * self.boxWidth)/self.dx).astype(int))
            else:
                self.integrationOrigin    = np.zeros(self.dimension, dtype=int)
            self.integrationConstant      = self.phase[self.integrationOrigin]

        self.velocities         = getCenteredGradient(self.phase, self.dx)
        self.volume             = self.dx**2
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
        self.limitGradient(density, densityGradient, self.dx)
        self.limitGradient(pressure, pressureGradient, self.dx)

        for i in range(self.dimension):
            velocityGradients[i] = getCenteredGradient(velocities[i], self.dx)
            self.limitGradient(velocities[i], velocityGradients[i], self.dx)

        #extrapolate
        for i in range(self.dimension):
            ddensity += velocities[i] * densityGradient[i] + density * velocityGradients[i][i]
            for j in range(self.dimension):
                dvelocities[i] += velocities[j] * velocityGradients[i][j]

            dvelocities[i] += pressureGradient[i]

        # extrapolate half-step in time
        density_prime    = density    - 0.5 * dt * ddensity
        velocities_prime = velocities - 0.5 * dt * dvelocities


        if self.fluidMode == c.INTEGRATE_V:
            if self.dimension == 1:
                self.phase = integration.antiderivative1D(velocities[0], dx = self.dx, C0 = self.integrationConstant, x0 = self.integrationOrigin[0])
            elif self.dimension == 2: 
                self.phase = integration.antiderivative2D(velocities[1], velocities[0], dx = self.dx, C0 = self.integrationConstant, x0 = self.integrationOrigin[0], y0 = self.integrationOrigin[1], debug=False)
            self.integrationConstant -= dt * (np.sum(velocities**2, axis = 0) / 2 + pressure + self.potential)[self.integrationOrigin] 
    
        else:
            ###UPDATE PHASE USING EULER
            self.phase -= dt * (np.sum(velocities**2, axis = 0) / 2 + pressure + self.potential)

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
        return self.phase

    def getAdaptiveTimeStep(self):
        t1 = 1/6 * self.eta*self.dx*self.dx
        t2 = 0.5 * self.dx/(self.dimension*(self.vmax + 1e-8))
        return np.min([t1, t2])

    def getName(self):
        return "MUSCL-Hancock"