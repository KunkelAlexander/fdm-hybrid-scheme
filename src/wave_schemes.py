
import numpy as np
import scipy
import matplotlib.pyplot as plt 

import src.fd as fd
import src.fd_1d as fd_1d
import src.fd_2d as fd_2d
import src.schemes as schemes


#Evolve wave function psi 
class WaveScheme(schemes.SchroedingerScheme):
    def __init__(self, config, generateIC):
        super().__init__(config, generateIC)
        self.cfl = .2
        self.friction = config["friction"]
        if self.friction > 0:
            print("Using wave scheme with non-zero friction. This results in imaginary time steps.")

    def step(self, dt):

        if not self.usePeriodicBC:
            self.setBoundaryConditions(self.psi)

        if self.friction > 0:
            dt *= 1j

        # (1/2) kick
        self.psi = np.exp(-1.0j * dt / 2.0 * self.potential) * self.psi

        # drift
        self.drift(dt)

        #update potential
        self.potential = self.computePotential(np.abs(self.psi) ** 2)

        #(1/2) kick
        self.psi = np.exp(-1.0j * dt / 2.0 * self.potential) * self.psi


        self.t += dt * self.getScaleFactor() ** 2

    def drift(self):
        raise NotImplementedError("Please Implement this method")

    def getDensity(self):
        return np.abs(self.psi) ** 2

    def getPhase(self):
        return fd.make_continuous(np.angle(self.psi))
        
    def getAdaptiveTimeStep(self):
        return 1/6*self.eta*self.dx*self.dx

    def setBoundaryConditions(self, psi):
        f = self.generateIC(*self.grid, self.dx, self.t)
        psi[self.boundary] = f[self.boundary]

class SpectralScheme(WaveScheme):
    def __init__(self, config, generateIC):
        super().__init__(config, generateIC)

    def drift(self, dt):
        # drift
        psihat = scipy.fft.fftn(self.psi, workers = self.workers)
        psihat = np.exp(dt * (-1.0j * self.kSq / 2.0)) * psihat
        self.psi = scipy.fft.ifftn(psihat, workers = self.workers)


    def getName(self):
        return "spectral scheme"

class FTCSScheme(WaveScheme):
    def __init__(self, config, generateIC):
        super().__init__(config, generateIC)
        self.cfl = .125
        
    def drift(self, dt):
        dx, u0 = self.dx, self.psi
        u1 = u0 + fd.solvePeriodicFTCSDiffusion(u0, dx = dx, dt = dt, coeff = self.c2_coeff, stencil = self.c2_stencil)
        self.psi = 0.5 * u0 + 0.5 * u1 + fd.solvePeriodicFTCSDiffusion(u1, dx = dx, dt = dt * 0.5, coeff = self.c2_coeff, stencil = self.c2_stencil)

    def getName(self):
        return "ftcs scheme"

class CNScheme(WaveScheme):
    def __init__(self, config, generateIC):
        super().__init__(config, generateIC)

        if self.dimension > 2:
            raise ValueError("CN scheme only supports 1D and 2D")


    def drift(self, dt):
        if self.dimension == 1:
            self.kin_A, self.kin_b = fd.createKineticLaplacian(
                self.N_ghost, dt, self.dx, periodicBC=self.usePeriodicBC
            )
        elif self.dimension == 2:
            self.kin_A_row, self.kin_b_row = fd.createKineticLaplacian(
                self.N_ghost, dt, self.dx, periodicBC=self.usePeriodicBC
            )
            self.kin_A_col, self.kin_b_col = fd.createKineticLaplacian(
                self.N_ghost, dt, self.dx, periodicBC=self.usePeriodicBC
            )

        # drift
        if self.usePeriodicBC:
            if self.dimension == 1:
                self.psi = fd_1d.solvePeriodicCNDiffusion(self.psi, self.kin_A, self.kin_b)
            elif self.dimension == 2:
                self.psi = fd_2d.solvePeriodicCNDiffusion(
                    self.psi, self.kin_A_row, self.kin_b_row, self.kin_A_col, self.kin_b_col
                )
        else:

            if self.dimension == 1:
                self.psi = fd_1d.solveDirichletCNDiffusion(self.psi, self.psi[0], self.psi[-1], self.kin_A, self.kin_b)
            elif self.dimension == 2:
                #self.psi = fd_2d.solveDirichletCNDiffusion(
                #    self.psi, self.kin_A_row, self.kin_b_row, self.kin_A_col, self.kin_b_col
                #)
                raise ValueError()

    def getName(self):
        return "crank-nicolson scheme"