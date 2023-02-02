
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

    def step(self, dt):

        if not self.usePeriodicBC:
            self.setBoundaryConditions(self.psi)

        # (1/2) kick
        self.psi = np.exp(-1.0j * dt / 2 * self.potential) * self.psi

        # drift
        self.drift(dt)

        #update potential
        self.potential = self.computePotential(np.abs(self.psi) ** 2)

        #(1/2) kick
        self.psi = np.exp(-1.0j * dt / 2 * self.potential) * self.psi


        self.t += dt * self.getScaleFactor() ** 2

    def drift(self):
        raise NotImplementedError("Please Implement this method")

    def getDensity(self):
        return np.abs(self.psi) ** 2

    def getPhase(self):
        return fd.make_continuous(np.angle(self.psi))
        
    def getAdaptiveTimeStep(self):
        t1 = self.C_parabolic * self.dx**2/self.eta
        if self.G > 0:
            t2 = self.C_potential    * self.hbar/np.max(np.abs(self.potential) + 1e-8)
        else:
            t2 = 1e4
        
        return np.min([t1, t2])

    def setBoundaryConditions(self, psi):
        f = self.generateIC(*self.grid, self.dx, self.t, self.m, self.hbar)
        psi[self.boundary] = f[self.boundary]

    def computeRelError(self):
        psi_ref = self.generateIC(*self.grid, self.dx, self.t, self.m, self.hbar)
        l_infty_diff = np.max(np.abs(np.abs(self.psi[self.inner])**2 - np.abs(psi_ref[self.inner])**2))
        l_infty_ref  = np.max(np.abs(psi_ref)**2)
        return l_infty_diff/l_infty_ref 
    
    def computeRMSError(self):
        psi_ref = self.generateIC(*self.grid, self.dx, self.t, self.m, self.hbar)
        d1 = np.abs(self.psi[self.inner])**2
        d2 = np.abs(psi_ref[self.inner])**2
        RMS = np.sqrt(np.sum((d1 - d2)**2))
        RMS /= len(d1)
        return RMS

class SpectralScheme(WaveScheme):
    def __init__(self, config, generateIC):
        super().__init__(config, generateIC)

    def drift(self, dt):
        # drift
        psihat = scipy.fft.fftn(self.psi, workers = self.workers)
        psihat = np.exp(dt * (-1.0j * self.eta * self.kSq / 2.0)) * psihat
        self.psi = scipy.fft.ifftn(psihat, workers = self.workers)


    def getName(self):
        return "spectral scheme"

class FTCSScheme(WaveScheme):
    def __init__(self, config, generateIC):
        super().__init__(config, generateIC)
        
    def drift(self, dt):
        dx, u0 = self.dx, self.psi
        u1 = u0 + self.eta * fd.solvePeriodicFTCSDiffusion(u0, dx = dx, dt = dt, coeff = self.c2_coeff, stencil = self.c2_stencil)
        self.psi = 0.5 * u0 + 0.5 * u1 + self.eta * fd.solvePeriodicFTCSDiffusion(u1, dx = dx, dt = dt * 0.5, coeff = self.c2_coeff, stencil = self.c2_stencil)

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
                self.N_ghost, dt, self.dx, periodicBC=self.usePeriodicBC, eta = self.eta
            )
        elif self.dimension == 2:
            self.kin_A_row, self.kin_b_row = fd.createKineticLaplacian(
                self.N_ghost, dt, self.dx, periodicBC=self.usePeriodicBC, eta = self.eta
            )
            self.kin_A_col, self.kin_b_col = fd.createKineticLaplacian(
                self.N_ghost, dt, self.dx, periodicBC=self.usePeriodicBC, eta = self.eta
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
                self.psi = fd_2d.solveDirichletCNDiffusion(
                    self.psi, self.kin_A_row, self.kin_b_row, self.kin_A_col, self.kin_b_col
                )
                raise ValueError()

    def getName(self):
        return "crank-nicolson scheme"
    

def LAP1(f, axis = 0):
    fm4 = np.roll(f, 4 * fd.ROLL_L, axis=axis)
    fm3 = np.roll(f, 3 * fd.ROLL_L, axis=axis)
    fm2 = np.roll(f, 2 * fd.ROLL_L, axis=axis)
    fm1 = np.roll(f, 1 * fd.ROLL_L, axis=axis)
    fp4 = np.roll(f, 4 * fd.ROLL_R, axis=axis)
    fp3 = np.roll(f, 3 * fd.ROLL_R, axis=axis)
    fp2 = np.roll(f, 2 * fd.ROLL_R, axis=axis)
    fp1 = np.roll(f, 1 * fd.ROLL_R, axis=axis)
    return 1.0/12.0 * ( - fm2 + 16.0*fm1 - 30.0*f - fp2 + 16.0*fp1 )

def LAP2(f, axis = 0): 
    fm4 = np.roll(f, 4 * fd.ROLL_L, axis=axis)
    fm3 = np.roll(f, 3 * fd.ROLL_L, axis=axis)
    fm2 = np.roll(f, 2 * fd.ROLL_L, axis=axis)
    fm1 = np.roll(f, 1 * fd.ROLL_L, axis=axis)
    fp4 = np.roll(f, 4 * fd.ROLL_R, axis=axis)
    fp3 = np.roll(f, 3 * fd.ROLL_R, axis=axis)
    fp2 = np.roll(f, 2 * fd.ROLL_R, axis=axis)
    fp1 = np.roll(f, 1 * fd.ROLL_R, axis=axis)
    return 1.0/144.0 * ( + fm4 - 32.0*fm3 + 316.0*fm2 - 992.0*fm1 + fp4 - 32.0*fp3 + 316.0*fp2 - 992.0*fp1 +  1414.0*f )

class GAMERScheme(WaveScheme):
    def __init__(self, config, generateIC):
        super().__init__(config, generateIC)
        
    def drift(self, dt):
        dx, u0 = self.dx, self.psi
        _Eta = self.eta 
        _dh  = 1/dx 
        Taylor3_Coeff = 1/6
        dT           = 0.5*dt*_Eta
        _Eta2_dh     = 0.5*_dh*_Eta
        Coeff1       = dT*_dh*_dh
        Coeff2       = Taylor3_Coeff*Coeff1**2


        re0 = np.real(u0)
        im0 = np.imag(u0)

        re1 = re0 - 0.5*Coeff1*LAP1( im0 ) - Coeff2*LAP2( re0 )
        im1 = im0 + 0.5*Coeff1*LAP1( re0 ) - Coeff2*LAP2( im0 )

        re2   = re0 - Coeff1*LAP1( im1 )
        im2   = im0 + Coeff1*LAP1( re1 )

        self.psi = re2 + 1j * im2

    def getName(self):
        return "gamer scheme"