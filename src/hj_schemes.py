import src.fv 
import src.fd
import src.fd_1d
import src.fd_2d
import src.schemes
import src.eno 

import numpy as np
import matplotlib.pyplot as plt


""" DEFINE SIMULATION """


#Pure advection equation d phi/ dt = grad (grad phase * phi)
class HJScheme(schemes.Scheme):
    def __init__(self, config, generateIC):
        super().__init__(config, generateIC)

        if not self.usePeriodicBC:
            raise ValueError("HJ scheme only supports periodic BC")

        phi, _ = self.generateIC(*self.grid, self.dx, self.t)
        self.fields = np.zeros((1, *phi.shape))
        self.fields[0] = phi

        self.returnDerivative = config["returnHJDerivative"]
        
    def getDensity(self):
        phi = self.fields[0].copy()
        if self.dimension != 1:
            phi = np.diagonal(phi)
    
        if self.returnDerivative:
            phi = fd.getDerivative(phi, self.dx, self.c1_stencil, self.c1_coeff, axis = 0)
        return phi 
        
    def getPhase(self):
        return self.fields[0] * 0


class UpwindScheme(HJScheme):

    def __init__(self, config, generateIC):
        super().__init__(config, generateIC)


    def getUpdatedFields(self, dt, fields):
        pc = fields[0]
        dx = self.dx

        dphi = np.zeros(pc.shape)

        for i in range(self.dimension):
            pr   = np.roll(pc, fd.ROLL_R, axis=i)
            pl   = np.roll(pc, fd.ROLL_L, axis=i)
            vc   = (pr - pl) / (2*dx)
            vf   = (pr - pc) / (dx)
            vb   = (pc - pl) / (dx)

            dphi += (np.minimum(vf, 0)**2 + np.maximum(vb, 0)**2)/2

        return dt * -np.array([dphi])


class ENOScheme(HJScheme):

    def __init__(self, config, generateIC):
        super().__init__(config, generateIC)
        self.coef = np.zeros((self.stencilOrder, self.N_ghost))
        self.lpis = np.zeros((self.stencilOrder, self.N_ghost), dtype=int)


    def getUpdatedFields(self, dt, fields):
        f = fields[0]
        dx = self.dx

        dphi = np.zeros(f.shape)

        for i in range(self.dimension):
            eno_f = eno.ENO(xx = self.grid[i], y = f, L = self.boxWidth, order = self.stencilOrder, direction =   1, lpis = self.lpis, coef = self.coef)
            eno_b = eno.ENO(xx = self.grid[i], y = f, L = self.boxWidth, order = self.stencilOrder, direction = - 1, lpis = self.lpis, coef = self.coef)
            up    = eno_f.dP(x = self.grid[i])
            um    = eno_b.dP(x = self.grid[i])

            dphi += (np.minimum(up, 0)**2 + np.maximum(um, 0)**2)/2

        return dt * -np.array([dphi])

class SOUScheme(HJScheme):

    def __init__(self, config, generateIC):
        super().__init__(config, generateIC)
        self.coef = np.zeros((self.stencilOrder, self.N_ghost))
        self.lpis = np.zeros((self.stencilOrder, self.N_ghost), dtype=int)


    def getUpdatedFields(self, dt, fields):
        pc = fields[0]
        dx = self.dx

        dphi = np.zeros(pc.shape)

        for i in range(self.dimension):
            pp  = np.roll(pc,     fd.ROLL_R, axis = i)
            pm  = np.roll(pc,     fd.ROLL_L, axis = i)
            p2m = np.roll(pc, 2 * fd.ROLL_L, axis = i)

            ql = (pc-pm)/((pm-p2m) + (((pm-p2m)==0) * 1e-8))
            qc = np.roll(ql, fd.ROLL_R, axis = i)

            Qc = 1/(qc + (qc == 0) * 1e-8)
            Qr = np.roll(Qc, fd.ROLL_R, axis = i)

            vm = 1/dx * ( 1 + 0.5 * self.limiter(qc) - 0.5 * self.limiter(ql)/(ql + (ql==0) * 1e-8)) * (pc - pm)
            vp = 1/dx * ( 1 + 0.5 * self.limiter(Qc) - 0.5 * self.limiter(Qr)/(Qr + (Qr==0) * 1e-8)) * (pp - pc)
            
            dphi   += (np.minimum(vp, 0)**2 + np.maximum(vm, 0)**2)/2

        return dt * -np.array([dphi])