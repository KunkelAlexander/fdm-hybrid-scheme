import fv 
import fd
import fd_1d
import fd_2d
import schemes
import config as configuration
import interpolation
import eno 

import numpy as np
import matplotlib.pyplot as plt


""" DEFINE SIMULATION """

#Pure advection equation d phi/ dt = grad (grad phase * phi)
class AdvectionScheme(schemes.Scheme):
    def __init__(self, config, generateIC):
        super().__init__(config, generateIC)

        phi, phase = self.generateIC(*self.grid, self.dx, self.t)
        self.fields = np.zeros((2, *phi.shape))
        self.fields[0] = phi
        self.fields[1] = phase

        #CFL constant for advection scheme
        self.cfl  = 0.99
        self.vmax = 1
        
    def getDensity(self):
        return self.fields[0]

    def getPhase(self):
        return self.fields[1]

    def getAdaptiveTimeStep(self):
        phase = self.fields[1]
        self.vmax = 0 

        for i in range(self.dimension):
            v = (np.roll(phase, fd.ROLL_R, axis = i) - np.roll(phase, fd.ROLL_L, axis = i))/(2*self.dx)
            self.vmax = np.maximum(self.vmax, np.max(np.abs(v[self.inner])))
            
        return self.cfl*self.dx/(0.1 + self.vmax)

    def setBoundaryConditions(self, fields):
        rho, phase = self.generateIC(*self.grid, self.dx, self.t)
        
        fields[0][self.boundary] = rho[self.boundary]
        fields[1] = phase


    def setPhase(self, fields):
        rho, phase = self.generateIC(*self.grid, self.dx, self.t)
        fields[1] = phase


class DonorCellScheme(AdvectionScheme):

    def __init__(self, config, generateIC):
        super().__init__(config, generateIC)


    def getUpdatedFields(self, dt, fields):
        self.setPhase(fields)
        phi_P, pc = fields
        dx = self.dx

        dphi = np.zeros(phi_P.shape)
        dpc  = np.zeros(pc.shape)

        for i in range(self.dimension):
            pr   = np.roll(pc, fd.ROLL_R, axis=i)
            pl   = np.roll(pc, fd.ROLL_L, axis=i)
            vc   = (pr - pl) / (2*dx)
            vr   = (pr - pc) / (dx)
            vl   = (pc - pl) / (dx)

            phi_E = np.roll(phi_P,  fd.ROLL_R, axis=i)
            phi_W = np.roll(phi_P,  fd.ROLL_L, axis=i)
            F_e   = vr
            F_w   = vl

            a_E = np.maximum(-F_e, 0)
            a_W = np.maximum(+F_w, 0)
            a_P = np.maximum(+F_e, 0) + np.maximum(-F_w,  0)

            phi_e = np.maximum(F_e, 0) * phi_P + np.minimum(F_e, 0) * phi_E
            phi_w = np.maximum(F_w, 0) * phi_W + np.minimum(F_w, 0) * phi_P

            dphi += (phi_e - phi_w)/dx

        return dt * -np.array([dphi, pc])

class LaxWendroffScheme(AdvectionScheme):

    def __init__(self, config, generateIC):
        super().__init__(config, generateIC)

    def getUpdatedFields(self, dt, fields):
        self.setPhase(fields)
        phi, phase = self.fields
        dx = self.dx

        dphi   = np.zeros(phi.shape)
        dphase = np.zeros(phi.shape)


        for i in range(self.dimension):
            phasep   = np.roll(phase, fd.ROLL_R, axis=i)
            phasem   = np.roll(phase, fd.ROLL_L, axis=i)
            phip   = np.roll(phi, fd.ROLL_R, axis=i)
            phim   = np.roll(phi, fd.ROLL_L, axis=i)

            #v_i-1/2
            vm2 = fd.getDerivative(phase, dx, self.b1_stencil, self.b1_coeff, axis = i)
            #v_i+1/2
            vp2 = np.roll(vm2, fd.ROLL_R, axis=i)
            #v_i
            v   = (vm2 + vp2)/2
            #v_i+1
            vp  = np.roll(v, fd.ROLL_R, axis=i)
            #v_i-1
            vm  = np.roll(v, fd.ROLL_L, axis=i)

            dphi -= - 1/(2*dx) * (vp * phip - vm * phim) + dt / (2*dx**2) * (vp2 * (vp * phip - v * phi) - vm2 * (v * phi - vm * phim))

        return -dt * np.array([dphi, dphase])

class SOLimiterScheme(AdvectionScheme):
    def __init__(self, config, generateIC):
        super().__init__(config, generateIC)

        self.limiter = schemes.FluxLimiters.SUPERBEE
        self.cfl = 0.4999

    def getUpdatedFields(self, dt, fields):
        self.setPhase(fields)
        rc, pc = fields
        dx     = self.dx
        dr     = np.zeros(rc.shape)
        dp     = np.zeros(pc.shape)

        for i in range(self.dimension):
            #p_i-1
            pl   = np.roll(pc, fd.ROLL_L, axis=i)

            #v_i-1/2
            vm   = fd.getDerivative(pc, dx, self.b1_stencil, self.b1_coeff, axis = i)
            #v_i+1/2
            vp   = np.roll(vm, fd.ROLL_R, axis=i)

            #r_i-1
            rl   = np.roll(rc,   fd.ROLL_L, axis=i)
            #r_i-2
            r2l  = np.roll(rc, 2*fd.ROLL_L, axis=i)
            #r_i+1
            rr   = np.roll(rc,   fd.ROLL_R, axis=i)
            
            #dr += 1 / (2*dx) * (3 * rc - 4 * rl + r2l)

            if self.debug:
                plt.title("rc - rl")
                plt.plot((rc-rl)[self.inner])
                plt.show()


                plt.title("rl - r2l")
                plt.plot((rl-r2l)[self.inner])
                plt.show()

            ql = (rc-rl)/((rl-r2l) + (((rl-r2l)==0) * 1e-8))
            qc = np.roll(ql, fd.ROLL_R, axis = i)

            Qc = 1/(qc + (qc == 0) * 1e-8)
            Qr = np.roll(Qc, fd.ROLL_R, axis = i)

            if self.debug:
                print("ql: ", ql[self.inner])
                print("limiter (qc)", self.limiter(qc)[self.inner])

                plt.title("ql")
                plt.plot(ql[self.inner])
                plt.show()
                plt.title("Limiter ql")
                plt.plot(self.limiter(qc)[self.inner])
                plt.show()
                plt.title("Limiter ql / rl")
                plt.plot((self.limiter(ql)/(ql + (ql==0) * 1e-8))[self.inner])
                plt.show()


                plt.title("Qc")
                plt.plot(Qc[self.inner])
                plt.show()
                plt.title("Limiter Qc")
                plt.plot(self.limiter(Qc)[self.inner])
                plt.show()

            #Slope-limited upwind gradient of density
            drm = 1/dx * ( 1 + 0.5 * self.limiter(qc) - 0.5 * self.limiter(ql)/(ql + (ql==0) * 1e-8)) * (rc - rl)
            
            #Slope-limited downwind gradient of density
            drp = 1/dx * ( 1 + 0.5 * self.limiter(Qc) - 0.5 * self.limiter(Qr)/(Qr + (Qr==0) * 1e-8)) * (rr - rc)

            if self.debug:
                plt.title("drm")
                plt.plot(drm[self.inner])
                plt.show()


            dr += np.maximum(vm, 0) * drm + np.minimum(vp, 0) * drp 

        dfields = -dt * np.array([dr, dp])

        return dfields

class MUSCLScheme(AdvectionScheme):
    def __init__(self, config, generateIC):
        super().__init__(config, generateIC)

        self.limiter = schemes.FluxLimiters.SUPERBEE

    def getUpdatedFields(self, dt, fields):
        self.setPhase(fields)
        rc, pc = fields
        dx     = self.dx
        dr     = np.zeros(rc.shape)
        dp     = np.zeros(pc.shape)

        for i in range(self.dimension):
            #r_i+1
            rf   = np.roll(rc, fd.ROLL_R, axis=i)
            #r_i-1
            rb   = np.roll(rc, fd.ROLL_L, axis=i)
            #r_i+2
            r2b  = np.roll(rc, 2*fd.ROLL_L, axis=i)
            #p_i-1
            pb   = np.roll(pc, fd.ROLL_L, axis=i)
            #v_i-1/2
            vm   = fd.getDerivative(pc, dx, self.b1_stencil, self.b1_coeff, axis = i)

            qm = ((rb - r2b) * (vm > 0) + (rf  - rc) * (vm < 0)) / ((rc - rb) + ((rc - rb) == 0) * 1e-8)

            #f_i-1/2
            fl  =  (np.maximum(vm, 0) * rb + np.minimum(vm, 0) * rc) \
                + 1 / 2 * np.abs(vm) *  (1 - np.abs(vm * dt/dx)) * self.limiter(qm) * (rc - rb)

            #f_i+1/2
            fr  = np.roll(fl, fd.ROLL_R, axis = i)
            dr += (fr - fl)/dx
                
        dfields = -dt * np.array([dr, dp])

        return dfields


class LaxFriedrichsScheme(AdvectionScheme):

    def __init__(self, config, generateIC):
        super().__init__(config, generateIC)
        self.cfl = 0.25

        self.interpolation_stencil = interpolation.stencil_ij(self.stencilOrder)

        if self.stencilOrder % 2 == 0:
            self.left_shift = int(self.stencilOrder/2 - 1)
        else:
            self.left_shift = int((self.stencilOrder-1)/2)

        self.f1_stencil, self.f1_coeff  = fd.getFiniteDifferenceCoefficients(derivative_order = 1, accuracy = self.stencilOrder, mode = fd.MODE_FORWARD)
        self.b1_stencil, self.b1_coeff  = fd.getFiniteDifferenceCoefficients(derivative_order = 1, accuracy = self.stencilOrder, mode = fd.MODE_BACKWARD)


    def getUpdatedFields(self, dt, fields):
        self.setPhase(fields)
        density, phase = fields
        dx = self.dx
        
        ddensity = np.zeros(density.shape)
        dphase   = np.zeros(phase.shape)

        self.vmax = 0

        for i in range(self.dimension):
            up2m    = fd.getDerivative (phase, dx, self.f1_stencil, self.f1_coeff, axis = i)
            um2p    = fd.getDerivative (phase, dx, self.b1_stencil, self.b1_coeff, axis = i)
            self.vmax = np.maximum(self.vmax, np.max(np.abs(0.5 * (up2m + um2p))))

            up2p    = np.roll(up2m, fd.ROLL_R, axis = i)
            um2m    = np.roll(um2p, fd.ROLL_L, axis = i)

            if self.debug:
                plt.title("up2m")
                plt.plot(up2m)
                plt.show()

                plt.title("um2p")
                plt.plot(um2p)
                plt.show()

            densityp2m  = interpolation.fixed_stencil_reconstruction(density, self.interpolation_stencil, left_shift = self.left_shift, axis = i, p2 = True)
            densitym2p  = interpolation.fixed_stencil_reconstruction(density, self.interpolation_stencil, left_shift = self.left_shift, axis = i, p2 = False)
            densityp2p  = np.roll(densitym2p, fd.ROLL_R, axis = i)
            densitym2m  = np.roll(densityp2m, fd.ROLL_L, axis = i)

            if self.debug:
                plt.title("densityp2m")
                plt.plot(densityp2m)
                plt.show()
                plt.title("densitym2p")
                plt.plot(densitym2p)
                plt.show()

            flux = fd.getLaxFriedrichsFlux

            fp = flux(densityp2m, densityp2p, up2m, up2p, dx, self.vmax)
            fm = flux(densitym2m, densitym2p, um2m, um2p, dx, self.vmax)

            flux = (fp - fm) / self.dx


            if self.debug:
                plt.title("flux")
                plt.plot(flux)
                plt.show()

            ddensity  += flux


        return dt * -np.array([ddensity, dphase])


class ENOScheme(AdvectionScheme):

    def __init__(self, config, generateIC):
        super().__init__(config, generateIC)

        if self.dimension != 1:
            raise ValueError("ENO advection only implemented for 1D case")



        self.interpolation_stencil = interpolation.stencil_ij(self.stencilOrder)

        if self.stencilOrder % 2 == 0:
            self.left_shift = int(self.stencilOrder/2 - 1)
        else:
            self.left_shift = int((self.stencilOrder-1)/2)

        self.vmax = 1

        self.coef    = np.zeros((self.stencilOrder, self.N_ghost), dtype=float)
        self.lpis    = np.zeros((self.stencilOrder, self.N_ghost), dtype=int)


    def getUpdatedFields(self, dt, fields):
        self.setPhase(fields)
        density, phase = fields
        dx = self.dx

        ddensity = np.zeros(density.shape)
        dphase   = np.zeros(phase.shape)

        for i in range(self.dimension):
            shiftx = self.ghostBoundarySize * dx 

            eno_f   = eno.ENO(xx = self.grid[i] + shiftx, y = phase, L = self.boxWidth, order = self.stencilOrder, direction =   1, lpis = self.lpis, coef = self.coef)
            eno_b   = eno.ENO(xx = self.grid[i] + shiftx, y = phase, L = self.boxWidth, order = self.stencilOrder, direction = - 1, lpis = self.lpis, coef = self.coef)
            up2m    = eno_f.dP(x = self.grid[i] + shiftx)
            um2p    = eno_b.dP(x = self.grid[i] + shiftx)
            up2p    = np.roll(up2m, fd.ROLL_R, axis = i)
            um2m    = np.roll(um2p, fd.ROLL_L, axis = i)


            if self.debug:
                plt.title("up2m")
                plt.plot(up2m)
                plt.show()

                plt.title("um2p")
                plt.plot(um2p)
                plt.show()

            eno_f   = eno.ENO(xx = self.grid[i] + shiftx, y = density, L = self.boxWidth + (2*self.ghostBoundarySize*dx), order = self.stencilOrder, direction =   1)#, lpis = self.lpis, coef = self.coef)
            eno_b   = eno.ENO(xx = self.grid[i] + shiftx, y = density, L = self.boxWidth + (2*self.ghostBoundarySize*dx), order = self.stencilOrder, direction = - 1)#, lpis = self.lpis, coef = self.coef)

            densityp2m  = eno_f.P(x = self.grid[i] + shiftx)
            densitym2p  = eno_b.P(x = self.grid[i] + shiftx)
            densityp2p  = np.roll(densitym2p, fd.ROLL_R, axis = i)
            densitym2m  = np.roll(densityp2m, fd.ROLL_L, axis = i)

            if self.debug:
                plt.title("densityp2m")
                plt.plot(densityp2m)
                plt.show()
                plt.title("densitym2p")
                plt.plot(densitym2p)
                plt.show()

            flux = fd.getLaxFriedrichsFlux

            fp = flux(densityp2m, densityp2p, up2m, up2p, dx, self.vmax)
            fm = flux(densitym2m, densitym2p, um2m, um2p, dx, self.vmax)

            flux = (fp - fm) / self.dx


            if self.debug:
                plt.title("flux")
                plt.plot(flux, label="1")
                plt.plot(-flux, label="2")
                plt.legend()
                plt.show()

            ddensity  = flux

        return -dt * np.array([ddensity, dphase])

