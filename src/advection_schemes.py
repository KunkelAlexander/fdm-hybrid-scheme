import numpy as np
import matplotlib.pyplot as plt

import src.schemes as schemes
import src.fd as fd
import src.interpolation as interpolation 
import src.eno as eno

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
        self.vmax = 1 
            
        return self.cfl*self.dx/(0.1 + self.vmax)

    def setBoundaryConditions(self, fields):
        rho, phase = self.generateIC(*self.grid, self.dx, self.t)
        
        fields[0][self.boundary] = rho[self.boundary]
        fields[1] = phase


    def setPhase(self, fields):
        rho, phase = self.generateIC(*self.grid, self.dx, self.t)
        fields[1] = phase




class CenteredDifferenceScheme(AdvectionScheme):

    def __init__(self, config, generateIC):
        super().__init__(config, generateIC)


    def getUpdatedFields(self, dt, fields):
        self.setPhase(fields)
        uc, pc = fields
        dx = self.dx 

        du   = np.zeros(uc.shape)
        dpc  = np.zeros(pc.shape)

        for i in range(self.dimension):
            pr    = np.roll(pc, fd.ROLL_R, axis=i)
            pl    = np.roll(pc, fd.ROLL_L, axis=i)
            ur    = np.roll(uc, fd.ROLL_R, axis=i)
            ul    = np.roll(uc, fd.ROLL_L, axis=i)
            vc    = (pr - pl) / (2*dx)
        
            du += vc* (ur - ul)/(2*dx)

        return -dt * np.array([du, dpc])

    def getName(self):
        return "centered difference scheme"



class UpwindScheme(AdvectionScheme):

    def __init__(self, config, generateIC):
        super().__init__(config, generateIC)


    def getUpdatedFields(self, dt, fields):
        self.setPhase(fields)
        uc, pc = fields
        dx = self.dx 

        du   = np.zeros(uc.shape)
        dpc  = np.zeros(pc.shape)

        for i in range(self.dimension):
            pr    = np.roll(pc, fd.ROLL_R, axis=i)
            pl    = np.roll(pc, fd.ROLL_L, axis=i)
            ur    = np.roll(uc, fd.ROLL_R, axis=i)
            ul    = np.roll(uc, fd.ROLL_L, axis=i)
            vc    = (pr - pl) / (2*dx)
        
            du +=  np.maximum(vc, 0) * (uc - ul)/dx + np.minimum(vc, 0)  * (ur - uc)/dx

        return -dt * np.array([du, dpc])

    def getName(self):
        return "upwind scheme"


class SecondOrderUpwindScheme(AdvectionScheme):

    def __init__(self, config, generateIC):
        super().__init__(config, generateIC)

    def getUpdatedFields(self, dt, fields):
        self.setPhase(fields)
        uc, pc = fields
        dx = self.dx 

        du   = np.zeros(uc.shape)
        dpc  = np.zeros(pc.shape)

        for i in range(self.dimension):
            pr    = np.roll(pc, fd.ROLL_R, axis=i)
            pl    = np.roll(pc, fd.ROLL_L, axis=i)
            vc    = (pr - pl) / (2*dx)

            uf = fd.getF2Gradient(uc, dx, axis=i)
            ub = fd.getB2Gradient(uc, dx, axis=i)
        
            du += np.maximum(vc, 0) * ub + np.minimum(vc, 0) * uf

        return -dt * np.array([du, dpc])

    def getName(self):
        return "second-order upwind scheme"


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

    def getName(self):
        return "conor-cell scheme"

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

    def getName(self):
        return "lax-wendroff scheme"

class SOLimiterScheme(AdvectionScheme):
    def __init__(self, config, generateIC):
        super().__init__(config, generateIC)
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

    def getName(self):
        return "second-order upwind scheme with limiter"

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

    def getName(self):
        return "muscl scheme"

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

    def getName(self):
        return "lax-friedrichs scheme"

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


    def getName(self):
        return "eno scheme"


### Implement piecewise parabolic method for advection scheme (Colella & Woodward, 1984 )
### ( https://doi.org/10.1016/0021-9991(84)90143-8 )
###
### Approximate density profile rho(xi) in each zone as 
### a(xi) = a_L + x(d_a + a_6 ( 1 - x ))
###       where x = (xi - xi_p)/dxi
###
### 

class PPMScheme(AdvectionScheme):

    def __init__(self, config, generateIC):
        super().__init__(config, generateIC)#
        self.eta1 = 20
        self.eta2 = 0.05
        self.epsilon = 0.01 

        self.fix1 = config["fix1"]
        self.fix2 = config["fix2"]

    def getUpdatedFields(self, dt, fields):
        self.setPhase(fields)
        density, phase = fields
        dxi = self.dx
        eta1    = self.eta1
        eta2    = self.eta2
        epsilon = self.epsilon 

        ddensity = np.zeros(density.shape)
        dphase   = np.zeros(phase.shape)

        for i in range(self.dimension):

            ### Face-centered velocities 
            #v_i-1/2
            vm2 = fd.getDerivative(phase, dxi, self.b1_stencil, self.b1_coeff, axis = i)
            #v_i+1/2
            vp2 = np.roll(vm2, fd.ROLL_R, axis=i)
            #v_i
            v   = (vm2 + vp2)/2

            v   = np.ones(v.shape)

            if self.debug:
                plt.title("v")
                plt.plot(v)
                plt.show()

            ### Density cell-averages
            #rho_i
            a   = density 
            #rho_i+1
            ap  = np.roll(a,     fd.ROLL_R) 
            #rho_i+2
            app = np.roll(a, 2 * fd.ROLL_R) 
            #rho_i-1
            am  = np.roll(a,     fd.ROLL_L)
            #rho_i-2
            amm = np.roll(a, 2 * fd.ROLL_L)
            

            # Average slope of parabola
            delta_a = 1/2 * (ap - am)
            delta_m = np.minimum(np.abs(delta_a), 2 * np.minimum(np.abs(a - am), np.abs(ap - a))) * np.sign(delta_a)

            if self.debug:
                plt.title("delta_a")
                plt.plot(delta_a)
                plt.show()
                plt.title("delta_m")
                plt.plot(delta_m)
                plt.show()
                plt.title("delta_a - delta_m")
                plt.plot(delta_a - delta_m)
                plt.show()

            cond          = ((ap - a) * ( a - am )) <= 0
            delta_m[cond] = 0

            delta_mm = np.roll(delta_m, fd.ROLL_L)
            delta_mp = np.roll(delta_m, fd.ROLL_R)

            ### Face-centered density approximations obtained via parabolic interpolation
            ### Yields continuous approximation to density 
            #rho_i+1/2
            #ap2 = 7/12 * ( a + ap ) - 1/12 * ( app + am )
            ap2 = a + 1/2 * ( ap - a ) - 1/6 * (delta_mp - delta_m)
            #rho_i-1/2
            am2 = np.roll(ap2, fd.ROLL_L) 

            ### Face-centered density approximations that take into account monotonicity
            ### Potentially discontinuous, shock-resolving approximation to density
            a_R = ap2
            a_L = am2

            if self.fix1: 
                ### Switch to different interpolation if we detect discontinuities in a 

                # Second derivative as measure for discontinuities
                d2_a  = 1/(6*dxi**2) * (ap - 2*a + am)
                d2_ap = np.roll(d2_a, fd.ROLL_R)
                d2_am = np.roll(d2_a, fd.ROLL_L)

                eta_bar = - ( (d2_ap - d2_am ) / ( 2 * dxi) ) * ( 2*dxi**3 / (ap - am) )

                cond1 = (-d2_ap * d2_am <= 0)
                cond2 = (np.abs(ap - am) - epsilon * np.minimum(np.abs(ap), np.abs(am)) <= 0) 

                eta_bar[cond1] = 0
                eta_bar[cond2] = 0 

                eta = np.maximum(0, np.minimum(eta1 * (eta_bar - eta2), 1))


                a_Ld = am + 1/2 * delta_mm
                a_Rd = ap - 1/2 * delta_mp

                a_L = a_L * ( 1 - eta ) + a_Ld * eta
                a_R = a_R * ( 1 - eta ) + a_Rd * eta

            if self.fix2: 
                ### Set coefficients of the interpolating parabola such that it does not overshoot
                
                # 1. If a is local extremum, set the interpolation function to be constant
                cond = ((a_R - a)*(a - a_L) <= 0) # cond == True <-> a is local extremum
                a_L[cond] = a[cond]
                a_R[cond] = a[cond]
                
                # 2. If a between a_R and a_L, but very close to them, the parabola might still overshoot
                cond = + (a_R - a_L)**2 / 6 < (a_R - a_L) * (a - 1/2 * (a_L + a_R))
                a_L[cond] = 3 * a[cond] - 2 * a_R[cond]
                cond = - (a_R - a_L)**2 / 6 > (a_R - a_L) * (a - 1/2 * (a_L + a_R))
                a_R[cond] = 3 * a[cond] - 2 * a_L[cond]


            ### Free parameters in approximation polynomial 
            ### a(xi) = a_L + x(d_a + a_6 ( 1 - x ))
            ### where x = (xi - xi_p)/dxi
            d_a =                  a_R - a_L 
            a_6 = 6 * (a - 1/2 * ( a_R + a_L))

            a_Lp = np.roll(a_L, fd.ROLL_R) 
            d_ap = np.roll(d_a, fd.ROLL_R)
            a_6p = np.roll(a_6, fd.ROLL_R)

            ### Compute density fluxes at i+1/2 as seen by cells centered at i (fp_R) and i + 1 (fp_L)
            y =  v * dt
            x =  y / dxi
            fp_L = a_R  - x/2 * (d_a  - ( 1 - 2/3 * x) * a_6 )

            y = -v * dt
            x =  y / dxi
            fp_R = a_Lp + x/2 * (d_ap + ( 1 - 2/3 * x) * a_6p)

            if self.debug:
                plt.title("fp_L")
                plt.plot(fp_L)
                plt.show()
                plt.title("fp_R")
                plt.plot(fp_R)
                plt.show()

            ### Enforce upwinding for density fluxes abar
            a_bar_p2 = np.zeros(fp_R.shape)

            a_bar_p2 = fp_L * ( v >= 0) + fp_R * ( v < 0 )

            a_bar_m2 = np.roll(a_bar_p2, fd.ROLL_L)


            if self.debug:
                plt.title("abar_p2")
                plt.plot(a_bar_p2)
                plt.show()
                plt.title("abar_m2")
                plt.plot(a_bar_m2)
                plt.show()
            
            ddensity = v / dxi * (a_bar_p2 - a_bar_m2)

        return -dt * np.array([ddensity, dphase])


    def getName(self):
        return "ppm scheme"