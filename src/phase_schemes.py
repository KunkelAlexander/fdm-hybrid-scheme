
import numpy as np

import src.fd as fd
import src.schemes as schemes

""" DEFINE SIMULATION """

#Evolve density and phase
class PhaseScheme(schemes.SchroedingerScheme):
    def __init__(self, config, generateIC):
        super().__init__(config, generateIC)

        self.fields    = np.zeros((2, *self.psi.shape))
        self.fields[0] = np.abs(self.psi) ** 2 * self.m
        self.fields[1] = fd.make_continuous(np.angle(self.psi))
        
        self.turnOffConvection = config["turnOffConvection"]
        self.turnOffDiffusion  = config["turnOffDiffusion"]
        self.friction          = config["friction"]
        self.C_velocity        = config["C_velocity"]
        self.C_acceleration    = config["C_acceleration"]

    def getDensity(self):
        return self.fields[0] / self.m

    def getPhase(self):
        return self.fields[1]

    def kick1(self, fields, dt):
        #kick by dt/2
        fields[1] -= dt/2.0 * self.potential
        return fields

    def kick2(self, fields, dt):
        #update potential
        self.potential = self.computePotential(fields[0])

        #kick by dt/2
        fields[1] -= dt/2.0 * self.potential
        return fields

    #Only required if we do not use periodic boundary conditions
    def setBoundaryConditions(self, fields):
        psi = self.generateIC(*self.grid, self.dx, self.t, self.m, self.hbar)
        fields[0][self.boundary] = (np.abs(psi)**2)[self.boundary] * self.m
        fields[1][self.boundary] = np.angle(psi)[self.boundary]
        #fields[1] = fd.make_boundary_continuous(fields[1])

    def getAdaptiveTimeStep(self):
        # Combination of 
        # CFL-condition for advection: dt < CFL * dx / (sum |v_i|)
        # CFL-condition for diffusion: dt < CFL * hbar/m * dx^2
        # CFL-condition based on acceleration for n-body methods
        # CFL-condition for gravitational methods
        #t1 = 0.125  * self.eta*self.dx*self.dx
        #t2 = .5 * 0.5 * self.dx/(self.dimension*(self.vmax + 1e-8))
        #t3 = .5 * 0.4 * (self.dx/(self.amax + 1e-8))**0.5

        self.vmax = 0
        self.amax = 0

        for i in range(self.dimension):
            pc  = self.fields[1]
            pp  = np.roll(pc, fd.ROLL_R, axis = i)
            pm  = np.roll(pc, fd.ROLL_L, axis = i)

            self.vmax = np.maximum(np.max(np.abs((pp - pm)/(2*self.dx))), self.vmax)
            self.amax = np.maximum(np.max(np.abs((pp - 2*pc + pm)/(self.dx**2))), self.amax)

        t1 = self.C_parabolic    * self.dx**2/self.eta
        t2 = self.C_velocity     * self.dx/(2 * self.dimension*(self.vmax + 1e-8)*self.eta)
        t3 = self.C_acceleration * (self.dx/(self.amax + 1e-8))**0.5
        if self.G > 0:
            t4 = self.C_potential    * self.hbar/np.max(np.abs(self.potential) + 1e-8)
        else:
            t4 = 1e4
        
        return np.min([t1, t2, t3, t4])
        

class UpwindScheme(PhaseScheme):
    def __init__(self, config, generateIC):
        super().__init__(config, generateIC)


    def getUpdatedFields(self, dt, fields):
        if self.outputTimestep:
            print(f"min density = {np.min(self.fields[0])} max density = {np.max(self.fields[0])}")

        density, phase = fields
        dx = self.dx

        ddensity = np.zeros(density.shape)
        dphase   = np.zeros(phase.shape)
        self.vmax = 0
        self.amax = 0

        sr = 0.5 * np.log(density)

        ## Classical friction term 
        if self.friction > 0:
            dphase += self.friction * (phase - np.mean(phase))

        for i in range(self.dimension):
            ### ROLL FIELDS ###

            #Phase
            pc  = phase
            pp = np.roll(phase, fd.ROLL_R, axis = i)
            pm = np.roll(phase, fd.ROLL_L, axis = i)


            vp = (pp - pc)/dx 
            vm = np.roll(vp, fd.ROLL_L, axis = i)

            a  = (pp - 2*pc + pm)/dx**2
            self.amax = np.maximum(np.max(np.abs(a)), self.amax)

            #Density 
            r  = density
            rf = np.roll(density, fd.ROLL_R, axis = i)
            rb = np.roll(density, fd.ROLL_L, axis = i)

            #Logarithm of density for quantum pressure
            srp = np.roll(sr, fd.ROLL_R, axis = i)
            srm = np.roll(sr, fd.ROLL_L, axis = i)

            ### COMPUTE UPWIND-FLUX FOR DENSITY ###

            fp = np.maximum(vp, 0) * r  + np.minimum(vp, 0) * rf
            fm = np.maximum(vm, 0) * rb + np.minimum(vm, 0) * r
            
            ddensity += (fp - fm) / dx

            if self.debug:
                D_sr      =  0.5/dx
                D_si      = -0.5/dx 
                F_sr      =  (srp - srm) / (2*dx)
                F_si      =  (vp + vm) /2
                peclet_sr =  F_sr /  D_sr
                peclet_si =  F_si /  D_si
                
                if (np.max(np.abs(peclet_sr)) > 2 or np.max(np.abs(peclet_si)) > 2):
                    print(f"Peclet S_r: max = {np.max(peclet_sr)} min = {np.min(peclet_sr)}, Peclet S_i: max = {np.max(peclet_si)} min = max = {np.min(peclet_si)}")


            ### COMPUTE QUANTUM PRESSURE ###
            if self.turnOffDiffusion is False:
                dphase -= 0.5/dx**2 * (0.25 * (srp - srm)**2 + (srp - 2 * sr + srm))

            ### COMPUTE OSHER-SETHIAN-FLUX ###

            if self.turnOffConvection is False:
                #Compute Osher-Sethian flux for phase
                dphase += (np.minimum(vp, 0)**2 + np.maximum(vm, 0)**2)/2

        return -dt * self.eta * np.array([ddensity, dphase])


    def getName(self):
        return "upwind scheme"

#Get high-order upwind drift by dt
def getHODrift(density, phase, dt, dx, eta, f1_stencil, f1_coeff, b1_stencil, b1_coeff, c1_stencil, c1_coeff, c2_stencil, c2_coeff, inner = None, turnOffConvection = False, turnOffDiffusion = False, friction = 0.0, limiter = schemes.FluxLimiters.VANLEER, limitHJ = True):
    ddensity = np.zeros(density.shape)
    dphase   = np.zeros(phase.shape)
    vmax = 0

    if friction > 0:
        dphase   += friction * (phase - np.mean(phase))

    for i in range(density.ndim):
        ### PHASE ###
        pc  = phase
        pp  = np.roll(phase, fd.ROLL_R, axis = i)
        pm  = np.roll(phase, fd.ROLL_L, axis = i)
        p2m = np.roll(pc, 2 * fd.ROLL_L, axis = i)

        ### DENSITY ###
        rc  = density
        rf  = np.roll(rc,   fd.ROLL_R, axis = i)
        rb  = np.roll(rc,   fd.ROLL_L, axis = i)
        r2b = np.roll(rc, 2*fd.ROLL_L, axis = i)

        ### COMPUTE CELL-FACE VELOCITIES ###
        vp = (pp - pc)/dx 
        vm = np.roll(vp, fd.ROLL_L, axis = i)

        if inner is not None:
            vmax = np.maximum(vmax, np.max(np.maximum(vp,vm)[inner]))
        else:
            vmax = np.maximum(vmax, np.max(np.maximum(vp,vm)[inner]))

        ### COMPUTE UPWIND-FLUX FOR DENSITY ###
        qm = ((rb - r2b) * (vm >= 0) + (rf  - rc) * (vm <= 0)) / ((rc - rb) + ((rc - rb) == 0) * 1e-8)

        #Store output of limiter for debugging
        limiterqm = limiter(qm)

        ### COMPUTE CELL-FACE DENSITY FLUXES ###
        fm  = (np.maximum(vm, 0) * rb + np.minimum(vm, 0) * rc) + 1 / 2 * np.abs(vm) *  (1 - np.abs(vm * dt/dx)) * limiterqm * (rc - rb)
        fp  = np.roll(fm, fd.ROLL_R, axis = i)
        
        if limitHJ:
            #Slope-limited second order gradients (useful if phase develops discontinuities, could happen in hybrid scheme with 2 pi jump)
            ql = (pc-pm)/((pm-p2m) + (((pm-p2m)==0) * 1e-8))
            qc = np.roll(ql, fd.ROLL_R, axis = i)
            Qc = 1/(qc + (qc == 0) * 1e-8)
            Qr = np.roll(Qc, fd.ROLL_R, axis = i)
            
            #Slope-limited upwind gradient of density
            vm = 1/dx * ( 1 + 0.5 * limiter(qc) - 0.5 * limiter(ql)/(ql + (ql==0) * 1e-8)) * (pc - pm)
            
            #Slope-limited downwind gradient of density
            vp = 1/dx * ( 1 + 0.5 * limiter(Qc) - 0.5 * limiter(Qr)/(Qr + (Qr==0) * 1e-8)) * (pp - pc)
        else:
            #Arbitrary order gradients (no slope limiting)
            vm = fd.getDerivative(pc, dx, b1_stencil, b1_coeff, axis = i)
            vp = fd.getDerivative(pc, dx, f1_stencil, f1_coeff, axis = i)


        ddensity += (fp - fm) / dx


        if not turnOffConvection:
            dphase   += (np.minimum(vp, 0)**2 + np.maximum(vm, 0)**2)/2

    if not turnOffDiffusion:
        dphase  += fd.getHOSqrtQuantumPressure(density, dx, c1_stencil, c1_coeff, c2_stencil, c2_coeff)

    return - dt * eta * ddensity, - dt * eta * dphase, vmax


class HOUpwindScheme(PhaseScheme):
    def __init__(self, config, generateIC):
        super().__init__(config, generateIC)
        self.limitHJ = config["limitHJ"]

    def getUpdatedFields(self, dt, fields):
        ddensity, dphase, vmax = getHODrift(fields[0], fields[1], dt, self.dx, self.eta, self.f1_stencil, self.f1_coeff, self.b1_stencil, self.b1_coeff, self.c1_stencil, self.c1_coeff, self.c2_stencil, self.c2_coeff, turnOffConvection = self.turnOffConvection, turnOffDiffusion = self.turnOffDiffusion, friction=self.friction, limiter = self.limiter, limitHJ = self.limitHJ)
        return np.array([ddensity, dphase])

    def getName(self):
        return "phase scheme"


class LaxWendroffUpwindScheme(PhaseScheme):

    def __init__(self, config, generateIC):
        super().__init__(config, generateIC)
        
        if self.stencilOrder % 2 == 0:
            self.left_shift = int(self.stencilOrder/2 - 1)
        else:
            self.left_shift = int((self.stencilOrder-1)/2)

        self.f1_stencil, self.f1_coeff  = fd.getFiniteDifferenceCoefficients(derivative_order = 1, accuracy = self.stencilOrder, mode = fd.MODE_FORWARD)
        self.b1_stencil, self.b1_coeff  = fd.getFiniteDifferenceCoefficients(derivative_order = 1, accuracy = self.stencilOrder, mode = fd.MODE_BACKWARD)
        
        print(self.f1_stencil, self.f1_coeff, self.b1_stencil, self.b1_coeff)


    def getUpdatedFields(self, dt, fields):
        density, phase = self.fields
        dx = self.dx

        ddensity = np.zeros(density.shape)
        dphase   = fd.getQuantumPressure(density, dx)


        ddensity2 = np.zeros(density.shape)
        dphase2   = dphase.copy()

        src = 0.5 * np.log(density)


        for i in range(self.dimension):
            ### ROLL FIELDS ###

            #Phase
            pc  = phase
            pp = np.roll(pc, fd.ROLL_R, axis = i)
            pm = np.roll(pc, fd.ROLL_L, axis = i)

            #Density 
            rc  = density
            rf = np.roll(rc, fd.ROLL_R, axis = i)
            rb = np.roll(rc, fd.ROLL_L, axis = i)

            #Logarithm of density for quantum pressure
            srp = np.roll(src, fd.ROLL_R, axis = i)
            srm = np.roll(src, fd.ROLL_L, axis = i)


            ### COMPUTE CELL-CENTERED-VELOCITIES ###
            vc  = (pp - pm)/(2*dx)
            vp  = np.roll(vc, fd.ROLL_R, axis = i)
            vm  = np.roll(vc, fd.ROLL_L, axis = i)

            ### COMPUTE CELL-FACE-VELOCITIES ###
            vp2 = (pp - pc)/dx 
            vm2 = (pc - pm)/dx
            
            self.vmax = np.maximum(self.vmax, np.max(np.abs(vc)))

            ### LAX-WENDROFF UPWINDING FOR DENSITY ###
            ddensity += + 1  / (2*dx) * (vp * rf - vm * rb)\
                       - dt / (2*dx**2) * (vp2 * (vp * rf - vc * rc) - vm2 * (vc * rc - vm * rb))

            vp4 = fd.getDerivative(pc, dx, self.f1_stencil, self.f1_coeff, axis = i, derivative_order=1)
            vm4 = fd.getDerivative(pc, dx, self.b1_stencil, self.b1_coeff, axis = i, derivative_order=1)


            ### COMPUTE QUANTUM PRESSURE ###
            #dphase -= 0.5/dx**2 * (0.25 * (srp - srm)**2 + (srp - 2 * src + srm))

            ### COMPUTE OSHER-SETHIAN-FLUX ###
            dphase += (np.minimum(vp4, 0)**2 + np.maximum(vm4, 0)**2)/2


        #c = np.logical_or((np.abs(dphase) > 10), np.abs(ddensity) > 10)
#
        ##if (np.sum(c) > 10):
        ##    self.cs.append(np.packbits(c, axis=None))
#
        #for i in range(self.dimension):
        #    #Phase
        #    pc  = phase
        #    pp = np.roll(pc, fd.ROLL_R, axis = i)
        #    pm = np.roll(pc, fd.ROLL_L, axis = i)
 #
        #    #Density 
        #    rc  = density
        #    rf = np.roll(density, fd.ROLL_R, axis = i)
        #    rb = np.roll(density, fd.ROLL_L, axis = i)
 #
        #    #Logarithm of density for quantum pressure
        #    srp = np.roll(src, fd.ROLL_R, axis = i)
        #    srm = np.roll(src, fd.ROLL_L, axis = i)
 #
        #    ### COMPUTE CELL-FACE-VELOCITIES ###
        #    vp = (pp - pc)/dx 
        #    vm = (pc - pm)/dx 
 #
        #    ### COMPUTE UPWIND-FLUX FOR DENSITY ###
        #    fp = np.maximum(vp, 0) * rc + np.minimum(vp, 0) * rf
        #    fm = np.maximum(vm, 0) * rb + np.minimum(vm, 0) * rc
        #    ddensity2 += (fp - fm) / dx
 #
        #    ### COMPUTE QUANTUM PRESSURE ###
        #    dphase2 -= 0.5/dx**2 * (0.25 * (srp - srm)**2 + (srp - 2 * src + srm))
 #
        #    ### COMPUTE SETHIAN-OSHER-FLUX ###
        #    dphase2 += (np.minimum(vp, 0)**2 + np.maximum(vm, 0)**2)/2
#
#
        #ddensity[c] = ddensity2[c]
        #dphase[c] = dphase2[c]

        return -dt * self.eta * np.array([ddensity, dphase])


#Evolve Sr = 0.5 * log(density) and Si = phase
class ConvectiveScheme(PhaseScheme):
    def __init__(self, config, generateIC):
        super().__init__(config, generateIC)

        self.fields[0] = 0.5 * np.log(self.fields[0])
        
    def getDensity(self):
        return np.exp(2 * self.fields[0])

    def setDensity(self, density):
        self.fields[0] = 0.5 * np.log(density)

    def setBoundaryConditions(self, fields):
        super().setBoundaryConditions(fields)
        fields[0][self.boundary] = 0.5 * np.log(fields[0][self.boundary])
    

    def kick2(self, fields, dt):
        #update potential
        self.potential = self.computePotential(np.exp(2 * fields[0]))

        #kick by dt/2
        fields[1] -= dt/2 * self.potential
        return field

class FTCSConvectiveScheme(ConvectiveScheme):
    def __init__(self, config, generateIC):
        super().__init__(config, generateIC)


    def getUpdatedFields(self, dt, fields):
        sr, si = fields
        dx = self.dx

        dsr     = np.zeros(self.psi.shape)
        dsi     = np.zeros(self.psi.shape)

        for i in range(self.dimension):
            vrc          = fd.getDerivative(sr, dx, self.c1_stencil, self.c1_coeff, axis = i, derivative_order=1)
            vic          = fd.getDerivative(si, dx, self.c1_stencil, self.c1_coeff, axis = i, derivative_order=1)
            laplacian_sr = fd.getDerivative(sr, dx, self.c2_stencil, self.c2_coeff, axis = i, derivative_order=2)
            laplacian_si = fd.getDerivative(si, dx, self.c2_stencil, self.c2_coeff, axis = i, derivative_order=2)

            dsr += (
                + 0.5 * laplacian_si
                + 1.0 * vic * vrc
            )

            dsi += (
                - 0.5 * laplacian_sr
                + 1.0 * vic * vic
                - 0.5 * (vic**2 + vrc**2)
            )

        dfields = -np.array([dsr, dsi])

        return self.eta * dfields * dt

class DonorCellConvectionScheme(ConvectiveScheme):
    def __init__(self, config, generateIC):
        super().__init__(config, generateIC)

    def getUpdatedFields(self, fields):
        sr, si = fields
        dx = self.dx
        density = np.exp(2 * sr)

        dsi    = np.zeros(self.psi.shape)
        dsr    = np.zeros(self.psi.shape)
        ddensity = np.zeros(self.psi.shape)
        dphase   = np.zeros(self.psi.shape)
        vr     = np.zeros(self.psi.shape)
        vi     = np.zeros(self.psi.shape)
        lapsr  = np.zeros(self.psi.shape)
        lapsi  = np.zeros(self.psi.shape)
        vrt    = np.zeros(self.psi.shape)
        vit    = np.zeros(self.psi.shape)

        dsi   = np.zeros(density.shape)


        for i in range(self.dimension):
            sir = np.roll(si, fd.ROLL_R, axis=i)
            sil = np.roll(si, fd.ROLL_L, axis=i)
            srr = np.roll(sr, fd.ROLL_R, axis=i)
            srl = np.roll(sr, fd.ROLL_L, axis=i)
            vrc = (srr - srl) / (2 * dx)
            vic = (sir - sil) / (2 * dx)
            vrf = (srr - sr) / (dx)
            vif = (sir - si) / (dx)
            vrb = (sr - srl) / (dx)
            vib = (si - sil) / (dx)
            lapsr = (srr - 2 * sr + srl) / dx ** 2
            lapsi = (sir - 2 * si + sil) / dx ** 2

            D_sr      =  0.5/dx
            D_si      = -0.5/dx 
            F_sr      =  vrc
            F_si      =  vic 
            
            peclet_sr =  F_sr /  D_sr
            peclet_si =  F_si /  D_si
            

            if (np.max(np.abs(peclet_sr)) > 2 or np.max(np.abs(peclet_si)) > 2):
                print(f"Peclet S_r: max = {np.max(peclet_sr)} min = {np.min(peclet_sr)}, Peclet S_i: max = {np.max(peclet_si)} min = max = {np.min(peclet_si)}")

            v1 = (np.abs(peclet_sr) <= 2) * vrc + (peclet_sr > 2) * vrf + (peclet_sr < -2)  * vrb
            v2 = (np.abs(peclet_si) <= 2) * vic + (peclet_si > 2) * vif + (peclet_si < -2)  * vib

            peclet_sir = peclet_si * peclet_sr
            v3 = (np.abs(peclet_sir) <= 4) * vrc * vic  + (peclet_sir > 4) * vrb * vib + (peclet_sir < -4) * vrf * vif
            convection_sr = v3
            convection_si = v2 * v2


            dsr += (
                + 0.5 * lapsi
                + 1.0 * convection_sr
            )
            dsi += (
                - 0.5 * lapsr
                + 1.0 * convection_si
                - 0.5 * (vrc ** 2 + vic ** 2)
            )

        dfields = -np.array([dsr, dsi])

        return self.eta * dfields

class ConvectiveScheme(ConvectiveScheme):
    def __init__(self, config, generateIC):
        super().__init__(config, generateIC)
        self.turnOffConvection      = config["turnOffConvection"]
        self.turnOffDiffusion       = config["turnOffDiffusion"]
        self.turnOffSource          = config["turnOffSource"]

    def getUpdatedFields(self, dt, fields):
        sr, si = fields
        dx = self.dx
        density = np.exp(2 * sr)
        self.vmax = 0

        dsi      = np.zeros(self.psi.shape)
        dsr      = np.zeros(self.psi.shape)
        ddensity = np.zeros(self.psi.shape)
        dphase   = np.zeros(self.psi.shape)
        vr       = np.zeros(self.psi.shape)
        vi       = np.zeros(self.psi.shape)
        lapsr    = np.zeros(self.psi.shape)
        lapsi    = np.zeros(self.psi.shape)
        vrt      = np.zeros(self.psi.shape)
        vit      = np.zeros(self.psi.shape)


        for i in range(self.dimension):
            vrc   = fd.getDerivative(sr, dx, self.c1_stencil, self.c1_coeff, axis = i, derivative_order=1)
            vrf   = fd.getDerivative(sr, dx, self.f1_stencil, self.f1_coeff, axis = i, derivative_order=1)
            vrb   = fd.getDerivative(sr, dx, self.b1_stencil, self.b1_coeff, axis = i, derivative_order=1)
            vic   = fd.getDerivative(si, dx, self.c1_stencil, self.c1_coeff, axis = i, derivative_order=1)
            vif   = fd.getDerivative(si, dx, self.f1_stencil, self.f1_coeff, axis = i, derivative_order=1)
            vib   = fd.getDerivative(si, dx, self.b1_stencil, self.b1_coeff, axis = i, derivative_order=1)
            lapsr = fd.getDerivative(sr, dx, self.c2_stencil, self.c2_coeff, axis = i, derivative_order=2)
            lapsi = fd.getDerivative(si, dx, self.c2_stencil, self.c2_coeff, axis = i, derivative_order=2)

            #D_sr      =  0.5/dx
            #D_si      = -0.5/dx 
            #F_sr      =  vrc
            #F_si      =  vic 
            #peclet_sr =  F_sr /  D_sr
            #peclet_si =  F_si /  D_si
            #Ff_sr     =  vrf 
            #Fb_sr     =  vrb 
            #Ff_si     =  vif 
            #Fb_si     =  vib 
#
            #if (np.max(np.abs(peclet_sr)) > 2 or np.max(np.abs(peclet_si)) > 2):
            #    print(f"Peclet S_r: max = {np.max(peclet_sr)} min = {np.min(peclet_sr)}, Peclet S_i: max = {np.max(peclet_si)} min = max = {np.min(peclet_si)}")


            if not self.turnOffDiffusion:
                dsr +=   0.5 * lapsi
                dsi += - 0.5 * lapsr

            if not self.turnOffConvection:
                dsr += 1.0 * vi * vr
                dsi += 1.0 * vi * vi

            if not self.turnOffSource:
                dsi += - 0.5 * (vi ** 2 + vr ** 2)

            self.vmax = np.maximum(self.vmax, np.max(np.abs(vi)))


        dfields = -np.array([dsr, dsi])

        return self.eta * dfields * dt
