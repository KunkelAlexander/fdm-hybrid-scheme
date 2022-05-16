
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt

import src.cosmology as cosmology
import src.fd as fd


class FluxLimiters(Enum):
    #First-order linear
    DONORCELL   = lambda r: 0
    #Second-order linear
    BEAMWARMING = lambda r: r
    LAXWENDROFF = lambda r: 1
    #Second-order nonlinear
    MINMOD      = lambda r: np.maximum(0, np.minimum(1, r))
    SUPERBEE    = lambda r: np.maximum(0, np.maximum(np.minimum(2*r, 1), np.minimum(r, 2)))
    MC          = lambda r: np.maximum(0, np.minimum(np.minimum((1 + r) / 2, 2), 2 * r))
    #Smooth limiters
    VANLEER     = lambda r: (r + np.abs(r))/(1 + np.abs(r))
    VANALBADA   = lambda r: (r**2 + r)/(1+r**2)
    #Family of limiters parametrised by a,
    ALFA        = lambda r, a: np.maximum(0, np.minimum(2*r, np.minimum(a*r+1-a, 2)))
    #Third order limiter of ALFA family (SMART) for a = 0.75
    SMART       = lambda r: np.maximum(0, np.minimum(2*r, np.minimum(.75*r+.25, 4)))
    #CFL-dependent (s = dt/dx) limiters
    ULTRABEE    = lambda r, s: np.maximum(0, np.maximum(np.minimum(2*r/s, 1),np.minimum(r, 2/(1-s))))
    ULTRAALFA   = lambda r, s, a: np.maximum(0, np.minimum(2*r/s, np.minimum(a*(r-1) + 1, 2/(1-s))))

fluxLimiterDictionary = {
    "DONORCELL"   : FluxLimiters.DONORCELL,   
    "BEAMWARMING" : FluxLimiters.BEAMWARMING,
    "LAXWENDROFF" : FluxLimiters.LAXWENDROFF,
    "MINMOD"      : FluxLimiters.MINMOD,     
    "VANALBADA"   : FluxLimiters.VANALBADA,     
    "SUPERBEE"    : FluxLimiters.SUPERBEE, 
    "MC"          : FluxLimiters.MC, 
    "VANLEER"     : FluxLimiters.VANLEER,
    "SMART"       : FluxLimiters.SMART
}

class Scheme:
    def __init__(self, config, generateIC):

        print(f"Constructing {self.getName()} scheme")

        self.config              = config

        # Set up time management
        self.t0                  = config["t0"]
        self.t                   = config["t0"]
        self.dt                  = config["dt"]
        self.tEnd                = config["tEnd"]
        self.useAdaptiveTimestep = config["useAdaptiveTimestep"]
        self.outputTimestep      = config["outputTimestep"]
        self.timeOrder           = config["timeOrder"]
        self.cfl                 = config["cfl"]
        self.ntmax               = config["maximumNumberOfTimesteps"]



        # Set up grid (evenly-spaced with or without ghost boundary)
        self.dimension           = config["dimension"]
        self.stencilOrder        = config["stencilOrder"] 

        # Store forward, backward and centered finite differences for convenience
        if self.stencilOrder % 2 == 0:
            self.left_shift = int(self.stencilOrder/2 - 1)
        else:
            self.left_shift = int((self.stencilOrder-1)/2)

        self.f1_stencil, self.f1_coeff  = fd.getFiniteDifferenceCoefficients(derivative_order = 1, accuracy = self.stencilOrder, mode = fd.MODE_FORWARD)
        self.b1_stencil, self.b1_coeff  = fd.getFiniteDifferenceCoefficients(derivative_order = 1, accuracy = self.stencilOrder, mode = fd.MODE_BACKWARD)
        self.c1_stencil, self.c1_coeff  = fd.getFiniteDifferenceCoefficients(derivative_order = 1, accuracy = self.stencilOrder + self.stencilOrder % 2, mode = fd.MODE_CENTERED)
        self.c2_stencil, self.c2_coeff  = fd.getFiniteDifferenceCoefficients(derivative_order = 2, accuracy = self.stencilOrder + self.stencilOrder % 2, mode = fd.MODE_CENTERED)

        self.limiter            = fluxLimiterDictionary[config["fluxLimiter"]]

        #Since we use np.roll for finite differences there is no need for a ghost boundary with periodic boundary conditions
        self.usePeriodicBC = config["usePeriodicBC"]
        if self.usePeriodicBC:
            self.ghostBoundarySize = 0
        else:
            self.ghostBoundarySize = self.timeOrder * self.stencilOrder


        # Set up simulation grid
        N              = config["resolution"]
        boxWidth       = config["domainSize"]
        self.boxWidth  = boxWidth
        self.innerN   = N
        self.dx        = boxWidth / N


        #Handle ghost boundary
        self.totalN  = self.innerN + 2 * self.ghostBoundarySize
        N            = self.totalN
        Ll           = -self.dx * self.ghostBoundarySize
        Lh           = self.boxWidth + self.dx * self.ghostBoundarySize

        #Create 1D grid
        xlin = np.linspace(Ll, Lh, num=N + 1)  # Note: x=0 & x=1 are the same point!
        xlin = xlin[0:N]  # chop off periodic point)

        self.boundaryColumns = np.concatenate([np.arange(self.ghostBoundarySize), \
                                            np.arange(self.innerN + self.ghostBoundarySize, self.innerN + 2 * self.ghostBoundarySize)])
        self.innerColumns    = np.arange(self.ghostBoundarySize, self.innerN + self.ghostBoundarySize)

        #Construct higher-dimensional grid 
        if self.dimension == 1:
            self.grid = [xlin]
            self.boundary = np.ix_(self.boundaryColumns)
            self.inner    = np.ix_(self.innerColumns)

        elif self.dimension == 2:
            self.grid = np.meshgrid(xlin, xlin)
            self.boundary = np.ix_(self.boundaryColumns, self.boundaryColumns)
            self.inner    = np.ix_(self.innerColumns, self.innerColumns)

        elif self.dimension == 3:
            self.grid = np.meshgrid(xlin, xlin, xlin, indexing='ij')
            self.boundary = np.ix_(self.boundaryColumns, self.boundaryColumns, self.boundaryColumns)
            self.inner    = np.ix_(self.innerColumns, self.innerColumns, self.innerColumns)

        else:
            raise ValueError("Dimension above 3 not supported")


        self.debug = config["debug"]
        self.generateIC = generateIC

        

    def getGrid(self):
        return self.grid

    def getTime(self):
        return self.t

    def getTimeStep(self): 
        if self.useAdaptiveTimestep:
            return self.getAdaptiveTimeStep()
        else:
            return self.dt 

    def getAdaptiveTimeStep(self):
        raise NotImplementedError("Please Implement this method")

    def getConfig(self):
        return self.config

    def getUpdatedFields(self, dt, fields):
        raise NotImplementedError("Please Implement this method")

    def setBoundaryConditions(self, fields):
        raise NotImplementedError("Please Implement this method")

    def getScaleFactor(self):
        return 1

    def run(self, tfin = None, enableBackward = False):
        if tfin is None:
            tfin = self.tEnd
        i = 0
        while(self.t < tfin - 1e-15):
            dt = self.getTimeStep()

            if (tfin - self.t < dt):
                dt = tfin - self.t

            self.step(dt)
            i += 1
            if i > self.ntmax:
                print("Maximum number of timesteps reached. Aborting.")
                break
        
        if enableBackward:
            while(tfin < self.t):
                dt = self.getTimeStep()

                if (self.t - tfin < dt):
                    dt = self.t - tfin

                self.step(-dt)
                i += 1
                if i > self.ntmax:
                    print("Maximum number of timesteps reached. Aborting.")
                    break
        print(f"Finished in {i} time steps")

    #Implement first to fourth order TVD-RK integrator by default
    #Can be overwritten in children classes to implement different time integration
    def step(self, dt):
        if not self.usePeriodicBC:
            self.setBoundaryConditions(self.fields)

        un = self.kick1(self.fields, dt)

        if self.outputTimestep:
            print(f"t = {self.t:.4f} dt = {dt:.4f} a = {self.getScaleFactor():.4f} ")

        if self.timeOrder == 1:
            un = un + 1 / 1 * self.getUpdatedFields(dt, un)

        elif self.timeOrder == 2:
            u1 = un + self.getUpdatedFields(dt, un)
            un = 1 / 2 * un + 1 / 2 * u1 + self.getUpdatedFields(1/2 * dt, u1)
            #u1 = un + self.getUpdatedFields(0.5 * dt, un)
            #un = un + self.getUpdatedFields(1.0 * dt, u1)

        elif self.timeOrder == 3:
            u1 = un + self.getUpdatedFields(dt, un)
            u2 = 3 / 4 * un + 1 / 4 * u1 +  self.getUpdatedFields(1/4 * dt, u1)
            un = 1 / 3 * un + 2 / 3 * u2 + self.getUpdatedFields(2/3 * dt, u2)

        elif self.timeOrder == 4:
            u1 = un + self.getUpdatedFields(0.39175222700392 * dt, un)
            u2 = (
                0.44437049406734 * un
                + 0.55562950593266 * u1
                + self.getUpdatedFields(0.36841059262959 * dt, u1)
            )
            u3 = (
                0.62010185138540 * un
                + 0.37989814861460 * u2
                + self.getUpdatedFields(0.25189177424738 * dt, u2)
            )
            u4 = (
                0.17807995410773 * un
                + 0.82192004589227 * u3
                + self.getUpdatedFields(0.54497475021237 * dt, u3)
            )
            un = (
                0.00683325884039 * un
                + 0.51723167208978 * u2
                + 0.12759831133288 * u3
                + self.getUpdatedFields(0.08460416338212 * dt, u3)
                + 0.34833675773694 * u4
                + self.getUpdatedFields(0.22600748319395 * dt, u4)
            )
        else:
            raise ValueError("Invalid time order")

        self.fields = self.kick2(un, dt)


        self.t += dt * self.getScaleFactor() ** 2

    #Dummy function for implementation of first kick in kick-drift-kick scheme
    def kick1(self, fields, dt):
        return fields

    #Dummy function for implementation of second kick in kick-drift-kick scheme
    #Here the gravitational potential should be updated
    def kick2(self, fields, dt):
        return fields

    def getName(self):
        raise NotImplementedError("Please Implement this method")

        
#Define complex wave function psi as well as cosmology and gravity
class SchroedingerScheme(Scheme):
    def __init__(self, config, generateIC):
        super().__init__(config, generateIC)

        self.hbar = config["hbar"]
        self.m    = config["m"]
        self.eta = self.hbar / self.m
        print(f"hbar/m is {self.eta}")

        self.psi = self.generateIC(*self.grid, self.dx, self.t, self.m, self.hbar)
        self.potential = np.zeros(self.psi.shape, dtype=np.float128)

        # Set up global parameters and constants
        self.G            = config["gravity"]
        self.useCosmology = config["useCosmology"]
        self.useHybrid = False


        self.C_potential       = config["C_potential"]
        self.C_parabolic       = config["C_parabolic"]


        if (self.G == 0) and self.useCosmology:
            raise ValueError(
                "Gravity required in expanding universe! Set config[Gravity] != 0."
            )

        if (self.G != 0) and not self.usePeriodicBC:
            raise ValueError(
                "Gravity only supported for periodic boundary conditions."
            )

        # Set up Fourier Space Variables for computing gravitational potential
        klin = 2.0 * np.pi / self.boxWidth * np.arange(-self.totalN / 2, self.totalN / 2)

        if self.dimension == 1:
            self.kx = np.fft.ifftshift(klin)
            self.momentumGrid = [self.kx]
            self.kSq = self.kx ** 2

        elif self.dimension == 2:
            kx, ky = np.meshgrid(klin, klin)
            self.kx, self.ky = np.fft.ifftshift(kx), np.fft.ifftshift(ky)
            self.momentumGrid = [self.kx, self.ky]
            self.kSq = self.kx ** 2 + self.ky ** 2

        elif self.dimension == 3:
            kx, ky, kz = np.meshgrid(klin, klin, klin, indexing='ij')
            self.kx, self.ky, self.kz = np.fft.ifftshift(kx), np.fft.ifftshift(ky), np.fft.ifftshift(kz)
            self.momentumGrid = [self.kx, self.ky, self.kz]
            self.kSq = self.kx ** 2 + self.ky ** 2 + self.kz ** 2

        else:
            raise ValueError("Dimension above 3 not supported")

        self.externalPotential = None 

        self.nThreads = config["nThreads"]
        if self.nThreads > 1:
            self.workers = self.nThreads # Tell scipy fft how many threads to use
        else:
            self.workers = None

    def setExternalPotentialFunction(self, potentialFunction):
        self.externalPotential = potentialFunction

    def computePotential(self, density):
        if np.isnan(density).any():
            print("Density array in computePotential contained nan")
            self.t = self.tEnd 
            return 

        V = np.zeros(density.shape)

        if self.G != 0:
            V += 1/self.eta * fd.computePotential(
                density, self.G * self.getScaleFactor(), self.kSq, self.workers
            )

        if self.externalPotential is not None:
            V += 1/self.hbar * self.externalPotential(*self.grid, self.m)

        self.potential = V

        return V 

    def getPotential(self):
        return self.potential

    def getScaleFactor(self):
        if self.useCosmology:
            return cosmology.getScaleFactor(self.t)
        else:
            return 1

    def getPsi(self):
        return self.psi 

    def setPsi(self, psi):
        self.psi = psi 
        if self.G != 0:
            self.potential = self.computePotential(np.abs(self.psi)**2)
