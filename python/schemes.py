import cosmology
import fv 
import fd
import fd_1d
import fd_2d
import numpy as np
import tree
import interpolation 
import config as configuration
import integration 
import multiprocessing
import animation 

from enum import Enum
import matplotlib.pyplot as plt


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
    SMART       = lambda r: np.maximum(0, np.minimum(2*r, np.minimum(.75*r+.25, 2)))
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
    "VANLEER"     : FluxLimiters.VANLEER
}

class Scheme:
    def __init__(self, config, generateIC):

        print("Constructing scheme")

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

        print("Setting up grid")

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

    def run(self, tfin = None):
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

    #Implement first to fourth order TVD-RK integrator by default
    #Can be overwritten in children classes to implement different time integration
    def step(self, dt):
        if not self.usePeriodicBC:
            self.setBoundaryConditions(self.fields)

        un = self.fields

        if self.outputTimestep:
            print(f"t = {self.t} dt = {dt} ")

        if self.timeOrder == 1:
            self.fields = un + 1 / 1 * self.getUpdatedFields(dt, un)

        elif self.timeOrder == 2:
            u1 = un + self.getUpdatedFields(dt, un)
            self.fields = 1 / 2 * un + 1 / 2 * u1 + self.getUpdatedFields(1/2 * dt, u1)

        elif self.timeOrder == 3:
            u1 = un + 1 / 1 * dt * self.getUpdatedFields(dt, un)
            u2 = 3 / 4 * un + 1 / 4 * u1 + self.getUpdatedFields( 1/4 * dt, u1)
            self.fields = 1 / 3 * un + 2 / 3 * u2 + self.getUpdatedFields(2/3 * dt, u2)

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
            self.fields = (
                0.00683325884039 * un
                + 0.51723167208978 * u2
                + 0.12759831133288 * u3
                + self.getUpdatedFields(0.08460416338212 * dt, u3)
                + 0.34833675773694 * u4
                + self.getUpdatedFields(0.22600748319395 * dt, u4)
            )
        else:
            raise ValueError("Invalid time order")


        self.t += dt * self.getScaleFactor() ** 2

    def getName(self):
        raise NotImplementedError("Please Implement this method")

        
#Define complex wave function psi as well as cosmology and gravity
class SchroedingerScheme(Scheme):
    def __init__(self, config, generateIC):
        super().__init__(config, generateIC)

        print("Reading in initial conditions")
        self.psi = self.generateIC(*self.grid, self.dx, self.t)
        self.potential = np.zeros(self.psi.shape, dtype=np.float128)
        self.eta = 1 #hbar / m

        # Set up global parameters and constants
        self.G            = config["gravity"]
        self.useCosmology = config["useCosmology"]
        self.useHybrid = False

        if (self.G == 0) and self.useCosmology:
            raise ValueError(
                "Gravity required in expanding universe! Set config[Gravity] != 0."
            )

        if (self.G != 0) and not self.usePeriodicBC:
            raise ValueError(
                "Gravity only supported for periodic boundary conditions."
            )

        print("Setting up fourier grid")
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
            print("Density array in compute potential contained nan")
            self.t = self.tEnd 
            return 

        V = np.zeros(density.shape)

        if self.G != 0:
            V += 1/self.eta * fd.computePotential(
                density, self.G * self.getScaleFactor(), self.kSq, self.workers
            )

        if self.externalPotential is not None:
            V += self.externalPotential(*self.grid)

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

class SubregionScheme:
    def __init__(self, position, size, N, grid, density, phase, V, dt, dx, debug):
        self.dt, self.dx, self.debug = dt, dx, debug
        self.innerN = N
        self.position = position
        self.size = size
        self.dimension = len(grid)
        self.grid = grid
        self.debug = debug
        self.secondOrder = True

        if self.debug:
            print(f"Pos {self.position}, size {self.size}")

        if self.dimension == 1:
            self.N_low = int(position / self.dx) % N
            self.N_high = int((position + size) / self.dx) % N
        elif self.dimension == 2:
            self.N_low = (position / self.dx).astype(int) % N
            self.N_high = ((position + size) / self.dx).astype(int) % N
        else:
            raise ValueError()

        self.sub_N = self.N_high - self.N_low

        if self.debug:
            print(f"N low  {self.N_low}, N high {self.N_high}, sub N {self.sub_N}")

        if self.dimension == 1:
            if self.N_low > self.N_high:
                self.sub_N += N
        elif self.dimension == 2:
            oob = self.N_low > self.N_high
            self.sub_N[oob] += N

        self.create_boundaries(self.N_low, self.N_high)

        # Compute psi and V in region
        if self.dimension == 1:
            xx = self.grid[0]
            self.sub_xx = xx[self.subx_]
            if debug:
                print("len sub xx", len(self.sub_xx))
            sub_density = density[self.subx_]
            sub_phase = phase[self.subx_]
            self.sub_V = V[self.subx_].copy()
        elif self.dimension == 2:
            xx, yy = self.grid
            self.sub_xx = xx[self.suby_, self.subx_]
            self.sub_yy = yy[self.suby_, self.subx_]
            sub_density = density[self.suby_, self.subx_]
            sub_phase = phase[self.suby_, self.subx_]
            self.sub_V = V[self.suby_, self.subx_].copy()

        self.sub_psi = np.sqrt(sub_density) * np.exp(1j * sub_phase)

        if debug:
            print("N_low", self.N_low, "N_high", self.N_high, "sub_N", self.sub_N)

        if self.dimension == 1:
            self.kin_A, self.kin_b = fd_1d.createKineticLaplacian(
                self.sub_N, self.dt, self.dx
            )
            self.D2 = fd_1d.createPotentialLaplacian(self.sub_N, self.dx)
        elif self.dimension == 2:
            self.kin_A_row, self.kin_b_row = fd.createKineticLaplacian(
                self.sub_N[1], self.dt, self.dx
            )
            self.kin_A_col, self.kin_b_col = fd.createKineticLaplacian(
                self.sub_N[0], self.dt, self.dx
            )
            self.D2_row = fd_2d.createPotentialLaplacian(self.sub_N[1], self.dx)
            self.D2_col = fd_2d.createPotentialLaplacian(self.sub_N[0], self.dx)

    def create_boundaries(self, N_low, N_high):
        if self.dimension == 1:
            self.subx_ = self.prange(N_low, N_high).flatten()
            self.subsubx_ = np.delete(self.subx_, [0, -1], axis=0)
            self.upbx_ = (N_high - 1) % self.innerN
            self.lobx_ = N_low % self.innerN
        elif self.dimension == 2:
            nly, nlx = N_low
            nhy, nhx = N_high
            self.suby_ = self.prange(nly, nhy).reshape(-1, 1)
            self.subx_ = self.prange(nlx, nhx).reshape(1, -1)
            self.subsubx_ = np.delete(self.subx_, [0, -1], axis=1)
            self.subsuby_ = np.delete(self.suby_, [0, -1], axis=0)
            self.upby_ = N_low[0] % self.innerN
            self.loby_ = (N_high[0] - 1) % self.innerN
            self.leby_ = self.prange(nly, nhy)
            self.riby_ = self.prange(nly, nhy)
            self.upbx_ = self.prange(nlx, nhx)
            self.lobx_ = self.prange(nlx, nhx)
            self.lebx_ = N_low[1] % self.innerN
            self.ribx_ = (N_high[1] - 1) % self.innerN

    def prange(self, lo, hi):
        if lo > hi:
            lo -= self.innerN
        return np.array(range(lo, hi)) % self.innerN

    def get_boundaries(self, f):
        if self.dimension == 1:
            return f[self.lobx_], f[self.upbx_]

        elif self.dimension == 2:
            leb = f[self.leby_, self.lebx_].copy()
            upb = f[self.upby_, self.upbx_].copy()
            lob = f[self.loby_, self.lobx_].copy()
            rib = f[self.riby_, self.ribx_].copy()
            return leb, upb, lob, rib

    def get_updated_region(self):
        if self.dimension == 1:
            return self.subsubx_

        elif self.dimension == 2:
            return self.subsuby_, self.subsubx_

        else:
            raise ValueError()


    def set_subregion_to_zero(self, f):
        if self.dimension == 1:
            f[self.subsubx_] = 0

        elif self.dimension == 2:
            f[self.subsuby_, self.subsubx_] = 0

        else:
            raise ValueError()

    def set_subregion(self, density, phase):
        if self.dimension == 1:
            # Update subregion using wave formulation
            density[self.subsubx_] = self.getDensity()[1:-1]
            phase[self.subsubx_] = self.getPhase()[1:-1]

        elif self.dimension == 2:
            # Update subregion using wave formulation
            density[self.subsuby_, self.subsubx_] = self.getDensity()[1:-1, 1:-1]
            phase[self.subsuby_, self.subsubx_] = self.getPhase()[1:-1, 1:-1]

        else:
            raise ValueError()

    def set_boundary_condition(self, density, phase, V):
        psi = np.sqrt(density) * np.exp(1j * phase)

        if self.dimension == 1:
            self.leb, self.rib = self.get_boundaries(psi)
            self.lebP, self.ribP = self.get_boundaries(phase)
            self.lebV, self.ribV = self.get_boundaries(V)
        elif self.dimension == 2:
            self.leb, self.upb, self.lob, self.rib = self.get_boundaries(psi)
            self.lebP, self.upbP, self.lobP, self.ribP = self.get_boundaries(phase)
            self.lebV, self.upbV, self.lobV, self.ribV = self.get_boundaries(V)

    def step(self, dt, G):
        dx = self.dx

        # plt.title("1")
        # plt.plot(self.sub_xx, np.angle(self.sub_psi))
        # plt.show()

        k = np.exp(-1.0j * dt / 2.0 * self.sub_V)

        if self.dimension == 1:
            leb = self.leb * k[0]
            rib = self.rib * k[-1]
        elif self.dimension == 2:
            leb = self.leb * k[:, 0]
            upb = self.upb * k[0, :]
            rib = self.rib * k[:, -1]
            lob = self.lob * k[-1, :]

        # 1/2 Kick
        self.sub_psi = k * self.sub_psi

        # plt.title("2")
        # plt.plot(self.sub_xx, np.angle(self.sub_psi))
        # plt.show()

        if self.dimension == 1:
            if self.secondOrder:
                self.sub_psi = fd_1d.solveDirichletCNDiffusion(
                    psi=self.sub_psi, leb=leb, rib=rib, A=self.kin_A, b=self.kin_b
                )
            else:
                self.sub_psi = fd_1d.solveDirichletFTCSDiffusion(self.sub_psi, leb = leb, rib = rib, dt = dt, dx = dx)

            self.sub_V = fd_1d.computeDirichletPotential(
                (np.abs(self.sub_psi) ** 2), self.lebV, self.ribV, G, self.D2
            )

        elif self.dimension == 2:
            if self.secondOrder:

                self.kin_A_row, self.kin_b_row = fd.createKineticLaplacian(
                    self.sub_N[1], dt, self.dx
                )
                self.kin_A_col, self.kin_b_col = fd.createKineticLaplacian(
                    self.sub_N[0], dt, self.dx
                )

                self.sub_psi = fd_2d.solveDirichletCNDiffusion(
                    self.sub_psi,
                    leb=leb,
                    upb=upb,
                    lob=lob,
                    rib=rib,
                    A_row=self.kin_A_row,
                    b_row=self.kin_b_row,
                    A_col=self.kin_A_col,
                    b_col=self.kin_b_col,
                )
            else:
                self.sub_psi = fd_2d.solveDirichletFTCSDiffusion(self.sub_psi, leb = leb, upb = upb, lob = lob, rib = rib, dt = dt, dx = dx)

            self.sub_V = fd_2d.computeDirichletPotential(
                self.sub_V,
                np.abs(self.sub_psi) ** 2,
                G,
                self.lebV,
                self.upbV,
                self.lobV,
                self.ribV,
                dx,
                self.D2_row,
                self.D2_col,
                10,
            )

        # plt.title("3")
        # plt.plot(self.sub_xx, np.angle(self.sub_psi))
        # plt.show()

        ##1/2 Kick
        self.sub_psi = np.exp(-1.0j * dt / 2.0 * self.sub_V) * self.sub_psi

        # plt.title("4")
        # plt.plot(self.sub_xx, np.angle(self.sub_psi))
        # plt.show()

    def getDensity(self):
        return np.abs(self.sub_psi) ** 2

    def getPhase(self):
        boundaryThickness = 7

        threshold = 100
        # Difference above which 2 pi is added
        max_diff = 5

        if self.dimension == 1:
            sub_theta = fd.make_1d_boundary_continuous(
                f=np.angle(self.sub_psi), boundaryThickness=boundaryThickness
            )

            for k in range(threshold):
                # Diff from outside to inside, if positive add 2 pi to inside
                diff = self.lebP - sub_theta[0]
                if np.abs(diff) < max_diff:
                    break
                sub_theta[:boundaryThickness] += np.sign(diff) * 2 * np.pi

            # plt.title("2")
            # plt.plot(self.sub_xx, sub_theta)
            # plt.show()

            for k in range(threshold):
                # Diff from outside to inside, if positive add 2 pi to inside
                diff = self.ribP - sub_theta[-1]
                if np.abs(diff) < max_diff:
                    break
                sub_theta[-boundaryThickness:] += np.sign(diff) * 2 * np.pi

            # plt.title("3")
            # plt.plot(self.sub_xx, sub_theta)
            # plt.show()

            return sub_theta

        elif self.dimension == 2:
            sub_theta = fd.make_2d_boundary_continuous(
                f=np.angle(self.sub_psi), boundaryThickness=boundaryThickness
            )

            # Maximum number of 2 pis to be added

            # Stitch discontinuous regions together
            for i in range(sub_theta.shape[0]):
                # L2R
                for k in range(threshold):
                    # Diff from outside to inside, if positive add 2 pi to inside
                    diff = self.lebP[i] - sub_theta[i, 0]
                    if np.abs(diff) < max_diff:
                        break
                    sub_theta[i, :boundaryThickness] += np.sign(diff) * 2 * np.pi
                # R2L
                for k in range(threshold):
                    # Diff from outside to inside, if positive add 2 pi to inside
                    diff = self.ribP[i] - sub_theta[i, -1]
                    if np.abs(diff) < max_diff:
                        break
                    sub_theta[i, -boundaryThickness:] += np.sign(diff) * 2 * np.pi
                    # Stitch discontinuous regions together
            for i in range(sub_theta.shape[1]):
                # T2B
                for k in range(threshold):
                    # Diff from outside to inside, if positive add 2 pi to inside
                    diff = self.upbP[i] - sub_theta[0, i]
                    if np.abs(diff) < max_diff:
                        break
                    sub_theta[:boundaryThickness, i] += np.sign(diff) * 2 * np.pi
                # BT2
                for k in range(threshold):
                    # Diff from outside to inside, if positive add 2 pi to inside
                    diff = self.lobP[i] - sub_theta[-1, i]
                    if np.abs(diff) < max_diff:
                        break
                    sub_theta[-boundaryThickness:, i] += np.sign(diff) * 2 * np.pi

            return sub_theta


def treeCallback(node, subregionScheme, phaseScheme):
    if phaseScheme.dimension == 1:
        subregion_size = node.boxWidth * phaseScheme.dx

    elif phaseScheme.dimension == 2:
        subregion_size = np.array([node.boxWidth, node.boxWidth]) * phaseScheme.dx

    subregion_position = node.N0 * phaseScheme.dx

    subregion = SubregionScheme(
        subregion_position,
        subregion_size,
        phaseScheme.innerN,
        phaseScheme.grid,
        phaseScheme.getDensity(),
        phaseScheme.getPhase(),
        phaseScheme.getPotential(),
        phaseScheme.dt,
        phaseScheme.dx,
        phaseScheme.debug,
    )

    return subregion


class HybridScheme(SchroedingerScheme):
    def __init__(self, phaseScheme, subregionScheme, c, generateIC):
        super().__init__(c, generateIC)

        if not self.usePeriodicBC:
            raise ValueError("Hybrid scheme only supports periodic BC")

        self.useHybrid       = True
        self.phaseScheme     = phaseScheme(c, generateIC)
        self.subregionScheme = subregionScheme
        self.subregions      = []

        self.useAdaptiveSubregions = c["useAdaptiveSubregions"]

        if self.useAdaptiveSubregions:
            if self.dimension == 1:
                N0 = 0
            elif self.dimension == 2:
                N0 = np.array([0, 0])

            print(f"Set up nD binary tree with N = {self.innerN}")

            self.binaryTree = tree.Node(
                level=0,
                N=self.innerN,
                N0=N0,
                boxWidth=self.innerN,
                dimension=self.dimension,
                createCallback=lambda node: treeCallback(
                    node, subregionScheme, self.phaseScheme
                ),
            )
        else:
            for subregion in self.c["subregions"]:
                if self.dimension == 1:
                    subregion_size = subregion[0] * self.boxWidth
                    subregion_position = subregion[1] * self.boxWidth

                elif self.dimension == 2:
                    subregion_size = np.array([subregion[0], subregion[1]]) * self.boxWidth
                    subregion_position = np.array([subregion[2], subregion[3]]) * self.boxWidth

                self.subregions.append(
                    self.subregionScheme(
                        subregion_position,
                        subregion_size,
                        self.innerN,
                        self.phaseScheme.grid,
                        self.phaseScheme.getDensity(),
                        self.phaseScheme.getPhase(),
                        self.phaseScheme.getPotential(),
                        self.dt,
                        self.dx,
                        self.debug,
                    )
                )

        #self.pool = multiprocessing.Pool(4)


    def step(self, dt):
        # Aliases to avoid writing self. everywhere
        dx, G = self.dx, self.G * self.getScaleFactor()

        density = self.phaseScheme.getDensity().copy()
        phase   = self.phaseScheme.getPhase().copy()
        V       = self.phaseScheme.getPotential()

        """""" """""" """""" """""" """"""
        """    CHECK TREE          """
        """""" """""" """""" """""" """"""

        if self.useAdaptiveSubregions:
            dsi = np.abs(fd.deltaSi(density, phase, dx))

            self.binaryTree.update([density, dsi])
            self.subregions = []
            self.binaryTree.getCallbacks(self.subregions)

        """""" """""" """""" """""" """"""
        """    OBTAIN IC           """
        """""" """""" """""" """""" """"""

        # Get boundary conditions from outer simulation at time t
        phase_region_to_update = []
        for subregion in self.subregions:
            subregion.set_boundary_condition(density, phase, V)
            phase_region_to_update.append(subregion.get_updated_region())

        """""" """""" """""" """""" """"""
        """ OUTER SIMULATION       """
        """""" """""" """""" """""" """"""

        self.phaseScheme.step(dt)

        if self.debug:
            plt.title("density before sub update")
            plt.imshow(self.phaseScheme.fields[0])
            plt.colorbar()
            plt.show()


            plt.title("phase after outer update")
            plt.imshow(self.phaseScheme.fields[1])
            plt.colorbar()
            plt.show()

        """""" """""" """""" """""" """"""
        """ UPDATE SUBREGIONS      """
        """""" """""" """""" """""" """"""

        for subregion in self.subregions:
            subregion.step(dt, G)

        """""" """""" """""" """""" """"""
        """   UPDATE OUTER         """
        """""" """""" """""" """""" """"""

        for sub in self.subregions:
            sub.set_subregion(self.phaseScheme.getDensity(), self.phaseScheme.getPhase())

        if self.debug:
            plt.title("density after sub update")
            plt.imshow(self.phaseScheme.fields[0])
            plt.colorbar()
            plt.show()

            plt.title("phase after sub update")
            plt.imshow(self.phaseScheme.fields[1])
            plt.colorbar()
            plt.show()


            plt.title("ddensity")
            plt.imshow(self.phaseScheme.fields[0] - density)
            plt.colorbar()
            plt.show()

            plt.title("dphase")
            plt.imshow(self.phaseScheme.fields[1] - phase)
            plt.colorbar()
            plt.show()

        self.t += dt * self.getScaleFactor() ** 2

    def getDensity(self):
        return self.phaseScheme.getDensity()

    def getPhase(self):
        return self.phaseScheme.getPhase()

    def setDensity(self, density):
        self.phaseScheme.setDensity(density)

    def setPhase(self, phase):
        self.phaseScheme.setPhase(phase)

    def getTimeStep(self):
        phase = self.phaseScheme.fields[1]
        self.vmax = np.zeros(phase.shape)

        for i in range(self.dimension):
            v = (np.roll(phase, fd.ROLL_R, axis=i) - np.roll(phase, fd.ROLL_L, axis=i))/(2*self.dx)
            self.vmax = np.maximum(self.vmax, np.abs(v))

        if self.debug:
            plt.title(f"vmax before considering subregions = {np.max(self.vmax)}")
            plt.imshow(self.vmax)
            plt.colorbar()
            plt.show()

        for sub in self.subregions:
            sub.set_subregion_to_zero(self.vmax)

        if self.debug:
            plt.title(f"vmax after considering subregions = {np.max(self.vmax)}")
            plt.imshow(self.vmax)
            plt.colorbar()
            plt.show()

        return self.cfl*self.eta*self.dx*self.dx/(1.0+self.eta*self.dx*np.max(self.vmax)*self.dimension)