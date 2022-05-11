from multiprocessing import Pool
from itertools import repeat
import matplotlib.patches as patches
import matplotlib.pyplot as plt 

import numpy as np
from enum import Enum

import src.fd as fd
import src.schemes as schemes 
import src.wave_schemes as wave_schemes 
import src.phase_schemes as phase_schemes 

wave_threshold = 0.0
splitting_threshold = 0.8
density_threshold = 1e-10
dphase_threshold = 20
interval_length_threshold = 16


class SolverType(Enum):
    NOSOLVER = 0
    WAVESOLVER = 1
    PHASESOLVER = 2


# This is the 2D simulation box:
#   <-------------------------------------> = totalN
#        <------------>                     = noneN
#   p = (2, 4)                              = nodePosition
#
#   0    4    9    14    19    24    29   34
#
# 0  000000000000000000000000000000000000000
# 1  000000000000000000000000000000000000000
# 2  00000pxxxxxxxxxxxxx00000000000000000000
# 3  00000xxxxxxxxxxxxxx00000000000000000000
# 4  00000xxxxxxxxxxxxxx00000000000000000000
# 5  00000xxxxxxxxxxxxxx00000000000000000000
# 6  00000xxxxxxxxxxxxxx00000000000000000000
# 7  00000xxxxxxxxxxxxxx00000000000000000000
# 8  000000000000000000000000000000000000000
# 9  000000000000000000000000000000000000000
# 10 000000000000000000000000000000000000000
# 11 000000000000000000000000000000000000000
# 12 000000000000000000000000000000000000000
# 13 000000000000000000000000000000000000000
# 14 000000000000000000000000000000000000000


class Node:
    def __init__(self, level, totalN, nodePosition, nodeN, dimension, createSolver):
        self.level = level
        self.children = []
        self.nodePosition = nodePosition
        self.nodeN = nodeN
        self.totalN = totalN
        self.dimension = dimension
        self.createSolver = createSolver
        self.subregion = createSolver(self, SolverType.NOSOLVER)
        if dimension < 3:
            self.minlevel = 1
        else:
            self.minlevel = 2

    def prange(self, low, high):
        if (low > high):
            low -= self.totalN
        return np.array(range(low, high)) % self.totalN

    def checkSplit(self, fields):
        sub = np.ix_(*[self.prange(self.nodePosition[i], self.nodePosition[i] + self.nodeN) for i in range(self.dimension)])

        density = fields[0][sub]
        dphase  = fields[1][sub]

        cond = np.logical_or((density < density_threshold),
                             (dphase  > dphase_threshold))
        return np.sum(cond)/self.nodeN**self.dimension

    def update(self, fields):
        # Check splitting condition
        ratio = self.checkSplit(fields)

        # If the level interference is below the threshold, we do not need subdivision or wave solver
        # For the sake of multiprocessing we still split it up
        if (ratio <= wave_threshold) and self.level >= self.minlevel:
            self.children = []
            if self.subregion.id != SolverType.PHASESOLVER:
                self.subregion = self.createSolver(
                    self, SolverType.PHASESOLVER)
            return

        # Else proceed to turn on wave solver and subdivide node
        half_length = int(self.nodeN/2)

        # Do not subdivide if resulting nodes are too small or most of the node requires wave solver
        if ((half_length < interval_length_threshold) or (ratio >= splitting_threshold)) and self.level >= self.minlevel:
            self.children = []
            if self.subregion.id != SolverType.WAVESOLVER:
                self.subregion = self.createSolver(self, SolverType.WAVESOLVER)
            return

        # If the node requires splitting and does not already have children, add children
        if not self.children:
            self.subregion = self.createSolver(self, SolverType.NOSOLVER)
            positions = []
            if self.dimension == 1:
                positions.append(self.nodePosition)
                positions.append(self.nodePosition + half_length)
            elif self.dimension == 2:
                positions.append(np.array([self.nodePosition[0],                self.nodePosition[1]]))
                positions.append(np.array([self.nodePosition[0],                self.nodePosition[1] + half_length]))
                positions.append(np.array([self.nodePosition[0] + half_length,  self.nodePosition[1]]))
                positions.append(np.array([self.nodePosition[0] + half_length,  self.nodePosition[1] + half_length]))
            elif self.dimension == 3:
                positions.append(np.array([self.nodePosition[0],                  self.nodePosition[1],                   self.nodePosition[2]]))
                positions.append(np.array([self.nodePosition[0],                  self.nodePosition[1] + half_length,     self.nodePosition[2]]))
                positions.append(np.array([self.nodePosition[0] + half_length,    self.nodePosition[1],                   self.nodePosition[2]]))
                positions.append(np.array([self.nodePosition[0] + half_length,    self.nodePosition[1] + half_length,     self.nodePosition[2]]))
                positions.append(np.array([self.nodePosition[0],                  self.nodePosition[1],                   self.nodePosition[2] + half_length]))
                positions.append(np.array([self.nodePosition[0],                  self.nodePosition[1] + half_length,     self.nodePosition[2] + half_length]))
                positions.append(np.array([self.nodePosition[0] + half_length,    self.nodePosition[1],                   self.nodePosition[2] + half_length]))
                positions.append(np.array([self.nodePosition[0] + half_length,    self.nodePosition[1] + half_length,     self.nodePosition[2] + half_length]))
            else:
                raise ValueError()

            for position in positions:
                self.children.append(Node(
                    self.level + 1, self.totalN, position, half_length, self.dimension, self.createSolver))

        # Update children
        for child in self.children:
            child.update(fields)

    def getSolvers(self, solvers=[]):
        if self.subregion.id != SolverType.NOSOLVER:
            solvers.append(self.subregion)

        for child in self.children:
            child.getSolvers(solvers)

    def getSubregions(self, subregions=[]):
        if self.subregion.id != SolverType.NOSOLVER:
            if self.children:
                raise RuntimeError("Node has subregion and children")
            subregions.append((self.nodePosition, self.nodeN,
                              self.subregion.id == SolverType.WAVESOLVER))

        for child in self.children:
            child.getSubregions(subregions)

    def getWaveVolumeFraction(self):
        totalFraction = 0

        if self.subregion.id == SolverType.WAVESOLVER:
            if self.children:
                raise RuntimeError("Node has subregion and children")
            totalFraction += (self.nodeN / self.totalN)**self.dimension
            
        for child in self.children:
            totalFraction += child.getWaveVolumeFraction()

        return totalFraction


# Print the Tree
    def plotTree(self, ax, root=False):
        if root:
            plt.title(f"Binary tree in {self.dimension}D")

            if self.dimension == 1:
                plt.xlabel("N")
                plt.ylabel("# levels")
            elif self.dimension == 2:
                ax.set_aspect("equal")
                ax.set_ylim([-10, self.nodeN + 10])
                ax.set_xlim([-10, self.nodeN + 10])

        for child in self.children:
            child.plotTree(ax)

        if self.dimension == 1:
            xx = np.arange(self.nodePosition, self.nodePosition + self.nodeN)
            if (self.subregion.id == SolverType.WAVESOLVER):
                ax.scatter(xx, self.level *
                           np.ones(xx.shape[0]), c="r", alpha=0.3)
            else:
                ax.scatter(xx, self.level *
                           np.ones(xx.shape[0]), c="k", alpha=0.3)

        elif self.dimension == 2 and len(self.children) == 0:
            if (self.subregion.id == SolverType.WAVESOLVER):
                c = "r"
            else:
                c = "k"

            ax.add_patch(patches.Rectangle(self.nodePosition, self.nodeN, self.nodeN,
                                           edgecolor=c,
                                           facecolor=c,
                                           fill=True,
                                           alpha=0.2))


class SubregionScheme:
    def __init__(self, dx, eta, dimension, subregionPosition, totalN, subregionN, stencilOrder, timeOrder, f1_stencil, f1_coeff, b1_stencil, b1_coeff, c1_stencil, c1_coeff, c2_stencil, c2_coeff):
        self.dx = dx
        self.eta = eta
        self.dimension = dimension
        self.subregionN = subregionN
        self.totalN = totalN
        self.ghostBoundarySize = timeOrder * stencilOrder
        self.boundaries = []

        # self.boundaries contains indices of subarrays that need updating at different levels of RK scheme from perspective of subregion itself
        for i in range(2 * timeOrder + 1):
            boundaryColumns = np.arange(
                i * stencilOrder, self.subregionN + 2 * self.ghostBoundarySize - i * stencilOrder)
            self.boundaries.append(
                np.ix_(*[boundaryColumns for j in range(self.dimension)]))


        boundaryColumns = np.arange(
            stencilOrder, self.subregionN + self.ghostBoundarySize - stencilOrder)
            
        self.debug_boundary = np.ix_(*[boundaryColumns for j in range(self.dimension)])

        self.f1_stencil, self.f1_coeff, self.b1_stencil, self.b1_coeff, self.c1_stencil, self.c1_coeff, self.c2_stencil, self.c2_coeff = f1_stencil, f1_coeff, b1_stencil, b1_coeff, c1_stencil, c1_coeff, c2_stencil, c2_coeff
        self.vmax = 0

        N0 = subregionPosition

        # self.outerFull and self.outerInner contains indices of subregion with and without ghost boundaries from perspective of outer simulation
        self.outerFull  = np.ix_(*[self.prange(N0[i] - self.ghostBoundarySize, N0[i] + subregionN + self.ghostBoundarySize) for i in range(self.dimension)])
        self.outerInner = np.ix_(*[self.prange(N0[i], N0[i] + subregionN) for i in range(self.dimension)])

    def prange(self, low, high):
        if (low > high):
            low -= self.totalN
        return np.array(range(low, high)) % self.totalN


class WaveScheme(SubregionScheme):
    def __init__(self, dx, eta, dimension, position, totalN, subregionN,  stencilOrder, f1_stencil, f1_coeff, b1_stencil, b1_coeff, c1_stencil, c1_coeff, c2_stencil, c2_coeff):
        super().__init__(dx, eta, dimension, position, totalN, subregionN,  stencilOrder, 2,
                         f1_stencil, f1_coeff, b1_stencil, b1_coeff, c1_stencil, c1_coeff, c2_stencil, c2_coeff)
        self.isWaveScheme = True
        self.id = SolverType.WAVESOLVER

    def drift(self, density, phase, dt):
        u0 = np.sqrt(density) * np.exp(1j * phase)
        # Update full array u0 at time t0 with timestep dt
        u1 = u0 + fd.solvePeriodicFTCSDiffusion(
            u0, dx=self.dx, dt=dt * 1.0, coeff=self.c2_coeff, stencil=self.c2_stencil)
        u2 = 0.5 * u0 + 0.5 * u1
        # Update u1 - stencilOrder points at each side in second time step
        # This avoids updating points in ghost boundary that are not required anymore
        u2[self.boundaries[1]] += fd.solvePeriodicFTCSDiffusion(
            u1[self.boundaries[1]], dx=self.dx, dt=dt * 0.5, coeff=self.c2_coeff, stencil=self.c2_stencil)

        # Return u2 - 2*stencilOrder points at each side = u2 - 2*ghostBoundarySize
        new_density = np.abs(u2[self.boundaries[2]])**2
        new_phase = self.getPhase(
            u2[self.boundaries[2]], phase[self.boundaries[2]])

        return new_density, new_phase, 0


    def getPhase(self, psi, phase):
        # Difference above which 2 pi is added
        maxdiff = 5

        sub_theta = np.angle(psi)
        #Check difference between old phase and updated phase at boundary
        diff = (phase - sub_theta)
        sub_theta += 2*np.pi * np.sign(diff) * ((np.abs(diff) // (2 * np.pi)) + ((np.abs(diff) % (2 * np.pi)) > maxdiff))

        return sub_theta

class PhaseScheme(SubregionScheme):
    def __init__(self, dx, eta, dimension, position, totalN, subregionN,  stencilOrder, f1_stencil, f1_coeff, b1_stencil, b1_coeff, c1_stencil, c1_coeff, c2_stencil, c2_coeff):
        super().__init__(dx, eta, dimension, position, totalN, subregionN,  stencilOrder, 2,
                         f1_stencil, f1_coeff, b1_stencil, b1_coeff, c1_stencil, c1_coeff, c2_stencil, c2_coeff)
        self.isWaveScheme = False
        self.id = SolverType.PHASESOLVER
        

    def drift(self, d0, p0, dt):
        dd0, dp0, v0 = phase_schemes.getHODrift(d0, p0, 1.0 * dt, self.dx, self.eta, self.f1_stencil, self.f1_coeff, self.b1_stencil,
                                               self.b1_coeff, self.c1_stencil, self.c1_coeff, self.c2_stencil, self.c2_coeff, self.boundaries[1])
        d1 = d0  + dd0
        p1 = p0  + dp0
        dd1, dp1, v1 = phase_schemes.getHODrift(d1[self.boundaries[1]], p1[self.boundaries[1]], 0.5 * dt, self.dx, self.eta, self.f1_stencil,
                                               self.f1_coeff, self.b1_stencil, self.b1_coeff, self.c1_stencil, self.c1_coeff, self.c2_stencil, self.c2_coeff, self.boundaries[2])

        d2 = 0.5 * d0 + 0.5 * d1
        p2 = 0.5 * p0 + 0.5 * p1
        d2[self.boundaries[1]] += dd1
        p2[self.boundaries[1]] += dp1

        return d2[self.boundaries[2]], p2[self.boundaries[2]], np.maximum(v0, v1)


class NoSolver:
    def __init__(self):
        self.id = SolverType.NOSOLVER


def createSolver(node, solvertype, hybridsolver):
    print("Create solver of type: ", solvertype)

    if (solvertype == SolverType.WAVESOLVER):
        return WaveScheme(
            dx=hybridsolver.dx,
            eta=hybridsolver.eta,
            dimension=hybridsolver.dimension,
            position=node.nodePosition,
            totalN=node.totalN,
            subregionN=node.nodeN,
            stencilOrder=hybridsolver.stencilOrder,
            f1_stencil=hybridsolver.f1_stencil,
            f1_coeff=hybridsolver.f1_coeff,
            b1_stencil=hybridsolver.b1_stencil,
            b1_coeff=hybridsolver.b1_coeff,
            c1_stencil=hybridsolver.c1_stencil,
            c1_coeff=hybridsolver.c1_coeff,
            c2_stencil=hybridsolver.c2_stencil,
            c2_coeff=hybridsolver.c2_coeff
        )
    elif (solvertype == SolverType.PHASESOLVER):
        return PhaseScheme(
            dx=hybridsolver.dx,
            eta=hybridsolver.eta,
            dimension=hybridsolver.dimension,
            position=node.nodePosition,
            totalN=node.totalN,
            subregionN=node.nodeN,
            stencilOrder=hybridsolver.stencilOrder,
            f1_stencil=hybridsolver.f1_stencil,
            f1_coeff=hybridsolver.f1_coeff,
            b1_stencil=hybridsolver.b1_stencil,
            b1_coeff=hybridsolver.b1_coeff,
            c1_stencil=hybridsolver.c1_stencil,
            c1_coeff=hybridsolver.c1_coeff,
            c2_stencil=hybridsolver.c2_stencil,
            c2_coeff=hybridsolver.c2_coeff
        )
    else:
        return NoSolver()




import multiprocessing

def init(a, b, c, d, e):
    global sharedDensity, sharedPhase, sharedDensityBuffer, sharedPhaseBuffer, fieldShape
    sharedDensity = a
    sharedPhase = b
    sharedDensityBuffer = c
    sharedPhaseBuffer = d
    fieldShape = e 

def parallelUpdateSubregion(solver, dt):
    #print("Read buffers")

    density        = np.frombuffer(sharedDensity).reshape(fieldShape)
    phase          = np.frombuffer(sharedPhase).reshape(fieldShape)
    densityBuffer  = np.frombuffer(sharedDensityBuffer).reshape(fieldShape)
    phaseBuffer    = np.frombuffer(sharedPhaseBuffer).reshape(fieldShape)

    #print("Done reading buffers")

    d, p, v = solver.drift(
        density[solver.outerFull], phase[solver.outerFull], dt)

    #print("Writing to buffers")

    densityBuffer[solver.outerInner] = d
    phaseBuffer[solver.outerInner]   = p

    return v


def updateSubregion(solver, dt, outerFields):
    density, phase, vmax = solver.drift(
        outerFields[0][solver.outerFull], outerFields[1][solver.outerFull], dt)

    return ((density, phase), solver.outerInner, vmax)

class HybridScheme(schemes.SchroedingerScheme):
    def __init__(self, c, generateIC):
        super().__init__(c, generateIC)

        if not self.usePeriodicBC:
            raise ValueError("Hybrid scheme only supports periodic BC")

        self.vmax = 0
        self.subGhostBoundarySize = self.stencilOrder * self.timeOrder
        self.treeUpdateFrequency = 10 
        self.treeUpdateCounter = 0

        self.useHybrid = c["useHybrid"]

        if self.nThreads > 1:
            print(f"Enabling multiprocessing via pool with {self.nThreads} processes")

            fieldShape  = self.psi.shape
            fieldLength = len(self.psi.flatten())
            sharedDensity       = multiprocessing.RawArray('d', fieldLength)
            sharedPhase         = multiprocessing.RawArray('d', fieldLength)
            sharedDensityBuffer = multiprocessing.RawArray('d', fieldLength)
            sharedPhaseBuffer   = multiprocessing.RawArray('d', fieldLength)

            self.density          = np.frombuffer(sharedDensity).reshape(fieldShape)
            self.phase            = np.frombuffer(sharedPhase).reshape(fieldShape)
            self.densityBuffer    = np.frombuffer(sharedDensityBuffer).reshape(fieldShape)
            self.phaseBuffer      = np.frombuffer(sharedPhaseBuffer).reshape(fieldShape)

            np.copyto(self.density, np.abs(self.psi) ** 2)
            np.copyto(self.phase, fd.make_continuous(np.angle(self.psi)))
            self.pool = Pool(self.nThreads, initializer = init, initargs=(sharedDensity, sharedPhase, sharedDensityBuffer, sharedPhaseBuffer, fieldShape))
        else:
            self.density = np.abs(self.psi) ** 2
            self.phase   = fd.make_continuous(np.angle(self.psi))

        print(f"Set up nD binary tree with N = {self.innerN}")

        self.binaryTree = Node(
            level=0,
            totalN=self.innerN,
            nodePosition=np.zeros(self.dimension, dtype=int),
            nodeN=self.innerN,
            dimension=self.dimension,
            createSolver=lambda node, solvertype: createSolver(
                node, solvertype, self)
        )

        self.subregions = []
        self.updateTree()

    def updateTree(self):
        self.binaryTree.update([self.density, np.abs(fd.deltaSi(self.density, self.phase, self.dx))])
        self.subregions = []
        self.binaryTree.getSolvers(self.subregions)

    def step(self, dt):
        if self.outputTimestep:
            print(f"t = {self.t} dt = {dt} min density = {np.min(self.density)} max density = {np.max(self.density)}")

        if self.useHybrid:
            self.treeUpdateCounter += 1
            if self.treeUpdateCounter % self.treeUpdateFrequency == 0:
                self.updateTree()
                self.treeUpdateCounter = 0

        # kick by dt/2
        self.phase -= dt/2 * self.potential

        if self.nThreads > 1:
            velocities = self.pool.starmap(parallelUpdateSubregion, zip(self.subregions, repeat(dt)))
            self.vmax = np.max(velocities)
            np.copyto(self.density, self.densityBuffer)
            np.copyto(self.phase  , self.phaseBuffer)
        else:
            updatedSubregions = [updateSubregion(
                sub, dt, self.fields) for sub in self.subregions]

            for fields, outerInner, vmax in updatedSubregions:
                self.vmax = np.maximum(vmax, self.vmax)
                self.fields[0][outerInner], self.fields[1][outerInner] = fields


        # update potential
        self.potential = self.computePotential(self.density)

        # kick by dt/2
        self.phase -= dt/2 * self.potential

        self.t += dt * self.getScaleFactor()**2

    def getTimeStep(self):
        #if self.vmax == 0:
        #    self.vmax = 1/self.dx
        #v = (self.vmax + 1/self.dx)*self.dimension
        #dt1 = .4 * self.dx / v #This is the time step for Osher sethian (1 >= 2 (dt/dx) |H_x| + 2 * (dt/dx) |H_y| + ...)
        #return dt1 

        self.vmaxp = self.vmax
        self.vmax = 0
        self.amax = 0

        for i in range(self.dimension):
            pc  = self.phase
            pp  = np.roll(pc, fd.ROLL_R, axis = i)
            pm  = np.roll(pc, fd.ROLL_L, axis = i)

            self.vmax += np.max(np.abs((pp - pm)/(2*self.dx)))
            self.amax = np.maximum(np.max(np.abs((pp - 2*pc + pm)/(self.dx**2))), self.amax)

        #dt < 0.25 * m/hbar * dx**2 from whistler-type waves
        #dt <= 1/2 * dx *(|H_x| + |H_y| + ...)^(-1) #Use |H_i| = |v_i|
        #dt <= 1/2 * dx *(sum_i |v_i|)
        t1 = 1/6 * self.eta*self.dx*self.dx
        t2 = 0.5 * self.dx/(self.vmax + 1e-8)
        t3 = 0.4 * (self.dx/(self.amax + 1e-8))**0.5
        #print("old vmax = ", self.vmaxp, " new vmax: ", self.vmax, " amax: ", self.amax, "Advection: ", t2, " Diffusion: ", t1, " Acceleration: ", t3)
        return np.min([t1, t2, t3])
        
    def getDensity(self):
        return self.density

    def getPhase(self):
        return self.phase

    def setDensity(self, density):
        self.density = density

    def setPhase(self, phase):
        self.phase = phase 

    def getName(self):
        return "hybrid scheme"
