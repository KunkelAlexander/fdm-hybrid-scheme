
import numpy as np
import matplotlib.pyplot as plt 

from IPython.display import display, Markdown

import src.wave_schemes as wave_schemes
import src.phase_schemes as phase_schemes
import src.fluid_schemes as fluid_schemes
import src.hybrid_scheme as hybrid_scheme 

import src.tests as tests
import src.config as config
import src.animation as animation 
import src.cosmology as cosmology 


test = tests.standingWave

def getBaseConfig():
    c = config.generateConfig(dt=1e-4, t0=0)
    c["dt"]                  = 1e-4
    c["domainSize"]          = 1
    c["xlim"]                = [0, 1]
    c["densityYlim"]         = [0, 2]
    c["resolution"]          = 128
    c["timeOrder"]           = 1
    c["dimension"]           = 1
    c["debug"]               = False
    c["slowDown"]            = 1
    c["tEnd"]                = 1
    c["outputTimestep"]      = False
    c["useAdaptiveTimestep"] = True
    c["usePeriodicBC"]       = False
    c["gravity"]             = 0
    c["nThreads"]            = 1
    c["fps"] = 10
    c["dpi"] = 80
    c["plotDebug"] = False
    c["m"] = 1
    c["hbar"] = 1
    return c


def ftcsConfig(c):
    c["stencilOrder"] = 2
    c["timeOrder"]    = 2
    c["C_parabolic"]  = 1/6

def hoftcsConfig(c):
    c["stencilOrder"] = 4
    c["timeOrder"]    = 2
    c["C_parabolic"]  = 1/6

def cnConfig(c):
    c["stencilOrder"] = 1
    c["timeOrder"]    = 1

def spectralConfig(c):
    c["usePeriodicBC"] = True


def upwindConfig(c):
    c["stencilOrder"] = 2
    c["timeOrder"]    = 4
    c["C_velocity"] = 0.1

def upwindWithFrictionConfig(c):
    c["stencilOrder"] = 1
    c["timeOrder"]    = 4
    c["friction"]     = 0.1

def upwindWithoutDiffusionConfig(c):
    c["stencilOrder"] = 1
    c["timeOrder"]    = 4
    c["turnOffDiffusion"] = True

def upwindWithoutConvectionConfig(c):
    c["stencilOrder"] = 1
    c["timeOrder"]    = 4
    c["turnOffConvection"] = True 

def hoUpwindConfig(c):
    c["stencilOrder"] = 2
    c["timeOrder"] = 4
    c["slowDown"] = 10

def hoUpwindWithFrictionConfig(c):
    c["stencilOrder"] = 2
    c["timeOrder"] = 4
    c["friction"]  = 0.1

def hoUpwindWithoutDiffusionConfig(c):
    c["stencilOrder"] = 4
    c["timeOrder"] = 2
    c["turnOffDiffusion"] = True

def hoUpwindWithoutConvectionConfig(c):
    c["stencilOrder"] = 4
    c["timeOrder"] = 2
    c["turnOffConvection"] = True 

def lwUpwindConfig(c):
    c["stencilOrder"] = 2
    c["timeOrder"] = 1

def ftcsConvectiveConfig(c):
    c["stencilOrder"] = 4
    c["timeOrder"] = 2

def ftcsConvectiveWithoutSourceConfig(c):
    c["stencilOrder"] = 4
    c["timeOrder"] = 2
    c["turnOffSource"] = True

def ftcsConvectiveWithoutConvectionConfig(c):
    c["stencilOrder"] = 4
    c["timeOrder"] = 2
    c["turnOffConvection"] = True 

def ftcsConvectiveWithoutDiffusionConfig(c):
    c["stencilOrder"] = 4
    c["timeOrder"] = 2
    c["turnOffDiffusion"] = True

def musclHancockConfig(c):
    c["stencilOrder"] = 2
    c["timeOrder"] = 2

def hybridConfig(c):
    c["timeOrder"] = 2
    c["stencilOrder"] = 4
    c["debug"] = False
    c["useAdaptiveSubregions"] = True
    c["outputTimestep"] = True
    c["useAdaptiveTimestep"] = True 
    c["nThreads"] = 2
    c["useHybrid"] = True

def standingWaveConfig(c):
    c["tEnd"] = 6
    c["slowDown"] = 1
    c["resolution"] = 64

def oscillatorConvergenceConfig(c):
    c["domainSize"] = 1
    c["xlim"] = [0, 1]
    c["densityYlim"] = [0, 10]
    c["tEnd"] = 1

def oscillatorEigenstateConfig(c):
    c["domainSize"] = 6
    c["xlim"] = [0, 6]
    c["densityYlim"] = [0, 10]

def oscillatorCoherentStateConfig(c):
    c["domainSize"] = 14
    c["xlim"] = [0, 14]
    c["densityYlim"] = [0, 10]
    c["plotPhaseMod2"] = True
    c["tEnd"] = 2 * np.pi
    c["resolution"] = 64

def fastOscillatorCoherentStateConfig(c):
    c["domainSize"] = 4
    c["xlim"] = [0, 4]
    c["densityYlim"] = [0, 10]
    c["tEnd"] = np.pi

def infiniteWellConfig(c):
    c["domainSize"] = 1
    c["xlim"] = [0, 1]
    c["densityYlim"] = [0, 4]
    c["usePeriodicBC"] = False

def perturbationWaveConfig(c):
    c["usePeriodicBC"] = True
    c["resolution"] = 32
    c["tEnd"] = 1/(2*np.pi*3*0.5)
    c["domainSize"] = 1
    N = 1
    k = 2*np.pi / (N * c["domainSize"])
    eta = c["hbar"]/c["m"]
    omega = 0.5 * eta * k**2
    tEnd = 2*np.pi / omega
    print("tEnd", tEnd)
    c["tEnd"] = tEnd
    #config["tEnd"] = 1/(2*np.pi*3*0.5) , omega = 0.5/Eta * sum k_i^2 t = 2pi/omega

    c["xlim"] = [0, 1]
    c["densityYlim"] = [0.98, 1.02]
    c["phaseYlim"] = [-0.01, 0.01]
    c["slowDown"] = 10/c["tEnd"] 

def stabilityTestConfig(c):
    c["usePeriodicBC"] = True
    c["resolution"] = 64
    c["tEnd"] = 10
    c["domainSize"] = 1
    c["xlim"] = [0, 1]
    c["densityYlim"] = [0, 20]
    c["phaseYlim"] = [-3.14, 3.14]
    c["slowDown"] = 1
    c["fps"] = 1
    c["gravity"] = 1

def accuracyTest1DConfig(c):
    c["usePeriodicBC"] = True
    c["resolution"] = 8
    N = 1
    k = 2*np.pi / (N * c["domainSize"])
    eta = 1
    omega = 0.5/eta * k**2
    tEnd = 2*np.pi / omega
    print("tEnd", tEnd)
    c["tEnd"] = tEnd
    c["domainSize"] = 1
    c["xlim"] = [0, 1]
    c["densityYlim"] = [0.95, 1.05]
    c["phaseYlim"] = [-0.05, 0.05]
    c["slowDown"] = 1
    c["fps"] = 1


def cosmoConfig(c):
    c["usePeriodicBC"] = True
    c["resolution"] = 128
    c["tEnd"] = 100
    c["domainSize"] = 10
    c["xlim"] = [0, 10]
    c["densityYlim"] = [0, 10]
    c["slowDown"] = 1
    c["gravity"] = 1
    c["fps"] = 2

def expandingCosmoConfig(c):
    c["usePeriodicBC"] = True
    c["resolution"] = 512
    c["tEnd"] = 10
    c["domainSize"] = 10
    c["xlim"] = [0, 10]
    c["densityYlim"] = [0, 10]
    c["slowDown"] = 10
    c["gravity"] = 1
    c["useCosmology"] = True

def li1Config(c):
    c["resolution"] = 128
    c["tEnd"] = .25
    c["domainSize"] = 4
    c["xlim"] = [0, 4]
    c["densityYlim"] = [0, 12]
    c["slowDown"] = 20

def periodicLi1Config(c):
    c["resolution"] = 128
    c["tEnd"] = .05
    c["domainSize"] = 4
    c["xlim"] = [0, 4]
    c["densityYlim"] = [0, 12]
    c["slowDown"] = 20

def hubbleExpansionConfig(c):
    c["resolution"] = 512
    c["tEnd"] = .2
    c["domainSize"] = 10
    c["xlim"] = [0, 10]
    c["densityYlim"] = [0, 25]
    c["slowDown"] = 50
    c["plotPhaseMod2"] = False 
    c["phaseYlim"] = [-40, 20]
    c["usePeriodicBC"] = False


def li2Config(c):
    c["t0"] = 0.0025
    c["tEnd"] = 0.0076
    c["resolution"] = 4096
    c["domainSize"] = 20
    c["xlim"] = [8.75, 11.25]
    c["usePeriodicBC"] = True
    c["densityYlim"] = [4.5, 6.5]
    c["phaseYlim"] = [-0.05, 0.05]
    c["slowDown"] = 1000

def li3Config(c):
    c["usePeriodicBC"] = True
    c["resolution"] = 512
    c["tEnd"] = .005
    c["domainSize"] = 1
    c["xlim"] = [0, 1]
    c["densityYlim"] = [0, 30]
    c["plotPhaseMod2"] = False
    c["phaseYlim"] = [-40, 50]
    c["slowDown"] = 2000


def travellingWavePacketConfig(c):
    c["usePeriodicBC"] = False
    c["resolution"] = 512
    c["tEnd"] = .005
    c["domainSize"] = 1
    c["xlim"] = [0, 1]
    c["densityYlim"] = [0, 16]
    c["slowDown"] = 2000


def stabilityTest2DConfig(c):
    c["usePeriodicBC"] = True
    c["dimension"] = 2
    c["resolution"] = 64
    c["tEnd"] = 25
    c["domainSize"] = 25
    c["xlim"] = [0, 25]
    c["densityYlim"] = [0, 20]
    c["phaseYlim"] = [-3.14, 3.14]
    c["slowDown"] = 1
    c["fps"] = 1
    c["gravity"] = 1


def accuracyTest2DConfig(c):
    c["usePeriodicBC"] = True
    c["dimension"] = 2
    c["resolution"] = 8
    N = 2
    k = 2*np.pi / (N * c["domainSize"])
    eta = 1
    omega = 0.5/eta * (2 * k**2)
    tEnd = 2*np.pi / omega
    print("tEnd", tEnd)
    c["tEnd"] = tEnd

    c["domainSize"] = 1
    c["xlim"] = [0, 1]
    c["densityYlim"] = [0.98, 1.02]
    c["phaseYlim"] = [-0.01, 0.01]
    c["plotDensityLogarithm"] = False
    c["slowDown"] = 1
    c["fps"] = 1

def perturbationWave2DConfig(c):
    c["dimension"] = 2
    c["usePeriodicBC"] = True
    c["resolution"] = 32
    N = 1
    k = 2*np.pi / (N * c["domainSize"])
    eta = 1
    omega = 0.5/eta * (2 * k**2)
    tEnd = 2*np.pi / omega
    print("tEnd", tEnd)
    c["tEnd"] = tEnd
    c["domainSize"] = 1
    c["xlim"] = [0, 1]
    c["densityYlim"] = [0.6, 1.4]
    c["slowDown"] = 5/c["tEnd"] 

def cosmo2DConfig(c):
    c["dimension"] = 2
    c["usePeriodicBC"] = True
    c["domainSize"] = 25
    c["resolution"] = 64
    c["tEnd"] = 1.5
    c["slowDown"] = 10
    c["plotPhaseMod2"] = False
    c["phaseYlim"] = [-50, 50]
    c["densityYlim"] = [0, 1]
    c["gravity"] = 1
    c["fps"] = 20

def cosmo2DExpansionConfig(c):
    cosmo2DConfig(c)
    c["useCosmology"] = True
    c["t0"] = cosmology.getTime(a = 0.1)
    c["t"]  = 1
    c["resolution"] = 64

def cosmo2DTestConfig(c):
    c["dimension"] = 2
    c["usePeriodicBC"] = True
    c["domainSize"] = 25
    c["resolution"] = 32
    c["tEnd"] = 1
    c["slowDown"] = 10
    c["plotPhaseMod2"] = False
    c["phaseYlim"] = [-50, 50]
    c["densityYlim"] = [0, 1]
    c["gravity"] = 1
    c["fps"] = 1



def perturbationWave3DConfig(c):
    c["dimension"] = 3
    c["usePeriodicBC"] = True
    c["resolution"] = 16
    N = 1
    k = 2*np.pi / (N * c["domainSize"])
    eta = 1
    omega = 0.5/eta * (3 * k**2)
    tEnd = 2*np.pi / omega
    print("tEnd", tEnd)
    c["tEnd"] = tEnd
    c["domainSize"] = 1
    c["xlim"] = [0, 1]
    c["densityYlim"] = [0.98, 1.02]
    c["phaseYlim"] = [-0.01, 0.01]
    c["slowDown"] = 5/c["tEnd"] 


def cosmo3DTestConfig(c):
    c["dimension"] = 3
    c["usePeriodicBC"] = True
    c["domainSize"] = 8
    c["resolution"] = 16
    c["tEnd"] = 1
    c["slowDown"] = 10
    c["plotPhaseMod2"] = False
    c["phaseYlim"] = [-50, 50]
    c["densityYlim"] = [0, 1]
    c["gravity"] = 1
    c["fps"] = 1


def accuracyTest3DConfig(c):
    c["usePeriodicBC"] = True
    c["dimension"] = 3
    c["resolution"] = 8
    N = 1
    k = 2*np.pi / (N * c["domainSize"])
    eta = 1
    omega = 0.5/eta * (3 * k**2)
    tEnd = 2*np.pi / omega
    print("tEnd", tEnd)
    c["tEnd"] = tEnd

    c["domainSize"] = 1
    c["xlim"] = [0, 1]
    c["densityYlim"] = [0.98, 1.02]
    c["phaseYlim"] = [-0.01, 0.01]
    c["plotDensityLogarithm"] = False
    c["slowDown"] = 1
    c["fps"] = 1

test_list = {
    "standing wave": [tests.standingWave, standingWaveConfig, None],
    "harmonic oscillator convergence": [tests.generate1DUniform, oscillatorConvergenceConfig, lambda x, m: tests.oscillatorPotential1D(x, m, x0 = 0.5)],
    "harmonic oscillator eigenstate": [tests.oscillatorEigenstate1D, oscillatorEigenstateConfig, lambda x, m: tests.oscillatorPotential1D(x, m, x0 = 3)],
    "harmonic oscillator coherent state": [tests.oscillatorCoherentState1D, oscillatorCoherentStateConfig, lambda x, m: tests.oscillatorPotential1D(x, m, x0 = 7)],
    "infinite well": [tests.infiniteWell1D, infiniteWellConfig, None],
    "gaussian wave packet": [lambda x, dx, t, m, hbar: tests.li1(x, dx, t, m, hbar, x0=2), li1Config, None],
    "periodic gaussian wave packet": [lambda x, dx, t, m, hbar: tests.periodicLi1(x, dx, t, m, hbar, x0=2, L = 4), periodicLi1Config, None],
    "hubble expansion": [lambda x, dx, t, m, hbar: tests.li1(x, dx, t, m, hbar, x0=2, eps = 1e-4), li1Config, None],
    "wide hubble expansion": [lambda x, dx, t, m, hbar: tests.li1(x, dx, t, m, hbar, x0=5, eps = 1e-4), hubbleExpansionConfig, None],
    "quasi-shock": [lambda x, dx, t, m, hbar: tests.li2(x, dx, t, m, hbar, x0 = 10), li2Config, None],
    "wave packet collision": [tests.li3, li3Config, None],
    "travelling wave packet": [tests.travellingWavePacket, travellingWavePacketConfig, None],
    "perturbation wave": [tests.cosmological1D, perturbationWaveConfig, None],
    "accuracy test 1D": [lambda xx, dx, t, m, hbar: tests.cosmological1D(xx, dx, t, m, hbar, eps=5e-3, Lx=1, N = 1), accuracyTest1DConfig, None],
    "cosmo 1D": [lambda xx, dx, t, m, hbar: tests.cosmological1D(xx, dx, t, m, hbar, eps=5e-3, Lx=10, N=10), cosmoConfig, None],
    "perturbation wave 2D": [lambda x, y, dx, t, m, hbar: tests.cosmological2D(x, y, dx, t, m, hbar, Lx = 1, Ly = 1, N = 1, eps=5e-3), perturbationWave2DConfig, None],
    "cosmo 2D": [lambda x, y, dx, t, m, hbar: tests.cosmological2D(x, y, dx, t, m, hbar, Lx = 25, Ly = 25, N = 10, eps= 5e-3), cosmo2DConfig, None],
    "cosmo 2D test": [lambda x, y, dx, t, m, hbar: tests.cosmological2D(x, y, dx, t, m, hbar, Lx = 25, Ly = 25, N = 5, eps= 5e-3), cosmo2DTestConfig, None],
    "cosmo 2D expansion": [lambda x, y, dx, t, m, hbar: tests.cosmological2D(x, y, dx, t, m, hbar, Lx = 25, Ly = 25, N = 5, eps= 5e-3), cosmo2DExpansionConfig, None],
    "accuracy test 2D": [lambda x, y, dx, t, m, hbar: tests.cosmological2D(x, y, dx, t, m, hbar, Lx = 1, Ly = 1, N = 1, eps= 5e-3), accuracyTest2DConfig, None],
    "stability test 2D": [lambda xx, yy, dx, t, m, hbar: tests.cosmological2D(xx, yy, dx, t, m, hbar, eps=3e-5, Lx=25, Ly=25, N = 10), stabilityTest2DConfig, None],
    "perturbation wave 3D": [lambda x, y, z, dx, t, m, hbar: tests.cosmological3D(x, y, z, dx, t, m, hbar, Lx = 1, Ly = 1, Lz = 1, N = 1, eps=5e-3), perturbationWave3DConfig, None],
    "cosmo 3D test": [lambda x, y, z, dx, t, m, hbar: tests.cosmological3D(x, y, z, dx, t, m, hbar, Lx = 8, Ly = 8, Lz = 8, N = 3, eps=5e-3), cosmo3DTestConfig, None],
    "accuracy test 3D": [lambda x, y, z, dx, t, m, hbar: tests.cosmological3D(x, y, z, dx, t, m, hbar, Lx = 1, Ly = 1, Lz = 1, N = 1, eps=5e-3), accuracyTest3DConfig, None],
}

nice_test_list = ["perturbation wave", "harmonic oscillator coherent state", "quasi-shock", "periodic gaussian wave packet", "hubble expansion", "wave packet collision"]
nice_scheme_list = ["hybrid", "wave-ftcs2", "wave-ftcs4", "phase-upwind", "phase-ho-upwind", "phase-ho-upwind without diffusion", "phase-ho-upwind without convection"]

def hoUpwindMCConfig(c):
    hoUpwindConfig(c)
    c["fluxLimiter"] = "MC"

def hoUpwindALBADAConfig(c):
    hoUpwindConfig(c)
    c["fluxLimiter"] = "VANALBADA"

def hoUpwindLEERConfig(c):
    hoUpwindConfig(c)
    c["fluxLimiter"] = "VANLEER"

scheme_list = {
    "hybrid":[hybrid_scheme.HybridScheme, hybridConfig],
    "wave-ftcs2": [wave_schemes.FTCSScheme, ftcsConfig],
    "wave-ftcs4": [wave_schemes.FTCSScheme, hoftcsConfig],
    "wave-crank-nicolson": [wave_schemes.CNScheme, cnConfig],
    "wave-spectral": [wave_schemes.SpectralScheme, spectralConfig],
    "phase-upwind": [phase_schemes.UpwindScheme, upwindConfig],
    "phase-upwind with friction": [phase_schemes.UpwindScheme, upwindWithFrictionConfig],
    "phase-upwind without quantum pressure": [phase_schemes.UpwindScheme, upwindWithoutDiffusionConfig],
    "phase-upwind without convection": [phase_schemes.UpwindScheme, upwindWithoutConvectionConfig],
    "phase-ho-upwind": [phase_schemes.HOUpwindScheme, hoUpwindConfig],
    "phase-ho-upwind_mc": [phase_schemes.HOUpwindScheme, hoUpwindMCConfig],
    "phase-ho-upwind_albada": [phase_schemes.HOUpwindScheme, hoUpwindALBADAConfig],
    "phase-ho-upwind_leer": [phase_schemes.HOUpwindScheme, hoUpwindLEERConfig],
    "phase-ho-upwind with friction": [phase_schemes.HOUpwindScheme, hoUpwindWithFrictionConfig],
    "phase-ho-upwind without diffusion": [phase_schemes.HOUpwindScheme, hoUpwindWithoutDiffusionConfig],
    "phase-ho-upwind without convection": [phase_schemes.HOUpwindScheme, hoUpwindWithoutConvectionConfig],
    "phase-lw-upwind": [phase_schemes.LaxWendroffUpwindScheme, lwUpwindConfig],
    "phase-ftcs-convective": [phase_schemes.FTCSConvectiveScheme, ftcsConvectiveConfig],
    "phase-ftcs-convective without diffusion": [phase_schemes.FTCSConvectiveScheme, ftcsConvectiveWithoutDiffusionConfig],
    "phase-ftcs-convective without convection": [phase_schemes.FTCSConvectiveScheme, ftcsConvectiveWithoutDiffusionConfig],
    "phase-ftcs-convective without source": [phase_schemes.FTCSConvectiveScheme, ftcsConvectiveWithoutSourceConfig],
    "fluid-muscl-hancock": [fluid_schemes.MUSCLHancock, musclHancockConfig],
}

def run(title, scheme, c, test, label, potential = None, createAnimation = False, useWaveSolver = False, suffix = ""):
    filename = title.lower().replace(" ", "_") + "_" + label.lower().replace(" ", "_")+suffix
    solver = scheme(c, test)
    solver.setExternalPotentialFunction(potential)

    if useWaveSolver:
        waveSolver = wave_schemes.SpectralScheme(c, test)
        waveSolver.setExternalPotentialFunction(potential)
    else:
        waveSolver = None
        
    if not createAnimation:
        i = 0
        solver.run()

        if useWaveSolver:
            waveSolver.run()


        print(f"Finished in {i} steps.")
        animation.drawFrame(solver = solver, label = title, analyticalSolution = test, filename = filename, waveSolver = waveSolver)
        plt.show()
    else:
        animation.createAnimation(solver = solver, analyticalSolution = test, filename = filename, waveSolver = waveSolver)

def runTest(test_name, scheme_name = None, createAnimation = False, useWaveSolver = False, suffix = "", extraConfig = None):
    test, testConfig, potential = test_list[test_name]

    display(Markdown('# ' + test_name))

    if scheme_name is None:
        for key, value in scheme_list.items():
            scheme, schemeConfig = value

            c = getBaseConfig()
            testConfig(c)
            schemeConfig(c)
            if extraConfig is not None:
                extraConfig(c)

            display(Markdown('## ' + key))

            run(key, scheme, c, test, key, potential, createAnimation, useWaveSolver, suffix)
    else:
        scheme, schemeConfig = scheme_list[scheme_name]

        c = getBaseConfig()
        testConfig(c)
        schemeConfig(c)
        if extraConfig is not None:
            extraConfig(c)

        display(Markdown('## ' + scheme_name))

        run(test_name, scheme, c, test, scheme_name, potential, createAnimation, useWaveSolver, suffix)
