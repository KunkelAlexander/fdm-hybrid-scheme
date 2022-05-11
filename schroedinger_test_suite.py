
import numpy as np
import matplotlib.pyplot as plt 

from IPython.display import display, Markdown, Latex

import src.wave_schemes as wave_schemes
import src.phase_schemes as phase_schemes
import src.fluid_schemes as fluid_schemes

import src.tests as tests
import src.config as config
import src.animation as animation 


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
    c["nThreads"]            = 4
    c["fps"] = 20
    return c


def ftcsConfig(c):
    c["stencilOrder"] = 6
    c["timeOrder"]    = 2

def cnConfig(c):
    c["stencilOrder"] = 1
    c["timeOrder"]    = 1

def spectralConfig(c):
    c["usePeriodicBC"] = True


def upwindConfig(c):
    c["stencilOrder"] = 1
    c["timeOrder"]    = 4

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
    c["stencilOrder"] = 4
    c["timeOrder"] = 2

def vhoUpwindConfig(c):
    c["stencilOrder"] = 4
    c["timeOrder"] = 4

def hoUpwindWithFrictionConfig(c):
    c["stencilOrder"] = 4
    c["timeOrder"] = 2
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
    c["tEnd"] = 4 * np.pi
    c["useAdaptiveYlim"] = True

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
    c["resolution"] = 64
    c["tEnd"] = 1/(2*np.pi*3*0.5)
    c["domainSize"] = 1
    N = 2
    k = 2*np.pi / (N * c["domainSize"])
    eta = 1
    omega = 0.5/eta * k**2
    tEnd = 2*np.pi / omega
    print("tEnd", tEnd)
    config["tEnd"] = tEnd
    #config["tEnd"] = 1/(2*np.pi*3*0.5) , omega = 0.5/Eta * sum k_i^2 t = 2pi/omega

    c["xlim"] = [0, 1]
    c["densityYlim"] = [0.98, 1.02]
    c["phaseYlim"] = [-0.01, 0.01]
    c["slowDown"] = 5/c["tEnd"] 

def stabilityTestConfig(c):
    c["usePeriodicBC"] = True
    c["resolution"] = 64
    c["tEnd"] = 100
    c["domainSize"] = 1
    c["xlim"] = [0, 1]
    c["densityYlim"] = [0, 20]
    c["phaseYlim"] = [-3.14, 3.14]
    c["slowDown"] = 1
    c["fps"] = 1
    c["gravity"] = 1

def accuracyTestConfig(c):
    c["usePeriodicBC"] = True
    c["resolution"] = 16
    N = 2
    k = 2*np.pi / (N * c["domainSize"])
    eta = 1
    omega = 0.5/eta * k**2
    tEnd = 2*np.pi / omega
    print("tEnd", tEnd)
    c["domainSize"] = 1
    c["xlim"] = [0, 1]
    c["densityYlim"] = [0.7, 1.3]
    c["phaseYlim"] = [-0.05, 0.05]
    c["slowDown"] = 1
    c["fps"] = 1


def solitonConfig(c):
    c["usePeriodicBC"] = True
    c["resolution"] = 128
    c["tEnd"] = 100
    c["domainSize"] = 10
    c["xlim"] = [0, 10]
    c["densityYlim"] = [0, 10]
    c["slowDown"] = 1
    c["gravity"] = 1
    c["fps"] = 2

def expandingSolitonConfig(c):
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
    c["densityYlim"] = [0, 25]
    c["slowDown"] = 20

def hubbleExpansionConfig(c):
    c["resolution"] = 512
    c["tEnd"] = .25
    c["domainSize"] = 10
    c["xlim"] = [0, 10]
    c["densityYlim"] = [0, 25]
    c["slowDown"] = 20


def li2Config(c):
    c["t0"] = 0.0025
    c["tEnd"] = 0.01
    c["resolution"] = 4096
    c["domainSize"] = 20
    c["xlim"] = [9, 11]
    c["usePeriodicBC"] = True
    c["densityYlim"] = [4.5, 6.5]
    c["phaseYlim"] = [-0.05, 0.05]
    c["slowDown"] = 500

def li3Config(c):
    c["usePeriodicBC"] = False
    c["resolution"] = 512
    c["tEnd"] = .005
    c["domainSize"] = 1
    c["xlim"] = [0, 1]
    c["densityYlim"] = [0, 16]
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
    c["resolution"] = 32
    c["tEnd"] = 100
    c["domainSize"] = 1
    c["xlim"] = [0, 1]
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
    c["densityYlim"] = [0.8, 1.2]
    c["phaseYlim"] = [-0.05, 0.05]
    c["plotDensityLogarithm"] = False
    c["slowDown"] = 1
    c["fps"] = 1

def perturbationWave2DConfig(c):
    c["dimension"] = 2
    c["usePeriodicBC"] = True
    c["resolution"] = 16
    c["tEnd"] = 1/(2*np.pi*3*0.5)
    c["domainSize"] = 1
    c["xlim"] = [0, 1]
    c["densityYlim"] = [0.6, 1.4]
    c["cfl"] = .2
    c["slowDown"] = 2/c["tEnd"] 

def soliton2DConfig(c):
    c["dimension"] = 2
    c["usePeriodicBC"] = True
    c["domainSize"] = 25
    c["resolution"] = 128
    c["tEnd"] = 1.5
    c["domainSize"] = 10
    c["slowDown"] = 5
    c["plotPhaseMod2"] = False
    c["phaseYlim"] = [-50, 50]
    c["densityYlim"] = [0, 1]
    c["gravity"] = 1


test_list = {
    "standing wave": [tests.standingWave, standingWaveConfig, None],
    "harmonic oscillator convergence": [tests.generate1DUniform, oscillatorConvergenceConfig, lambda x: tests.oscillatorPotential1D(x, x0 = 0.5)],
    "harmonic oscillator eigenstate": [tests.oscillatorEigenstate1D, oscillatorEigenstateConfig, lambda x: tests.oscillatorPotential1D(x, x0 = 3)],
    "harmonic oscillator coherent state": [tests.oscillatorCoherentState1D, oscillatorCoherentStateConfig, lambda x: tests.oscillatorPotential1D(x, x0 = 7)],
    "infinite well": [tests.infiniteWell1D, infiniteWellConfig, None],
    "gaussian wave packet": [lambda x, dx, t: tests.li1(x, dx, t, x0=2), li1Config, None],
    "hubble expansion": [lambda x, dx, t: tests.li1(x, dx, t, x0=2, eps = 1e-4), li1Config, None],
    "wide hubble expansion": [lambda x, dx, t: tests.li1(x, dx, t, x0=5, eps = 1e-4), hubbleExpansionConfig, None],
    "quasi-shock": [lambda x, dx, t: tests.li2(x, dx, t, x0 = 10), li2Config, None],
    "wave packet collision": [tests.li3, li3Config, None],
    "travelling wave packet": [tests.travellingWavePacket, travellingWavePacketConfig, None],
    "perturbation wave": [tests.cosmological1D, perturbationWaveConfig, None],
    "stability test": [lambda xx, dx, t: tests.cosmological1D(xx, dx, t, eps=1e-1, Lx=1, N = 1), stabilityTestConfig, None],
    "accuracy test": [lambda xx, dx, t: tests.cosmological1D(xx, dx, t, eps=1e-1, Lx=1, N = 2), accuracyTestConfig, None],
    "soliton": [lambda xx, dx, t: tests.cosmological1D(xx, dx, t, eps=5e-3, Lx=10, N=10), solitonConfig, None],
    "expanding_soliton": [lambda xx, dx, t: tests.cosmological1D(xx, dx, t, eps=5e-3, Lx=10, N=10), expandingSolitonConfig, None],
    "perturbation wave 2D": [lambda x, y, dx, t: tests.cosmological2D(x, y, dx, t, Lx = 1, Ly = 1, N = 3, eps= 0.1), perturbationWave2DConfig, None],
    "soliton 2D": [lambda x, y, dx, t: tests.cosmological2D(x, y, dx, t, Lx = 25, Ly = 25, N = 10, eps= 5e-3), soliton2DConfig, None],
    "accuracy test 2D": [lambda xx, yy, dx, t: tests.cosmological2D(xx, yy, dx, t, eps=1e-1, Lx=1, Ly=1, N = 2), accuracyTest2DConfig, None],
}


scheme_list = {
    "wave-ftcs (forward in time, centered in space)": [wave_schemes.FTCSScheme, ftcsConfig],
    "wave-crank-nicolson": [wave_schemes.CNScheme, cnConfig],
    "wave-spectral": [wave_schemes.SpectralScheme, spectralConfig],
    "phase-upwind": [phase_schemes.UpwindScheme, upwindConfig],
    "phase-upwind with friction": [phase_schemes.UpwindScheme, upwindWithFrictionConfig],
    "phase-upwind without quantum pressure": [phase_schemes.UpwindScheme, upwindWithoutDiffusionConfig],
    "phase-upwind without convection": [phase_schemes.UpwindScheme, upwindWithoutConvectionConfig],
    "phase-ho-upwind": [phase_schemes.HOUpwindScheme, hoUpwindConfig],
    "phase-ho-upwind with friction": [phase_schemes.HOUpwindScheme, hoUpwindWithFrictionConfig],
    "phase-ho-upwind without diffusion": [phase_schemes.HOUpwindScheme, hoUpwindWithoutDiffusionConfig],
    "phase-ho-upwind without convection": [phase_schemes.HOUpwindScheme, hoUpwindWithoutConvectionConfig],
    "phase-vho-upwind": [phase_schemes.HOUpwindScheme, vhoUpwindConfig],
    "phase-lw-upwind": [phase_schemes.LaxWendroffUpwindScheme, lwUpwindConfig],
    "phase-ftcs-convective": [phase_schemes.FTCSConvectiveScheme, ftcsConvectiveConfig],
    "phase-ftcs-convective without diffusion": [phase_schemes.FTCSConvectiveScheme, ftcsConvectiveWithoutDiffusionConfig],
    "phase-ftcs-convective without convection": [phase_schemes.FTCSConvectiveScheme, ftcsConvectiveWithoutDiffusionConfig],
    "phase-ftcs-convective without source": [phase_schemes.FTCSConvectiveScheme, ftcsConvectiveWithoutSourceConfig],
    "fluid-muscl-hancock": [fluid_schemes.MUSCLHancock, musclHancockConfig],
}

def run(title, scheme, c, test, label, potential = None, createAnimation = False, useWaveSolver = False):
    filename = title.lower().replace(" ", "_") + "_" + label.lower().replace(" ", "_")
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
        animation.createAnimation(solver = solver, label = title, analyticalSolution = test, filename = filename, waveSolver = waveSolver)

def runTest(test_name, scheme_name = None, createAnimation = False, useWaveSolver = False):
    test, testConfig, potential = test_list[test_name]

    display(Markdown('# ' + test_name))

    if scheme_name is None:
        for key, value in scheme_list.items():
            scheme, schemeConfig = value

            c = getBaseConfig()
            testConfig(c)
            schemeConfig(c)

            display(Markdown('## ' + key))

            run(title, scheme, c, test, key, potential, createAnimation, useWaveSolver)
    else:
        scheme, schemeConfig = scheme_list[scheme_name]

        c = getBaseConfig()
        testConfig(c)
        schemeConfig(c)

        display(Markdown('## ' + scheme_name))

        run(test_name, scheme, c, test, scheme_name, potential, createAnimation, useWaveSolver)
