

import src.config as config 
import src.animation as animation 
import matplotlib.pyplot as plt 
import numpy as np

import src.advection_schemes as advection_schemes
import src.advection_tests as advection_tests

algorithms = {
    "centered difference":advection_schemes.CenteredDifferenceScheme,
    "upwind": advection_schemes.UpwindScheme,
    "second-order upwind": advection_schemes.SecondOrderUpwindScheme,
    "sou with limiter": advection_schemes.SOLimiterScheme,
    "lax-wendroff": advection_schemes.LaxWendroffScheme, 
    "muscl": advection_schemes.MUSCLScheme,
    "ppm": advection_schemes.PPMScheme
}

tests = {
    "gaussian": advection_tests.gaussian1D,
    "reverse gaussian": lambda xx, dx, t: advection_tests.gaussian1D(xx, dx, t, v = -1, x0=4),
    "tophat": advection_tests.tophat1D
}


def getBaseConfig():
    c = config.generateConfig(dt=1e-4, t0=0)
    c["dt"] = 1e-3
    c["domainSize"] = 5
    c["xlim"] =  [0, 5]
    c["densityYlim"] = [0, 1.6]
    c["resolution"] = 256
    c["timeOrder"] = 1
    c["stencilOrder"] = 2
    c["dimension"] = 1
    c["debug"] = False
    c["slowDown"] = 1
    c["tEnd"] = 1
    c["gravity"] = 1
    c["outputTimestep"] = False
    c["alpha"] = 0.1
    c["useAdaptiveTimestep"] = True
    c["usePeriodicBC"] = False
    c["dpi"] = 200
    c["size"] = (3.54 * 2, 3.54)#(3.54 * 1.5, 3.54)
    c["savePlots"] = True
    c["cfl"] = 0.1
    c["plotPhase"] = False 
    c["plotDebug"] = False
    return c


def interactive_advection_test(t, resolution, time_order, scheme, test, limiter, take_snapshot):
    c = getBaseConfig()
    c["resolution"] = resolution
    c["timeOrder"] = time_order
    c["tEnd"] = t
    c["savePlots"] = take_snapshot
    c["fluxLimiter"] = limiter
    c["plotPhase"] = False 
    c["plotDebug"] = False
    if take_snapshot:
        c["dpi"] = 600
        c["figsize"] = [3.54 * 2, 3.54]
    else:
        c["dpi"] = 80
        c["figsize"] = [3.54 * 2, 3.54]

    solver = algorithms[scheme](c, tests[test])
    solver.run()
    animation.drawFrame(solver, advection=True)
    plt.show()
