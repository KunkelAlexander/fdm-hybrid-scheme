import schemes
import tests
import config
import analysis
import animation

import numpy as np

config, psi, density, phase = animation.loadRun("2d_runs/cosmo_convective_6.npz")
config["dimensions"] = 2
config["stencilOrder"] = 4
config["dt"] = 1e-5
config["dpi"] = 80
config["slowDown"] = 1
hsolver = schemes.HybridScheme(schemes.ConvectiveScheme, schemes.SubregionScheme, config, tests.cosmological2D)
solver = schemes.ConvectiveScheme(config, tests.cosmological2D)
fsolver = schemes.FourierScheme(config, tests.cosmological2D)
solver.setDensity(density)
solver.setPhase(phase) 
hsolver.setDensity(density)
hsolver.setPhase(phase) 
fsolver.setPsi(psi)
animation.createAnimation(solver, "convective scheme", config, tests.cosmological2D, "cosmo_convective_after_breakdown", fsolver)