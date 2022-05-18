
#set scheme in config
HIGH_ORDER_PHASE_SCHEME = 0
PHASE_SCHEME            = 1
FLUID_SCHEME            = 2
CN_SCHEME               = 3

scheme_names = ["HOPS", "PS", "FS", "CNS"]

#different fluxes in phase scheme
LAX_FRIEDRICHS_FLUX = 0
GODUNOV_FLUX        = 1

#set whether fluid scheme integrates velocity field or evolves phase field
INTEGRATE_V      = 0
EVOLVE_PHASE     = 1


def generateConfig(t0=0, resolution=512, dt=1e-4):
    return {
        # Grid
        "dimension": 1,
        "resolution": resolution,
        "domainSize": 1,
        "usePeriodicBC": True,
        "ghostBoundarySize": 3,

        # Time management
        "t0": t0,  # current time of the simulation
        "tEnd": 1.0,  # time at which simulation ends
        "dt": dt,  # timestep
        "slowDown": 1,  # 0.03s of simulation correspond to 3 s of animation
        "fps": 20,  # ms, 50 frames are 1s of animation
        "useAdaptiveTimestep": False, 
        "cfl": 1,
        "maximumNumberOfTimesteps": 100000,

        # Scheme options
        "timeOrder": 1,
        "stencilOrder": 1,
        "fluxLimiter": "VANALBADA",

        # Gravity & cosmology
        "hbar" : 1,
        "m" : 1,
        "gravity": 0,  # Turn on self-gravity
        "useCosmology": False,

        # Multithreading (for FFT and hybrid scheme)
        "nThreads": 1,

        # External potential 
        "externalPotential": None, 

        # Turn on debug output
        "debug": False,
        "outputTimestep": False,

        # Subregion options
        "useHybrid": False,
        "useAdaptiveSubregions": False,
        "subregions": [],  # size_y, size_x, position_y, position_x relative in fraction of window size
        "windowUpdateFrequency": 100,
        "WindowSize": 0.5,

        # 1D Hybrid second order options
        "mode": 0,  # 0 is first-order upwind, 1 is first-order upwind with naive differencing, 2 is n-th order with ENO
        "enoOrder": 1,
        "timeOrder": 1,
        "rhoOrder": 1,
        "modifiedPQN": 3,
        "switchingThreshold": 0.04,

        # Animation options
        "xlim": [0, 10],
        "densityYlim": [0, 1],
        "phaseYlim": [-3.14, 3.14],
        "useAdaptiveYlim": False,
        "plotDensityLogarithm": True,
        "plotPhaseMod2": True,
        #Overwrite default settings
        #"dpi" : 600, 
        #"figsize": [3.54, 3.54],
        "plotTime" : True,
        "plotDebug": True,
        "plotPhase": True,
        "savePlots": False,

        #Phase scheme options
        "artificialDiffusion"   : 1,
        "fluxMode"              : LAX_FRIEDRICHS_FLUX,
        "SPeriodicBoundary"     : True, 
        "modifiedPQN"           : 3,
        "turnOffConvection"     : False,
        "turnOffDiffusion"      : False,
        "turnOffSource"         : False,
        "friction"              : 0.0,
        "limitHJ"               : True,

        #CFL conditions
        "C_parabolic"           : 1/6,
        "C_velocity"            : 1,
        "C_acceleration"        : 0.4,
        "C_potential"           : 2 * 3.14159 * 0.3,


        #Fluid scheme options
        "fluidMode"             : INTEGRATE_V,
        "useSlopeLimiting"      : False, #Fluid-only option
        "maxSpeedC"             : 1.0, #Fluid-only option
    }
