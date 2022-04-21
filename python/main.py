import analysis
import config
import numpy as np
import schemes
import tests
import convection_tests
import animation 

import matplotlib.pyplot as plt 

config = config.generateConfig(dt=1e-4, t0=0)


def resetConfig(config):
    config["dt"] = 1e-3
    config["domainSize"] = 1#2*np.pi
    config["resolution"] = 16
    config["timeOrder"] = 2
    config["tEnd"] = 1/(2*np.pi*3*0.5)
    #config["tEnd"] = 1/(2*np.pi*3*0.5) # For cosmo tests at L = 10, in general k = 2 pi / L, omega = 0.5/Eta * sum k_i^2 t = 2pi/omega
    config["stencilOrder"] = 2
    config["gravity"] = 0
    config["alpha"] = 2
    config["outputTimestep"] = False
    config["useAdaptiveTimestep"] = True

resetConfig(config)

if 0:
    solver = schemes.ConvectiveScheme(config, tests.periodicLi1)
    for i in range(10):
        solver.step(solver.dt)
    animation.drawFrame("li1_order_4_1", solver, tests.periodicLi1)


if 0:
    solver = schemes.ConvectiveScheme(config, tests.cosmological1D)
    for i in range(10):
        solver.step(solver.dt)
    animation.plotFields("cosmo_nograv_order_4_1", solver, tests.cosmological1D)


if 0:
    solver = schemes.ConvectiveScheme(config, tests.periodicLi3)
    for i in range(10):
        solver.step(solver.dt)
    animation.plotFields("li3_order_4_1", solver, tests.periodicLi3)

if 0:
    analysis.accuracyTest(
        "periodicLi1",
        schemes.ConvectiveScheme,
        config,
        tests.periodicLi1,
        tests.periodicLi1,
        resetConfig,
        N=5,
        halfDT=False,
        halfDX=True,
        NT=10,
        plotRef=True,
        debug=False,
        chopBoundary=False,
    )

if 0:
    analysis.accuracyTest(
        "periodicLi1_wave",
        schemes.WaveScheme,
        config,
        tests.periodicLi1,
        tests.periodicLi1,
        resetConfig,
        N=5,
        halfDT=False,
        halfDX=True,
        NT=10,
        plotRef=True,
        debug=False,
        chopBoundary=False,
    )

if 0:
    dimensions = [3]
    tests_1d  = [tests.cosmological1D, tests.periodicLi1, tests.periodicLi3]
    test_names_1d = ["cosmo", "li1", "li3"]
    tests_2d  = [tests.cosmological2D]
    test_names_2d = ["cosmo"]
    tests_3d  = [tests.cosmological3D]
    test_names_3d = ["cosmo"]

    schemes = [schemes.FourierScheme, schemes.ConvectiveScheme, schemes.UpwindScheme]
    scheme_names  = ["spectral", "convective", "phase"]
    mode = ["stencilOrder"]
    mode_names = ["space"]

    for dimension in dimensions:
        config["dimension"] = dimension
        if dimension == 1:
            test_names = test_names_1d
            tests      = tests_1d
        elif dimension == 2: 
            test_names = test_names_2d
            tests      = tests_2d
        elif dimension == 3: 
            test_names = test_names_3d
            tests      = tests_3d

        for test_name, test in zip(test_names, tests):
            for mode_name, mode in zip(mode_names, mode):
                for scheme_name, scheme in zip(scheme_names, schemes):
                    analysis.accuracyTest(
                        mode_name + "_" + test_name + "_" + scheme_name,
                        scheme,
                        config,
                        test,
                        test,
                        resetConfig,
                        N=5,
                        halfDT=False,
                        halfDX=True,
                        NT=10,
                        plotRef=True,
                        debug=False,
                        chopBoundary=False,
                        mode = mode
                    )

if 1:
    N = 3
    dimensions = [2]
    tests_1d  = [tests.cosmological1D]
    test_names_1d = ["cosmological initial condition without gravity"]
    tests_2d  = [tests.cosmological2D]
    test_names_2d = ["cosmological initial condition without gravity"]
    tests_3d  = [tests.cosmological3D]
    test_names_3d = ["cosmological initial condition without gravity"]

    schemes_ = [schemes.UpwindScheme, schemes.LaxWendroffUpwindScheme, schemes.HOUpwindScheme]#, schemes.CNScheme, schemes.MixedScheme, schemes.CDPhaseScheme]
    scheme_names  = ["convective", "upwind", "lax-wendroff", "ho-upwind"]#, "cranck-nicolson", "mixed", "cd-phase"]
    modes = ["stencilOrder"]
    mode_names = ["spatial and temporal accuracy"]

    for dimension in dimensions:
        print(f"\n\nMake {dimension}D plots")
        config["dimension"] = dimension
        if dimension == 1:
            test_names = test_names_1d
            tests      = tests_1d
        elif dimension == 2: 
            test_names = test_names_2d
            tests      = tests_2d
        elif dimension == 3: 
            test_names = test_names_3d
            tests      = tests_3d

        for test_name, test in zip(test_names, tests):
            print(f"\nRun test {test_name}")

            for mode_name, mode in zip(mode_names, modes):

                print(f"Mode {mode_name}")

                truncations_errors = []
                labels = []


                halfDT = False 
                halfDX = False

                if mode_name == "spatial accuracy":
                    halfDX = True
                elif mode_name == "temporal accuracy":
                    halfDT = True
                else:
                    halfDX = True 
                    halfDT = True


                for scheme_name, scheme in zip(scheme_names, schemes_):
                    resetConfig(config)
                    p = analysis.analyseError(
                        scheme,
                        config,
                        test,
                        test,
                        N=N,
                        halfDT=halfDT,
                        halfDX=halfDX,
                        NT=10,
                        debug=False,
                        chopBoundary=False#,
                        #waveScheme=schemes.FourierScheme
                    )

                    truncations_errors.append(p)
                    labels.append(scheme_name)

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6), dpi = 80)
                plt.suptitle(test_name + ": " + mode_name)
                xx = np.arange(N)

                for i, p in enumerate(truncations_errors):
                    ax1.plot(xx, np.log(p)/np.log(2), label=labels[i])
                ax1.legend()

                ax1.set_xlabel(
                    f"Decrease dx by factor 1/2 = {halfDX}, decrease dt by factor 1/2 = {halfDT}"
                )

                ax1.set_ylabel(r"$\log_2$ (RMS truncation error per temporal and spatial step)")
                ax1.set_xticks(np.arange(N))
                ax1.set_xticklabels(np.arange(N))

                xx = np.arange(N - 1)

                ax2.set_xlabel(
                    f"Decrease dx by factor 1/2 = {halfDX}, decrease dt by factor 1/2 = {halfDT}"
                )

                ax2.set_ylabel(r"order")
                ax2.set_xticks(np.arange(N))
                ax2.set_xticklabels(np.arange(N))
                ax2.set_yticks(np.arange(8))
                ax2.set_yticklabels(np.arange(8))
                ax2.set_ylim(0, 7)

                for i, p in enumerate(truncations_errors):
                    error = np.log(p)/np.log(2)
                    order = np.roll(error, -1) - error
                    ax2.plot(xx, np.abs(order[:-1]), label=labels[i])
                ax2.legend()
                
                filename = str(dimension) + "d_" + mode_name + "_" + test_name
                plt.savefig("error_analysis/" + filename  + ".jpg")
                plt.clf()
                plt.close()

if 0:
    analysis.accuracyTest(
        "cosmo_nograv",
        schemes.ConvectiveScheme,
        config,
        tests.cosmological2D,
        tests.cosmological2D,
        resetConfig,
        N=5,
        halfDT=False,
        halfDX=True,
        NT=10,
        plotRef=True,
        debug=False,
        chopBoundary=False,
    )

if 0:
    N = 3
    dimensions = [1, 2]
    tests_1d  = [convection_tests.periodicTest1D]
    test_names_1d = ["test 1"]
    tests_2d  = [convection_tests.periodicTest2D]
    test_names_2d = ["test 2"]

    schemes_ = [schemes.AdvectionScheme, schemes.HOAdvectionScheme, schemes.LaxWendroffScheme]
    scheme_names  = ["advection", "ho advection", "lax-wendroff"]
    modes = ["stencilOrder"]
    mode_names = ["spatial accuracy"]

    for dimension in dimensions:
        print(f"\n\nMake {dimension}D plots")
        config["dimension"] = dimension
        if dimension == 1:
            test_names = test_names_1d
            tests      = tests_1d
        elif dimension == 2: 
            test_names = test_names_2d
            tests      = tests_2d
        elif dimension == 3: 
            test_names = test_names_3d
            tests      = tests_3d

        for test_name, test in zip(test_names, tests):
            print(f"\nRun test {test_name}")

            for mode_name, mode in zip(mode_names, modes):

                print(f"Mode {mode_name}")

                truncations_errors = []
                labels = []


                halfDT = False 
                halfDX = False

                if mode_name == "spatial accuracy":
                    halfDX = True
                elif mode_name == "temporal accuracy":
                    halfDT = True
                else:
                    halfDX = True 
                    halfDT = True


                for scheme_name, scheme in zip(scheme_names, schemes_):
                    resetConfig(config)
                    p = analysis.analyseError(
                        scheme,
                        config,
                        test,
                        test,
                        N=N,
                        halfDT=halfDT,
                        halfDX=halfDX,
                        NT=10,
                        debug=False,
                        chopBoundary=False,
                        advection=True
                        #waveScheme=schemes.FourierScheme
                    )

                    truncations_errors.append(p)
                    labels.append(scheme_name)

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6), dpi = 80)
                plt.suptitle(test_name + ": " + mode_name)
                xx = np.arange(N)

                for i, p in enumerate(truncations_errors):
                    ax1.plot(xx, np.log(p)/np.log(2), label=labels[i])
                ax1.legend()

                ax1.set_xlabel(
                    f"Decrease dx by factor 1/2 = {halfDX}, decrease dt by factor 1/2 = {halfDT}"
                )

                ax1.set_ylabel(r"$\log_2$ (RMS truncation error per temporal and spatial step)")
                ax1.set_xticks(np.arange(N))
                ax1.set_xticklabels(np.arange(N))

                xx = np.arange(N - 1)

                ax2.set_xlabel(
                    f"Decrease dx by factor 1/2 = {halfDX}, decrease dt by factor 1/2 = {halfDT}"
                )

                ax2.set_ylabel(r"order")
                ax2.set_xticks(np.arange(N))
                ax2.set_xticklabels(np.arange(N))
                ax2.set_yticks(np.arange(8))
                ax2.set_yticklabels(np.arange(8))
                ax2.set_ylim(0, 7)

                for i, p in enumerate(truncations_errors):
                    error = np.log(p)/np.log(2)
                    order = np.roll(error, -1) - error
                    ax2.plot(xx, np.abs(order[:-1]), label=labels[i])
                ax2.legend()
                
                filename = str(dimension) + "d_" + mode_name + "_" + test_name
                plt.savefig("error_analysis/" + filename  + ".jpg")
                plt.clf()
                plt.close()