import matplotlib.pyplot as plt
import numpy as np

figsize = (8, 6)
dpi = 100


def analyseError(
    solver,
    config,
    generateIC,
    analyticalSolution,
    N=5,
    halfDT=True,
    halfDX=True,
    NT=10,
    debug=False,
    chopBoundary=False,
    waveScheme = None, 
    advection = False
):
    NX = config["resolution"]
    L = config["domainSize"]
    tEnd = config["tEnd"]
    dx = L / NX
    dt = config["dt"]
    error = np.zeros(N)

    for i in range(N):
        config["resolution"] = NX
        config["dt"] = dt
        sol = solver(config, generateIC)
        if waveScheme is not None:
            wavesolver = waveScheme(config, generateIC)

        while sol.getTime() < tEnd:
            delt = sol.getTimeStep()
            sol.step(delt)
            if waveScheme is not None:
                wavesolver.step(delt)

        if waveScheme is None:
            if advection:
                ana, _  = analyticalSolution(*sol.grid, sol.dx, sol.t)
            else:
                ana = np.abs(analyticalSolution(*sol.grid, sol.dx, sol.t)) ** 2
        else:
            ana = np.abs(wavesolver.getPsi())**2

        num = sol.getDensity()

        length = NX

        if chopBoundary:
            boundaryEffect = int(np.maximum(sol.stencilOrder, 8) * NT * sol.timeOrder)
            lb = boundaryEffect
            ub = NX - boundaryEffect
            length = NX - 2 * boundaryEffect
            if ub < lb:
                raise ValueError(
                    f"Boundary effects too strong ({boundaryEffect}). Choose bigger grid!"
                )

            if sol.dimension == 1:
                ana = ana[lb:ub]
                num = num[lb:ub]
            elif sol.dimension == 2:
                ana = ana[lb:ub, lb:ub]
                num = num[lb:ub, lb:ub]
            elif sol.dimension == 3:
                ana = ana[lb:ub, lb:ub, lb:ub]
                num = num[lb:ub, lb:ub, lb:ub]
            else:
                raise ValueError(
                    f"Boundary chopping unsupported in {sol.dimension} dimensions"
                )

        error[i] = np.sum(np.abs(num - ana)) / (length ** (sol.dimension))
        if debug:
            print("dx ", dx, " dt ", dt)
            print("error [i]: ", i, error[i])
        if halfDT and not halfDX:
            NT *= 2
            dt /= 2
        if halfDX and not halfDT:
            NX *= 2
            dx /= 2
        if halfDX and halfDT: 
            NX *= 2
            dx /= 2

            NT *= 4
            dt /= 4
            
    return error


def accuracyTest(
    filename,
    solver,
    config,
    generateIC,
    analyticalSolution,
    reset,
    N,
    halfDT,
    halfDX,
    NT,
    plotRef=True,
    debug=False,
    chopBoundary=False,
    mode = "stencilOrder",
    advection = False
):
    reset(config)
    config[mode] = 2
    p1 = analyseError(
        solver,
        config,
        generateIC,
        analyticalSolution,
        N=N,
        halfDT=halfDT,
        halfDX=halfDX,
        NT=NT,
        debug=debug,
        chopBoundary=chopBoundary,
        advection=advection
    )
    reset(config)
    config[mode] = 4
    p2 = analyseError(
        solver,
        config,
        generateIC,
        analyticalSolution,
        N=N,
        halfDT=halfDT,
        halfDX=halfDX,
        NT=NT,
        debug=debug,
        chopBoundary=chopBoundary,
        advection=advection
    )


    plt.figure(figsize=figsize, dpi=dpi)
    plt.title("Truncation error")
    plt.xlabel(
        f"Decrease dx by factor 1/2 = {halfDX}, decrease dt by factor 1/2 = {halfDT}"
    )
    xx = np.array(range(N))
    plt.plot(xx, np.log(p1) / np.log(2), label="2nd order")
    plt.plot(xx, np.log(p2) / np.log(2), label="4th order")
    if plotRef:
        plt.plot(xx, np.log(p1[0] * 1 / 2 ** (xx * 2)) / np.log(2), label="Quadratic")
        plt.plot(xx, np.log(p1[0] * 1 / 2 ** (xx * 4)) / np.log(2), label="Quartic")
    plt.legend()
    plt.savefig("1d_plots/" + filename + ".jpg")
    plt.clf()
    plt.close()
    return p1
