from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import AxesGrid
import os

import src.fd as fd


"""""" """""" """""" """""" """"""
"""""" """"" Timing """ """""" ""
"""""" """""" """""" """""" """"""


def getNumberOfFrames(config):
    t = config["t0"]
    tEnd = config["tEnd"]
    slowDown = config["slowDown"]
    fps = config["fps"]
    frames = int(np.floor((tEnd - t) * slowDown * fps))
    return frames


def getTimePerFrame(config):
    slowDown = config["slowDown"]
    fps = config["fps"]
    tUpdate = 1 / (fps * slowDown)
    return tUpdate


"""""" """""" """""" """""" """"""
"""""" """"" Stability """ """""" ""
"""""" """""" """""" """""" """"""


def computeEnergy(density, phase, potential, dx):
    I = 0
    J = 0
    for i in range(density.ndim):
        I += 0.5 * fd.getCenteredGradient(np.sqrt(density), dx, axis=i)**2
        J += 0.5 * density * fd.getCenteredGradient(phase, dx, axis=i)**2
    W = 0.5 * density * potential

    E = np.mean(I + J + W)
    return E


def computeTruncationError(solver, density, density_ref):
    rms_error = np.sum(np.abs(density[solver.inner] - density_ref[solver.inner])) / (
        (solver.t / solver.dt) * solver.innerN ** (solver.dimension)
    )
    return rms_error


"""""" """""" """""" """""" """"""
"""""" """""  1D    """ """""" ""
"""""" """""" """""" """""" """"""

plt.rcParams["font.size"] = 8
plt.rcParams['font.family'] = 'serif'
#plt.rcParams['font.serif'] = 'Ubuntu'
#plt.rcParams['font.monospace'] = 'Ubuntu Mono'
#plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['figure.titlesize'] = 12
plt.rcParams["font.family"] = "DejaVu Sans"



def create1DFrame(
    solver,
    label,
    config,
    analyticalSolution,
    filename,
    waveSolver=None,
    advection=False,
):
    plotPhase = config["plotPhase"]
    plotDebug = config["plotDebug"]

    xlim = config["xlim"]
    ylim = config["densityYlim"]
    ylim2 = config["phaseYlim"]
    mod2Phase = config["plotPhaseMod2"]
    useAdaptiveYlim = config["useAdaptiveYlim"]

    nrows = 1
    ncols = 1
    if plotPhase:
        ncols = 2
    if plotDebug:
        nrows = 2

    # prep figure
    if "figsize" in config:
        figsize = np.array(config["figsize"])
    else:
        figsize = [3.54, 3.54]

    figsize[0] *= ncols
    figsize[1] *= nrows

    if "dpi" in config:
        dpi = config["dpi"]
    else:
        dpi = 600

    fig = plt.figure(figsize=figsize, dpi=dpi)

    grid = plt.GridSpec(nrows, ncols, wspace=0.0, hspace=0.4)
    ax1 = fig.add_subplot(grid[0, 0])

    # Switch to axis 1 for density plots
    plt.sca(ax1)
    # Clear axis
    plt.cla()
    # Plot background
    (im1,) = plt.plot([], [], "ro",  ms=3)
    (im3,) = plt.plot([], [], "b",   ms=1)

    plt.xticks()
    plt.yticks()

    # Set axis ratios
    plt.xlim(xlim)
    plt.xlabel("$x$")

    if advection:
        plt.ylabel(r"$u(x)$")
    else:
        plt.ylabel(r"density $\rho$")

    if not useAdaptiveYlim:
        plt.ylim(ylim)

    if plotPhase:
        ax2 = fig.add_subplot(grid[0, 1])

        # Switch to axis 1
        plt.sca(ax2)
        # Clear axis
        plt.cla()

        if waveSolver is None:
            label2 = "analytical solution"
        else:
            label2 = "wave scheme"

        # Plot background
        (im2,) = plt.plot([], [], "ro", ms=3, label=label)
        (im4,) = plt.plot([], [], "bo", ms=1, label=label2)

        plt.xticks()
        plt.yticks()
        # Set axis ratios
        plt.xlim(xlim)
        plt.xlabel("$x$")
        plt.ylabel(r"phase $S$")
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position("right")

        leg = plt.legend(loc="upper right")
        leg.get_frame().set_linewidth(0.0)

        if not useAdaptiveYlim:
            plt.ylim(ylim2)
    else:
        # Draw invisible plots in order for the handles to exist
        (im2,) = plt.plot([], [], "ro", alpha=1)
        (im4,) = plt.plot([], [], "bo", alpha=1)

    if plotDebug:
        ax3 = fig.add_subplot(grid[1, :])

        # Switch to axis 1
        plt.sca(ax3)
        # Clear axis
        plt.cla()
        # Plot background
        (im5,) = plt.plot([], [], "ro", ms=3, label="density")
        (im6,) = plt.plot([], [], "bo", ms=1, label="phase")

        # Set axis ratios
        plt.xlim(xlim)
        plt.ylabel(r"relative error")
        plt.xlabel("$x$")

        if not useAdaptiveYlim:
            plt.ylim([0, .1])
    else:
        # Draw invisible plots in order for the handles to exist
        (im5,) = plt.plot([], [], "ro", alpha=1)
        (im6,) = plt.plot([], [], "bo", alpha=1)

    time_text = ax1.text(0.02, 0.9, "", transform=ax1.transAxes)

    if plotDebug:
        norm_text = ax3.text(0.02, 0.8, "", transform=ax3.transAxes)
    else:
        norm_text = ax1.text(0.02, 0.90, "", transform=ax1.transAxes, alpha=0)

    # initialization function: plot the background of each frame
    def init():
        im1.set_data([], [])
        im2.set_data([], [])
        im3.set_data([], [])
        im4.set_data([], [])
        im5.set_data([], [])
        im6.set_data([], [])

        time_text.set_text("")
        norm_text.set_text("")

        return im1, im2, im3, im4, im5, im6, time_text, norm_text


    global subregion_patches
    subregion_patches = []

    def draw(i):
        global subregion_patches

        density = solver.getDensity()
        phase = solver.getPhase()
        if not advection:
            potential = solver.getPotential()
        t = solver.getTime()
        a = solver.getScaleFactor()
        xx = solver.getGrid()[0]
        dx = solver.dx

        # Get phase or analytical solution
        if waveSolver is None:
            if advection:
                psi_ref, phase_ref = analyticalSolution(xx, dx, t)
                density_ref = psi_ref
            else:
                psi_ref = analyticalSolution(xx, dx, t, solver.m, solver.hbar)
                density_ref = np.abs(psi_ref) ** 2
        else:
            psi_ref = waveSolver.getPsi()
            density_ref = np.abs(psi_ref) ** 2

        im1.set_data(xx, density)
        im3.set_data(xx, density_ref)

        if plotPhase:
            if mod2Phase:
                im2.set_data(xx, np.angle(np.exp(1j * phase)))
                phase_ref = np.angle(psi_ref)
            else:
                im2.set_data(xx, phase - phase[0])
                phase_ref = fd.make_1d_continuous(np.angle(psi_ref))

            #contphase = fd.make_1d_continuous(phase.copy())

            im4.set_data(xx, phase_ref)

        if plotDebug:
            rms_error = 0

            if t != 0:
                rms_error = computeTruncationError(
                    solver, density, density_ref)

            density_relerr = np.abs(density - density_ref) / \
                (density_ref + 1e-8 * (density_ref == 0))
            im5.set_data(xx, density_relerr)

            if plotPhase:
                E1 = computeEnergy(density, phase, potential, solver.dx)
                E2 = computeEnergy(density_ref, phase_ref,
                                   potential, solver.dx)

                phase_relerr = np.abs(phase - phase_ref) / \
                    (phase_ref + 1e-8 * (phase_ref == 0))
                im6.set_data(xx, phase_relerr)

        for patch in subregion_patches:
            patch.remove()

        subregion_patches = []
    

        if hasattr(solver, "useHybrid"):
            if solver.useHybrid:
                for subregion in solver.subregions:
                    pos = subregion.outerInner[0]
                    if subregion.isWaveScheme:
                        c = "blue"
                    else:
                        c = "white"

                    pol1 = ax1.axvspan(
                        pos[0] * solver.dx, pos[-1] * solver.dx, color=c, alpha=0.3)
                    subregion_patches.append(pol1)

                    if plotPhase:
                        pol2 = ax2.axvspan(
                            pos[0] * solver.dx, pos[-1] * solver.dx, color=c, alpha=0.3)
                        subregion_patches.append(pol2)

                    if plotDebug:
                        pol3 = ax3.axvspan(
                            pos[0] * solver.dx, pos[-1] * solver.dx, color=c, alpha=0.3)
                        subregion_patches.append(pol3)

        time_text.set_text("time t = %.5f" % (t))

        if plotDebug:
            debug_information = r"$\int \mathrm{d}x|\psi|^2$" + "= %.3f, %.3f (wave/ana, phase), RMS error = .%.3e" % (
                np.mean(density_ref), np.mean(density), rms_error)

            if plotPhase:
                debug_information += f"\n E_num = {E1:.4f}, E_ana = {E2:.4f}"

            norm_text.set_text(debug_information)

        if solver.config["savePlots"]:
            plt.savefig(f"plots/1d/{filename}.pdf", bbox_inches='tight')

        if os.path.exists(f"runs/1d/{filename}"):
            np.savez_compressed(
                f"runs/1d/{filename}/{i}.npz",
                config=np.array(list(config.items()), dtype=object),
                t=t,
                a=a,
                density=density,
                phase=phase,
                psi=psi_ref,
            )

        return im1, im2, im3, im4, im5, im6, time_text, norm_text

    return fig, init, draw


"""""" """""" """""" """""" """"""
"""""" """""  2D    """ """""" ""
"""""" """""" """""" """""" """"""


def create2DFrame(
    solver,
    label,
    config,
    analyticalSolution,
    filename,
    waveSolver=None,
    advection=False,
):

    logscaling = config["plotDensityLogarithm"]
    mod2Phase = config["plotPhaseMod2"]

    # prep figure
    if "figsize" in config:
        figsize = config["figsize"]
    else:
        figsize = [3.54*3, 3.54*2]

    if "dpi" in config:
        dpi = config["dpi"]
    else:
        dpi = 600

    fig = plt.figure(figsize=figsize, dpi=dpi)
    # grid = AxesGrid(
    #    fig,
    #    111,
    #    nrows_ncols=(2, 3),
    #    axes_pad=0.25,
    #    cbar_mode="edge",
    #    cbar_location="right",
    #    cbar_pad=0.1,
    # )

    grid = AxesGrid(
        fig,
        (0.075, 0.075, 0.85, 0.85),
        nrows_ncols=(2, 3),
        axes_pad=0,
        share_all=True,
        cbar_location="right",
        cbar_mode="edge",
        cbar_size="5%",
        cbar_pad="0%",
    )

    for ax in grid:
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        # ax.set_axis_off()
        ax.set_aspect("equal")

    ax1, ax2, ax3, ax4, ax5, ax6 = grid

    N = solver.getConfig()["resolution"]

    ll, hl = config["densityYlim"]
    if not config["plotPhaseMod2"]:
        pl, ph = config["phaseYlim"]
    else:
        pl, ph = -np.pi, np.pi

    # Plot background
    im1 = ax1.imshow(np.random.random((N, N)),
                     cmap="inferno", vmin=ll, vmax=hl)
    #title1 = ax1.set_title(r"$\log_{10}(|\psi|^2)$ at t = $0$")
    im2 = ax2.imshow(np.random.random((N, N)),
                     cmap="inferno", vmin=ll, vmax=hl)
    #title2 = ax2.set_title(r"$\log_{10}(|\psi|^2)$ at t = $0$")
    im3 = ax3.imshow(np.random.random((N, N)), cmap="gray",    vmin=0, vmax=1)
    #title3 = ax3.set_title(r"$\log_{10}(|\psi|^2)$ at t = $0$")

    im4 = ax4.imshow(np.random.random((N, N)),
                     cmap="inferno", vmin=pl, vmax=ph)
    #title4 = ax4.set_title(r"${\rm angle}(\psi)$ at t = $0$")
    im5 = ax5.imshow(np.random.random((N, N)),
                     cmap="inferno", vmin=pl, vmax=ph)
    #title5 = ax5.set_title(r"${\rm angle}(\psi)$ at t = $0$")
    im6 = ax6.imshow(np.random.random((N, N)), cmap="gray",    vmin=0, vmax=1)
    #title6 = ax6.set_title(r"${\rm angle}(\psi)$ at t = $0$")

    suptitle = ax6.text(.05, .85, "", fontsize=10, c="w", bbox=dict(
        facecolor='gray', alpha=0.5), transform=ax.transAxes)

    cbar1 = plt.colorbar(im1, cax=grid.cbar_axes[0], ticks=[
                         0, 0.25, 0.5, 0.75, 1])
    cbar1.ax.set_yticklabels(
        ['', '$10^{0}$', '$10^{1}$', '$10^{2}$', '$10^{4}$'])
    cbar1.set_label(r"density $\rho$", size=12)
    cbar2 = plt.colorbar(im4, cax=grid.cbar_axes[1])
    cbar2.set_label(r"phase $S$", size=12)

    # cax = inset_axes(ax6, "4%", "210%", loc=3, bbox_to_anchor=(1.25, 0, 1, 1),
    #                bbox_transform=ax6.transAxes, borderpad=0.0)
    #cbar2 = plt.colorbar(im6, cax=cax, ticks=[0, .25, .5, .75, 1])
    #cbar2.ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])

    subregion_patches = []

    # initialization function: plot the background of each frame
    def init():
        im1.set_data(np.random.random((N, N)))
        im2.set_data(np.random.random((N, N)))
        im4.set_data(np.random.random((N, N)))
        im5.set_data(np.random.random((N, N)))
        im5.set_data(np.random.random((N, N)))
        im6.set_data(np.random.random((N, N)))
        return (im1, im2, im4, im5, im5, im6)

    def draw(i):
        # Phase scheme
        density = solver.getDensity().astype(np.float64)
        phase = solver.getPhase().astype(np.float64)
        if not advection:
            potential = solver.getPotential().astype(np.float64)
        xx, yy = solver.getGrid()
        t = solver.getTime()
        a = solver.getScaleFactor()
        dx = solver.dx

        if waveSolver is None:
            if advection:
                psi_ref, phase_ref = analyticalSolution(xx, yy, solver.dx, solver.t)
                density_ref = psi_ref.astype(np.float64)

            else:
                psi_ref = analyticalSolution(xx, yy, solver.dx, solver.t, solver.m, solver.hbar)
                density_ref = (np.abs(psi_ref) ** 2)#.astype(np.float64)
                phase_ref = np.angle(psi_ref)#.astype(np.float64)
        else:
            psi_ref = waveSolver.getPsi()
            density_ref = (np.abs(psi_ref) ** 2)#.astype(np.float64)
            phase_ref = np.angle(psi_ref)#.astype(np.float64)

        rms_error = 0

        if t != 0:
            rms_error = np.sqrt(np.sum((density[solver.inner] - density_ref[solver.inner]) ** 2)) / (
                (solver.t / solver.dt) * solver.innerN ** (solver.dimension)
            )

        rho_min = np.min(density_ref)
        rho_max = np.max(density_ref)
        rho_ratio = rho_max / rho_min

        for p in subregion_patches:
            p.remove()

        subregion_patches.clear()

        if hasattr(solver, "useHybrid"):
            if solver.useHybrid:
                subregions = []
                wave_scheme_subregion_counter = 0
                solver.binaryTree.getSubregions(subregions)
                for N0, N, isWaveScheme in subregions:
                    if isWaveScheme:
                        c = "white"
                        rect1 = patches.Rectangle(
                            np.flip(N0), N, N, linewidth=.5, edgecolor=c, facecolor='none')
                        rect2 = patches.Rectangle(
                            np.flip(N0), N, N, linewidth=.5, edgecolor=c, facecolor='none')
                        pol1 = ax1.add_patch(rect1)
                        pol2 = ax4.add_patch(rect2)
                        subregion_patches.append(rect1)
                        subregion_patches.append(rect2)
                        wave_scheme_subregion_counter += 1
                print(
                    f"{wave_scheme_subregion_counter} wave scheme subregions with {solver.binaryTree.getWaveVolumeFraction() * 100} percent of volume.")

        t = solver.getTime()
        a = solver.getScaleFactor()

        if logscaling:
            im1.set_array((np.log10(density) + 1) / 4)
            im2.set_array((np.log10(density_ref) + 1)/4)
            im3.set_array(
                np.abs(density - density_ref)
                / (density_ref + 1e-8 * (density_ref == 0))
            )
        else:
            im1.set_array(density)
            im2.set_array(density_ref)
            im3.set_array(
                np.abs(density - density_ref)
                / (density_ref + 1e-8 * (density_ref == 0))
            )

        if mod2Phase:
            im4.set_array(np.angle(np.exp(1j * phase)))
            im5.set_array(phase_ref)
        else:
            im4.set_array(phase)
            phase_ref = fd.make_2d_continuous(phase_ref)
            phase_ref += phase[0, 0] - phase_ref[0, 0]
            im5.set_array(phase_ref)

        im6.set_array(np.abs(np.angle(np.exp(1j * (phase - phase_ref))
                                      )/(np.abs(phase_ref) + 1e-8 * (phase_ref == 0))))

        current_time = ""
        if "plotTime" in config:
            if config["plotTime"]:
                current_time += (
                    "t = "
                    + f"{t:.3f}"
                    + f"; a = {a:.3f}"
                )

        if config["plotDebug"]:

            current_time += (
                r"\n$\delta / \rho$ = "
                + f"{rho_ratio:.1f}"
                + r" and $\rho_{min}$ = "
                + f"{rho_min:.2e} \n"
                + r"total mass "
                + "= %.3f, %.3f (wave/ana, phase) \nRMS error = .%.3e"
                % (np.mean(density_ref), np.mean(density), rms_error)
            )
            if not advection:
                E1 = computeEnergy(density, phase, potential, solver.dx)
                E2 = computeEnergy(density_ref, phase_ref,
                                   potential, solver.dx)
                current_time += (
                    f"\n E_num = {E1:.4f} E_ana = {E2:.4f}"
                )

        suptitle.set_text(current_time)

        #title1.set_text(r"$\log_{10}(|\psi|^2)$ for " + label)
        #title4.set_text(r"$angle(\psi)$ for " + label)
#
        # if waveSolver is None:
        #    title2.set_text(r"$\log_{10}(|\psi|^2)$ for analytical solution")
        #    title5.set_text(r"$angle(\psi)$ for analytical solution")
        # else:
        #    title2.set_text(r"$\log_{10}(|\psi|^2)$ for wave scheme")
        #    title5.set_text(r"$angle(\psi)$ for wave scheme")
#
        #title3.set_text(r"relative density error")
        #title6.set_text(r"relative phase error")

        if solver.config["savePlots"]:
            if os.path.exists(f"plots/2d"):
                plt.savefig(f"plots/2d/{filename}.pdf", bbox_inches="tight")

        if os.path.exists(f"runs/2d/{filename}"):
            np.savez_compressed(
                f"runs/2d/{filename}/{i}.npz",
                config=np.array(list(config.items()), dtype=object),
                t=t,
                a=a,
                density=density,
                phase=phase,
                psi=psi_ref,
            )

        return (im1, im2, im4, im5, im5, im6)

    return fig, init, draw

# Plot average density in x-y-plane as projection along the z-axis


def getProjection(f, axis=2):
    return np.mean(f, axis=axis)


def create3DFrame(
    solver,
    label,
    config,
    analyticalSolution,
    filename,
    waveSolver=None,
    advection=False,
    projectionAxis=2,
    saveFields=False
):

    logscaling = config["plotDensityLogarithm"]
    mod2Phase = config["plotPhaseMod2"]

    # prep figure
    fig = plt.figure(figsize=(15, 15), dpi=config["dpi"])
    grid = AxesGrid(
        fig,
        111,
        nrows_ncols=(2, 3),
        axes_pad=0.25,
        cbar_mode="edge",
        cbar_location="right",
        cbar_pad=0.1,
    )

    for ax in grid:
        ax.set_axis_off()
        ax.set_aspect("equal")

    ax1, ax2, ax3, ax4, ax5, ax6 = grid

    N = solver.getConfig()["resolution"]

    ll, hl = config["densityYlim"]
    if not config["plotPhaseMod2"]:
        pl, ph = config["phaseYlim"]
    else:
        pl, ph = -np.pi, np.pi

    # Plot background
    im1 = ax1.imshow(np.random.random((N, N)),
                     cmap="inferno", vmin=ll, vmax=hl)
    title1 = ax1.set_title(r"$\log_{10}(|\psi|^2)$ at t = $0$")
    im2 = ax2.imshow(np.random.random((N, N)),
                     cmap="inferno", vmin=ll, vmax=hl)
    title2 = ax2.set_title(r"${\rm angle}(\psi)$ at t = $0$")
    im3 = ax3.imshow(np.random.random((N, N)), cmap="gray", vmin=0, vmax=1)
    title3 = ax3.set_title(r"${\rm angle}(\psi)$ at t = $0$")

    im4 = ax4.imshow(np.random.random((N, N)),
                     cmap="inferno", vmin=pl, vmax=ph)
    title4 = ax4.set_title(r"${\rm angle}(\psi)$ at t = $0$")
    im5 = ax5.imshow(np.random.random((N, N)),
                     cmap="inferno", vmin=pl, vmax=ph)
    title5 = ax5.set_title(r"${\rm angle}(\psi)$ at t = $0$")
    im6 = ax6.imshow(np.random.random((N, N)), cmap="gray", vmin=0, vmax=1)
    title6 = ax6.set_title(r"${\rm angle}(\psi)$ at t = $0$")

    suptitle = ax6.text(.05, .8, "", fontsize=10, c="w", bbox=dict(
        facecolor='gray', alpha=0.5), transform=ax.transAxes)

    cbar1 = plt.colorbar(im1, cax=grid.cbar_axes[0], ticks=[
                         0, 0.25, 0.5, 0.75, 1])
    cbar1.ax.set_yticklabels(['-1', '0', '1', '2', '3'])
    plt.colorbar(im4, cax=grid.cbar_axes[1])

    cax = inset_axes(ax6, "3%", "207%", loc=3, bbox_to_anchor=(1.25, 0, 1, 1),
                     bbox_transform=ax6.transAxes, borderpad=0.0)
    cbar2 = plt.colorbar(im6, cax=cax, ticks=[0, .25, .5, .75, 1])
    cbar2.ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])

    subregion_patches = []

    # initialization function: plot the background of each frame
    def init():
        im1.set_data(np.random.random((N, N)))
        im2.set_data(np.random.random((N, N)))
        im4.set_data(np.random.random((N, N)))
        im5.set_data(np.random.random((N, N)))
        im5.set_data(np.random.random((N, N)))
        im6.set_data(np.random.random((N, N)))
        return (im1, im2, im4, im5, im5, im6)

    def draw(i):
        # Phase scheme
        density = solver.getDensity().copy().astype(float)
        phase = solver.getPhase().copy().astype(float)
        xx, yy, zz = solver.getGrid()
        t = solver.getTime()
        a = solver.getScaleFactor()
        dx = solver.dx

        if waveSolver is None:
            if advection:
                psi_ref, phase_ref = analyticalSolution(
                    xx, yy, zz, dx, solver.t)
                density_ref = psi_ref

            else:
                psi_ref = analyticalSolution(xx, yy, zz, dx, solver.t, solver.m, solver.hbar)
                density_ref = np.abs(psi_ref) ** 2
                phase_ref = np.angle(psi_ref)
        else:
            psi_ref = waveSolver.getPsi()
            density_ref = np.abs(psi_ref) ** 2
            phase_ref = np.angle(psi_ref)

        rms_error = 0

        if t != 0:
            rms_error = np.sqrt(np.sum((density[solver.inner] - density_ref[solver.inner]) ** 2)) / (
                (solver.t / solver.dt) * solver.innerN ** (solver.dimension)
            )

        rho_min = np.min(density_ref)
        rho_max = np.max(density_ref)
        rho_ratio = rho_max / rho_min

        for p in subregion_patches:
            p.remove()

        subregion_patches.clear()

        if hasattr(solver, "useHybrid"):
            if solver.useHybrid:
                subregions = []
                solver.binaryTree.getSubregions(subregions)
                waveSchemeRegionCounter = 0
                for N0, N, isWaveScheme in subregions:
                    if isWaveScheme:
                        c = "red"
                        waveSchemeRegionCounter += 1
                        print(
                            f"Subregion with x {N0[0]}, y {N0[1]}, z {N0[2]}, N {N}")
                        rect1 = patches.Rectangle(
                            (N0[1], N0[0]), N, N, linewidth=1, edgecolor=c, facecolor='none')
                        rect2 = patches.Rectangle(
                            (N0[1], N0[0]), N, N, linewidth=1, edgecolor=c, facecolor='none')
                        pol1 = ax1.add_patch(rect1)
                        pol2 = ax4.add_patch(rect2)
                        subregion_patches.append(rect1)
                        subregion_patches.append(rect2)
                print(f"{waveSchemeRegionCounter} wave scheme subregions with {solver.binaryTree.getWaveVolumeFraction() * 100} percent of volume.")
        t = solver.getTime()
        a = solver.getScaleFactor()

        # Average along z-axis
        density_projection = np.mean(np.log10(density), axis=2)

        if logscaling:
            im1.set_array(getProjection(
                (np.log10(density) + 1) / 4, projectionAxis))
            im2.set_array(getProjection(
                (np.log10(density_ref) + 1)/4, projectionAxis))
            im3.set_array(getProjection(
                np.abs(density - density_ref)
                / (density_ref + 1e-8 * (density_ref == 0)), projectionAxis)
            )
        else:
            im1.set_array(getProjection(density, projectionAxis))
            im2.set_array(getProjection(density_ref, projectionAxis))
            im3.set_array(getProjection(
                np.abs(density - density_ref)
                / (density_ref + 1e-8 * (density_ref == 0)), projectionAxis)
            )

        if mod2Phase:
            im4.set_array(getProjection(
                np.angle(np.exp(1j * phase)), projectionAxis))
            im5.set_array(getProjection(phase_ref, projectionAxis))
        else:
            im4.set_array(getProjection(phase, projectionAxis))
            phase_ref = fd.make_continuous(phase_ref)
            phase_ref += phase[0, 0, 0] - phase_ref[0, 0, 0]
            im5.set_array(getProjection(phase_ref, projectionAxis))

        im6.set_array(getProjection(np.abs(np.angle(np.exp(1j * (phase - phase_ref))) /
                      (np.abs(phase_ref) + 1e-8 * (phase_ref == 0))), projectionAxis))

        current_time = (
            "t = "
            + f"{t:.3f}"
            + f"; a = {a:.3f} \n"
            + r"$\delta / \rho$ = "
            + f"{rho_ratio:.1f}"
            + r" and $\rho_{min}$ = "
            + f"{rho_min:.2e} \n"
            + r"total mass "
            + "= %.3f, %.3f (wave/ana, phase) \nRMS error = .%.3e"
            % (np.mean(density_ref), np.mean(density), rms_error)
        )

        suptitle.set_text(current_time)

        title1.set_text(r"$\log_{10}(|\psi|^2)$ for " + label)

        if waveSolver is None:
            title2.set_text(r"analytical solution")
            title5.set_text(r"analytical solution")
        else:
            title2.set_text(r"wave scheme")
            title5.set_text(r"wave scheme")

        title3.set_text(r"relative density error")

        title4.set_text(r"$angle(\psi)$ for " + label)

        title6.set_text(r"relative phase error")

        print("Save fig")
        plt.savefig(f"plots/3d/{filename}/axis={projectionAxis}.jpg")

        if saveFields:
            np.savez_compressed(
                f"runs/3d/{filename}/{i}.npz",
                config=np.array(list(config.items()), dtype=object),
                t=t,
                a=a,
                density=density,
                phase=phase,
                psi=psi_ref,
            )

        return (im1, im2, im4, im5, im5, im6)

    return fig, init, draw


def createFrame(
    solver,
    label,
    config,
    analyticalSolution,
    filename,
    waveSolver=None,
    advection=False,
    projectionAxis=0,
):
    if config["dimension"] == 1:
        return create1DFrame(
            solver, label, config, analyticalSolution, filename, waveSolver, advection
        )
    elif config["dimension"] == 2:
        return create2DFrame(
            solver, label, config, analyticalSolution, filename, waveSolver, advection
        )
    elif config["dimension"] == 3:
        return create3DFrame(
            solver, label, config, analyticalSolution, filename, waveSolver, advection, projectionAxis=projectionAxis, saveFields=True
        )
    else:
        raise ValueError()


def drawFrame(
    solver,
    analyticalSolution=None,
    label=None,
    filename=None,
    waveSolver=None,
    advection=False,
    projectionAxis=0,
):
    if analyticalSolution is None:
        analyticalSolution = solver.generateIC
    if label is None:
        label = solver.getName()
    if filename is None:
        filename = solver.getName().replace(" ", "_")

    config = solver.config


    fig, init, draw = createFrame(
        solver, label, config, analyticalSolution, filename, waveSolver, advection, projectionAxis
    )
    init()
    draw(0)


def createAnimation(
    solver,
    analyticalSolution=None,
    label=None,
    filename=None,
    waveSolver=None,
    advection=False,
):
    if analyticalSolution is None:
        analyticalSolution = solver.generateIC
    if label is None:
        label = solver.getName()
    if filename is None:
        filename = solver.getName().replace(" ", "_")

    config = solver.config
    dim = config["dimension"]

    # Create folders for run files and plots
    os.makedirs(f"plots/{dim}d/{filename}", exist_ok=True)
    os.makedirs(f"runs/{dim}d/{filename}", exist_ok=True)
    print("filename ", filename)

    fig, init, draw = createFrame(
        solver, label, config, analyticalSolution, filename, waveSolver, advection
    )

    def animate(i):
        # Update solutions
        t = 0
        tfin = getTimePerFrame(config)
        while(True):
            dt = solver.getTimeStep()

            if (t + dt > tfin) and (t < tfin):
                dt = tfin - t
            solver.step(dt)
            if waveSolver is not None:
                waveSolver.step(dt)
            t += dt * solver.getScaleFactor()**2
            if t >= tfin:
                break

        obj = draw(i)
        plt.savefig(f"plots/{dim}d/{filename}/{i}.jpg",
                    bbox_inches="tight", dpi=config["dpi"])
        return obj

    frames = getNumberOfFrames(config)
    interval = getTimePerFrame(config) * 1000
    fps = config["fps"]

    print(
        f"Number of frames: {frames}, time per frame (ms) = {interval} and frames per second {fps}")
    print("Create animation with configuration: ", config)

    # blit=True re-d raws only the parts that have changed.
    anim = animation.FuncAnimation(
        fig, animate, frames=frames, interval=interval, blit=True, init_func=init, repeat_delay = 1000, repeat =True
    )
    writergif = animation.PillowWriter(fps=fps)
    anim.save(f"gifs/{dim}d/{filename}.gif", writer=writergif)


def loadRun(filename):
    data = np.load(filename, allow_pickle=True)
    t, psi, density, phase = (
        data["t"],
        data["psi"],
        data["density"],
        data["phase"],
    )
    config = dict(data["config"])
    config["t0"] = float(t)
    return config, psi, density, phase