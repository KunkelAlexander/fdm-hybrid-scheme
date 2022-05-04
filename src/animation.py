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
    t        = config["t0"]
    tEnd     = config["tEnd"]
    slowDown = config["slowDown"]
    fps      = config["fps"]
    frames = int(np.floor((tEnd - t) * slowDown * fps))
    return frames

def getTimePerFrame(config):
    slowDown = config["slowDown"]
    fps      = config["fps"]
    tUpdate  = 1 / (fps * slowDown)
    return tUpdate


"""""" """""" """""" """""" """"""
"""""" """""  1D    """ """""" ""
"""""" """""" """""" """""" """"""


def create1DFrame(
    solver,
    label,
    config,
    analyticalSolution,
    filename,
    waveSolver=None, 
    advection = False, 
):
    xlim            = config["xlim"]
    ylim            = config["densityYlim"]
    ylim2           = config["phaseYlim"]
    mod2Phase       = config["plotPhaseMod2"]
    useAdaptiveYlim = config["useAdaptiveYlim"]

    # prep figure
    fig = plt.figure(figsize=(9, 18), dpi=config["dpi"])
    grid = plt.GridSpec(3, 1, wspace=0.0, hspace=0.2)
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[1, 0])
    ax3 = fig.add_subplot(grid[2, 0])

    # Switch to axis 1
    plt.sca(ax1)
    # Clear axis
    plt.cla()
    # Plot background
    (im1,) = plt.plot([], [], "ro", ms=3, label=label)
    if waveSolver is None:
        label2 = "analytical solution"
    else:
        label2 = "wave scheme"
    (im3,) = plt.plot([], [], "b", ms=1, label=label2)

    # highlight1 = plt.axvspan(0, 0, color='black', alpha=0.3)
    plt.legend()

    # Switch to axis 1
    plt.sca(ax2)
    # Clear axis
    plt.cla()
    # Plot background
    (im2,) = plt.plot([], [], "ro", ms=3)
    (im4,) = plt.plot([], [], "bo", ms=1)
    # highlight2 = plt.axvspan(0, 0, color='black', alpha=0.3)


    # Switch to axis 1
    plt.sca(ax3)
    # Clear axis
    plt.cla()
    # Plot background
    (im5,) = plt.plot([], [], "ro", ms=3, label = "density")
    (im6,) = plt.plot([], [], "bo", ms=1, label = "phase")
    plt.legend(loc = "upper right")
    # highlight2 = plt.axvspan(0, 0, color='black', alpha=0.3)
    # Set axis ratios

    # Set axis ratios
    ax1.set_xlim(xlim)
    ax1.set_title(r"$|\psi|^2$")
    ax2.set_xlim(xlim)
    ax2.set_title(r"${\rm angle}(\psi)$")
    ax3.set_xlim(xlim)
    ax3.set_title(r"relative error")
    if not useAdaptiveYlim:
        ax1.set_ylim(ylim)
        ax2.set_ylim(ylim2)
    ax3.set_ylim([0, .1])


    time_text = ax3.text(0.02, 0.95, "", transform=ax3.transAxes)
    norm_text = ax3.text(0.02, 0.90, "", transform=ax3.transAxes)

    # initialization function: plot the background of each frame

    def init():
        ax1.set_title(r"$|\psi|^2$")
        ax2.set_title(r"${\rm angle}(\psi)$")
        ax3.set_title(r"relative error")

        im1.set_data([], [])
        im2.set_data([], [])
        im3.set_data([], [])
        im4.set_data([], [])
        im5.set_data([], [])
        im6.set_data([], [])

        time_text.set_text("")
        norm_text.set_text("")

        subregion_patches = []

        # , highlight1, highlight2
        return im1, im2, im3, im4, im5, im6, *subregion_patches, time_text, norm_text

    def draw(i):
        # Get fluid solution
        density = solver.getDensity()
        phase = solver.getPhase()
        t = solver.getTime()
        a = solver.getScaleFactor()
        xx = solver.getGrid()[0]
        dx = solver.dx

        im1.set_data(xx, density)

        if mod2Phase:
            im2.set_data(xx, np.angle(np.exp(1j * phase)))
        else:
            im2.set_data(xx, phase)

        # Get phase or analytical solution
        if waveSolver is None:
            if advection:
                psi_ref, phase_ref = analyticalSolution(xx, dx, t)
                density_ref = psi_ref 
            else:
                psi_ref = analyticalSolution(xx, dx, t)
                density_ref = np.abs(psi_ref) ** 2
        else:
            psi_ref = waveSolver.getPsi()
            density_ref = np.abs(psi_ref) ** 2


        if not advection:
            if mod2Phase:
                phase_ref = np.angle(psi_ref)
            else:
                phase_ref = fd.make_1d_continuous(np.angle(psi_ref))

        rms_error = 0

        if t != 0:
            rms_error = np.sqrt(np.sum((density[solver.inner]- density_ref[solver.inner]) ** 2)) / (
                (solver.t / solver.dt) * solver.innerN ** (solver.dimension)
            )
        im3.set_data(xx, density_ref)
        im4.set_data(xx, phase_ref)

        density_relerr = np.abs(density - density_ref)/(density_ref + 1e-8 * (density_ref == 0))
        phase_relerr   = np.abs(phase   - phase_ref)  /(phase_ref + 1e-8 * (phase_ref == 0))

        im5.set_data(xx, density_relerr)
        im6.set_data(xx, phase_relerr)


        subregion_patches = []

        if hasattr(solver, "useHybrid"):
            if solver.useHybrid:
                for subregion in solver.subregions:
                    pos = subregion.outerInner[0]
                    if subregion.isWaveScheme:
                        c = "blue"
                    else:
                        c = "gray"
                    pol1 = ax1.axvspan(pos[0] * solver.dx, pos[-1] * solver.dx, color=c, alpha=0.3)
                    pol2 = ax2.axvspan(pos[0] * solver.dx, pos[-1] * solver.dx, color=c, alpha=0.3)
                    pol3 = ax3.axvspan(pos[0] * solver.dx, pos[-1] * solver.dx, color=c, alpha=0.3)
                    subregion_patches.append(pol1)
                    subregion_patches.append(pol2)
                    subregion_patches.append(pol3)

        time_text.set_text("Time t = %.5f" % (t))
        norm_text.set_text(
            r"$\int \mathrm{d}x|\psi|^2$"
            +" = %.3f, %.3f (wave/ana, phase), RMS error = .%.3e"
            % (np.mean(density_ref), np.mean(density), rms_error)
        )

        if os.path.exists(f"plots/1d/{filename}/"):
            plt.savefig(f"plots/1d/{filename}.jpg")

        if os.path.exists(f"runs/1d/{filename}/"):
            np.savez_compressed(
                f"runs/1d/{filename}/{i}.npz",
                config=np.array(list(config.items()), dtype=object),
                t=t,
                a=a,
                density=density,
                phase=phase,
                psi=psi_ref,
            )

        # , highlight1, highlight2
        return im1, im2, im3, im4, im5, im6, *subregion_patches, time_text, norm_text

    return fig, init, draw


"""""" """""" """""" """""" """"""
"""""" """""  2D    """ """""" ""
"""""" """""" """""" """""" """"""

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def create2DFrame(
    solver,
    label,
    config,
    analyticalSolution,
    filename,
    waveSolver=None,
    advection = False, 
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
    im1 = ax1.imshow(np.random.random((N, N)), cmap="inferno", vmin = ll, vmax = hl)
    title1 = ax1.set_title(r"$\log_{10}(|\psi|^2)$ at t = $0$")
    im2 = ax2.imshow(np.random.random((N, N)), cmap="inferno", vmin = ll, vmax = hl)
    title2 = ax2.set_title(r"${\rm angle}(\psi)$ at t = $0$")
    im3 = ax3.imshow(np.random.random((N, N)), cmap="gray", vmin = 0, vmax = 1)
    title3 = ax3.set_title(r"${\rm angle}(\psi)$ at t = $0$")

    im4 = ax4.imshow(np.random.random((N, N)), cmap="inferno", vmin=pl, vmax=ph)
    title4 = ax4.set_title(r"${\rm angle}(\psi)$ at t = $0$")
    im5 = ax5.imshow(np.random.random((N, N)), cmap="inferno", vmin=pl, vmax=ph)
    title5 = ax5.set_title(r"${\rm angle}(\psi)$ at t = $0$")
    im6 = ax6.imshow(np.random.random((N, N)), cmap="gray", vmin=0, vmax=1)
    title6 = ax6.set_title(r"${\rm angle}(\psi)$ at t = $0$")

    suptitle = ax6.text(.05, .8, "", fontsize=10, c="w", bbox=dict(facecolor='gray', alpha=0.5), transform=ax.transAxes)

    cbar1 = plt.colorbar(im1, cax=grid.cbar_axes[0], ticks = [0, 0.25, 0.5, 0.75, 1])
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
        density = solver.getDensity()
        phase   = solver.getPhase()
        xx, yy  = solver.getGrid()
        t = solver.getTime()
        a = solver.getScaleFactor()
        dx = solver.dx

        if waveSolver is None:
            if advection:
                psi_ref, phase_ref = analyticalSolution(xx, yy, dx, solver.t)
                density_ref = psi_ref
        
            else:
                psi_ref = analyticalSolution(xx, yy, dx, solver.t)
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
                for N0, N, isWaveScheme in subregions:
                    if isWaveScheme:
                        c = "red"
                        rect1 = patches.Rectangle(np.flip(N0), N, N, linewidth=1, edgecolor=c, facecolor='none')
                        rect2 = patches.Rectangle(np.flip(N0), N, N, linewidth=1, edgecolor=c, facecolor='none')
                        pol1 = ax1.add_patch(rect1)
                        pol2 = ax4.add_patch(rect2)
                        subregion_patches.append(rect1)
                        subregion_patches.append(rect2)
#
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

        im6.set_array(np.abs(np.angle(np.exp(1j * (phase - phase_ref)))/(np.abs(phase_ref) + 1e-8 * (phase_ref == 0))))

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

        plt.savefig(f"plots/2d/{filename}.jpg")

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

#Plot average density in x-y-plane as projection along the z-axis
def getProjection(f, axis=2):
    return np.mean(f, axis=axis)

def create3DFrame(
    solver,
    label,
    config,
    analyticalSolution,
    filename,
    waveSolver=None,
    advection = False, 
    projectionAxis = 2,
    saveFields = False
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
    im1 = ax1.imshow(np.random.random((N, N)), cmap="inferno", vmin = ll, vmax = hl)
    title1 = ax1.set_title(r"$\log_{10}(|\psi|^2)$ at t = $0$")
    im2 = ax2.imshow(np.random.random((N, N)), cmap="inferno", vmin = ll, vmax = hl)
    title2 = ax2.set_title(r"${\rm angle}(\psi)$ at t = $0$")
    im3 = ax3.imshow(np.random.random((N, N)), cmap="gray", vmin = 0, vmax = 1)
    title3 = ax3.set_title(r"${\rm angle}(\psi)$ at t = $0$")

    im4 = ax4.imshow(np.random.random((N, N)), cmap="inferno", vmin=pl, vmax=ph)
    title4 = ax4.set_title(r"${\rm angle}(\psi)$ at t = $0$")
    im5 = ax5.imshow(np.random.random((N, N)), cmap="inferno", vmin=pl, vmax=ph)
    title5 = ax5.set_title(r"${\rm angle}(\psi)$ at t = $0$")
    im6 = ax6.imshow(np.random.random((N, N)), cmap="gray", vmin=0, vmax=1)
    title6 = ax6.set_title(r"${\rm angle}(\psi)$ at t = $0$")

    suptitle = ax6.text(.05, .8, "", fontsize=10, c="w", bbox=dict(facecolor='gray', alpha=0.5), transform=ax.transAxes)

    cbar1 = plt.colorbar(im1, cax=grid.cbar_axes[0], ticks = [0, 0.25, 0.5, 0.75, 1])
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
        density     = solver.getDensity().copy().astype(float)
        phase       = solver.getPhase().copy().astype(float)
        xx, yy, zz  = solver.getGrid()
        t = solver.getTime()
        a = solver.getScaleFactor()
        dx = solver.dx

        if waveSolver is None:
            if advection:
                psi_ref, phase_ref = analyticalSolution(xx, yy, zz, dx, solver.t)
                density_ref = psi_ref
        
            else:
                psi_ref = analyticalSolution(xx, yy, zz, dx, solver.t)
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
                        print(f"Subregion with x {N0[0]}, y {N0[1]}, z {N0[2]}, N {N}")
                        rect1 = patches.Rectangle((N0[1], N0[0]), N, N, linewidth=1, edgecolor=c, facecolor='none')
                        rect2 = patches.Rectangle((N0[1], N0[0]), N, N, linewidth=1, edgecolor=c, facecolor='none')
                        pol1 = ax1.add_patch(rect1)
                        pol2 = ax4.add_patch(rect2)
                        subregion_patches.append(rect1)
                        subregion_patches.append(rect2)
                print(f"{waveSchemeRegionCounter} wave scheme subregions with {solver.binaryTree.getWaveVolumeFraction() * 100} percent of volume.")
        t = solver.getTime()
        a = solver.getScaleFactor()

        #Average along z-axis
        density_projection = np.mean(np.log10(density), axis = 2)

        if logscaling:
            im1.set_array(getProjection((np.log10(density) + 1) / 4, projectionAxis))
            im2.set_array(getProjection((np.log10(density_ref) + 1)/4, projectionAxis))
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
            im4.set_array(getProjection(np.angle(np.exp(1j * phase)), projectionAxis))
            im5.set_array(getProjection(phase_ref, projectionAxis))
        else:
            im4.set_array(getProjection(phase, projectionAxis))
            phase_ref = fd.make_continuous(phase_ref)
            phase_ref += phase[0, 0, 0] - phase_ref[0, 0, 0]
            im5.set_array(getProjection(phase_ref, projectionAxis))

        im6.set_array(getProjection(np.abs(np.angle(np.exp(1j * (phase - phase_ref)))/(np.abs(phase_ref) + 1e-8 * (phase_ref == 0))), projectionAxis))

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
    advection = False,
    projectionAxis = 0,
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
    label = None,
    filename = None,
    waveSolver=None,
    advection=False, 
    projectionAxis = 0,
):
    if analyticalSolution == None:
        analyticalSolution = solver.generateIC
    if label == None:
        label = solver.getName()
    if filename == None:
        filename = solver.getName().replace(" ", "_")

    config = solver.config

    fig, init, draw = createFrame(
        solver, label, config, analyticalSolution, filename, waveSolver, advection, projectionAxis
    )
    init()
    draw(0)


def createAnimation(
    solver,
    analyticalSolution = None,
    label = None,
    filename = None,
    waveSolver=None,
    advection=False, 
    ):
    if analyticalSolution == None:
        analyticalSolution = solver.generateIC
    if label == None:
        label = solver.getName()
    if filename == None:
        filename = solver.getName().replace(" ", "_")

    config = solver.config
    dim = config["dimension"]

    #Create folders for run files and plots 
    os.makedirs(f"plots/{dim}d/{filename}", exist_ok=True)
    os.makedirs(f"runs/{dim}d/{filename}", exist_ok=True)


    fig, init, draw = createFrame(
        solver, label, config, analyticalSolution, filename, waveSolver, advection
    )


    def animate(i):
        # Update solutions
        t    = 0
        tfin = getTimePerFrame(config)
        while(True):
            dt = solver.getTimeStep()

            if (t + dt > tfin) and (t < tfin):
                dt = tfin - t
            solver.step(dt)
            if waveSolver is not None:
                waveSolver.step(dt)
            t += dt 
            if t >= tfin:
                 break

        obj = draw(i)
        plt.savefig(f"plots/{dim}d/{filename}/{i}.jpg", bbox_inches="tight", dpi=config["dpi"])
        return obj

    frames   = getNumberOfFrames(config)
    interval = getTimePerFrame(config) * 1000
    fps      = config["fps"]

    print(f"Number of frames: {frames}, time per frame (ms) = {interval} and frames per second {fps}")
    print("Create animation with configuration: ", config)

    # blit=True re-d raws only the parts that have changed.
    anim = animation.FuncAnimation(
        fig, animate, frames=frames, interval=interval, blit=True, init_func=init
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


def hinton(matrix, max_weight=None, ax=None):
    """Draw Hinton diagram for visualizing a weight matrix."""
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = 2 ** np.ceil(np.log2(np.abs(matrix).max()))

    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x, y), w in np.ndenumerate(matrix):
        color = 'white' if w > 0 else 'black'
        size = np.sqrt(abs(w) / max_weight)
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()


#if __name__ == '__main__':
#    # Fixing random state for reproducibility
#    np.random.seed(19680801)
#
#    hinton(np.random.rand(20, 20) - 0.5)