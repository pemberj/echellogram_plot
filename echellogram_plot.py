import numpy as np
from matplotlib import pyplot as plt
from PyEchelle import *
import pyzdde.zdde as pyz

# Optional: plot style and font
import matplotlib.font_manager as fm
fm.FontManager.addfont(fm.fontManager, path="./Quicksand-Regular.ttf")
from jakeStyle import plotStyle
plt.style.use(plotStyle)


# Produce the correct colours from wavelength specification
def wavelength_to_rgb(wavelength, gamma=0.8):

    '''
    This converts a given wavelength of light to an
    appropriate RGB color value. The wavelength should be given
    in nanometers in the range from 380 nm through 750 nm
    (789 THz through 400 THz). Wavelengths outside this range
    will return (0,0,0) colour == black.

    Based on code by Dan Bruton
    http://www.physics.sfasu.edu/astro/color/spectra.html
    '''

    if wavelength >= 380 and wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.
        B = (1. * attenuation) ** gamma

    elif wavelength >= 440 and wavelength <= 490:
        R = 0.
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.

    elif wavelength >= 490 and wavelength <= 510:
        R = 0.
        G = 1.
        B = (-(wavelength - 510) / (510 - 490)) ** gamma

    elif wavelength >= 510 and wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.
        B = 0.

    elif wavelength >= 580 and wavelength <= 645:
        R = 1.
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.

    elif wavelength >= 645 and wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.
        B = 0.

    else:
        R = 0.
        G = 0.
        B = 0.

    return (R, G, B)



# Interesting stellar spectral lines
lines = {
    # Balmer lines of Hydrogen
    "H$\\alpha$":   6564.8131,
    "H$\\beta$":    4863.3582,
    "H$\\gamma$":   4341.2202,
    "H$\\delta$":   4101.7103,

    # Sodium
    "Na $D_1$":     5895.92,
    "Na $D_2$":     5889.95,
    "NaI":          8191.2515,

    # Silicon
    "SiIII$_1$":    4552.622,
    "SiIII$_2$":    4567.840,
    "SiIII$_3$":    4574.757,

    # Calcium misc
    "CaI$_{422}$":  4226.7270,
    "CaI$_{616}$":  6164.2055,

    # Calcium-II doublet
    "CaII-K":       3933.6614,
    "CaII-H":       3968.4673,

    # Calcium Triplet
    "CaII$_1$":     8498.02,
    "CaII$_2$":     8542.09,
    "CaII$_3$":     8662.140,

    # Iron
    "FeI$_1$":      4384.8318,
    "FeI$_2$":      4406.0371,
    "FeI$_{869}$":  8691.3867,
    "FeH":          9940,

    # Helium d-line
    "He-d":         5875.618,

    "MgI":          5174.141,

    "Rb-B$_1$":     7929.781,
    "Rb-B$_2$":     7969.7918,

    "CaH2":         6831.862,
    "CaH3":         6976.9239,
}




def trace_lines(line_list, Echelle, FSRonly=True):
    print("Tracing spectral lines...")
    lines_xy = []

    for o in list(Echelle.Orders.values()):
        Echelle.ln.zSetSurfaceParameter(Echelle.zmx_nsurf, 2, o.m)
        Echelle.ln.zPushLens()

        if FSRonly:
            max_wl = o.minFSRwl
            min_wl = o.maxFSRwl
            # NOTE: if orders are defined as negative, FSR wavelengths are backwards
            # i.e. minFSRwl > maxFSRwl
        else:
            min_wl = o.minWL
            max_wl = o.maxWL

        for line_name, wl_angstrom in line_list.items():
            line_wl = wl_angstrom / 10_000. # Get wavelengths in microns

            if line_wl > min_wl and line_wl < max_wl:
                Echelle.ln.zSetWave(1, line_wl, 1.)
                Echelle.ln.zGetUpdate()
                rayTraceData = Echelle.ln.zGetTrace(1, 0, -1, 0, 0, 0, 0)
                if type(rayTraceData) is int:
                    print("error")
                    break
                else:
                    x = rayTraceData[2]
                    y = rayTraceData[3]
                    print(f"{line_name}, {line_wl:.3f}um, found in order {o.m} at ({x},{y})")

                    lines_xy.append([line_name, o.m, line_wl, [x, y]])

    return lines_xy


def plot_echellogram(ax: plt.axis,
                     SpectralFormat: list,
                     lines_xy: list,
                     detector_corners: list):
    """
    Produce a nice plot of the echellogram/spectral layout of the spectrograph,
    with interesting spectral lines labelled

    Args:
        axis (plt.axis): The matplotlib.pyplot.axis object onto which to plot
        SpectralFormat (list): A list or array of coordinates of echelle orders,
                               from PyEchelle
        lines_xy (list): A list of names and coordinates of interesting spectral
                         lines, pre-traced
        detector_corners (list): Locations of the detector corners (mm units)
    """


    working_m = 0
    order_xs, order_ys = [], []
    order_wls = []


    b = abs(detector_corners[0]) * 1.25
    ax.set_xlim(-b, b)
    ax.set_ylim(-b, b)


    for i, point in enumerate(SpectralFormat):
        m = point[0]

        if m != working_m:
            col = wavelength_to_rgb(np.mean(np.array(order_wls) * 1000.))
            if np.sum(col) == 0:
                ax.plot(order_ys, order_xs, color="k", lw=1.2)
            else:
                ax.plot(order_ys, order_xs, color=col, lw=1.2)
            if len(order_xs) > 0 and m%5 == 0:
                if order_xs[0] > ax.get_xlim()[0] and order_xs[0] < ax.get_xlim()[1]:
                    if order_ys[0] > ax.get_ylim()[0] and order_ys[0] + 5 < ax.get_ylim()[1]:
                        text_string = f"{order_wls[0] * 1000.:.0f}"
                        if abs(m) == 90:
                            text_string = "$\lambda=$" + text_string + "nm"
                        text_string = "  " + text_string
                        ax.text(order_ys[0], order_xs[0], text_string, size=12,
                                horizontalalignment="left", verticalalignment="center")

                if order_xs[0] > ax.get_xlim()[0] and order_xs[0] < ax.get_xlim()[1]:
                    if order_ys[0] - 3 > ax.get_ylim()[0] and order_ys[0] < ax.get_ylim()[1]:
                        text_string = f"{abs(m):.0f}"
                        if abs(m) == 90:
                            text_string = "$m=$" + text_string
                        text_string += "  "
                        ax.text(order_ys[-1], order_xs[-1], text_string, size=12,
                                horizontalalignment="right", verticalalignment="center")

            working_m = m
            order_xs, order_ys = [], []
            order_wls = []
            order_xs.append(point[2])
            order_ys.append(point[3])
            order_wls.append(point[1])

        if m == working_m:
            order_xs.append(point[2])
            order_ys.append(point[3])
            order_wls.append(point[1])

        # The very last order (reddest or bluest depending on negative or positive order numbers)
        if i == len(SpectralFormat) - 1:
            order_xs.append(point[2])
            order_ys.append(point[3])
            order_wls.append(point[1])

            col = wavelength_to_rgb(np.mean(np.array(order_wls) * 1000.))
            if np.sum(col) == 0:
                ax.plot(order_ys, order_xs, color="k", lw=1.2)
            else:
                ax.plot(order_ys, order_xs, color=col, lw=1.2)

    for line in lines_xy:
        col = wavelength_to_rgb(line[2] * 1000.)
        if line[3][0] > ax.get_xlim()[0] and line[3][0] < ax.get_xlim()[1]:
            if line[3][1] > ax.get_ylim()[0] and line[3][1] < ax.get_ylim()[1]:
                ax.text(line[3][1], line[3][0], line[0], size=12,
                        horizontalalignment="center", verticalalignment="center",
                        bbox=dict(facecolor="w", edgecolor="w", pad=1., alpha=0.75))
                #ax.plot(line[3][1], line[3][0], markersize=20, marker=".", alpha=0.5, color=col)

    # Detector
    b = abs(detector_corners[0])
    vert_x = [-b, b, b, -b]
    vert_y = [-b, -b, b, b]
    detector = plt.Polygon(np.transpose([vert_x, vert_y]), closed=True, ls="--", fill=None, lw=4, alpha=0.1)
    ax.add_artist(detector)

    # b4k = 30.72
    # vert_x = [-b4k, b4k, b4k, -b4k]
    # vert_y = [-b4k, -b4k, b4k, b4k]
    # detector4k = plt.Polygon(np.transpose([vert_x, vert_y]), closed=True, ls="--", fill=None, lw=4, alpha=0.1)
    # ax.add_artist(detector4k)

    ax.set_xticks([])
    ax.set_yticks([])

    for border in ["top", "bottom", "left", "right"]:
        ax.spines[border].set_linewidth(0)



if __name__ == "__main__":

    ln = pyz.createLink()
    spec = Echelle(ln, "Spectrograph")
    spec.analyseZemaxFile(echellename="Echelle", blazename="Blaze", gammaname="Gamma")

    spec.minord = -35
    spec.maxord = -94
    spec.setCCD(CCD(10560, 10560, 9, name="10k x 10k 9um CCD"))
    detector_corners = spec.CCD.extent

    spec.calc_wl()
    SpectralFormat = spec.do_spectral_format(nPerOrder=11)
    lines_xy = trace_lines(lines, spec, FSRonly=True)

    fig = plt.figure(figsize=(10,10))
    ax = fig.gca()
    plot_echellogram(ax, SpectralFormat, lines_xy, detector_corners)
    plt.show()